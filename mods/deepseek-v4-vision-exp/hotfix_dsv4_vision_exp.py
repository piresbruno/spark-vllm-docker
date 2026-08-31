#!/usr/bin/env python3
"""Install native DeepSeek-V4-Flash-Vision-Exp image support into the vLLM install.

Port of MiaAI-Lab DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
``patches/hotfix-dsv4-vision-exp.py`` (feat/vision-exp, PR #164) onto the
spark-vllm-docker B12X serving stack. Upstream targets the Anemll 0.1.1
image; this port keeps the patch logic and fail-closed drift detection but
locates the target files inside the vLLM package instead of assuming
Anemll's paths.

The stock ``DeepseekV4ForCausalLM`` is text-only. Vision-Exp ships a
32-layer ViT + Aligner and ``<｜deepseek_image｜>`` prompt tokens. This
startup patch:

1. Appends a fail-closed import hook to the DeepSeek-V4 model module that
   constructs the tower, maps ``vision.*`` / ``aligner.*`` / ``image_*`` /
   ``bias_vl`` weights, and registers a vLLM multimodal processor.
2. Remaps DSpark draft ``ffn.gate.bias_vl`` →
   ``e_score_correction_bias_vl`` (the stock loader only rewrote names
   ending in ``.ffn.gate.bias``).
3. When the encoder carries Anemll-style image-placeholder rejection, relax
   it so OpenAI ``image_url`` parts survive the chat parser and enforce the
   official rule (images in ``user`` messages only). On a stock encoder —
   e.g. the B12X stack's ``deepseek_v4_encoding.py``, which has no such
   logic — the encoding patch is not applicable and is skipped: there is
   nothing to relax, and the role restriction is then not hard-enforced by
   the server.

Video is not wired: the official weights, ``encoding/``, and ``inference/``
have no video encoder. GIF is decoded as a still RGB frame.

Modes:
  (no args)   Locate targets, apply the patch (idempotent), fail on drift.
  --status    Exit 0 only when every patch marker is present.
  --locate    Print the resolved target paths without patching.

Environment overrides:
  VLLM_PACKAGE_ROOT          vLLM package directory (skips discovery).
  VLLM_SITE_PACKAGES         Site-packages root containing ``vllm/``.
  VISION_EXP_PATCHES_DIR     Directory holding the ``vision_exp`` package
                             (default: next to this script).
  VISION_EXP_MODEL_FILE      Explicit DeepSeek-V4 model module path.
  VISION_EXP_ENCODING_FILE   Explicit deepseek_v4 encoding module path.
  VISION_EXP_DSPARK_FILE     Explicit DSpark draft module path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PATCHES_DIR_ENV = "VISION_EXP_PATCHES_DIR"
MODEL_FILE_ENV = "VISION_EXP_MODEL_FILE"
ENCODING_FILE_ENV = "VISION_EXP_ENCODING_FILE"
DSPARK_FILE_ENV = "VISION_EXP_DSPARK_FILE"

MODEL_MARK = "# [vision-exp-hotfix] native DeepSeek-V4-Flash-Vision-Exp image tower"
ENC_MARK = "# [vision-exp-hotfix] allow vLLM-inserted image placeholders"
ENC_ROLE_MARK = "# [vision-exp-hotfix] images only in user messages"
DSPARK_MARK = "# [vision-exp-hotfix] remap ffn.gate.bias_vl"

DSPARK_GATE_BIAS_OLD = '''                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]'''

DSPARK_GATE_BIAS_NEW = f'''                if name.endswith(".ffn.gate.bias_vl"):
                    name = name.replace(
                        ".ffn.gate.bias_vl",
                        ".ffn.gate.e_score_correction_bias_vl",
                    )
                elif name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                if name not in params_dict:
                    continue  {DSPARK_MARK}
                param = params_dict[name]'''

CONTENT_CHECK = (
    "if isinstance(content, str) and IMAGE_PLACEHOLDER in content:"
)
CONTENT_CHECK_NEW = (
    "if False and isinstance(content, str) and IMAGE_PLACEHOLDER in content:"
    f"  {ENC_MARK}"
)
REASONING_CHECK = (
    "if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:"
)
REASONING_CHECK_NEW = (
    "if False and isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:"
    f"  {ENC_MARK}"
)
TEXT_CHECK = "if IMAGE_PLACEHOLDER in text:"
TEXT_CHECK_NEW = f"if False and IMAGE_PLACEHOLDER in text:  {ENC_MARK}"


def model_inject(patches_dir: Path) -> str:
    parent = str(patches_dir.parent)
    return f'''
{MODEL_MARK}
import sys as _dspark_vision_sys
if {parent!r} not in _dspark_vision_sys.path:
    _dspark_vision_sys.path.insert(0, {parent!r})
from vision_exp.apply import apply_vision_exp as _dspark_apply_vision_exp
_dspark_apply_vision_exp(
    DeepseekV4Model=DeepseekV4Model,
    DeepseekV4ForCausalLM=DeepseekV4ForCausalLM,
    DeepseekV4MoE=DeepseekV4MoE,
)
'''


ENC_ROLE_INJECT = f'''
{ENC_ROLE_MARK}
def _dspark_vision_value_has_image(value) -> bool:
    if isinstance(value, str):
        return IMAGE_PLACEHOLDER in value or "<image>" in value
    if not isinstance(value, list):
        return False
    for block in value:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("image", "image_url"):
            return True
        text = block.get("text") or ""
        if isinstance(text, str) and (
            IMAGE_PLACEHOLDER in text or "<image>" in text
        ):
            return True
        nested = block.get("content")
        if isinstance(nested, list) and _dspark_vision_value_has_image(nested):
            return True
    return False


def _validate_no_image_sp_tokens(msg):
    """Official restriction: images in user messages only (system/assistant → 400)."""
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError(
            "reasoning_content contains image special token "
            + repr(IMAGE_PLACEHOLDER)
        )
    role = msg.get("role")
    if role in ("user", "developer"):
        return
    if _dspark_vision_value_has_image(msg.get("content")) or _dspark_vision_value_has_image(
        msg.get("content_blocks")
    ):
        raise ValueError(
            "Images are supported in user messages only: "
            "images in " + repr(role) + " messages return a 400 error."
        )
'''


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def vllm_package_root() -> Path:
    """Locate the installed vLLM package without importing it.

    Importing vLLM initializes CUDA, which must not happen while cluster
    containers are still being prepared, so discovery is filesystem-only.
    """
    root = _env_path("VLLM_PACKAGE_ROOT")
    if root is not None:
        return root
    site = _env_path("VLLM_SITE_PACKAGES")
    if site is not None:
        return site / "vllm"
    spec = __import__("importlib.util", fromlist=["find_spec"]).find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        return Path(spec.submodule_search_locations[0])
    return Path("/usr/local/lib/python3.12/dist-packages/vllm")


def _candidates(
    root: Path,
    patterns: tuple[str, ...],
    required_alternatives: tuple[tuple[str, ...], ...],
) -> tuple[list[Path], list[Path]]:
    """Files matching a name pattern, split by content validation.

    The first list holds files that contain all snippets of any required
    set; the alternative sets keep re-discovery working after a first
    application, when the patched file carries the patch markers instead
    of the original snippet it replaced (e.g. the dspark gate-bias remap).
    The second list holds name matches whose content did not validate;
    callers hand those to the patcher so drift produces a precise error
    instead of an opaque "no target" failure.
    """
    content_matches: list[Path] = []
    name_matches: list[Path] = []
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern)):
            if not candidate.is_file() or candidate in content_matches or candidate in name_matches:
                continue
            try:
                text = candidate.read_text()
            except OSError:
                continue
            if any(
                all(snippet in text for snippet in required)
                for required in required_alternatives
            ):
                content_matches.append(candidate)
            else:
                name_matches.append(candidate)
    return content_matches, name_matches


def _discover(
    root: Path,
    env_name: str,
    patterns: tuple[str, ...],
    required_alternatives: tuple[tuple[str, ...], ...],
) -> Path:
    explicit = _env_path(env_name)
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(
                f"FATAL: {env_name}={explicit} does not exist or is not a file"
            )
        return explicit
    content_matches, name_matches = _candidates(
        root, patterns, required_alternatives
    )
    if content_matches:
        if len(content_matches) > 1:
            listing = "\n  ".join(str(match) for match in content_matches)
            raise SystemExit(
                f"FATAL: ambiguous {env_name} targets under {root}:\n  {listing}\n"
                f"Set {env_name} to pick one."
            )
        return content_matches[0]
    if name_matches:
        # A file exists at an expected location but its content drifted.
        # Return it so the patch functions fail with the specific drift
        # status instead of an opaque discovery error.
        return name_matches[0]
    raise SystemExit(
        f"FATAL: no {env_name} target under {root} matching {patterns} "
        f"with required content {required_alternatives!r}. The installed vLLM does "
        "not appear to carry the DeepSeek-V4 modules this hotfix "
        "patches; set the env override to the exact file path."
    )


def locate_targets() -> tuple[Path, Path, Path, Path]:
    patches = _env_path(PATCHES_DIR_ENV) or (
        Path(__file__).resolve().parent / "vision_exp"
    )
    root = vllm_package_root()
    model = _discover(
        root,
        MODEL_FILE_ENV,
        (
            "models/deepseek_v4/nvidia/model.py",
            "model_executor/models/deepseek_v4/nvidia/model.py",
            "model_executor/models/deepseek_v4*/model.py",
            "models/deepseek_v4*/**/model.py",
        ),
        (
            ("class DeepseekV4ForCausalLM", "class DeepseekV4MoE"),
            (MODEL_MARK, "class DeepseekV4ForCausalLM", "class DeepseekV4MoE"),
        ),
    )
    encoding = _discover(
        root,
        ENCODING_FILE_ENV,
        (
            "tokenizers/deepseek_v4_encoding.py",
            "tokenizers/deepseek_v4*/encoding.py",
            "**/deepseek_v4*encoding*.py",
        ),
        (
            ("IMAGE_PLACEHOLDER",),
            ("IMAGE_PLACEHOLDER", ENC_MARK, ENC_ROLE_MARK),
        ),
    )
    dspark = _discover(
        root,
        DSPARK_FILE_ENV,
        (
            "models/deepseek_v4/nvidia/dspark.py",
            "model_executor/models/deepseek_v4/nvidia/dspark.py",
            "models/deepseek_v4*/**/dspark.py",
        ),
        (
            (DSPARK_GATE_BIAS_OLD,),
            (DSPARK_MARK, DSPARK_GATE_BIAS_NEW),
        ),
    )
    return patches, model, encoding, dspark


def patch_model_text(source: str, patches_dir: Path) -> tuple[str, str]:
    if MODEL_MARK in source:
        return source, "skipped"
    if "class DeepseekV4ForCausalLM" not in source or "class DeepseekV4MoE" not in source:
        return source, "drift:missing-dsv4-class"
    updated = source.rstrip() + "\n" + model_inject(patches_dir)
    compile(updated, "model.py", "exec")
    return updated, "applied"


def patch_encoding_text(source: str) -> tuple[str, str]:
    if (
        ENC_MARK in source
        and ENC_ROLE_MARK in source
        and CONTENT_CHECK_NEW in source
    ):
        return source, "skipped"
    if "IMAGE_PLACEHOLDER" not in source:
        # Stock encoder (no Anemll-style image-placeholder rejection).
        # Upstream relaxed that rejection so image_url parts survive the
        # chat parser; a stock encoder has nothing to relax and also does
        # not call the injected role validator, so patching is a no-op.
        return source, "not-applicable:stock-encoder"
    if ENC_MARK not in source:
        missing = []
        for old, new in (
            (CONTENT_CHECK, CONTENT_CHECK_NEW),
            (REASONING_CHECK, REASONING_CHECK_NEW),
            (TEXT_CHECK, TEXT_CHECK_NEW),
        ):
            if source.count(old) != 1:
                missing.append(f"{old!r}={source.count(old)}")
                continue
            source = source.replace(old, new, 1)
        if missing:
            return source, "drift:" + ",".join(missing)
    if ENC_ROLE_MARK not in source:
        source = source.rstrip() + "\n" + ENC_ROLE_INJECT
    compile(source, "encoding.py", "exec")
    return source, "applied"


def patch_dspark_text(source: str) -> tuple[str, str]:
    if DSPARK_MARK in source and DSPARK_GATE_BIAS_NEW in source:
        return source, "skipped"
    if source.count(DSPARK_GATE_BIAS_OLD) != 1:
        return source, (
            "drift:dspark-gate-bias-remap="
            f"{source.count(DSPARK_GATE_BIAS_OLD)}"
        )
    updated = source.replace(DSPARK_GATE_BIAS_OLD, DSPARK_GATE_BIAS_NEW, 1)
    compile(updated, "dspark.py", "exec")
    return updated, "applied"


def _write(path: Path, original: str, updated: str, status: str) -> None:
    if status == "applied":
        path.write_text(updated)
    elif status != "skipped" and not status.startswith("not-applicable"):
        raise SystemExit(f"FATAL: {path} {status}")
    print(f"vision-exp hotfix {path.name:40s}: {status}")


def _read_if_file(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        patches, model, encoding, dspark = locate_targets()
        model_src = _read_if_file(model)
        encoding_src = _read_if_file(encoding)
        dspark_src = _read_if_file(dspark)
        print(
            "vision-exp model.py                    :",
            "APPLIED" if MODEL_MARK in model_src else "NOT APPLIED",
        )
        if "IMAGE_PLACEHOLDER" in encoding_src:
            enc_ok = ENC_MARK in encoding_src and ENC_ROLE_MARK in encoding_src
            print(
                "vision-exp encoding.py                 :",
                "APPLIED" if enc_ok else "NOT APPLIED",
            )
        else:
            enc_ok = True
            print(
                "vision-exp encoding.py                 :",
                "N/A (stock encoder: no image-placeholder logic)",
            )
        print(
            "vision-exp dspark.py                   :",
            "APPLIED" if DSPARK_MARK in dspark_src else "NOT APPLIED",
        )
        ok = MODEL_MARK in model_src and enc_ok and DSPARK_MARK in dspark_src
        return 0 if ok else 1

    if len(sys.argv) > 1 and sys.argv[1] == "--locate":
        patches, model, encoding, dspark = locate_targets()
        print(f"patches : {patches}")
        print(f"model   : {model}")
        print(f"encoding: {encoding}")
        print(f"dspark  : {dspark}")
        return 0

    patches, model_path, encoding_path, dspark_path = locate_targets()

    if not (patches / "apply.py").is_file() or not (patches / "vision.py").is_file():
        print(f"FATAL: Vision-Exp overlay missing under {patches}", file=sys.stderr)
        return 1

    model_src = model_path.read_text()
    model_new, model_status = patch_model_text(model_src, patches)
    _write(model_path, model_src, model_new, model_status)

    enc_src = encoding_path.read_text()
    enc_new, enc_status = patch_encoding_text(enc_src)
    _write(encoding_path, enc_src, enc_new, enc_status)

    dspark_src = dspark_path.read_text()
    dspark_new, dspark_status = patch_dspark_text(dspark_src)
    _write(dspark_path, dspark_src, dspark_new, dspark_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
