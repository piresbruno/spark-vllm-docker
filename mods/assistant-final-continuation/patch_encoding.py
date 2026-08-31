#!/usr/bin/env python3
"""Emit a generation header when a request ends with a closed assistant turn
(optionally followed by a trailing latest_reminder) — DeepSeek-V4.

Port of the MiaAI-Lab issue #52 fix onto the eugr/spark-arena B12X image:

  <site-packages>/vllm/tokenizers/deepseek_v4_encoding.py

Without a generation header the prompt ends on a bare EOS (or a bare reminder
after the closed turn) and the model generates from a dead state: immediate
EOS, or raw DSML markup emitted as text. The fix extends the
"append Assistant + thinking token" transition to an assistant-final turn and
to a latest_reminder whose immediate predecessor is an assistant turn.

Fail-closed: after writing, the patched encoder is imported and three shapes
are rendered to confirm the header terminates assistant-final renders and that
a user->latest_reminder tail stays untouched; on any failure the original bytes
are restored and the script exits nonzero.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

MARK = "[assistant-final-hotfix]"

OLD = '''    elif messages[index].get("role") in ["user", "developer"]:
        # Normal generation: append Assistant + thinking token
'''

NEW = '''    elif messages[index].get("role") in ["user", "developer"] or (
        # [assistant-final-hotfix] A request may legitimately end with an assistant turn
        # (harness retry, continuation), optionally annotated by a
        # trailing latest_reminder harness message. Without a generation
        # header the prompt ends on a bare EOS (or a bare reminder after
        # the closed turn) and the model generates from a dead state:
        # immediate EOS, or raw DSML markup emitted as text. A reminder
        # tail directly after user/developer already ends inside the
        # pending generation slot and must stay byte-identical.
        messages[index].get("role") == "assistant"
        and index == len(messages) - 1
    ) or (
        messages[index].get("role") == "latest_reminder"
        and index == len(messages) - 1
        and index > 0
        and messages[index - 1].get("role") == "assistant"
    ):
        # Normal generation: append Assistant + thinking token
'''


def apply_text(src: str) -> tuple[str, str]:
    if MARK in src:
        return src, "skipped"
    if OLD not in src:
        return src, "missing"
    return src.replace(OLD, NEW, 1), "applied"


def _self_check(target: Path) -> tuple[bool, str]:
    """Import the patched encoder and confirm the fix renders correctly."""
    spec = importlib.util.spec_from_file_location("enc_check", target)
    if spec is None or spec.loader is None:
        return False, "cannot load module spec"
    enc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(enc)
        base = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "A finished answer."},
        ]
        rendered = enc.encode_messages(base, "thinking", reasoning_effort="high")
        reminded = enc.encode_messages(
            base + [{"role": "latest_reminder", "content": "Fresh context."}],
            "thinking",
            reasoning_effort="high",
        )
        intact = enc.encode_messages(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "latest_reminder", "content": "Fresh context."},
            ],
            "thinking",
            reasoning_effort="high",
        )
        speaker = getattr(enc, "ASSISTANT_SP_TOKEN", None)
        thinking = getattr(enc, "thinking_start_token", None)
    except Exception as err:
        return False, f"self-check raised {type(err).__name__}: {err}"
    if not speaker or not thinking:
        return False, "generation-header tokens are unavailable"

    def _ends_with_header(out) -> bool:
        if isinstance(out, str):
            return out.endswith(speaker + thinking)
        if isinstance(out, (list, tuple)):
            return list(out[-2:]) == [speaker, thinking]
        raise TypeError(f"unexpected encoder output type: {type(out).__name__}")

    try:
        valid = _ends_with_header(rendered) and _ends_with_header(reminded)
        untouched = intact.count(speaker) == 1 and not _ends_with_header(intact)
    except TypeError as err:
        return False, str(err)
    if not valid:
        return False, f"generation header does not terminate render: {rendered[-80:]!r}"
    if not untouched:
        return False, "transition widened too far: user->latest_reminder tail changed"
    return True, "generation headers terminate assistant-final renders"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate compatibility without writing"
    )
    args = parser.parse_args()

    if not args.target.is_file():
        print(
            "[assistant-final-continuation ERROR] target not found: %s" % args.target,
            file=sys.stderr,
        )
        return 1

    original = args.target.read_text(encoding="utf-8")
    patched, status = apply_text(original)

    if args.check:
        if status == "missing":
            print(
                "[assistant-final-continuation ERROR] incompatible %s: anchor not found"
                % args.target,
                file=sys.stderr,
            )
            return 1
        state = "already patched" if status == "skipped" else "compatible"
        print(f"[assistant-final-continuation] {args.target} is {state}.")
        return 0

    if status == "skipped":
        ok, why = _self_check(args.target)
        if not ok:
            print(
                "[assistant-final-continuation ERROR] already patched but self-check failed: %s"
                % why,
                file=sys.stderr,
            )
            return 1
        print("[assistant-final-continuation] already patched; skipping.")
        return 0

    if status == "missing":
        print(
            "[assistant-final-continuation ERROR] refusing to patch %s: anchor not found"
            % args.target,
            file=sys.stderr,
        )
        return 1

    args.target.write_text(patched, encoding="utf-8")
    ok, why = _self_check(args.target)
    if ok:
        print(f"[assistant-final-continuation] Patched {args.target}.")
        return 0

    # Fail closed: restore the original bytes and never leave a bad patch.
    args.target.write_text(original, encoding="utf-8")
    print(
        "[assistant-final-continuation ERROR] self-check failed, original restored (%s): %s"
        % (args.target, why),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
