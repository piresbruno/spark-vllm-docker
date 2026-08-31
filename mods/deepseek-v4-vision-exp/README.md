# DeepSeek-V4-Flash-Vision-Exp native image hotfix

Port of MiaAI-Lab `DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` `feat/vision-exp`
(PR #164, `patches/hotfix-dsv4-vision-exp.py` + `patches/vision_exp/`) onto
the spark-vllm-docker B12X serving stack. The upstream port targets the
Anemll 0.1.1 image; this port keeps the patch logic and fail-closed drift
detection but locates the target files inside the installed vLLM package.

The stock `DeepseekV4ForCausalLM` is text-only. The Vision-Exp checkpoint
ships a 32-layer ViT + Aligner and `<｜deepseek_image｜>` prompt tokens.
The startup patch:

1. Appends a fail-closed import hook to the DeepSeek-V4 model module that
   constructs the tower, maps `vision.*` / `aligner.*` / `image_*` /
   `bias_vl` weights, and registers a vLLM multimodal processor
   (`vision_exp/apply.py`, `vision.py` — pure-PyTorch ViT + Aligner).
2. Remaps the DSpark draft `ffn.gate.bias_vl` →
   `e_score_correction_bias_vl` and skips hash-MoE gate biases that have no
   registered parameter (Vision-Exp layers 0–2 quirk).
3. When the encoder carries Anemll-style image-placeholder rejection, relax
   it so OpenAI `image_url` parts survive the chat parser and enforce the
   official rule: images in `user` messages only (`system` / `assistant` →
   HTTP 400). On a **stock encoder** — e.g. the B12X stack's
   `deepseek_v4_encoding.py`, which has no such logic — the encoding patch
   is reported `not-applicable:stock-encoder` and skipped: there is nothing
   to relax, and the role restriction is then not hard-enforced server-side
   (send images in user messages only, matching the official API contract).

Video is not wired: the official weights have no video encoder; GIF is
decoded as a still RGB frame.

## Files

- `hotfix_dsv4_vision_exp.py` — fail-closed patcher. Modes: apply (no
  args, idempotent), `--status`, `--locate`. Target discovery and
  content-drift checks fail the boot with a `FATAL` message instead of
  serving a model that silently drops images.
- `vision_exp/` — ViT + Aligner tower, weight-name filters, and the vLLM
  multimodal processor (image preprocessing, patch grid layout, dummy
  inputs). Copied from upstream `feat/vision-exp`.

## Target discovery

Without configuration the hotfix finds the installed vLLM package
(filesystem-only; vLLM is never imported, so CUDA cannot initialize) and
locates:

- the module defining `DeepseekV4ForCausalLM` and `DeepseekV4MoE`
- the `deepseek_v4` tokenizer encoding module defining `IMAGE_PLACEHOLDER`
- the DSpark draft module carrying the `.ffn.gate.bias` remap

If a target is missing or ambiguous, the mod exits non-zero. A located
module whose content drifted from the expected snippets also fails closed —
except a stock encoder without any image-placeholder logic, which is
reported `not-applicable:stock-encoder` and skipped. Overrides:

| Variable | Meaning |
|----------|---------|
| `VLLM_PACKAGE_ROOT` | vLLM package directory (skips discovery) |
| `VLLM_SITE_PACKAGES` | Site-packages root containing `vllm/` |
| `VISION_EXP_PATCHES_DIR` | Directory holding the `vision_exp` package |
| `VISION_EXP_MODEL_FILE` | Explicit DeepSeek-V4 model module path |
| `VISION_EXP_ENCODING_FILE` | Explicit deepseek_v4 encoding module path |
| `VISION_EXP_DSPARK_FILE` | Explicit DSpark draft module path |
| `VLLM_DEEPSEEK_VISION_EXP` | Set to `0` to skip the mod entirely |

## Serving requirements (paired with the recipe)

- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (official checkpoint
  revision `86f746b3…` upstream).
- DSpark speculative depth: Vision-Exp has `num_nextn_predict_layers=3`, so
  `num_speculative_tokens` must be ≥ 5 (the checkpoint `dspark_block_size`)
  **and divisible by 3** — use 6.
- Images per request: cap with `--limit-mm-per-prompt '{"image":8}'`.
- The ViT/Aligner weights consume additional VRAM relative to 0731; if the
  KV pool cannot hold the target context, lower `max_model_len` or raise
  `gpu_memory_utilization` cautiously.
