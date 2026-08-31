#!/bin/bash
set -euo pipefail

PREFIX="[deepseek-v4-vision-exp]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOTFIX="$MOD_DIR/hotfix_dsv4_vision_exp.py"

# Opt out without removing the mod from the recipe.
if [ "${VLLM_DEEPSEEK_VISION_EXP:-1}" = "0" ]; then
    echo "$PREFIX Disabled via VLLM_DEEPSEEK_VISION_EXP=0; skipping."
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to patch vLLM." >&2
    exit 1
fi

if [ ! -f "$HOTFIX" ]; then
    echo "$PREFIX hotfix script not found: $HOTFIX" >&2
    exit 1
fi

if [ ! -f "$MOD_DIR/vision_exp/apply.py" ] || [ ! -f "$MOD_DIR/vision_exp/vision.py" ]; then
    echo "$PREFIX vision_exp overlay package missing under $MOD_DIR" >&2
    exit 1
fi

echo "=== DeepSeek-V4-Flash-Vision-Exp native image hotfix mod ==="

# Fail-closed discovery: resolves the DeepSeek-V4 model/encoding/dspark
# modules inside the installed vLLM package (no vLLM import, so no CUDA
# initialization while containers are being prepared).
python3 "$HOTFIX" --locate

python3 "$HOTFIX"
python3 "$HOTFIX" --status

# Best-effort pycache cleanup for the patched modules so the next vLLM
# import recompiles instead of silently reusing stale bytecode.
if [ -n "${VLLM_PACKAGE_ROOT:-}" ]; then
    VLLM_ROOT="$VLLM_PACKAGE_ROOT"
elif [ -n "${VLLM_SITE_PACKAGES:-}" ]; then
    VLLM_ROOT="$VLLM_SITE_PACKAGES/vllm"
else
    VLLM_ROOT=$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(0)
print(spec.submodule_search_locations[0])
PY
)
fi
if [ -d "$VLLM_ROOT" ]; then
    find "$VLLM_ROOT/models" "$VLLM_ROOT/model_executor" "$VLLM_ROOT/tokenizers" \
        -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
fi

echo "$PREFIX OK: Vision-Exp ViT/Aligner tower, image_url processor, and"
echo "$PREFIX DSpark bias_vl remap are installed (fail-closed startup patch)."
echo "$PREFIX Images are accepted in user messages only; there is no video"
echo "$PREFIX encoder in the official weights. Opt out with"
echo "$PREFIX VLLM_DEEPSEEK_VISION_EXP=0."
