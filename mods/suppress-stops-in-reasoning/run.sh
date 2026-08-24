#!/bin/bash
set -euo pipefail

PREFIX="[suppress-stops-in-reasoning]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="$MOD_DIR/patch_detokenizer.py"

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to locate and patch vLLM." >&2
    exit 1
fi

if [ ! -f "$PATCHER" ]; then
    echo "$PREFIX patcher not found: $PATCHER" >&2
    exit 1
fi

# VLLM_PACKAGE_ROOT is useful for tests and unusual image layouts. Normally,
# discover the package without importing vLLM so CUDA initialization cannot
# occur while cluster containers are still being prepared.
if [ -z "${VLLM_PACKAGE_ROOT:-}" ]; then
    if [ -n "${VLLM_SITE_PACKAGES:-}" ]; then
        VLLM_PACKAGE_ROOT="${VLLM_SITE_PACKAGES}/vllm"
    else
        VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vllm package not found")
print(spec.submodule_search_locations[0])
PY
)
    fi
fi

TARGET="$VLLM_PACKAGE_ROOT/v1/engine/detokenizer.py"

if [ ! -f "$TARGET" ]; then
    echo "$PREFIX detokenizer not found at $TARGET" >&2
    exit 1
fi

echo "=== suppress-stops-in-reasoning mod ==="

python3 "$PATCHER" --check "$TARGET"
python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

find "$(dirname "$TARGET")" -name "__pycache__" -type d \
    -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX OK: client stop strings stay dormant until </think>."
echo "$PREFIX Opt out with VLLM_SUPPRESS_STOPS_IN_REASONING=0 (or DSPARK_SUPPRESS_STOPS_IN_REASONING=0)."
