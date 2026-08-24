#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/tool-call-truncation/run.sh"
PATCHER="$PROJECT_DIR/mods/tool-call-truncation/patch_serving.py"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

VLLM_ROOT="$TMP_DIR/site-packages/vllm"
TARGET="$VLLM_ROOT/entrypoints/openai/chat_completion/serving.py"
mkdir -p "$(dirname "$TARGET")"

# Generate a mock that contains the patcher's exact anchors (drift-proof).
python3 - "$PATCHER" "$TARGET" <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("patch_serving", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

mock = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    + m.HELPER_ANCHOR
    + "\n"
    + m.STREAMING_OLD
    + "\n\n"
    + m.NOSTREAM_OLD
    + "\n"
)
Path(sys.argv[2]).write_text(mock)
PY

# First application must patch.
first_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'Patched' <<< "$first_output"

grep -Fq '# [issue55-hotfix] tool-call truncation safety' "$TARGET"
grep -Fq '_dsml_issue55_json_ok' "$TARGET"
grep -Fq 'str(output.finish_reason) != "length"' "$TARGET"

# Second application must be idempotent.
second_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'already patched' <<< "$second_output"

echo "test_tool_call_truncation_mod.sh: OK"
