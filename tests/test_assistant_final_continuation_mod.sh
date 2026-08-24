#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/assistant-final-continuation/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

VLLM_ROOT="$TMP_DIR/site-packages/vllm"
TARGET="$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py"
mkdir -p "$(dirname "$TARGET")"

# Minimal-but-functional encoder that mirrors the real transition logic.
cat > "$TARGET" <<'EOF'
ASSISTANT_SP_TOKEN = "<|Assistant|>"
USER_SP_TOKEN = "<|User|>"
LATEST_REMINDER_SP_TOKEN = "<|latest_reminder|>"
thinking_start_token = "<think>"
thinking_end_token = "</think>"
eos_token = "<|end|>"


def render_message(index, messages, thinking_mode, drop_thinking=True, reasoning_effort=None):
    prompt = ""
    msg = messages[index]
    role = msg.get("role")
    content = msg.get("content") or ""

    if role == "system":
        prompt += content
    elif role in ("user", "developer"):
        prompt += USER_SP_TOKEN + content
    elif role == "assistant":
        prompt += ASSISTANT_SP_TOKEN + content + eos_token
    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + content
    else:
        raise NotImplementedError(role)

    if index + 1 < len(messages) and messages[index + 1].get("role") not in ["assistant", "latest_reminder"]:
        return prompt

    elif messages[index].get("role") in ["user", "developer"]:
        # Normal generation: append Assistant + thinking token
        prompt += ASSISTANT_SP_TOKEN
        prompt += thinking_start_token

    return prompt


def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,
                    add_default_bos_token=True, reasoning_effort=None):
    out = ""
    for i in range(len(messages)):
        out += render_message(i, messages, thinking_mode, drop_thinking, reasoning_effort)
    return out
EOF

# First application must patch (and pass the self-check).
first_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'Patched' <<< "$first_output"

grep -Fq '[assistant-final-hotfix]' "$TARGET"
grep -Fq 'messages[index].get("role") == "assistant"' "$TARGET"

# Second application must be idempotent (and still pass the self-check).
second_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'already patched' <<< "$second_output"

echo "test_assistant_final_continuation_mod.sh: OK"
