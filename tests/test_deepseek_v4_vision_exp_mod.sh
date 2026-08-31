#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/deepseek-v4-vision-exp/run.sh"
HOTFIX="$PROJECT_DIR/mods/deepseek-v4-vision-exp/hotfix_dsv4_vision_exp.py"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

VLLM_ROOT="$TMP_DIR/site-packages/vllm"
MODEL_TARGET="$VLLM_ROOT/models/deepseek_v4/nvidia/model.py"
ENC_TARGET="$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py"
DSPARK_TARGET="$VLLM_ROOT/models/deepseek_v4/nvidia/dspark.py"
mkdir -p "$(dirname "$MODEL_TARGET")" "$(dirname "$ENC_TARGET")"

cat > "$MODEL_TARGET" <<'EOF'
# SPDX-License-Identifier: Apache-2.0
class DeepseekV4Model:
    def __init__(self, *, vllm_config, prefix=""):
        pass


class DeepseekV4MoE:
    def __init__(self, vllm_config, prefix=""):
        pass


class DeepseekV4ForCausalLM:
    pass
EOF

cat > "$ENC_TARGET" <<'EOF'
# SPDX-License-Identifier: Apache-2.0
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def validate_message(msg):
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError("placeholder in content")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError("placeholder in reasoning_content")
    text = msg.get("text", "")
    if IMAGE_PLACEHOLDER in text:
        raise ValueError("placeholder in text")
EOF

cat > "$DSPARK_TARGET" <<'EOF'
# SPDX-License-Identifier: Apache-2.0
def load_weights(self, weights):
    params_dict = dict(self.named_parameters())
    for name, weight in weights:
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
EOF

# Discovery must resolve all three targets.
locate_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" python3 "$HOTFIX" --locate)
grep -Fq "$MODEL_TARGET" <<< "$locate_output"
grep -Fq "$ENC_TARGET" <<< "$locate_output"
grep -Fq "$DSPARK_TARGET" <<< "$locate_output"

# First application must patch.
first_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
[ "$(grep -Fc ': applied' <<< "$first_output")" -eq 3 ]
[ "$(grep -Fc ': skipped' <<< "$first_output")" -eq 0 ]

grep -Fq '[vision-exp-hotfix] native DeepSeek-V4-Flash-Vision-Exp image tower' "$MODEL_TARGET"
grep -Fq 'vision_exp.apply import apply_vision_exp' "$MODEL_TARGET"
grep -Fq '[vision-exp-hotfix] allow vLLM-inserted image placeholders' "$ENC_TARGET"
grep -Fq '[vision-exp-hotfix] images only in user messages' "$ENC_TARGET"
grep -Fq '[vision-exp-hotfix] remap ffn.gate.bias_vl' "$DSPARK_TARGET"
grep -Fq '.ffn.gate.e_score_correction_bias_vl' "$DSPARK_TARGET"

# --status must report every patch applied.
VLLM_PACKAGE_ROOT="$VLLM_ROOT" python3 "$HOTFIX" --status >/dev/null

# Second application must be idempotent.
second_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
[ "$(grep -Fc ': skipped' <<< "$second_output")" -eq 3 ]
[ "$(grep -Fc ': applied' <<< "$second_output")" -eq 0 ]

# Opt-out knob must skip patching without touching targets.
cat > "$MODEL_TARGET" <<'EOF'
class DeepseekV4ForCausalLM:
    pass


class DeepseekV4MoE:
    pass
EOF
opt_out_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" VLLM_DEEPSEEK_VISION_EXP=0 bash "$MOD")
grep -Fq 'Disabled via VLLM_DEEPSEEK_VISION_EXP=0' <<< "$opt_out_output"
! grep -Fq '[vision-exp-hotfix]' "$MODEL_TARGET"

# Drift must fail closed: a model module without the DSV4 classes is fatal.
cat > "$MODEL_TARGET" <<'EOF'
class SomethingElse:
    pass
EOF
if VLLM_PACKAGE_ROOT="$VLLM_ROOT" python3 "$HOTFIX" >"$TMP_DIR/drift.out" 2>&1; then
    echo "drifted model.py must fail closed" >&2
    exit 1
fi
grep -Fq 'drift:missing-dsv4-class' "$TMP_DIR/drift.out"

# Missing target must fail closed: no DSpark module means no patch.
rm -f "$DSPARK_TARGET"
if VLLM_PACKAGE_ROOT="$VLLM_ROOT" python3 "$HOTFIX" >"$TMP_DIR/missing.out" 2>&1; then
    echo "missing dspark.py must fail closed" >&2
    exit 1
fi
grep -Fq 'VISION_EXP_DSPARK_FILE' "$TMP_DIR/missing.out"

# Stock encoder (no image-placeholder logic, B12X layout): encoding patch is
# not applicable, model + dspark still apply, and --status exits 0.
cat > "$MODEL_TARGET" <<'EOF'
# SPDX-License-Identifier: Apache-2.0
class DeepseekV4Model:
    def __init__(self, *, vllm_config, prefix=""):
        pass


class DeepseekV4MoE:
    def __init__(self, vllm_config, prefix=""):
        pass


class DeepseekV4ForCausalLM:
    pass
EOF
cat > "$DSPARK_TARGET" <<'EOF'
# SPDX-License-Identifier: Apache-2.0
def load_weights(self, weights):
    params_dict = dict(self.named_parameters())
    for name, weight in weights:
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
EOF
cat > "$ENC_TARGET" <<'EOF'
# SPDX-License-Identifier: Apache-2.0
# Stock deepseek_v4 encoder: no image placeholder handling at all.
def encode(text):
    return text
EOF
stock_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'not-applicable:stock-encoder' <<< "$stock_output"
[ "$(grep -Fc ': applied' <<< "$stock_output")" -eq 2 ]
VLLM_PACKAGE_ROOT="$VLLM_ROOT" python3 "$HOTFIX" --status >/dev/null
grep -Fq '[vision-exp-hotfix] native DeepSeek-V4-Flash-Vision-Exp image tower' "$MODEL_TARGET"
! grep -Fq '[vision-exp-hotfix]' "$ENC_TARGET"

# Re-run on the stock-encoder layout must stay idempotent.
stock_second=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
[ "$(grep -Fc ': skipped' <<< "$stock_second")" -eq 2 ]
[ "$(grep -Fc ': applied' <<< "$stock_second")" -eq 0 ]

echo "test_deepseek_v4_vision_exp_mod.sh: OK"
