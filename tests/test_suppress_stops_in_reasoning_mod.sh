#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/suppress-stops-in-reasoning/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

VLLM_ROOT="$TMP_DIR/site-packages/vllm"
TARGET="$VLLM_ROOT/v1/engine/detokenizer.py"
mkdir -p "$(dirname "$TARGET")"

cat > "$TARGET" <<'EOF'
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from abc import ABC, abstractmethod

import tokenizers
from tokenizers import Tokenizer
from transformers import TokenizersBackend


class IncrementalDetokenizer:
    def __init__(self):
        self.token_ids = []

    @classmethod
    def from_new_request(cls, tokenizer, request):
        if tokenizer is None:
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    def __init__(self, request):
        super().__init__()

        self.stop = request.sampling_params.stop
        self.min_tokens = request.sampling_params.min_tokens
        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""

    def update(self, new_token_ids, stop_terminated):
        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += "x"

        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
            )
        return stop_string


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    pass


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    pass


def check_stop_strings(output_text, new_char_count, stop):
    return None
EOF

# First application must patch.
first_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'Patched' <<< "$first_output"

grep -Fq 'import os' "$TARGET"
grep -Fq '_maybe_enable_reasoning_stop_guard' "$TARGET"
grep -Fq '_reasoning_stop_guard: bool = False' "$TARGET"
grep -Fq '_reasoning_end_str: str = "</think>"' "$TARGET"

# Second application must be idempotent.
second_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'already patched' <<< "$second_output"

echo "test_suppress_stops_in_reasoning_mod.sh: OK"
