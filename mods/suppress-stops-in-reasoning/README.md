# suppress-stops-in-reasoning

Keep client stop strings dormant until `</think>` so a think-in-prompt
request does not finish mid-reasoning with `content: null`.

## Why

vLLM v1 matches `stop` against the whole decoded stream. When a request
starts inside `<think>` (e.g. `--default-chat-template-kwargs.thinking=true`
with a prompt that opens reasoning), the chain-of-thought often restates
harness stop strings such as `Question:` or an lm-eval `stop[:4]` prefix. The
detokenizer then fires the stop mid-reason, so the reply arrives with
`content: null`.

The guard arms only when the **last prompt token** is the reasoning start
marker. EOS / `max_tokens` are unchanged. Under speculative decoding, when
`</think>` arrives in the same chunk as the last think tokens,
`stop_check_offset` jumps past the marker so a stop in that tail cannot fire.

## What it patches

`vllm/v1/engine/detokenizer.py` (single file, four anchors). It is a faithful
port of the `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` and
`tonyd2wild` "suppress-stops-in-reasoning" fix, re-anchored for the 0.27-era
detokenizer (`TokenizersBackend` dispatch, `import sys` already present).

## Runtime controls

- `VLLM_SUPPRESS_STOPS_IN_REASONING=0` or
  `DSPARK_SUPPRESS_STOPS_IN_REASONING=0` disables the guard process-wide
  (default is enabled).
- The guard prefers `--reasoning-config` `reasoning_start_str` /
  `reasoning_end_str`; when those are empty it falls back to
  `<think>` / `</think>`.

## Usage

```bash
./launch-cluster.sh \
  --apply-mod mods/suppress-stops-in-reasoning \
  exec \
  vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4",...}' \
    ...
```

Or reference it from a recipe:

```yaml
mods:
  - mods/suppress-stops-in-reasoning
```

## Scope and limitations

- It only changes stop-string *timing* inside the reasoning segment; it does
  not alter token generation, EOS, or `max_tokens`.
- It relies on the checkpoint/template opening reasoning with `<think>` (or
  the configured `reasoning_start_str`); requests that never enter a
  reasoning segment are unaffected.
- The guard is armed at request construction from the prompt token ids; a
  prompt whose final token is not the reasoning start marker is unaffected.
