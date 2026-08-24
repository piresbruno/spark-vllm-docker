# tool-call-truncation

When `max_tokens` truncates a request mid-tool-call, report
`finish_reason="length"` (not `"tool_calls"`) and drop any tool-call argument
that does not parse as JSON.

## Why

DeepSeek-V4 tool calls use a DSML argument block. When the engine stops for
`length` in the middle of one, stock vLLM still reports
`finish_reason="tool_calls"`, and the streaming flush can hand the client a
`tool_calls[].function.arguments` that is not valid JSON. A harness that
replays that broken call then gets the whole transcript rejected with
HTTP 400 ("Unterminated string …").

## What it patches

`vllm/entrypoints/openai/chat_completion/serving.py` (three hunks), gated
entirely on the engine's own `FinishReason.LENGTH` so a normal
model-terminated tool call is byte-for-byte unchanged:

1. a module helper `_dsml_issue55_json_ok` (after `import asyncio`);
2. the streaming final-chunk path — `finish_reason` becomes `"length"` on
   truncation and non-JSON `tool_calls` are stripped from the trailing delta;
3. the non-streaming path — same length gate + unparseable-args drop.

It is a faithful port of the `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
issue #55 mitigation.

## Usage

```bash
./launch-cluster.sh \
  --apply-mod mods/tool-call-truncation \
  exec \
  vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    ...
```

Or reference it from a recipe:

```yaml
mods:
  - mods/tool-call-truncation
```

## Scope and limitations

- Only the `finish_reason` verdict and the flushed tool-call args are touched
  when the engine reports `length`; reasoning, content, and stop handling are
  unchanged.
- It assumes the OpenAI chat-completions serving path (`serving.py`) — the
  `/v1/responses` path is out of scope.
