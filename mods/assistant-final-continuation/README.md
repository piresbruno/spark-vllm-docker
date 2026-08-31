# assistant-final-continuation

Emit a generation header (`<｜Assistant｜><think>`) when a request ends with a
closed assistant turn — optionally annotated by a trailing `latest_reminder` —
so the model does not generate from a dead state.

## Why

`render_message()` only appends the generation header when the trailing message
is `user` or `developer`. When a harness retries after a mid-stream error and
re-sends its partial assistant turn, the prompt ends on a bare EOS (or a bare
reminder after the closed turn) and the model responds with an empty turn or
hallucinated DSML markup. One bad turn self-sustains the loop.

Reopening the trailing turn with `wo_eos` is *not* the fix (a complete
assistant turn has nothing left to add and emits EOS immediately). Appending a
fresh generation header after the closed turn is correct for both shapes.

## What it patches

`vllm/tokenizers/deepseek_v4_encoding.py` — the transition in
`render_message()` that appends `ASSISTANT_SP_TOKEN` + the thinking token is
extended to fire when:

- the final message is an `assistant` turn, or
- the final message is a `latest_reminder` whose immediate predecessor is an
  `assistant` turn.

Every other rendering path is byte-identical (mid-transcript assistant turns,
reminders after `user`/`developer`).

## Fail-closed self-check

After writing, the patched encoder is imported and three shapes are rendered:

1. assistant-final must end with the generation header,
2. assistant-final + trailing `latest_reminder` must end with the header,
3. user→`latest_reminder` tail must keep its single in-slot header (no second
   header appended).

On any failure the original bytes are restored and the mod exits nonzero.

## Usage

```bash
./launch-cluster.sh \
  --apply-mod mods/assistant-final-continuation \
  exec \
  vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
    --tokenizer-mode deepseek_v4 \
    ...
```

Or reference it from a recipe:

```yaml
mods:
  - mods/assistant-final-continuation
```

## Scope and limitations

- Only the final message of a request is affected, and only the two shapes
  above; every other rendering path is untouched.
- It relies on the image's own `deepseek_v4_encoding.py` (the refactored
  DeepSeek-V4 encoder). If a different encoder is loaded at runtime, the mod
  must target that file instead.
