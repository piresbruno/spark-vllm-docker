#!/usr/bin/env python3
"""Report finish_reason=length (not "tool_calls") when max_tokens truncates a
tool call mid-flight, and drop non-JSON tool-call args (DeepSeek-V4 DSML).

Port of the MiaAI-Lab issue #55 mitigation onto the eugr/spark-arena B12X image:

  <site-packages>/vllm/entrypoints/openai/chat_completion/serving.py

When the engine truncates for ``length``, stock vLLM still reports
``finish_reason="tool_calls"`` and the streaming flush can hand the client a
``tool_calls[].function.arguments`` that does not parse as JSON, so a replaying
harness gets HTTP 400 ("Unterminated string ...").

Two monotonically-safer edits, gated entirely on the engine's own
``FinishReason.LENGTH``:
  1. Streaming final chunk: report "length" and strip non-JSON tool_calls.
  2. Non-streaming: don't claim "tool_calls" and drop unparseable args.

A normal model-terminated tool call is unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARK = "# [issue55-hotfix] tool-call truncation safety"

HELPER_ANCHOR = "import asyncio\n"
HELPER_NEW = '''import asyncio
# [issue55-hotfix] tool-call truncation safety
def _dsml_issue55_json_ok(s):
    """True iff `s` parses as a complete JSON value. None/'' are ok."""
    if s is None or s == "":
        return True
    try:
        import json as _j
        _j.loads(s)
        return True
    except Exception:
        return False
'''

STREAMING_OLD = '''                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )'''

STREAMING_NEW = '''                        # [issue55-hotfix] gate the tool_calls verdict on engine length,
                        # and strip non-JSON args from the trailing delta.
                        if (
                            tools_streamed[i]
                            and not tool_choice_function_name
                            and str(output.finish_reason) != "length"
                        ):
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                str(output.finish_reason)
                                if output.finish_reason
                                else "stop"
                            )
                        if (
                            str(output.finish_reason) == "length"
                            and delta_message is not None
                            and getattr(delta_message, "tool_calls", None)
                        ):
                            _kept = [
                                tc
                                for tc in delta_message.tool_calls
                                if getattr(tc, "function", None)
                                and _dsml_issue55_json_ok(tc.function.arguments)
                            ]
                            delta_message.tool_calls = _kept or None'''

NOSTREAM_OLD = '''            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",'''

NOSTREAM_NEW = '''            # [issue55-hotfix] if the engine truncated for length, do not
            # claim tool_calls ended cleanly, and drop unparseable args.
            if str(output.finish_reason) == "length":
                is_finish_reason_tool_calls = False
                if getattr(message, "tool_calls", None):
                    message.tool_calls = [
                        tc
                        for tc in message.tool_calls
                        if getattr(tc, "function", None)
                        and _dsml_issue55_json_ok(tc.function.arguments)
                    ] or None
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else str(output.finish_reason)
                if output.finish_reason
                else "stop",'''

HUNKS = [
    ("module helper", HELPER_ANCHOR, HELPER_NEW),
    ("streaming final chunk", STREAMING_OLD, STREAMING_NEW),
    ("non-streaming final choice", NOSTREAM_OLD, NOSTREAM_NEW),
]


def apply_text(src: str) -> tuple[str, str]:
    if MARK in src and "_dsml_issue55_json_ok" in src:
        return src, "skipped"
    missing = []
    out = src
    for label, old, new in HUNKS:
        if new in out:
            continue
        if old not in out:
            missing.append(label)
            continue
        out = out.replace(old, new, 1)
    if missing:
        return src, "missing:" + ",".join(missing)
    if out == src:
        return src, "skipped"
    return out, "applied"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate compatibility without writing"
    )
    args = parser.parse_args()

    if not args.target.is_file():
        print(
            "[tool-call-truncation ERROR] target not found: %s" % args.target,
            file=sys.stderr,
        )
        return 1

    original = args.target.read_text(encoding="utf-8")
    patched, status = apply_text(original)

    if args.check:
        if status.startswith("missing"):
            print(
                "[tool-call-truncation ERROR] incompatible %s: %s"
                % (args.target, status),
                file=sys.stderr,
            )
            return 1
        state = "already patched" if status == "skipped" else "compatible"
        print(f"[tool-call-truncation] {args.target} is {state}.")
        return 0

    if status == "skipped":
        print("[tool-call-truncation] already patched; skipping.")
        return 0
    if status.startswith("missing"):
        print(
            "[tool-call-truncation ERROR] refusing to patch %s: %s"
            % (args.target, status),
            file=sys.stderr,
        )
        return 1

    temporary = args.target.with_suffix(args.target.suffix + ".tool-truncation.tmp")
    temporary.write_text(patched, encoding="utf-8")
    temporary.replace(args.target)
    print(f"[tool-call-truncation] Patched {args.target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
