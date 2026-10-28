"""
mock_model_client.py - deterministic model client for AutoGen (Task 14)
=======================================================================
AutoGen agents need an object implementing ChatCompletionClient.
Under MOCK_LLM there is no real model, so this class returns scripted,
rule-based replies. It has two roles:

  role="reviewer" -> reads DRAFT + CONTEXT, flags sentences the context
                     does not support (plain-text reply)
  role="editor"   -> returns the verdict as JSON:
                     {"approved", "final_answer", "reason"}

No network, no API key. Rules, not intelligence - say so in the README.
"""

import json
import re
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient, CreateResult, ModelInfo, RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema

STOPWORDS = {
    "that", "this", "with", "from", "have", "will", "your", "their", "they",
    "been", "were", "which", "when", "than", "into", "also", "such", "must",
    "does", "each", "other", "within", "after", "before", "about", "there",
}
SUPPORT_THRESHOLD = 0.6   # share of a sentence's content words found in context


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 3 and w not in STOPWORDS}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _section(blob: str, start: str, end: Optional[str]) -> str:
    pattern = rf"{start}\s*(.*?)\s*(?:{end}|\Z)" if end else rf"{start}\s*(.*)"
    m = re.search(pattern, blob, re.DOTALL)
    return m.group(1).strip() if m else ""


def find_unsupported(draft: str, context: str) -> list[str]:
    """Sentences of the draft whose content words are mostly absent from context."""
    ctx = _words(context)
    bad = []
    for s in _sentences(draft):
        w = _words(s)
        if w and len(w & ctx) / len(w) < SUPPORT_THRESHOLD:
            bad.append(s)
    return bad


class MockChatCompletionClient(ChatCompletionClient):
    def __init__(self, role: str):
        assert role in ("reviewer", "editor")
        self.role = role
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.calls = 0

    # ---- the reply logic ------------------------------------------------
    def _reply(self, messages: Sequence[Any]) -> str:
        blob = "\n".join(str(getattr(m, "content", "")) for m in messages)
        draft   = _section(blob, "DRAFT:", "CONTEXT:")
        context = _section(blob, "CONTEXT:", "REVIEWER_FLAGS:|REVIEW:")
        bad = find_unsupported(draft, context)

        if self.role == "reviewer":
            if not bad:
                return "REVIEW: OK - every claim in the draft is supported by the context."
            return "REVIEW: UNSUPPORTED CLAIMS\n" + "\n".join(f"- {s}" for s in bad)

        if not bad:
            verdict = {"approved": True, "final_answer": draft,
                       "reason": "Reviewer found no unsupported claims; draft kept unchanged."}
        else:
            kept = [s for s in _sentences(draft) if s not in bad]
            verdict = {"approved": False, "final_answer": " ".join(kept),
                       "reason": f"Removed {len(bad)} claim(s) not supported by retrieved context."}
        return json.dumps(verdict)

    # ---- ChatCompletionClient interface (matches autogen-core 0.7.5) ----
    async def create(self, messages: Sequence[Any], *, tools: Sequence[Union[Tool, ToolSchema]] = [],
                     tool_choice: Any = "auto", json_output: Any = None,
                     extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Optional[CancellationToken] = None) -> CreateResult:
        self.calls += 1
        text = self._reply(messages)
        usage = RequestUsage(prompt_tokens=len(str(messages)) // 4,
                             completion_tokens=len(text) // 4)
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._usage.completion_tokens + usage.completion_tokens)
        return CreateResult(finish_reason="stop", content=text, usage=usage, cached=False)

    async def create_stream(self, messages: Sequence[Any], *, tools: Sequence[Union[Tool, ToolSchema]] = [],
                            tool_choice: Any = "auto", json_output: Any = None,
                            extra_create_args: Mapping[str, Any] = {},
                            cancellation_token: Optional[CancellationToken] = None
                            ) -> AsyncGenerator[Union[str, CreateResult], None]:
        yield await self.create(messages, tools=tools, json_output=json_output)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages: Sequence[Any], *, tools: Sequence[Union[Tool, ToolSchema]] = []) -> int:
        return len(str(messages)) // 4

    def remaining_tokens(self, messages: Sequence[Any], *, tools: Sequence[Union[Tool, ToolSchema]] = []) -> int:
        return 100000

    @property
    def capabilities(self) -> Any:
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True,
                         family="unknown", structured_output=True)