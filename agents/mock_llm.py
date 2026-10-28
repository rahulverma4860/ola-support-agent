"""
mock_llm.py — Deterministic MOCK_LLM for zero-API operation (Task 7)
=====================================================================
Extends crewai.llms.base_llm.BaseLLM as required by the brief.

Two known pitfalls (both produce silent wrong output, not a crash):

PITFALL 1 — "Observation:" appears in CrewAI's own ReAct system prompt
  template: "Observation: the result of the action"
  Wrong fix: search ALL messages for "Observation:" → matches template on
             turn 1 before any tool runs → agent skips tool call entirely
  Correct  : only trust Observation: found in a user/human message that
             appears AFTER an assistant message containing "Action:" and
             "Action Input:" — proof that a real tool cycle completed

PITFALL 2 — dispatching tools by name substring
  Wrong fix: if "lookup" in tool_name → matches rag_lookup incorrectly
  Correct  : dispatch by the tool's declared argument schema
             record_id in params → ticket lookup tool
             query in params     → RAG tool
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from crewai.llms.base_llm import BaseLLM
except ImportError:
    # Fallback if import path differs across CrewAI versions
    BaseLLM = object


class MockLLM(BaseLLM):
    """
    Deterministic mock LLM. No API keys, no network calls.
    Implements the ReAct loop CrewAI expects:
      Turn 1 (no prior tool call) → emit Thought + Action + Action Input
      Turn 2 (tool result present) → emit Thought + Final Answer
    """

    def __init__(self):
        try:
            super().__init__(model="mock-llm-v1")
        except Exception:
            pass   # BaseLLM constructor varies across versions

    # ── Required method ───────────────────────────────────────────────

    def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        callbacks=None,
        available_functions=None,
        **kwargs,
    ) -> str:
        """
        Called by CrewAI each time the agent needs to 'think'.
        Returns a string formatted as either:
          Thought → Action → Action Input   (tool call)
          Thought → Final Answer            (done)
        """

        # ── Step 1: find the original task ────────────────────────────
        # The first user/human message contains the agent's task.
        original_task = ""
        for msg in messages:
            if msg.get("role") in ("user", "human"):
                original_task = msg.get("content", "")
                break

        # ── Step 2: detect a REAL tool result ─────────────────────────
        # A real Observation lives in a user/human message that appears
        # AFTER an assistant message that contains "Action:" AND
        # "Action Input:". The system-prompt template's "Observation:"
        # never follows such an assistant message.
        real_observation = self._find_real_observation(messages)

        # ── Step 3: if tool already ran → Final Answer ────────────────
        if real_observation:
            obs_text = real_observation.split("Observation:", 1)[-1].strip()
            return (
                "Thought: I now have all the information I need.\n"
                f"Final Answer: {obs_text}"
            )

        # ── Step 4: no tool yet → decide which to call ────────────────
        if tools:
            return self._choose_tool_call(original_task, tools)

        # ── Step 5: no tools available → direct answer ────────────────
        return (
            "Thought: I can answer directly from my knowledge.\n"
            "Final Answer: Based on Ola's support policies, your request has "
            "been noted. Please contact support@ola.com for further assistance."
        )

    # ── Helper: detect a real (post-action) observation ──────────────

    def _find_real_observation(self, messages: list[dict]) -> str | None:
        """
        Walk the message list looking for this pattern:
          ... assistant message with 'Action:' + 'Action Input:' ...
          ... immediately followed by user message with 'Observation:' ...
        Returns the observation content, or None if not found.
        """
        for i, msg in enumerate(messages):
            is_assistant_with_action = (
                msg.get("role") == "assistant"
                and "Action:" in msg.get("content", "")
                and "Action Input:" in msg.get("content", "")
            )
            if is_assistant_with_action and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if (
                    next_msg.get("role") in ("user", "human")
                    and "Observation:" in next_msg.get("content", "")
                ):
                    return next_msg["content"]
        return None

    # ── Helper: choose and format a tool call ─────────────────────────

    def _choose_tool_call(self, task: str, tools: list[dict]) -> str:
        """
        Dispatch by argument SCHEMA, not by tool name.
        Iterates tools and checks which parameter names they declare.
        """
        for tool_schema in tools:
            func   = tool_schema.get("function", {})
            params = func.get("parameters", {}).get("properties", {})
            name   = func.get("name", "unknown_tool")

            # ── Dispatch rule: record_id in params → ticket lookup ───
            if "record_id" in params:
                # Extract ticket ID from the task (OLA-XXXX pattern)
                match     = re.search(r'OLA-\d{4}', task, re.IGNORECASE)
                record_id = match.group().upper() if match else "OLA-0001"
                return (
                    f"Thought: I need to look up ticket {record_id}.\n"
                    f"Action: {name}\n"
                    f'Action Input: {{"record_id": "{record_id}"}}'
                )

            # ── Dispatch rule: query in params → RAG tool ───────────
            if "query" in params:
                # Use the task as the search query (clean it up)
                clean = task[:120].replace('"', "'").replace("\n", " ").strip()
                return (
                    "Thought: I need to retrieve relevant policy information "
                    "from the knowledge base.\n"
                    f"Action: {name}\n"
                    f'Action Input: {{"query": "{clean}"}}'
                )

        # No matching tool schema found
        return (
            "Thought: I cannot find a suitable tool for this task.\n"
            "Final Answer: I was unable to process this request with available tools."
        )

    # ── Optional methods CrewAI may call ─────────────────────────────

    def get_context_window_size(self) -> int:
        return 8000

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_model_name(self) -> str:
        return "mock-llm-v1"


# ── CLI demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = MockLLM()

    print("=== Turn 1: no prior tool call — should emit Action ===")
    msgs_turn1 = [
        {"role": "system",    "content": "You are a support agent. Observation: the result of the action"},
        {"role": "user",      "content": "What is the SLA for Priority 1 tickets?"},
    ]
    tools_example = [
        {"function": {"name": "rag_policy_lookup",
                      "parameters": {"properties": {"query": {"type": "string"}}}}}
    ]
    response1 = llm.call(msgs_turn1, tools=tools_example)
    print(response1)
    print()

    print("=== Turn 2: real tool result present — should emit Final Answer ===")
    msgs_turn2 = [
        {"role": "system",    "content": "You are a support agent. Observation: the result of the action"},
        {"role": "user",      "content": "What is the SLA for Priority 1 tickets?"},
        {"role": "assistant", "content": "Thought: I need to look this up.\nAction: rag_policy_lookup\nAction Input: {\"query\": \"SLA Priority 1\"}"},
        {"role": "user",      "content": "Observation: Priority 1 tickets must be resolved within 4 hours."},
    ]
    response2 = llm.call(msgs_turn2, tools=tools_example)
    print(response2)
    print()

    print("=== Ticket lookup dispatch (record_id in schema) ===")
    msgs_ticket = [
        {"role": "user", "content": "Check the status of ticket OLA-0006"},
    ]
    tools_lookup = [
        {"function": {"name": "check_support_ticket_status",
                      "parameters": {"properties": {"record_id": {"type": "string"}}}}}
    ]
    response3 = llm.call(msgs_ticket, tools=tools_lookup)
    print(response3)