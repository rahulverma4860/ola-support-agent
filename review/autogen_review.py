"""
autogen_review.py - AutoGen review stage after the crew's draft (Task 14)
=========================================================================
Team: RoundRobinGroupChat([Policy-Compliance-Reviewer, Final-Editor]),
bounded with max_turns=2 so each agent speaks exactly once.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage

from review.mock_model_client import MockChatCompletionClient


class Verdict(BaseModel):
    approved    : bool = Field(description="True if the draft is approved unchanged")
    final_answer: str  = Field(description="Approved or revised answer text")
    reason      : str  = Field(description="Why it was approved or revised")


def build_team() -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=MockChatCompletionClient(role="reviewer"),
        system_message="Check every claim in the DRAFT against the CONTEXT. List unsupported claims.",
    )
    editor = AssistantAgent(
        name="Final_Editor",
        model_client=MockChatCompletionClient(role="editor"),
        system_message="Using the review, return the final verdict as structured output.",
        output_content_type=Verdict,
    )
    return RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[Verdict]],
    )


async def review_draft_async(draft: str, context: str) -> Verdict:
    team = build_team()
    task = f"DRAFT:\n{draft}\n\nCONTEXT:\n{context}"
    result = await team.run(task=task)
    for msg in reversed(result.messages):
        if isinstance(msg, StructuredMessage):
            return msg.content
    raise RuntimeError("Final_Editor produced no structured verdict")


def review_draft(draft: str, context: str) -> Verdict:
    """Sync wrapper. Call from threads/CLI, not from inside a running event loop."""
    return asyncio.run(review_draft_async(draft, context))