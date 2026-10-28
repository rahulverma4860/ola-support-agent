"""
schemas.py — Pydantic structured output schema (Task 9)
=======================================================
Every crew response is validated against CrewResponse.
This enforces consistent output shape regardless of which
agent path produced the answer.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CrewResponse(BaseModel):
    """Structured output schema for every crew response."""

    query: str = Field(
        description="The original user query (after PII masking)"
    )
    answer: str = Field(
        description="The final answer text"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="KB document IDs used (e.g. ['02_sla_by_severity'])"
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="Ticket ID if a lookup was performed (e.g. 'OLA-0012')"
    )
    ticket_status: Optional[str] = Field(
        default=None,
        description="Ticket status if looked up (e.g. 'Resolved')"
    )
    escalation_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Escalation score 0–1 if computed"
    )
    grounded: bool = Field(
        default=True,
        description="True if answer is grounded in retrieved KB content"
    )
    agent_path: str = Field(
        default="rag",
        description="Which path answered: 'rag', 'lookup', or 'both'"
    )


def validate_response(data: dict) -> CrewResponse:
    """
    Validate a dict against CrewResponse.
    Raises ValidationError if required fields are missing or wrong type.
    Pydantic fills in defaults for optional fields automatically.
    """
    return CrewResponse(**data)


# ── CLI demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Valid response — policy question
    r1 = validate_response({
        "query"  : "What is the SLA for Priority 1?",
        "answer" : "Priority 1 tickets must be resolved within 4 hours.",
        "sources": ["02_sla_by_severity"],
        "grounded": True,
        "agent_path": "rag",
    })
    print("✓ Policy response valid:")
    print(f"  {r1.model_dump()}\n")

    # Valid response — ticket lookup
    r2 = validate_response({
        "query"           : "What is the status of OLA-0006?",
        "answer"          : "Ticket OLA-0006 is Resolved. Escalation score: 0.69.",
        "ticket_id"       : "OLA-0006",
        "ticket_status"   : "Resolved",
        "escalation_score": 0.69,
        "agent_path"      : "lookup",
    })
    print("✓ Ticket lookup response valid:")
    print(f"  {r2.model_dump()}\n")

    # Invalid — missing required 'answer' field
    try:
        validate_response({"query": "test"})
    except Exception as e:
        print(f"✓ Correctly caught invalid response: {type(e).__name__}")