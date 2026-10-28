"""
tools.py - Two tools for the CrewAI agents
==========================================
Tool 1: rag_policy_lookup(query)          - grounded retrieval (cached, Task 16)
Tool 2: check_support_ticket_status(id)   - ticket lookup + escalation score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))   # project root FIRST

from crewai.tools import tool
from rag.retriever import GroundedRetriever
from dataset import SUPPORT_TICKETS, escalation_score
from governance.cache import grounded_cache

_retriever = GroundedRetriever(strategy="fixed")
_ticket_map = {t["record_id"]: t for t in SUPPORT_TICKETS}


@tool("rag_policy_lookup")
def rag_policy_lookup(query: str) -> str:
    """
    Retrieve relevant Ola support policy information for a given query.
    Use this tool for questions about SLAs, refund policies, escalation
    procedures, VIP handling, communication channels, or any other
    Ola support policy topic.
    Returns grounded policy text or a fallback if no match is found.
    """
    result = grounded_cache.get_or_compute(query, lambda: _retriever.answer(query))

    if result["grounded"]:
        context = result["chunks_used"][0] if result["chunks_used"] else ""
        sources = ", ".join(result["sources"])
        return f"[Sources: {sources}]\n\n{context}"

    return ("No relevant policy found in the knowledge base for this query. "
            "Please contact support directly.")


@tool("check_support_ticket_status")
def check_support_ticket_status(record_id: str) -> dict:
    """
    Look up the current status of a specific Ola support ticket by its
    record ID (format: OLA-XXXX, e.g. OLA-0012).
    Returns the ticket status, resolution time, escalation flag,
    and a designed escalation_score in [0, 1].
    Use this tool only when the user provides a specific ticket ID.
    """
    record_id = record_id.strip().upper()

    if record_id not in _ticket_map:
        return {"error": f"Ticket '{record_id}' not found in the system.",
                "record_id": record_id}

    ticket = _ticket_map[record_id]
    score  = escalation_score(ticket)

    return {
        "record_id"            : record_id,
        "category"             : ticket["category"],
        "status"               : ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "days_since_created"   : ticket["days_since_created"],
        "escalated"            : ticket["escalated"],
        "escalation_score"     : score,
        "recommend_escalation" : score > 0.5,
        "score_formula"        : "0.6 × escalated + 0.4 × min(days/26, 1.0)",
    }


if __name__ == "__main__":
    print("=== Tool 1: RAG policy lookup ===")
    print(rag_policy_lookup.run("What is the SLA for Priority 1 tickets?")[:200])
    print("\n=== Tool 2: Ticket status lookup ===")
    print(check_support_ticket_status.run("OLA-0006"))
    print("\n=== Tool 2: Non-existent ticket ===")
    print(check_support_ticket_status.run("OLA-9999"))