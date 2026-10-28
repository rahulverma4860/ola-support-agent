"""
memory.py — In-process session memory (Task 8)
==============================================
Uses LangChain's InMemoryChatMessageHistory for per-session history.
State lives in RAM — it does NOT survive a process restart.
That is sufficient for this task (the brief says "in-process only").

Note on RunnableWithMessageHistory:
  LangChain raises a LangChainDeprecationWarning pointing to LangGraph's
  persistence layer. This is EXPECTED and does NOT need to be silenced —
  the class still functions correctly. The brief acknowledges this.
"""

import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# InMemoryChatMessageHistory stores messages in a plain list in RAM
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
except ImportError:
    from langchain.memory import ChatMessageHistory   # older langchain


# ── Session store ─────────────────────────────────────────────────────
# A plain dict keyed by session_id.
# session_id is just a string — e.g. "session_001" or "user_abc"
# Each value is a ChatMessageHistory (list of HumanMessage + AIMessage)

_session_store: dict[str, BaseChatMessageHistory] = {}


# ── Public API ────────────────────────────────────────────────────────

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get or create a history object for the given session_id.
    First call creates an empty history.
    Subsequent calls with the same session_id return the SAME object.
    """
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def add_exchange(session_id: str, human: str, ai: str) -> None:
    """
    Record one human → AI exchange in the session.
    Call this after every crew response to keep history up to date.
    """
    history = get_session_history(session_id)
    history.add_message(HumanMessage(content=human))
    history.add_message(AIMessage(content=ai))


def get_history_text(session_id: str) -> str:
    """
    Return the full conversation history as plain text.
    Used to inject context into the agent's task description.
    Format:
      Human: <query>
      AI   : <answer>
      Human: <query>
      AI   : <answer>
    """
    history  = get_session_history(session_id)
    lines    = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        else:
            lines.append(f"AI   : {msg.content}")
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    """
    Wipe all history for a session.
    Used to demonstrate a 'fresh conversation' (second transcript for Task 8).
    """
    if session_id in _session_store:
        del _session_store[session_id]


def session_exists(session_id: str) -> bool:
    return session_id in _session_store and bool(_session_store[session_id].messages)


# ── CLI demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    SESSION = "demo_session"

    print("=== Multi-turn session (state persists) ===\n")

    # Turn 1
    add_exchange(SESSION, "What is the SLA for Priority 1?",
                 "Priority 1 tickets must be resolved within 4 hours.")
    print("After turn 1:")
    print(get_history_text(SESSION))

    # Turn 2 — references previous context
    add_exchange(SESSION, "What about Priority 2?",
                 "Priority 2 tickets have a 24-hour resolution SLA.")
    print("\nAfter turn 2:")
    print(get_history_text(SESSION))

    # Turn 3 — agent can 'remember' turn 1
    add_exchange(SESSION, "So Priority 1 is faster?",
                 "Yes — Priority 1 (4 hours) is significantly faster than Priority 2 (24 hours).")
    print("\nAfter turn 3 (history shows continuity):")
    print(get_history_text(SESSION))

    # ── Show fresh session ────────────────────────────────────────────
    print("\n\n=== Fresh session (state correctly absent) ===\n")
    FRESH_SESSION = "fresh_session"

    if not session_exists(FRESH_SESSION):
        print(f"  Session '{FRESH_SESSION}' is NEW — no history yet.")
    history_text = get_history_text(FRESH_SESSION)
    print(f"  History: '{history_text}' (empty as expected)")

    # Adding to fresh session doesn't bleed from the other session
    add_exchange(FRESH_SESSION, "Hello, what is the refund policy?",
                 "Refund requests must be submitted within 7 days of the ride.")
    print(f"\n  After first exchange in fresh session:")
    print(f"  {get_history_text(FRESH_SESSION)}")

    print(f"\n  Original session still intact:")
    print(f"  {get_history_text(SESSION)[:80]}...")