"""
logging_config.py — Structured JSON-Lines logging (Task 12)
============================================================
Every request produces one JSON-Lines entry written to:
  - Console (stdout) for development visibility
  - logs/requests.jsonl for persistent storage

Format (one JSON object per line):
  {
    "timestamp"  : "2026-10-19T10:30:00.123Z",
    "trace_id"   : "uuid4 string — unique per request",
    "session_id" : "which session this belongs to",
    "query"      : "user query WITH PII already masked",
    "duration_ms": 312,
    "grounded"   : true,
    "agent_path" : "rag" | "lookup" | "both",
    "blocked"    : false,
    "level"      : "INFO"
  }

CRITICAL: query field is always PII-masked before being logged.
A phone number must NEVER appear in a log file in cleartext.
"""

import logging
import uuid
import time
from pathlib import Path
from pythonjsonlogger import jsonlogger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.guardrails import mask_pii

# ── Log file setup ─────────────────────────────────────────────────────
LOGS_DIR  = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE  = LOGS_DIR / "requests.jsonl"

# ── Logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger("ola_support_api")
logger.setLevel(logging.INFO)

# JSON formatter — formats each log record as a JSON object
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    rename_fields={"asctime": "timestamp", "levelname": "level"},
)

# File handler — writes JSON-Lines to disk
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler — for development visibility
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Prevent log records from bubbling to the root logger
logger.propagate = False


# ── Public functions ───────────────────────────────────────────────────

def generate_trace_id() -> str:
    """Generate a unique ID for one request. Used to link all log entries."""
    return str(uuid.uuid4())


def log_request(
    trace_id   : str,
    session_id : str,
    raw_query  : str,     # NEVER log this directly — always mask first
    duration_ms: float,
    result     : dict,
) -> None:
    """
    Write one structured log entry for a completed request.
    PII masking is applied here — raw_query is masked before writing.
    Even if the caller passes an unmasked query, this function masks it.
    """
    # ALWAYS mask before logging — defence in depth
    masked_query, redacted = mask_pii(raw_query)

    logger.info(
        "request_completed",
        extra={
            "trace_id"   : trace_id,
            "session_id" : session_id,
            "query"      : masked_query,          # PII-safe
            "pii_redacted": len(redacted) > 0,    # bool: was anything masked?
            "duration_ms": round(duration_ms, 2),
            "grounded"   : result.get("grounded", False),
            "agent_path" : result.get("agent_path", "unknown"),
            "blocked"    : result.get("blocked", False),
            "has_ticket" : result.get("ticket_id") is not None,
        }
    )


def log_error(trace_id: str, session_id: str, error: str) -> None:
    """Log a request that failed with an error."""
    logger.error(
        "request_failed",
        extra={
            "trace_id"  : trace_id,
            "session_id": session_id,
            "error"     : error,
        }
    )


# ── CLI demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Logging to: {LOG_FILE}\n")

    # Simulate 3 requests
    for query, session in [
        ("What is the SLA for Priority 1?",         "session_001"),
        ("My number is +91-9876543210. Refund?",    "session_002"),
        ("Check ticket OLA-0006",                   "session_001"),
    ]:
        trace  = generate_trace_id()
        result = {"grounded": True, "agent_path": "rag", "blocked": False}
        log_request(trace, session, query, 245.5, result)

    print(f"\nLog entries written to {LOG_FILE}")
    print("Contents:")
    print(LOG_FILE.read_text(encoding="utf-8"))