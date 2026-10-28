"""
main.py — FastAPI deployment (Task 11)
======================================
Endpoints:
  POST /ask                    — single-turn query via HTTP
  POST /add-document           — add a new KB document at runtime
  WebSocket /ws/chat/{session} — real-time multi-turn chat

Run: uvicorn api.main:app --reload
Test POST: curl -X POST http://localhost:8000/ask \
           -H "Content-Type: application/json" \
           -d '{"query": "What is the SLA for Priority 1?"}'
"""

import os
import sys
import time
from fastapi.concurrency import run_in_threadpool


os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"]        = "true"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from agents.crew         import run_crew
from agents.guardrails   import apply_input_guardrails
from api.logging_config  import generate_trace_id, log_request, log_error
from governance.cache import grounded_cache

# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Ola Support Agent API",
    description = "Domain support agent — policy Q&A and ticket lookup",
    version     = "1.0.0",
)


# ── Request / Response models (Pydantic) ──────────────────────────────

class AskRequest(BaseModel):
    query     : str            = Field(..., description="Customer's question")
    session_id: Optional[str]  = Field(default="default", description="Session ID for memory")


class AskResponse(BaseModel):
    trace_id        : str
    query           : str           # masked
    answer          : str
    grounded        : bool
    agent_path      : str
    ticket_id       : Optional[str] = None
    escalation_score: Optional[float] = None
    blocked         : bool = False


class AddDocumentRequest(BaseModel):
    title  : str = Field(..., description="Document title (used as doc_id)")
    content: str = Field(..., description="Full document text")


class AddDocumentResponse(BaseModel):
    success      : bool
    doc_id       : str
    chunks_added : int
    message      : str


class WebSocketMessage(BaseModel):
    answer     : str
    grounded   : bool
    ticket_id  : Optional[str] = None
    blocked    : bool = False


# ── POST /ask ─────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Single-turn query endpoint.
    Applies guardrails, runs crew, logs the request, returns structured response.
    """
    trace_id   = generate_trace_id()
    start_time = time.time()

    try:
        result = run_crew(request.query, session_id=request.session_id)
        duration_ms = (time.time() - start_time) * 1000
        log_request(trace_id, request.session_id, request.query, duration_ms, result)

        return AskResponse(
            trace_id         = trace_id,
            query            = result["query"],      # already masked by run_crew
            answer           = result["answer"],
            grounded         = result.get("grounded", False),
            agent_path       = result.get("agent_path", "rag"),
            ticket_id        = result.get("ticket_id"),
            escalation_score = result.get("escalation_score"),
            blocked          = result.get("blocked", False),
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(trace_id, request.session_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /add-document ────────────────────────────────────────────────

@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    """
    Add a new document to the knowledge base at runtime.
    Chunks it with both strategies and upserts into both ChromaDB collections.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from rag.chunker  import fixed_size_chunks, sentence_chunks
        from rag.indexer  import (get_collection, MODEL_NAME,
                                   CHROMA_PATH_FIXED, CHROMA_PATH_SENTENCE,
                                   COLLECTION_FIXED, COLLECTION_SENTENCE)

        # Build a doc dict in the same format chunker expects
        doc = {
            "doc_id": request.title.lower().replace(" ", "_")[:50],
            "source": "runtime_addition",
            "text"  : request.content,
        }

        fixed_chunks = fixed_size_chunks(doc)
        sent_chunks  = sentence_chunks(doc)
        all_chunks   = fixed_chunks + sent_chunks

        model = SentenceTransformer(MODEL_NAME)

        # Upsert into fixed collection
        col_fixed = get_collection("fixed")
        if fixed_chunks:
            embs = model.encode([c["text"] for c in fixed_chunks]).tolist()
            col_fixed.upsert(
                ids        = [c["chunk_id"] for c in fixed_chunks],
                embeddings = embs,
                documents  = [c["text"] for c in fixed_chunks],
                metadatas  = [{"doc_id": c["doc_id"], "source": c["source"],
                               "strategy": "fixed"} for c in fixed_chunks],
            )

        # Upsert into sentence collection
        col_sent = get_collection("sentence")
        if sent_chunks:
            embs = model.encode([c["text"] for c in sent_chunks]).tolist()
            col_sent.upsert(
                ids        = [c["chunk_id"] for c in sent_chunks],
                embeddings = embs,
                documents  = [c["text"] for c in sent_chunks],
                metadatas  = [{"doc_id": c["doc_id"], "source": c["source"],
                               "strategy": "sentence"} for c in sent_chunks],
            )
        
        grounded_cache.clear()   # knowledge base changed: drop cached answers
        
        return AddDocumentResponse(
            success     = True,
            doc_id      = doc["doc_id"],
            chunks_added= len(all_chunks),
            message     = (f"Added {len(fixed_chunks)} fixed + "
                          f"{len(sent_chunks)} sentence chunks for '{doc['doc_id']}'"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket /ws/chat/{session_id} ───────────────────────────────────

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    Real-time multi-turn chat via WebSocket.

    Client sends: plain text query
    Server sends: JSON with answer, grounded, ticket_id, blocked

    Handles WebSocketDisconnect gracefully:
      When a client disconnects (closes browser, network drop),
      we catch the exception and log it — the server keeps running
      for all other connected clients.
    """
    await websocket.accept()

    try:
        while True:
            # Wait for the client to send a message
            query = await websocket.receive_text()

            if not query.strip():
                continue

            trace_id   = generate_trace_id()
            start_time = time.time()

            try:
                result = await run_in_threadpool(run_crew, query, session_id)
                duration_ms = (time.time() - start_time) * 1000
                log_request(trace_id, session_id, query, duration_ms, result)

                # Send response back to client as JSON string
                response = WebSocketMessage(
                    answer    = result["answer"],
                    grounded  = result.get("grounded", False),
                    ticket_id = result.get("ticket_id"),
                    blocked   = result.get("blocked", False),
                )
                await websocket.send_text(response.model_dump_json())

            except WebSocketDisconnect:
                raise

            except Exception as e:
                log_error(trace_id, session_id, str(e))
                await websocket.send_text(
                    f'{{"error": "{str(e)}", "trace_id": "{trace_id}"}}'
                )

    except WebSocketDisconnect:
        # Client disconnected — this is normal (browser closed, timeout, etc.)
        # We catch it here so the server keeps running for other clients.
        print(f"[WebSocket] Client disconnected: session={session_id}")

    except Exception as e:
        print(f"[WebSocket] Unexpected error for session={session_id}: {e}")


# ── Health check ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Ola Support Agent API"}


# ── Run directly ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)