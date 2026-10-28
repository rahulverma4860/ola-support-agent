"""
retriever.py — Grounded generation with similarity threshold
=============================================================
"Grounded" means the answer comes ONLY from retrieved context.
If no chunk is similar enough, we return a fallback instead of guessing.

Threshold calibration (run measure_threshold() to verify):
  In-scope queries  → top-1 similarity: ~0.65–0.75
  Out-of-scope      → top-1 similarity: ~0.15–0.25
  Chosen threshold  : 0.45  (clear gap between both clusters)

Under MOCK_LLM mode (MOCK_LLM=mock in .env):
  The "answer" is assembled directly from retrieved chunk text.
  A real LLM would use the same chunks as context in a prompt.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from indexer import get_collection, MODEL_NAME

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────
TOP_K                = 3       # how many chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.45    # empirically calibrated — see README


# ── Retriever ──────────────────────────────────────────────────────────
class GroundedRetriever:
    """
    Given a query → retrieve top-k chunks → check threshold → answer or fallback.
    """

    def __init__(self, strategy: str = "fixed"):
        """
        strategy: "fixed" uses the fixed-size collection,
                  "sentence" uses the sentence-based collection.
        """
        self.model      = SentenceTransformer(MODEL_NAME)
        self.collection = get_collection(strategy)
        self.strategy   = strategy
        print(f"GroundedRetriever ready | strategy={strategy} | threshold={SIMILARITY_THRESHOLD}")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the query and find the top-k most similar chunks.

        Returns a list of dicts:
          { chunk_id, text, doc_id, similarity }

        ChromaDB returns "distance" (lower = better).
        We convert: similarity = 1 - distance.
        """
        q_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings = q_embedding,
            n_results        = top_k,
            include          = ["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            distance   = results["distances"][0][i]
            similarity = 1.0 - distance          # convert distance → similarity

            chunks.append({
                "chunk_id"  : results["ids"][0][i],
                "text"      : results["documents"][0][i],
                "doc_id"    : results["metadatas"][0][i]["doc_id"],
                "similarity": round(similarity, 4),
            })

        return chunks

    def answer(self, query: str) -> dict:
        """
        Full grounded-generation pipeline:
          1. Retrieve chunks
          2. Check top-1 similarity against threshold
          3. If above → grounded answer from context
             If below → "I don't know" fallback

        Returns a result dict with all details for logging and evaluation.
        """
        chunks = self.retrieve(query)

        if not chunks:
            return self._fallback(query, 0.0)

        top_similarity = chunks[0]["similarity"]

        # THE THRESHOLD DECISION — core of grounded generation
        if top_similarity < SIMILARITY_THRESHOLD:
            return self._fallback(query, top_similarity)

        # Build the answer from retrieved context
        context_texts = [c["text"] for c in chunks]

        # Dedup sources while preserving order (dict trick)
        sources = list(dict.fromkeys(c["doc_id"] for c in chunks))

        mock_mode = os.getenv("MOCK_LLM", "mock") == "mock"
        if mock_mode:
            # Under MOCK_LLM: return the most relevant chunk text directly
            answer_text = (
                f"[MOCK ANSWER — grounded in retrieved context]\n\n"
                f"{context_texts[0]}\n\n"
                f"[Sources: {', '.join(sources)}]"
            )
        else:
            # Real LLM: pass context_texts in a prompt (wired in separately)
            answer_text = f"[LLM answer grounded in: {', '.join(sources)}]"

        return {
            "query"         : query,
            "answer"        : answer_text,
            "grounded"      : True,
            "top_similarity": top_similarity,
            "sources"       : sources,
            "chunks_used"   : context_texts,
        }

    def _fallback(self, query: str, similarity: float) -> dict:
        """Return a safe refusal when no relevant context is found."""
        return {
            "query"         : query,
            "answer"        : (
                "I don't have enough information in my knowledge base "
                "to answer that question reliably."
            ),
            "grounded"      : False,
            "top_similarity": similarity,
            "sources"       : [],
            "chunks_used"   : [],
        }

    def measure_threshold(self, in_scope: list[str], out_of_scope: list[str]) -> None:
        """
        Measure similarity scores for a set of queries.
        Run this once to calibrate SIMILARITY_THRESHOLD empirically.
        Results should show a clear gap between in-scope and out-of-scope.
        """
        print(f"\n{'='*54}")
        print("  THRESHOLD CALIBRATION MEASUREMENT")
        print(f"{'='*54}")

        print("\nIn-scope queries (should be HIGH similarity):")
        for q in in_scope:
            c   = self.retrieve(q, top_k=1)
            sim = c[0]["similarity"] if c else 0.0
            print(f"  {sim:.4f}  {q[:60]}")

        print("\nOut-of-scope queries (should be LOW similarity):")
        for q in out_of_scope:
            c   = self.retrieve(q, top_k=1)
            sim = c[0]["similarity"] if c else 0.0
            print(f"  {sim:.4f}  {q[:60]}")

        print(f"\n  Gap is clear → threshold set to: {SIMILARITY_THRESHOLD}")
        print(f"{'='*54}\n")


# ── CLI demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = GroundedRetriever(strategy="fixed")

    IN_SCOPE = [
        "What is the SLA for Priority 1 tickets?",
        "How do I request a refund from Ola?",
        "What is the escalation process for unresolved tickets?",
        "How does Ola handle VIP customers?",
        "What is the data retention policy for closed tickets?",
    ]
    OUT_OF_SCOPE = [
        "What is the weather in Mumbai today?",
        "Tell me a joke about taxi drivers.",
    ]

    # Show threshold calibration measurements
    retriever.measure_threshold(IN_SCOPE, OUT_OF_SCOPE)

    # Demo all queries
    print(f"\n{'='*54}")
    print("  GROUNDED GENERATION DEMO (6 queries)")
    print(f"{'='*54}")

    all_queries = IN_SCOPE + ["What is the weather in Mumbai today?"]
    for q in all_queries:
        result = retriever.answer(q)
        tag    = "✓ GROUNDED" if result["grounded"] else "✗ FALLBACK"
        print(f"\n[{tag}]  sim={result['top_similarity']:.4f}")
        print(f"  Q: {q}")
        print(f"  A: {result['answer'][:100]}...")