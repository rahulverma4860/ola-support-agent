"""
evaluator.py — Precision and recall for both chunking strategies
================================================================
We evaluate at the DOCUMENT level (not chunk level):
  1. Retrieve top-3 chunks from a collection for each query
  2. Map chunk_ids → parent doc_ids (using metadata)
  3. Deduplicate: 3 chunks from same doc = 1 retrieved doc
  4. Compare retrieved docs to ground-truth relevant docs
  5. Compute precision and recall

Precision = TP / (total retrieved docs)
Recall    = TP / (total relevant docs)

Ground truth (EVAL_QUERIES below) is manually defined:
  which KB document(s) contain the correct answer for each query?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
from indexer import get_collection, MODEL_NAME

TOP_K = 3   # retrieve top 3 chunks per query


# ── Ground truth ──────────────────────────────────────────────────────
# "relevant" = set of doc_ids (KB filenames without .txt) that answer the query
EVAL_QUERIES = [
    {
        "query"  : "What is the SLA for Priority 1 tickets?",
        "relevant": {"02_sla_by_severity"},
    },
    {
        "query"  : "How do I request a refund from Ola?",
        "relevant": {"04_refund_compensation_policy"},
    },
    {
        "query"  : "What is the escalation process for unresolved tickets?",
        "relevant": {"03_escalation_matrix"},
    },
    {
        "query"  : "How does Ola handle VIP customers?",
        "relevant": {"10_vip_customer_handling"},
    },
    {
        "query"  : "What is the data retention policy for closed tickets?",
        "relevant": {"12_data_retention_policy"},
    },
    {
        # Out-of-scope: relevant is empty — no KB doc answers this
        "query"  : "What is the weather in Mumbai today?",
        "relevant": set(),
    },
]


# ── Evaluate one collection ────────────────────────────────────────────
def evaluate_collection(strategy: str, model: SentenceTransformer) -> dict:
    """
    Run all EVAL_QUERIES against the given collection strategy.
    Returns per-query results and averages.
    """
    collection = get_collection(strategy)
    results    = []

    for item in EVAL_QUERIES:
        query    = item["query"]
        relevant = item["relevant"]      # ground truth doc set

        # Embed query and retrieve top-k chunks
        q_emb = model.encode([query]).tolist()
        raw   = collection.query(
            query_embeddings = q_emb,
            n_results        = TOP_K,
            include          = ["metadatas", "distances"],
        )

        # Step 1: Map chunks → parent docs (deduplicate)
        # If chunks 0, 1, 2 all come from doc A → retrieved = {A}
        retrieved_docs = set()
        for meta in raw["metadatas"][0]:
            retrieved_docs.add(meta["doc_id"])

        # Step 2: True positives
        tp = len(retrieved_docs & relevant)   # intersection

        # Step 3: Precision and Recall
        precision = tp / len(retrieved_docs) if retrieved_docs else 0.0
        if len(relevant) == 0:
            # Out-of-scope: if system retrieves nothing relevant,
            # that's correct behaviour — recall is 1.0 by convention
            recall = 1.0
        else:
            recall = tp / len(relevant)

        results.append({
            "query"         : query,
            "relevant_docs" : relevant,
            "retrieved_docs": retrieved_docs,
            "tp"            : tp,
            "precision"     : round(precision, 4),
            "recall"        : round(recall,    4),
        })

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall    = sum(r["recall"]    for r in results) / len(results)

    return {
        "strategy"     : strategy,
        "per_query"    : results,
        "avg_precision": round(avg_precision, 4),
        "avg_recall"   : round(avg_recall,    4),
    }


# ── Print report ───────────────────────────────────────────────────────
def print_report(report: dict) -> None:
    SEP = "=" * 62
    print(f"\n{SEP}")
    print(f"  EVAL REPORT — strategy: {report['strategy'].upper()}")
    print(SEP)

    for r in report["per_query"]:
        retrieved_str = sorted(r["retrieved_docs"])
        relevant_str  = sorted(r["relevant_docs"])
        print(f"\n  Q : {r['query'][:58]}...")
        print(f"      relevant  : {relevant_str}")
        print(f"      retrieved : {retrieved_str}")
        n_ret, n_rel = len(r["retrieved_docs"]), len(r["relevant_docs"])
        print(f"      TP        = |retrieved AND relevant| = {r['tp']}")
        print(f"      Precision = TP / retrieved = {r['tp']}/{n_ret} = {r['precision']:.4f}")
        if n_rel:
            print(f"      Recall    = TP / relevant  = {r['tp']}/{n_rel} = {r['recall']:.4f}")
        else:
            print("      Recall    = no relevant docs exist -> 1.0000 by convention")

    print(f"\n  ─────────────────────────────────────")
    print(f"  Average Precision : {report['avg_precision']:.4f}")
    print(f"  Average Recall    : {report['avg_recall']:.4f}")
    print(SEP)


# ── Recommendation ─────────────────────────────────────────────────────
def compare_and_recommend(fixed: dict, sentence: dict) -> None:
    f, s = fixed, sentence
    print(f"\n{'='*62}\n  RECOMMENDATION\n{'='*62}")
    print(f"\n  Fixed-size   -> Precision={f['avg_precision']:.4f}  Recall={f['avg_recall']:.4f}")
    print(f"  Sentence     -> Precision={s['avg_precision']:.4f}  Recall={s['avg_recall']:.4f}\n")
    if (f["avg_precision"], f["avg_recall"]) >= (s["avg_precision"], s["avg_recall"]):
        win, lose, w, l = "fixed-size", "sentence-based", f, s
    else:
        win, lose, w, l = "sentence-based", "fixed-size", s, f
    print(f"  Recommendation: the {win} strategy is recommended for deployment.")
    if w["avg_precision"] != l["avg_precision"]:
        print(f"  It has higher average precision ({w['avg_precision']:.4f} vs {l['avg_precision']:.4f}), "
              f"so it retrieves fewer unrelated documents next to the correct one;")
        print(f"  average recall is {w['avg_recall']:.4f} vs {l['avg_recall']:.4f} for the {lose} strategy.")
    elif w["avg_recall"] != l["avg_recall"]:
        print(f"  Average precision is equal ({w['avg_precision']:.4f}); it has higher average recall "
              f"({w['avg_recall']:.4f} vs {l['avg_recall']:.4f}).")
    else:
        print(f"  Both strategies measured the same (precision {w['avg_precision']:.4f}, "
              f"recall {w['avg_recall']:.4f}); fixed-size is chosen as the default.")
    print("=" * 62)


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)

    fixed_report    = evaluate_collection("fixed",    model)
    sentence_report = evaluate_collection("sentence", model)

    print_report(fixed_report)
    print_report(sentence_report)
    compare_and_recommend(fixed_report, sentence_report)