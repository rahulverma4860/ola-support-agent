"""review/demo.py - Task 14 evidence: one approved draft, one revised draft."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import GroundedRetriever
from review.autogen_review import review_draft

retriever = GroundedRetriever(strategy="fixed")

CASES = [
    # (label, query, draft)
    ("CASE 1 - clean draft",
     "What is the SLA for Priority 1 tickets?",
     "Priority 1 tickets must be fully resolved within four hours of creation."),
    ("CASE 2 - draft with an injected ungrounded claim",
     "How do I request a refund from Ola?",
     "Ola issues refunds when a service failure is verified to be the fault of the platform or a driver. "
     "Customers also receive a free airport transfer voucher for every refund."),
]

for label, query, draft in CASES:
    chunks  = retriever.retrieve(query, top_k=3)
    context = " ".join(c["text"] for c in chunks)
    verdict = review_draft(draft, context)

    print("=" * 70)
    print(label)
    print(f"Query   : {query}")
    print(f"Sources : {[c['doc_id'] for c in chunks]}")
    print(f"Draft   : {draft}")
    print(f"Verdict : approved={verdict.approved}")
    print(f"Final   : {verdict.final_answer}")
    print(f"Reason  : {verdict.reason}")
print("=" * 70)