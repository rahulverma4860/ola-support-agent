"""review/demo.py - Task 14 evidence: one approved draft, one revised draft."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import GroundedRetriever
from review.autogen_review import review_draft

# Sentence-based chunks keep whole sentences, so a draft built from them can be checked fairly.
retriever = GroundedRetriever(strategy="sentence")

INJECTED = "Customers also receive a free airport transfer voucher for every refund."

CASES = [
    ("CASE 1 - clean draft", "What is the SLA for Priority 1 tickets?", None),
    ("CASE 2 - draft with an injected ungrounded claim", "How do I request a refund from Ola?", INJECTED),
]


def first_full_sentence(chunk_text):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", chunk_text) if len(s) > 40]
    return sentences[0].split("\n\n")[-1]    # drop the document title if it is glued on


for label, query, injected in CASES:
    chunks  = retriever.retrieve(query, top_k=3)
    context = " ".join(c["text"] for c in chunks)
    draft   = first_full_sentence(chunks[0]["text"])
    if injected:
        draft = draft + " " + injected
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