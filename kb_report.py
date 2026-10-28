"""kb_report.py - Task 2 evidence: list the knowledge-base documents with sentence counts."""
import re
from pathlib import Path

docs = sorted(Path("knowledge_base").glob("*.txt"))
print(f"Knowledge base: {len(docs)} documents\n")
print(f"{'document':<42}{'sentences':>10}{'words':>8}")
for p in docs:
    text = p.read_text(encoding="utf-8").strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    print(f"{p.stem:<42}{len(sentences):>10}{len(text.split()):>8}")