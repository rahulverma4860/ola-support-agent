"""
chunker.py — Two chunking strategies for the knowledge base
============================================================
Strategy 1: Fixed-size with overlap
  - chunk_size = 300 chars, overlap = 50 chars
  - step = 300 - 50 = 250  (how far the window slides each time)

Strategy 2: Sentence-based
  - Split text into individual sentences first
  - Group 3 sentences together into one chunk
  - No sentence is ever cut in the middle
"""

import re
import sys
from pathlib import Path
from typing import TypedDict

# Make sure Python can find sibling files when run directly
sys.path.insert(0, str(Path(__file__).parent))

# ── KB directory ──────────────────────────────────────────────────────
# __file__ is rag/chunker.py → .parent is rag/ → .parent is ola-support-agent/
KB_DIR = Path(__file__).parent.parent / "knowledge_base"

# ── Chunking constants ────────────────────────────────────────────────
FIXED_CHUNK_SIZE    = 300   # characters per fixed chunk
FIXED_OVERLAP       = 50    # characters shared between consecutive chunks
SENTENCES_PER_CHUNK = 3     # sentences grouped per sentence-based chunk


# ── Chunk schema ──────────────────────────────────────────────────────
class Chunk(TypedDict):
    chunk_id  : str   # unique ID e.g. "01_ticket_priority_classification_fixed_0"
    doc_id    : str   # parent document name (filename without .txt)
    source    : str   # full path to the source file
    strategy  : str   # "fixed" or "sentence"
    text      : str   # the actual text content of this chunk


# ── Document loader ───────────────────────────────────────────────────
def load_documents() -> list[dict]:
    """
    Read all .txt files from knowledge_base/ and return a list of dicts.
    sorted() ensures consistent order across operating systems.
    """
    docs = []
    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        docs.append({
            "doc_id": path.stem,   # "01_ticket_priority_classification" (no extension)
            "source": str(path),
            "text"  : text,
        })
    print(f"Loaded {len(docs)} documents from {KB_DIR}")
    return docs


# ── Strategy 1: Fixed-size with overlap ──────────────────────────────
def fixed_size_chunks(doc: dict) -> list[Chunk]:
    """
    Slide a window of FIXED_CHUNK_SIZE characters across the text,
    moving STEP characters at a time (STEP = chunk_size - overlap).

    Example (chunk_size=10, overlap=3):
      text : "abcdefghij klmnopqrst"
      step : 10 - 3 = 7
      chunk 0: chars 0-9   → "abcdefghij"
      chunk 1: chars 7-16  → "hij klmnop"   ← "hij" is the 3-char overlap
      chunk 2: chars 14-23 → "opqrst"
    """
    text      = doc["text"]
    step      = FIXED_CHUNK_SIZE - FIXED_OVERLAP
    chunks    = []
    start     = 0
    chunk_num = 0

    while start < len(text):
        end        = start + FIXED_CHUNK_SIZE
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(
                chunk_id = f"{doc['doc_id']}_fixed_{chunk_num}",
                doc_id   = doc["doc_id"],
                source   = doc["source"],
                strategy = "fixed",
                text     = chunk_text,
            ))
            chunk_num += 1

        start += step   # slide the window forward

    return chunks


# ── Strategy 2: Sentence-based ────────────────────────────────────────
def sentence_chunks(doc: dict) -> list[Chunk]:
    """
    Split the document into individual sentences, then group
    SENTENCES_PER_CHUNK sentences into one chunk.

    The regex r'(?<=[.!?])\\s+' means:
      (?<=[.!?])  → lookbehind: only split if preceded by . ! or ?
      \\s+        → split at one or more whitespace characters

    This keeps the punctuation attached to the sentence before the split.
    """
    text = doc["text"]

    # Split into sentences, keep punctuation attached
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences     = [s.strip() for s in raw_sentences if s.strip()]

    chunks    = []
    chunk_num = 0

    # Group into batches of SENTENCES_PER_CHUNK
    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        group      = sentences[i : i + SENTENCES_PER_CHUNK]
        chunk_text = " ".join(group)

        if chunk_text:
            chunks.append(Chunk(
                chunk_id = f"{doc['doc_id']}_sent_{chunk_num}",
                doc_id   = doc["doc_id"],
                source   = doc["source"],
                strategy = "sentence",
                text     = chunk_text,
            ))
            chunk_num += 1

    return chunks


# ── Public API ────────────────────────────────────────────────────────
def get_all_chunks() -> tuple[list[Chunk], list[Chunk]]:
    """
    Load all documents and return two lists:
      - all_fixed    : every doc chunked with fixed-size strategy
      - all_sentence : every doc chunked with sentence strategy
    """
    docs         = load_documents()
    all_fixed    = []
    all_sentence = []

    for doc in docs:
        all_fixed.extend(fixed_size_chunks(doc))
        all_sentence.extend(sentence_chunks(doc))

    return all_fixed, all_sentence


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fixed, sentence = get_all_chunks()

    print(f"\nFixed-size chunks  : {len(fixed)}")
    print(f"Sentence chunks    : {len(sentence)}")

    print(f"\nExample fixed chunk [0]:")
    print(f"  chunk_id : {fixed[0]['chunk_id']}")
    print(f"  doc_id   : {fixed[0]['doc_id']}")
    print(f"  text     : {fixed[0]['text'][:120]}...")

    print(f"\nExample sentence chunk [0]:")
    print(f"  chunk_id : {sentence[0]['chunk_id']}")
    print(f"  doc_id   : {sentence[0]['doc_id']}")
    print(f"  text     : {sentence[0]['text'][:120]}...")