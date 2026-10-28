"""
indexer.py — Embed chunks and store in two ChromaDB collections
================================================================
SentenceTransformer model: all-MiniLM-L6-v2
  - Free, runs locally (no API key)
  - Downloads once (~80MB), cached in ~/.cache/
  - Produces 384-dimensional vectors
  - Fast on CPU, good quality for English text

Two separate ChromaDB collections:
  - "ola_fixed"    → fixed-size chunks
  - "ola_sentence" → sentence-based chunks

ChromaDB stores each record as:
  id         : chunk_id (unique string)
  embedding  : the 384-float vector
  document   : the original chunk text (for retrieval)
  metadata   : doc_id, source, strategy (for filtering and evaluation)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

# Allow imports from sibling files
sys.path.insert(0, str(Path(__file__).parent))
from chunker import get_all_chunks, Chunk

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────
MODEL_NAME          = "all-MiniLM-L6-v2"
COLLECTION_FIXED    = "ola_fixed"
COLLECTION_SENTENCE = "ola_sentence"

# ChromaDB stores data on disk at these paths (created automatically)
CHROMA_PATH_FIXED    = os.getenv("CHROMA_PATH_FIXED",    "./chroma_fixed")
CHROMA_PATH_SENTENCE = os.getenv("CHROMA_PATH_SENTENCE", "./chroma_sentence")


# ── Embedding model ────────────────────────────────────────────────────
def get_embedding_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model.
    First call downloads it; subsequent calls load from cache.
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


# ── Build one collection ───────────────────────────────────────────────
def build_index(
    chunks          : list[Chunk],
    model           : SentenceTransformer,
    chroma_path     : str,
    collection_name : str,
) -> chromadb.Collection:
    """
    Embed all chunks and upsert into a ChromaDB collection.

    Steps:
      1. Create (or open) a persistent ChromaDB client at chroma_path
      2. Get or create the named collection
      3. Embed all chunk texts in one batch (faster than one-by-one)
      4. Upsert: insert new, update existing (safe to re-run)
    """
    print(f"\nBuilding collection: '{collection_name}'")
    print(f"  Chunks to index : {len(chunks)}")
    print(f"  Storage path    : {chroma_path}")

    # PersistentClient saves data to disk so it survives restarts
    client     = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},   # cosine similarity metric
    )

    # Embed all texts at once — much faster than a loop
    texts      = [c["text"] for c in chunks]
    print(f"  Embedding chunks (this may take a moment)...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # upsert = "update or insert"
    collection.upsert(
        ids        = [c["chunk_id"] for c in chunks],
        embeddings = embeddings.tolist(),    # numpy array → plain list
        documents  = [c["text"]    for c in chunks],
        metadatas  = [
            {
                "doc_id"  : c["doc_id"],
                "source"  : c["source"],
                "strategy": c["strategy"],
            }
            for c in chunks
        ],
    )

    print(f"  ✓ Collection '{collection_name}': {collection.count()} records")
    return collection


# ── Build both collections ─────────────────────────────────────────────
def build_both_indexes() -> tuple[chromadb.Collection, chromadb.Collection]:
    """Load chunks, embed, and build both ChromaDB collections."""
    model                       = get_embedding_model()
    fixed_chunks, sent_chunks   = get_all_chunks()

    col_fixed    = build_index(fixed_chunks, model, CHROMA_PATH_FIXED,    COLLECTION_FIXED)
    col_sentence = build_index(sent_chunks,  model, CHROMA_PATH_SENTENCE, COLLECTION_SENTENCE)

    return col_fixed, col_sentence


# ── Load existing collection (used by retriever and evaluator) ─────────
def get_collection(strategy: str = "fixed") -> chromadb.Collection:
    """
    Open an already-built collection from disk.
    Call build_both_indexes() first if collections don't exist yet.
    """
    if strategy == "fixed":
        path, name = CHROMA_PATH_FIXED,    COLLECTION_FIXED
    else:
        path, name = CHROMA_PATH_SENTENCE, COLLECTION_SENTENCE

    client = chromadb.PersistentClient(path=path)
    return client.get_collection(name=name)


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    col_fixed, col_sentence = build_both_indexes()

    print(f"\n{'='*50}")
    print(f"  INDEX SUMMARY")
    print(f"{'='*50}")
    print(f"  Fixed collection    : {col_fixed.count()} chunks")
    print(f"  Sentence collection : {col_sentence.count()} chunks")

    # Quick sanity check — query both with the same question
    model      = SentenceTransformer(MODEL_NAME)
    test_query = "What is the SLA for Priority 1 tickets?"
    q_emb      = model.encode([test_query]).tolist()

    print(f"\n  Sanity check: '{test_query}'")
    for col, label in [(col_fixed, "fixed"), (col_sentence, "sentence")]:
        result   = col.query(query_embeddings=q_emb, n_results=1)
        top_doc  = result["metadatas"][0][0]["doc_id"]
        top_text = result["documents"][0][0][:80]
        dist     = result["distances"][0][0]
        sim      = round(1.0 - dist, 4)
        print(f"\n  [{label}]")
        print(f"    top doc  : {top_doc}")
        print(f"    sim      : {sim}")
        print(f"    text     : {top_text}...")