"""
cache.py - in-memory response cache for the grounded-generation step (Task 16)
=============================================================================
Key   : normalized query text (lowercase, collapsed whitespace, trailing
        punctuation stripped), so trivial variations share one entry.
Value : the result dict from GroundedRetriever.answer().
Why   : a repeated question skips the embedding + ChromaDB search entirely.

Notes
  - The tool only ever receives the PII-masked query, so keys never contain
    a raw phone number.
  - The cache must be cleared when the knowledge base changes (/add-document),
    otherwise it would keep serving answers from the old knowledge base.
  - Bounded size with oldest-entry eviction; lock-protected because FastAPI
    runs endpoints in worker threads.
"""

import re
import threading
from typing import Any, Callable


class ResponseCache:
    def __init__(self, max_entries: int = 256):
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def normalize(query: str) -> str:
        q = re.sub(r"\s+", " ", query.strip().lower())
        return q.rstrip("?!. ")

    def get_or_compute(self, query: str, compute: Callable[[], Any]) -> Any:
        key = self.normalize(query)
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
        value = compute()            # slow work runs outside the lock
        with self._lock:
            self.misses += 1
            if len(self._store) >= self.max_entries:
                self._store.pop(next(iter(self._store)))   # evict oldest
            self._store[key] = value
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._store)}


# One shared instance for the whole process
grounded_cache = ResponseCache()