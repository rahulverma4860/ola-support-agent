"""Task 16 evidence: a repeated query is served from cache, skipping the retriever."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.tools import rag_policy_lookup, _retriever
from governance.cache import grounded_cache

# Count how many times the real retriever (embedding + ChromaDB) is actually called
calls = {"n": 0}
_original = _retriever.answer
def counted(q):
    calls["n"] += 1
    return _original(q)
_retriever.answer = counted

grounded_cache.clear()
Q = "What is the SLA for Priority 1 tickets?"

t = time.perf_counter(); r1 = rag_policy_lookup.run(Q)
t1 = time.perf_counter() - t
print(f"1st call (cold)           : {t1*1000:8.2f} ms | retriever calls so far: {calls['n']}")

t = time.perf_counter(); r2 = rag_policy_lookup.run(Q)
t2 = time.perf_counter() - t
print(f"2nd call (identical)      : {t2*1000:8.2f} ms | retriever calls so far: {calls['n']}")

t = time.perf_counter(); r3 = rag_policy_lookup.run("  what is the SLA for priority 1 tickets??  ")
t3 = time.perf_counter() - t
print(f"3rd call (messy variant)  : {t3*1000:8.2f} ms | retriever calls so far: {calls['n']}")

print("\nSame answer returned each time:", r1 == r2 == r3)
print("Cache stats:", grounded_cache.stats())
print(f"Speed-up on repeat: {t1/t2:,.0f}x")