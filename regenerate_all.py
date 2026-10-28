"""
regenerate_all.py - rebuild the index and re-create every transcript that depends
on the knowledge base. Everything is written as UTF-8 by Python itself, so no
PowerShell encoding problems. Run from the project root:  python regenerate_all.py
Stop any running uvicorn server first (it locks the ChromaDB files).
"""
import os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
TR   = ROOT / "transcripts"
PY   = sys.executable
ENV  = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
status = {}


def run(label, args, outfile=None, timeout=1200):
    t0 = time.time()
    print(f"[{label}] running ...", flush=True)
    try:
        p = subprocess.run([PY, *args], cwd=ROOT, env=ENV, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if outfile:
            (TR / outfile).write_bytes(p.stdout)
        ok = p.returncode == 0
    except Exception as e:
        ok = False
        print(f"    error: {e}")
    status[label] = ok
    print(f"[{label}] {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)", flush=True)
    return ok


def reset_index():
    for d in ("chroma_fixed", "chroma_sentence"):
        shutil.rmtree(ROOT / d, ignore_errors=False) if (ROOT / d).exists() else None


# ---------------------------------------------------------------- 1. KB + index
run("write knowledge base", ["setup_kb.py"])
try:
    reset_index()
except Exception as e:
    sys.exit(f"Could not delete chroma_fixed/chroma_sentence ({e}).\n"
             "Stop any running uvicorn server and close other Python windows, then run again.")

run("task02 kb report",       ["kb_report.py"],                 "task02_knowledge_base.txt")
run("task03 indexing",        ["rag/indexer.py"],               "task03_indexing.txt")
run("task03 chunking",        ["rag/chunker.py"],               "task03_chunking.txt")
run("task04 grounded gen",    ["rag/retriever.py"],             "task04_grounded_generation.txt")
run("task05 chunking eval",   ["rag/evaluator.py"],             "task05_chunking_eval.txt")
run("task06 tools",           ["agents/tools.py"],              "task06_tools.txt")
run("task07 crew",            ["agents/crew.py"],               "task7_crew_demo.txt")
run("task13 evaluation",      ["eval/evaluator.py"],            "task13_eval.txt")
run("task14 autogen review",  ["review/demo.py"],               "task14_autogen_review.txt")
run("task16 caching",         ["governance/demo_cache.py"],     "task16_caching.txt")

# ---------------------------------------------------------------- 2. API + clients
print("[task11 api] starting server ...", flush=True)
srv_log = open(TR / "task11_server_output.txt", "wb")
server = subprocess.Popen([PY, "-m", "uvicorn", "api.main:app", "--port", "8000"],
                          cwd=ROOT, env=ENV, stdout=srv_log, stderr=subprocess.STDOUT)
up = False
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2); up = True; break
    except Exception:
        time.sleep(1)
status["task11 server up"] = up
if up:
    run("task11 websocket", ["ws_test.py"],   "task11_websocket.txt")
    run("task11 http",      ["http_demo.py"], "task11_http_endpoints.txt")
server.terminate()
try: server.wait(timeout=15)
except Exception: server.kill()
srv_log.close()
time.sleep(2)

# leave a clean index behind (http_demo's /add-document test document is removed)
try:
    reset_index()
    run("final clean index rebuild", ["rag/indexer.py"])
except Exception as e:
    print(f"(could not rebuild the index at the end: {e})")


# ---------------------------------------------------------------- 3. report
def text(name):
    p = TR / name
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

print("\n" + "=" * 66 + "\n  VERIFICATION REPORT (paste everything below)\n" + "=" * 66)
print("\nstep status:", {k: ("ok" if v else "FAILED") for k, v in status.items()})

print("\n-- transcript health --")
for p in sorted(TR.glob("*.txt")):
    t = p.read_text(encoding="utf-8", errors="replace")
    flags = []
    if not t.strip(): flags.append("EMPTY")
    if "Traceback" in t: flags.append("TRACEBACK")
    if "charmap" in t: flags.append("CHARMAP")
    if "\u00c3" in t or "\ufffd" in t: flags.append("MOJIBAKE")
    print(f"  {p.name:<34}{p.stat().st_size:>8} bytes  {' '.join(flags) or 'ok'}")

print("\n-- task02 sentence counts --")
for l in text("task02_knowledge_base.txt").splitlines():
    if re.search(r"\d+\s+\d+\s*$", l) or l.startswith("Knowledge"): print("  " + l)
counts = [int(m.group(1)) for m in re.finditer(r"^\S+\s+(\d+)\s+\d+\s*$", text("task02_knowledge_base.txt"), re.M)]
print("  all documents within 2-5 sentences:", bool(counts) and all(2 <= c <= 5 for c in counts))

print("\n-- task04 similarity measurements --")
for l in text("task04_grounded_generation.txt").splitlines():
    if re.match(r"^\s+\d\.\d{4}\s{2}", l): print("  " + l.strip())

print("\n-- task05 averages --")
for l in text("task05_chunking_eval.txt").splitlines():
    if "Average" in l or "Recommendation:" in l: print("  " + l.strip())

print("\n-- task13 evaluation --")
t13 = text("task13_eval.txt").splitlines()
sims, cur = {}, None
for l in t13:
    m = re.search(r"\[(PASS|FAIL)\] Q(\d+):", l)
    if m:
        cur = int(m.group(2))
        if m.group(1) == "FAIL": print("  FAIL:", l.strip())
    m = re.search(r"sim=([0-9.]+)", l)
    if m and cur: sims[cur] = float(m.group(1))
    if "judge: wrong source" in l: print("        ", l.strip())
    if re.match(r"^\s+(Accuracy|Grounding|Completeness|Safety)\s+:", l): print("  " + l.strip())
ins  = [sims[i] for i in range(1, 13) if i in sims]
oos  = [sims[i] for i in (13, 15) if i in sims]
if ins and oos:
    print(f"  in-scope top-1 similarity range : {min(ins):.4f} - {max(ins):.4f}")
    print(f"  out-of-scope top-1 similarity   : max {max(oos):.4f}")
    print(f"  threshold 0.45 separates them   : {min(ins) > 0.45 > max(oos)}")

print("\n-- task14 verdicts --")
for l in text("task14_autogen_review.txt").splitlines():
    if l.startswith(("CASE", "Verdict")): print("  " + l)

print("\n-- task16 --")
for l in text("task16_caching.txt").splitlines():
    if "retriever calls" in l or "Cache stats" in l or "Same answer" in l: print("  " + l.strip())

c7 = text("task7_crew_demo.txt")
print("\n-- task07 crew --")
print("  RAG TOOL OUTPUT boxes :", c7.count("RAG TOOL OUTPUT"))
print("  escalation_score 0.6462:", c7.count("0.6462"))
print("  PHONE_REDACTED         :", c7.count("[PHONE_REDACTED]"))
print("  injection blocked      :", "Blocked: True" in c7)

s11, w11, h11 = text("task11_server_output.txt"), text("task11_websocket.txt"), text("task11_http_endpoints.txt")
print("\n-- task11 --")
print("  disconnect lines   :", re.findall(r"session=(\w+)", "\n".join(l for l in s11.splitlines() if "Client disconnected" in l)))
print("  request_failed     :", s11.count("request_failed"))
print("  http 200 responses :", h11.count("HTTP 200"))
print("  ws turns answered  :", "Turn 1 answer" in w11 and "Turn 2 ticket" in w11 and "Client 2 answer" in w11)

raw = [p.name for p in TR.glob("*.txt") if "9876543210" in p.read_text(encoding="utf-8", errors="replace")]
print("\n-- files containing the raw demo phone number (task10 demo is expected):", raw)
print("=" * 66)