"""Task 15 evidence: an oversized request is rejected, not silently run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from governance.budget_cap import check_budget
from agents.crew import run_crew

normal = "What is the SLA for Priority 1 tickets?"
print("NORMAL request budget:", check_budget(normal, ""))

oversized = "Tell me about Priority 1 tickets. " + "Please explain in detail. " * 400
print(f"\nOVERSIZED request: {len(oversized)} characters")
print("budget:", check_budget(oversized, ""))

result = run_crew(oversized, session_id="budget_demo")
print("\nrun_crew result:")
print("  blocked   :", result["blocked"])
print("  agent_path:", result["agent_path"])
print("  answer    :", result["answer"])