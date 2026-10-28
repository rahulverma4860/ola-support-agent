"""Task 15 evidence: the lookup tool cannot be given to any other agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.crew  import composer_agent          # import runs the startup audit
from agents.tools import check_support_ticket_status
from governance.least_autonomy import attach_tools

print("Attempt: give check_support_ticket_status to the Composer agent")
try:
    attach_tools(composer_agent, [check_support_ticket_status])
    print("NOT BLOCKED  <-- this would be a failure")
except PermissionError as e:
    print("BLOCKED:", e)

print("Composer tools after the attempt:", composer_agent.tools)