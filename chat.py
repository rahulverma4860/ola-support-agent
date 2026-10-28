"""
chat.py — Interactive terminal chat with the Ola Support Agent
Run: python chat.py
Type your question and press Enter. Type 'quit' to exit.
"""

import os
import sys

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"]        = "true"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from agents.crew import run_crew

SESSION_ID = "interactive_session"

print("\n" + "="*60)
print("  Ola Support Agent — Interactive Chat")
print("  Type your question and press Enter.")
print("  Type 'quit' or 'exit' to stop.")
print("="*60 + "\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit", "bye"):
        print("Agent: Goodbye! Have a great day.")
        break

    print("\nAgent: thinking...\n")

    result = run_crew(user_input, session_id=SESSION_ID)

    print(f"Agent: {result['answer']}")

    # Show extra info if it was a ticket lookup
    if result.get("ticket_id"):
        print(f"       [Ticket: {result['ticket_id']} | Score: {result.get('escalation_score', 'N/A')}]")

    print()