"""
least_autonomy.py - Application-layer governance: least autonomy (Task 15)
==========================================================================
Principle: an agent gets only the tools its job needs.
Rule enforced here: ONLY the Lookup Agent may hold check_support_ticket_status.

Two mechanisms:
  attach_tools()            - the only sanctioned way to give an agent tools;
                              raises PermissionError on a violation.
  enforce_least_autonomy()  - startup audit of every agent's tool list;
                              raises PermissionError if any agent breaks the rule.
"""

RESTRICTED_TOOLS: dict[str, set[str]] = {
    # tool name                  : agent roles allowed to hold it
    "check_support_ticket_status": {"Ola Ticket Status Specialist"},
}


def _tool_name(tool) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))


def find_violations(agent_role: str, tools) -> list[str]:
    """Restricted tools in `tools` that `agent_role` is NOT allowed to hold."""
    bad = []
    for t in tools or []:
        name = _tool_name(t)
        allowed = RESTRICTED_TOOLS.get(name)
        if allowed is not None and agent_role not in allowed:
            bad.append(name)
    return bad


def attach_tools(agent, tools) -> None:
    """Give `agent` these tools, unless that breaks least autonomy."""
    bad = find_violations(agent.role, tools)
    if bad:
        raise PermissionError(
            f"Least-autonomy violation: agent '{agent.role}' may not hold {bad}. "
            f"Allowed holders: {[sorted(RESTRICTED_TOOLS[b]) for b in bad]}"
        )
    agent.tools = list(tools)


def enforce_least_autonomy(agents) -> None:
    """Audit all agents. Print who holds what; raise if anyone breaks the rule."""
    print("\n[Governance] Least-autonomy audit")
    problems = []
    for a in agents:
        names = [_tool_name(t) for t in (a.tools or [])]
        print(f"  {a.role:<34} tools={names}")
        problems += [(a.role, n) for n in find_violations(a.role, a.tools)]
    if problems:
        raise PermissionError(f"Least-autonomy audit failed: {problems}")
    print("[Governance] Audit passed: restricted tools held only by permitted agents\n")