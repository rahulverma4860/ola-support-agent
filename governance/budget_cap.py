"""
budget_cap.py - Runtime-layer governance: per-request token/cost budget (Task 15)
=================================================================================
A request that would exceed the budget is REJECTED up front, before any tool
call or crew run, instead of silently running over budget.

Token count is an estimate (characters / 4), the usual rough rule for English.
The price is illustrative: MOCK_LLM costs nothing, so this shows how a cost cap
would work with a real model.
"""

CHARS_PER_TOKEN        = 4
MAX_TOKENS_PER_REQUEST = 1500
RESERVED_TOKENS        = 600      # headroom for tool output, task templates, agent prompts
PRICE_PER_1K_TOKENS    = 0.002    # illustrative USD
MAX_COST_PER_REQUEST   = MAX_TOKENS_PER_REQUEST / 1000 * PRICE_PER_1K_TOKENS


class BudgetExceededError(Exception):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def check_budget(*parts: str) -> dict:
    """Estimate the request's tokens and cost; report whether it fits."""
    tokens = sum(estimate_tokens(p) for p in parts if p) + RESERVED_TOKENS
    return {
        "estimated_tokens"  : tokens,
        "estimated_cost_usd": round(tokens / 1000 * PRICE_PER_1K_TOKENS, 6),
        "limit_tokens"      : MAX_TOKENS_PER_REQUEST,
        "limit_cost_usd"    : round(MAX_COST_PER_REQUEST, 6),
        "within_budget"     : tokens <= MAX_TOKENS_PER_REQUEST,
    }


def enforce_budget(*parts: str) -> dict:
    """Return the estimate if it fits; raise BudgetExceededError if not."""
    r = check_budget(*parts)
    if not r["within_budget"]:
        raise BudgetExceededError(
            f"estimated {r['estimated_tokens']} tokens "
            f"(${r['estimated_cost_usd']}) exceeds the per-request budget of "
            f"{r['limit_tokens']} tokens (${r['limit_cost_usd']})"
        )
    return r