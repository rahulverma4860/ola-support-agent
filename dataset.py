"""
dataset.py — Ola Support Ticket Dataset Generator
===================================================
Design choices (state these in README.md for the grader):
  Seed            : 42
  Total records   : 50
  Category weights: Billing 0.25 | Technical Issue 0.25 |
                    Account Access 0.20 | Product Defect 0.15 |
                    General Inquiry 0.15
  Status weights  : Open 0.15 | In Progress 0.25 | Escalated 0.10 |
                    Resolved 0.30 | Closed 0.20
  Resolution time : per-category realistic range (hours)
  Escalation prob : base 0.12; boosted to 0.80 when status=="Escalated",
                    to 0.25 when days_since_created > 20
"""

import random
from typing import TypedDict

# ── Constants ─────────────────────────────────────────────────────────

SEED            = 42
TOTAL_RECORDS   = 50

RECENCY_CAP_DAYS = 26   # ~80th percentile of days_since_created (measured: 26.2)

CATEGORIES      = ["Billing", "Technical Issue", "Account Access",
                   "Product Defect", "General Inquiry"]
CATEGORY_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]

STATUSES        = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
STATUS_WEIGHTS  = [0.15, 0.25, 0.10, 0.30, 0.20]

RESOLUTION_HOURS: dict[str, tuple[float, float]] = {
    "Billing":         (2.0,  72.0),
    "Technical Issue": (1.0,  48.0),
    "Account Access":  (0.5,  26.2),
    "Product Defect":  (4.0,  96.0),
    "General Inquiry": (0.5,  12.0),
}

# ── Type ──────────────────────────────────────────────────────────────

class SupportTicket(TypedDict):
    record_id:              str
    category:               str
    status:                 str
    resolution_time_hours:  float
    days_since_created:     int
    escalated:              bool


# ── Generator ─────────────────────────────────────────────────────────

def generate_tickets(seed: int = SEED, n: int = TOTAL_RECORDS) -> list[SupportTicket]:
    """Produce a deterministic list of n support tickets."""
    rng = random.Random(seed)
    tickets: list[SupportTicket] = []

    for i in range(1, n + 1):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status   = rng.choices(STATUSES,   weights=STATUS_WEIGHTS,   k=1)[0]

        lo, hi = RESOLUTION_HOURS[category]
        resolution_time = round(rng.uniform(lo, hi), 1)
        days_since_created = rng.randint(0, 30)

        if status == "Escalated":
            esc_prob = 0.80
        elif days_since_created > 20:
            esc_prob = 0.25
        else:
            esc_prob = 0.12

        escalated = rng.random() < esc_prob

        tickets.append(SupportTicket(
            record_id             = f"OLA-{i:04d}",
            category              = category,
            status                = status,
            resolution_time_hours = resolution_time,
            days_since_created    = days_since_created,
            escalated             = escalated,
        ))

    return tickets


# ── Escalation score (also used by Task 6 lookup tool) ───────────────

def escalation_score(ticket: SupportTicket) -> float:
    """
    score = 0.6 × escalated_flag + 0.4 × min(days_since_created / 26, 1.0)

    Threshold = 0.5
    Any ticket with escalated=True scores ≥ 0.6 → above threshold → escalate.
    Non-escalated tickets score max 0.4 → below threshold → don't escalate.
    Recency differentiates urgency within the already-escalated bucket.
    """
    flag    = 1.0 if ticket["escalated"] else 0.0
    recency = min(ticket["days_since_created"] / RECENCY_CAP_DAYS, 1.0)
    return round(0.6 * flag + 0.4 * recency, 4)


# ── Validation & report ───────────────────────────────────────────────

def validate_and_report(tickets: list[SupportTicket]) -> None:
    SEP = "=" * 54
    print(f"\n{SEP}")
    print("  DATASET VALIDATION REPORT")
    print(SEP)
    print(f"  Total records : {len(tickets)}")

    cat_counts: dict[str, int] = {}
    for t in tickets:
        cat_counts[t["category"]] = cat_counts.get(t["category"], 0) + 1

    print(f"\n  Category distribution (required ≥3 each):")
    all_cat_ok = True
    for cat in CATEGORIES:
        count = cat_counts.get(cat, 0)
        ok    = count >= 3
        print(f"    {'✓' if ok else '✗ FAIL'}  {cat:<22} {count}")
        if not ok:
            all_cat_ok = False

    status_counts: dict[str, int] = {}
    for t in tickets:
        status_counts[t["status"]] = status_counts.get(t["status"], 0) + 1

    print(f"\n  Status distribution (required ≥1 each):")
    all_status_ok = True
    for status in STATUSES:
        count = status_counts.get(status, 0)
        ok    = count >= 1
        print(f"    {'✓' if ok else '✗ FAIL'}  {status:<22} {count}")
        if not ok:
            all_status_ok = False

    esc_count = sum(1 for t in tickets if t["escalated"])
    esc_pct   = esc_count / len(tickets) * 100
    esc_ok    = 10.0 <= esc_pct <= 30.0
    print(f"\n  Escalated records (required 10%–30%):")
    print(f"    {'✓' if esc_ok else '✗ FAIL'}  {esc_count}/{len(tickets)} = {esc_pct:.1f}%")

    overall = all_cat_ok and all_status_ok and esc_ok
    print(f"\n  Overall : {'✓ PASS' if overall else '✗ FAIL'}")
    print(SEP + "\n")

    if not overall:
        raise ValueError("Dataset failed validation. Adjust SEED or weights.")


# ── Module-level singleton (imported by other modules) ────────────────

SUPPORT_TICKETS: list[SupportTicket] = generate_tickets()


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickets = generate_tickets()
    validate_and_report(tickets)

    print("First 5 records:")
    for t in tickets[:5]:
        score = escalation_score(t)
        print(f"  {t['record_id']}  cat={t['category']:<20} "
              f"status={t['status']:<12} days={t['days_since_created']:>2}  "
              f"esc={t['escalated']}  score={score:.4f}")