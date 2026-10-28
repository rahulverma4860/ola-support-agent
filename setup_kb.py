"""
setup_kb.py - writes the 12 knowledge-base documents (2-5 sentences each).
Run: python setup_kb.py   (overwrites knowledge_base/*.txt)
"""
from pathlib import Path

KB_DIR = Path("knowledge_base")
KB_DIR.mkdir(exist_ok=True)

docs = {
"01_ticket_priority_classification.txt": """Ticket Priority Classification Rules

Ola classifies every support ticket into one of four priority levels based on the severity of the issue. Priority 1 (Critical) covers issues that completely prevent a rider or driver from using the platform, such as a payment failure that blocks all bookings or a driver locked out of their account, and needs an initial response within one hour. Priority 2 (High) covers significant degradation such as inaccurate GPS tracking or a refund that is not processing, with an initial response within four hours. Priority 3 (Medium) covers non-blocking issues such as a missing receipt or a promo code that fails to apply, with a response within twenty-four hours, and Priority 4 (Low) covers general inquiries, feature requests and feedback, with a response within forty-eight hours. The system auto-classifies tickets by keywords and category, agents may upgrade a priority when the context warrants it, and any account with three or more escalations in thirty days is raised to Priority 2.""",

"02_sla_by_severity.txt": """SLA Policy by Severity

Ola's Service Level Agreements (SLAs) set the maximum time from ticket creation to final resolution for each priority level. Priority 1 tickets must be fully resolved within four hours of creation, and if that is not possible the ticket is automatically escalated to the on-call senior support manager. Priority 2 tickets have a twenty-four hour resolution SLA, and partial workarounds are acceptable as long as the root cause is fixed within that window. Priority 3 tickets have a seventy-two hour (three business day) SLA, and Priority 4 tickets are resolved within seven calendar days. The SLA clock pauses while a ticket is in "Awaiting Customer Response" status, and breaches trigger an alert to the team lead and are reviewed in the weekly operations meeting.""",

"03_escalation_matrix.txt": """Escalation Matrix

Ola's escalation matrix sets out who resolves a ticket at each tier and when it must be escalated to the next one. Tier 1 (Front-line Agent) handles all new tickets within documented authority, such as standard refunds up to the approved amount or password resets, and must escalate to Tier 2 when the SLA cannot be met, the customer asks for a supervisor, a safety incident is involved, or the amount exceeds the agent's limit. Tier 2 (Senior Support Specialist) handles complex billing disputes, safety incidents, driver deactivation complaints and cases that need engineering or operations support. Tier 3 (Support Manager) takes cases that Tier 2 cannot resolve, including potential legal liability, media attention, VIP customers and systemic issues affecting many accounts, and can issue credits above the standard limit. Every escalation must be documented with a reason code in the ticket before the handoff.""",

"04_refund_compensation_policy.txt": """Refund and Compensation Policy

Ola issues refunds or compensation credits when a service failure is verified to be the fault of the platform or a driver. Full fare refunds are granted when a driver cancels after accepting with no alternative offered, when the driver deviates significantly from the agreed route, or when a technical error causes a double charge. Partial refunds of up to 50% of the fare apply to verified route inefficiencies, waits more than fifteen minutes beyond the expected time, or verified driver-quality complaints. Minor inconveniences receive wallet compensation credits that cannot be withdrawn as cash, and refund requests must be submitted within seven days of the ride, with later requests reviewed case by case on compelling evidence. Cash refunds reach the original payment method in five to seven business days, wallet credits are instant, and fraudulent claims may lead to account suspension.""",

"05_communication_channel_policy.txt": """Customer Communication Channel Policy

Ola offers support through in-app chat, email (support@ola.com) and a phone line reserved for Priority 1 and Priority 2 issues. In-app chat is available twenty-four hours a day and is the recommended channel because it links the conversation to the customer's ride history. Email is monitored during business hours with a four-hour first-response target, and Priority 3 or 4 tickets sent by email or phone may be redirected to in-app chat. All interactions are logged in the customer's account record regardless of channel, so continuity is kept when a customer switches channels. Agents must never share personal contact details and must keep all communication within official Ola channels.""",

"06_business_hours_holiday_support.txt": """Business Hours and Holiday Support Policy

Ola's standard support team works Monday to Saturday during business hours of 8:00 AM to 10:00 PM in the local time zone of the customer's registered city. Outside those hours a reduced in-app chat team handles only Priority 1 and Priority 2 issues, and Priority 3 and 4 tickets are queued for the next business day. On national public holidays a skeleton crew covers critical issues, and non-urgent tickets carry over to the next working day without SLA penalties. During peak periods such as festive seasons or city-wide promotions Ola may add temporary staff, and customers with urgent problems can use the in-app emergency escalation button to reach the on-call Priority 1 team at any hour. Agents must log shift start and end times accurately so that SLA calculations reflect real coverage.""",

"07_repeat_complaint_handling.txt": """Repeat Complaint Handling Policy

When the same customer submits three or more tickets on the same root issue within thirty days, the system tags the account as a repeat complainant and notifies the senior support specialist. The specialist must review the earlier tickets, confirm the root cause was genuinely addressed, and upgrade the priority and send a new resolution plan within two hours if it was not. No agent may close a repeat-complainant ticket without a root-cause note explaining why the issue recurred. Five or more tickets on one issue within sixty days go to the Support Manager, who may escalate to the product or engineering team, and repeats linked to the same driver or vehicle go to the Driver Quality team. Repeat complainants are never penalised or deprioritised, because the goal is to find and fix recurring platform problems.""",

"08_service_credit_policy.txt": """Service Credit Policy

Ola may issue service credits to a customer's wallet as a goodwill gesture when a failure falls short of a cash refund but still caused measurable inconvenience. Front-line agents can issue up to INR 150 per incident without approval, and larger credits need Tier 2 approval. Credits are one-time, non-transferable and expire ninety days after issue. Typical cases are an app crash mid-booking, a driver cancelling with under two minutes' notice on a rainy day, or a promotional discount that failed because of a system error. Credits are not issued for incidents already resolved by a full cash refund, and an account that has received three credits within thirty days needs manager review before another.""",

"09_feedback_collection_process.txt": """Feedback Collection Process

Ola collects customer feedback through a post-ride rating prompt (one to five stars with an optional comment) shown within thirty minutes of ride completion, a post-resolution satisfaction survey sent when a ticket is closed, and a quarterly Net Promoter Score (NPS) survey sent to a random sample of active users. Post-ride ratings help the Driver Quality team monitor drivers, and patterns of negative feedback trigger coaching or review. Post-resolution surveys measure support quality and are reviewed weekly by the Support Manager. Customers can also send unsolicited feedback at any time through the in-app "Share Feedback" feature, which is tagged as a General Inquiry, and all feedback is anonymised before it is shared with product teams. A customer who rates a resolution one or two stars is automatically offered a callback from a senior agent within twenty-four hours.""",

"10_vip_customer_handling.txt": """VIP Customer Handling Policy

Ola's VIP tier includes customers with more than two hundred rides in the past twelve months, corporate account holders with active travel agreements, and customers flagged as VIP by the Support Manager. VIP accounts carry a badge in the ticket system, any VIP ticket is upgraded to at least Priority 2, and it must be handled by a Senior Support Specialist (Tier 2) or above. VIP customers get priority queue placement in every support channel. Senior Specialists may approve credits and compensation up to twice the standard limit without manager sign-off, and if a VIP issue breaches its SLA the Support Manager is notified and must follow up personally. VIP status is reviewed quarterly, and customers who fall below the usage threshold may be demoted with sixty days' notice.""",

"11_outage_communication_protocol.txt": """Outage Communication Protocol

When an outage or service degradation affects more than five percent of active users, Ola activates its outage communication protocol. Within fifteen minutes of the engineering team confirming the incident, the Support Manager must post a status update on the in-app status banner and the official Ola account on Twitter/X, and then update at least every thirty minutes until it is resolved. The final update must give the root cause in plain language and a prevention plan. Agents handling tickets during an outage must acknowledge the system status and avoid promising resolution times that engineering has not confirmed, and SLA timers for non-VIP tickets are paused until the outage is marked resolved. A post-incident summary goes to the Support and Product teams within forty-eight hours, and customers who contacted support during the outage receive a proactive service credit.""",

"12_data_retention_policy.txt": """Data Retention Policy for Support Tickets

Ola retains support ticket data, including conversation history, attachments and agent notes, for three years from the date the ticket was closed, after which the records are permanently deleted from primary storage. Anonymised aggregate data such as category, resolution time and satisfaction score may be kept indefinitely for analytics, and financial records tied to refunds or credits follow local tax rules and may be kept for up to seven years. Customers can request their own ticket history under applicable data protection laws, and Ola fulfils such requests within thirty days through the in-app "My Data" feature or the Data Privacy team. Deletion requests apply only to primary records, because anonymised aggregates and legally retained financial records are kept. Agents can access ticket history only while employed at Ola, and access is revoked immediately on separation with devices wiped within twenty-four hours.""",
}

for filename, content in docs.items():
    (KB_DIR / filename).write_text(content.strip(), encoding="utf-8")
    print(f"  OK  {filename}")
print(f"\nWrote {len(docs)} files to {KB_DIR.resolve()}")