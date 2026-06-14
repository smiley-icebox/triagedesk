"""Sample data: a few known tickets, plus a labeled message set for evaluation.

Two distinct purposes:
  1. SAMPLE_TICKETS — planted with fixed ids so the query flow has real tickets to
     look up in a demo, and so the IDOR-scoping behaviour is demonstrable (one
     ticket belongs to a DIFFERENT customer and must NOT be visible to the demo
     customer). Run `seed()` to load them into a clean DB.
  2. EVAL_SET — messages with their correct labels, used by evaluation.py to score
     classification accuracy and routing. This is the project's "test case
     coverage for classification logic" from the brief.

Run as a script (`python seed_data.py`) to (re)build the demo database.
"""

import db
from config import (
    LABEL_GENERAL,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LABEL_QUERY,
    STATUS_IN_PROGRESS,
    STATUS_OPEN,
    STATUS_RESOLVED,
)

# The demo customer the Streamlit UI acts as by default. In production this would
# come from an authenticated session, never be a constant.
DEMO_CUSTOMER_ID = "cust_1001"
DEMO_CUSTOMER_NAME = "Jordan"

# A second customer, used ONLY to prove read-scoping: ticket 940011 belongs to
# them, so a lookup of 940011 by the demo customer must return "not found".
OTHER_CUSTOMER_ID = "cust_2002"

SAMPLE_TICKETS = [
    {
        "ticket_id": "650932",  # matches the brief's worked example
        "customer_id": DEMO_CUSTOMER_ID,
        "customer_name": DEMO_CUSTOMER_NAME,
        "issue": "Debit card replacement had not arrived.",
        "status": STATUS_RESOLVED,
    },
    {
        "ticket_id": "100428",
        "customer_id": DEMO_CUSTOMER_ID,
        "customer_name": DEMO_CUSTOMER_NAME,
        "issue": "Disputed transaction on credit card statement.",
        "status": STATUS_IN_PROGRESS,
    },
    {
        "ticket_id": "781205",
        "customer_id": DEMO_CUSTOMER_ID,
        "customer_name": DEMO_CUSTOMER_NAME,
        "issue": "Unable to update mailing address in the mobile app.",
        "status": STATUS_OPEN,
        # A deadline in the past so the SLA breach check has something to flag.
        "sla_due_at": "2020-01-01T00:00:00+00:00",
    },
    {
        "ticket_id": "940011",  # belongs to ANOTHER customer — scoping demo
        "customer_id": OTHER_CUSTOMER_ID,
        "customer_name": "Sam",
        "issue": "Loan statement download fails.",
        "status": STATUS_OPEN,
    },
]

# Labeled classification cases. Kept balanced across the labels and
# deliberately varied in phrasing so accuracy isn't measured on near-duplicates.
EVAL_SET = [
    # --- positive_feedback ---
    ("Thanks for sorting out my net banking login issue.", LABEL_POSITIVE),
    ("Just wanted to say your support team was fantastic today.", LABEL_POSITIVE),
    ("Really appreciate how quickly you resolved my dispute!", LABEL_POSITIVE),
    ("Great service — the new card arrived faster than expected.", LABEL_POSITIVE),
    ("Kudos to your team, the mobile app update is excellent.", LABEL_POSITIVE),
    # --- negative_feedback ---
    ("My debit card replacement still hasn't arrived.", LABEL_NEGATIVE),
    ("I was charged twice for the same transaction and I'm frustrated.", LABEL_NEGATIVE),
    ("The mobile app keeps crashing when I try to pay a bill.", LABEL_NEGATIVE),
    ("Nobody has gotten back to me about my loan application.", LABEL_NEGATIVE),
    ("My account was locked for no reason and it's unacceptable.", LABEL_NEGATIVE),
    # --- query ---
    ("Could you check the status of ticket 650932?", LABEL_QUERY),
    ("What's the current status of my ticket 100428?", LABEL_QUERY),
    ("Any update on ticket number 781205?", LABEL_QUERY),
    ("Can you tell me where my support request stands?", LABEL_QUERY),
    ("I'd like an update on the ticket I opened last week.", LABEL_QUERY),
    # --- general_query (RAG; product-depth extension beyond the 3 core classes) ---
    ("What are your international transaction fees?", LABEL_GENERAL),
    ("How do I reset my online banking password?", LABEL_GENERAL),
    ("What's the daily ATM withdrawal limit?", LABEL_GENERAL),
    ("What are your branch hours on Saturday?", LABEL_GENERAL),
]

# Ambiguous / out-of-scope messages that SHOULD route to a human (low confidence),
# not be forced into one of the happy-path classes. evaluation.py reports how many of
# these the system correctly declines to auto-handle.
ESCALATION_SET = [
    "hello",
    "asdkfj435",
    "What's the weather like today?",
    "Do you have any job openings?",
]


def seed() -> None:
    """Rebuild a clean demo database with the sample tickets."""
    db.reset_db()
    db.init_db()
    db.load_tickets(SAMPLE_TICKETS)


if __name__ == "__main__":
    seed()
    tickets = db.list_tickets()  # unscoped: show everything we planted
    print(f"Seeded {len(tickets)} tickets into {db.DB_PATH}:")
    for t in tickets:
        print(f"  #{t['ticket_id']}  {t['status']:<12}  ({t['customer_id']})  {t['issue']}")
