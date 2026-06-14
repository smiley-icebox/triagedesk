"""Outbound notifications — abstracted behind an interface.

When a ticket is opened, a real system tells the customer (email/SMS) and often
alerts a team. The CHANNEL is an integration detail that shouldn't leak into the
handlers, so it sits behind a Notifier interface. The default ConsoleNotifier
writes to a local log (runs anywhere, no accounts); an SMTP or Twilio notifier is
"implement Notifier.send and point the factory at it" — the same seam pattern as
the storage repository.

Like the data layer, send() never raises into the request path: a notification
failure must not break the customer's interaction.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
NOTIFY_PATH = os.path.join(LOG_DIR, "notifications.log")


class Notifier(ABC):
    @abstractmethod
    def send(self, channel: str, recipient: str, subject: str, body: str) -> bool:
        """Deliver a message. Returns success; never raises."""


class ConsoleNotifier(Notifier):
    """Writes notifications to logs/notifications.log. The local stand-in for a real
    email/SMS provider — proves the wiring without an external account."""

    def send(self, channel, recipient, subject, body):
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "channel": channel,
            "recipient": recipient,
            "subject": subject,
            "body": body,
        }
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(NOTIFY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            return True
        except Exception:
            return False


# Swap point: a real deployment returns an SmtpNotifier/TwilioNotifier here,
# selected by env, with no change to the handlers that call notify().
_notifier: Notifier | None = None


def get_notifier() -> Notifier:
    global _notifier
    if _notifier is None:
        _notifier = ConsoleNotifier()
    return _notifier


def notify_ticket_created(customer_name: str, customer_id: str, ticket_id: str) -> bool:
    """Tell the customer a ticket was opened. Convenience wrapper over the notifier."""
    return get_notifier().send(
        channel="email",
        recipient=customer_id,
        subject=f"Support ticket #{ticket_id} created",
        body=f"Hi {customer_name}, we've opened ticket #{ticket_id} and our team will "
             f"follow up shortly.",
    )


def read_recent(limit: int = 50) -> list[dict]:
    """Recent notifications for the dashboard."""
    if not os.path.exists(NOTIFY_PATH):
        return []
    try:
        with open(NOTIFY_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        return list(reversed([json.loads(x) for x in lines if x.strip()]))[:limit]
    except Exception:
        return []


def clear() -> None:
    if os.path.exists(NOTIFY_PATH):
        os.remove(NOTIFY_PATH)
