"""Authentication — where customer identity actually comes from.

The original demo let the UI type any customer_id into a text box. That's an IDOR
hole: identity asserted by the client is identity you can forge. The fix is the
one every real system uses — the customer authenticates, and the server derives
their id from the verified session. Every downstream ticket lookup is scoped to
THAT id, never to anything the client supplied.

What's real here:
  - passwords are never stored or compared in plaintext; we store a per-user salt
    and a PBKDF2-HMAC-SHA256 hash, and verify with a constant-time comparison.
  - a successful login yields a Session carrying the trusted customer_id.

What's a demo shortcut (and labeled as such): the user directory is seeded in
memory from demo credentials. A real system gets users from an identity provider
(or its own users table) and would NEVER hold the plaintext password even briefly,
as we do here only to generate the demo hashes at import.
"""

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timezone

_PBKDF2_ROUNDS = 200_000


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS).hex()


@dataclass(frozen=True)
class _User:
    customer_id: str
    name: str
    salt: bytes
    pw_hash: str


@dataclass(frozen=True)
class Session:
    """A verified login. customer_id here is TRUSTED — derived from authentication,
    not from client input. This is what handlers scope their reads to.

    `issued_at` (UTC ISO-8601) is stamped at login so an idle/absolute-timeout check
    can be added later. PRODUCTION GAPS (out of scope for this demo, but required for
    a real users path): login rate-limiting / lockout against brute force, session
    expiry enforcement, and a real identity provider instead of the in-memory
    demo directory below."""

    customer_id: str
    customer_name: str
    username: str
    issued_at: str = ""

    def is_expired(self, max_idle_seconds: int) -> bool:
        """Hook for session expiry (not enforced in the demo). True if older than
        max_idle_seconds. Returns False if no issued_at was recorded."""
        if not self.issued_at:
            return False
        try:
            issued = datetime.fromisoformat(self.issued_at)
            return (datetime.now(timezone.utc) - issued).total_seconds() > max_idle_seconds
        except ValueError:
            return False


def _seed_user(username: str, password: str, customer_id: str, name: str) -> tuple[str, _User]:
    # Demo-only: hash a known password at import. A real system stores the hash
    # produced at registration and never sees the plaintext here.
    salt = os.urandom(16)
    return username, _User(customer_id, name, salt, _hash_password(password, salt))


# Demo directory. Passwords are documented in the README as demo credentials.
_USERS: dict[str, _User] = dict(
    [
        _seed_user("jordan", "demo123", "cust_1001", "Jordan"),
        _seed_user("sam", "demo123", "cust_2002", "Sam"),
    ]
)


def authenticate(username: str, password: str) -> Session | None:
    """Verify credentials and return a Session, or None on failure.

    Constant-time hash comparison so a timing side-channel can't leak which part
    of the credential was wrong. Unknown usernames still run a hash to avoid a
    user-enumeration timing difference.
    """
    user = _USERS.get((username or "").strip().lower())
    if user is None:
        # Run a dummy hash so response time doesn't reveal whether the user exists.
        _hash_password(password or "", b"0" * 16)
        return None
    candidate = _hash_password(password or "", user.salt)
    if not hmac.compare_digest(candidate, user.pw_hash):
        return None
    return Session(
        customer_id=user.customer_id, customer_name=user.name,
        username=username.strip().lower(),
        issued_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def demo_usernames() -> list[str]:
    """For the login UI's convenience dropdown (demo only)."""
    return list(_USERS.keys())
