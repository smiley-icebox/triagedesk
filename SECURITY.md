# Security Policy

TriageDesk is an **educational / portfolio project** (an Applied GenAI capstone). It
is **not a production system** and processes only synthetic demo data — there are no
real customers, accounts, or secrets in this repository.

## Reporting a vulnerability

If you find a security issue you'd still like to report, please use **GitHub's private
vulnerability reporting** (the repository's **Security** tab → *Report a vulnerability*)
rather than opening a public issue. I'll respond as time allows — this is a personal
project, not a maintained service.

## Known, intentional limitations (by design, for a demo)

These are deliberate scope choices, documented here and in the code/`WRITEUP.md` so
they're not mistaken for oversights. A real deployment would close each one:

- **Auth is a demo directory.** Users and the password `demo123` are seeded in memory
  (`auth.py`); a real system uses an identity provider. Passwords *are* hashed properly
  (PBKDF2-HMAC-SHA256, per-user salt, constant-time compare) — but there is **no login
  rate-limiting / lockout** and **no enforced session expiry** (`Session.is_expired`
  exists as a hook but isn't wired in).
- **PII at rest is not redacted.** A complaint's raw text is stored in `support_tickets`
  and shown in the UI. Redaction is best-effort and applied only at the **log** boundary
  (`observability.py`); production would redact at the storage boundary and encrypt at
  rest.
- **Prompt-injection defense is structural, not perfect.** The real protection is that
  the classifier returns a constrained `{label, confidence}` and *code* routes from it,
  so the worst an injection achieves is a misroute (never data exfiltration or an
  invented ticket). A regex pre-filter catches blatant attempts; it is not exhaustive.
- **SQLite, single-node.** Fine for a demo; production would use a managed database with
  proper access controls, backups, and encryption.

## What *is* handled

- Ticket reads/writes are scoped to the authenticated `customer_id` (no IDOR); only the
  agent role can drive ticket lifecycle.
- No secrets are committed — `.env` is gitignored; the LLM API key lives only locally.
- Parameterized SQL throughout; the data layer never raises into the request path.
