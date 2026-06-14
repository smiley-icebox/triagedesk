"""Shared pytest fixtures.

Every test runs against an ISOLATED, freshly-seeded database and trace log in a
temp dir — so tests never touch the real support.db / logs, and never depend on
each other's writes. The db/observability modules read their paths from module
globals, so monkeypatching those globals redirects all of their file I/O.
"""

import pytest

import config
import notifier
import observability
import repository
import seed_data


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    # Repoint the DB at a temp file and force the repository singleton to rebuild
    # against it, so each test gets an isolated, freshly-migrated, seeded database.
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    repository.reset_repository_singleton()
    # Keep tests offline by default: template responses, no live API calls. Tests
    # that exercise the LLM responder inject a fake llm directly.
    monkeypatch.setattr(config, "USE_LLM_RESPONSES", False)
    # Redirect ALL on-disk side effects to the temp dir — traces AND notifications —
    # so tests never touch the project's real logs/ (handle_negative writes both).
    monkeypatch.setattr(observability, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(observability, "TRACE_PATH", str(tmp_path / "traces.jsonl"))
    monkeypatch.setattr(notifier, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(notifier, "NOTIFY_PATH", str(tmp_path / "notifications.log"))
    seed_data.seed()  # clean DB with the known sample tickets
    yield
    repository.reset_repository_singleton()
