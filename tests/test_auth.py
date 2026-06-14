"""Auth: credential verification and no-plaintext-storage."""

import auth


def test_valid_login_returns_session_with_trusted_id():
    sess = auth.authenticate("jordan", "demo123")
    assert sess is not None
    assert sess.customer_id == "cust_1001"
    assert sess.customer_name == "Jordan"


def test_roles_are_assigned_from_the_directory():
    # Role is trusted (from auth), and gates privileged actions in the UI.
    cust = auth.authenticate("jordan", "demo123")
    assert cust.role == "customer" and cust.is_agent is False
    agent = auth.authenticate("agent", "demo123")
    assert agent is not None and agent.role == "agent" and agent.is_agent is True


def test_wrong_password_rejected():
    assert auth.authenticate("jordan", "wrong") is None


def test_unknown_user_rejected():
    assert auth.authenticate("nobody", "demo123") is None


def test_username_is_case_insensitive():
    assert auth.authenticate("JORDAN", "demo123") is not None


def test_passwords_are_not_stored_in_plaintext():
    # The stored hash must not equal the password, and must be a 64-char sha256 hex.
    user = auth._USERS["jordan"]
    assert user.pw_hash != "demo123"
    assert len(user.pw_hash) == 64
    assert all(c in "0123456789abcdef" for c in user.pw_hash)
