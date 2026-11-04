from kb_server.services.passwords import Passwords


def test_the_right_password_verifies(passwords: Passwords) -> None:
    assert passwords.verify(passwords.hash("secret password"), "secret password")


def test_a_wrong_password_does_not(passwords: Passwords) -> None:
    assert not passwords.verify(passwords.hash("secret password"), "guess")


def test_nothing_verifies_without_a_hash(passwords: Passwords) -> None:
    # not even the dummy password the missing hash is checked against
    assert not passwords.verify(None, "not a real password")
