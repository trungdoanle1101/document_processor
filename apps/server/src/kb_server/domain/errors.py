class InvalidRequest(ValueError):
    """The request breaks a rule the service checks."""


class NotAllowed(Exception):
    """The principal may not do this."""


class EmailTaken(Exception):
    """A user with this email already exists."""


class CredentialNotFound(Exception):
    """No such credential, or not the caller's."""


class NotAuthenticated(Exception):
    """No credential, or one that isn't usable."""
