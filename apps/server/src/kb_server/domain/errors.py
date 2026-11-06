class AdminExists(Exception):
    """The system already has an admin, so the first one can't be created."""


class EmailTaken(Exception):
    """A user with this email already exists."""


class GroupNameTaken(Exception):
    """A group with this name already exists."""


class CollectionSlugTaken(Exception):
    """A collection of this slug already exists."""


class GroupNotFound(Exception):
    """No such group."""


class CollectionNotFound(Exception):
    """No such collection."""


class GroupInUse(Exception):
    """Group is use by at least one collection."""


class InvalidRequest(ValueError):
    """The request breaks a rule the service checks."""


class NotAllowed(Exception):
    """The principal may not do this."""


class CredentialNotFound(Exception):
    """No such credential, or not the caller's."""


class NotAuthenticated(Exception):
    """No credential, or one that isn't usable."""


class UserNotFound(Exception):
    """No user found for the given information."""


class LoginFailed(Exception):
    """The email and password don't match an active user."""
