"""Credential errors. Messages and attributes never contain credential values."""

SOURCES = ("openrouter-chat", "openai-chat", "tavily-chat")


class CredentialConfigError(RuntimeError):
    """Invalid credential configuration. The process must not start with it."""


class CredentialUnavailableError(RuntimeError):
    """A required provider credential is not loaded, is stale or was revoked."""

    code = "credentials_unavailable"
    retryable = False

    def __init__(self, source=None):
        self.source = source if source in SOURCES else None
        super().__init__("Research credentials unavailable")


class CredentialPolicyError(RuntimeError):
    """A request or configuration tried to supply its own provider credential."""

    code = "credential_override_forbidden"
    retryable = False

    def __init__(self):
        super().__init__("Provider credentials cannot be supplied with the request")


CREDENTIAL_ERRORS = (CredentialUnavailableError, CredentialPolicyError)


def find_credential_error(error):
    """Return the credential error in an exception chain (SDKs may wrap it)."""
    seen = set()
    for _ in range(8):
        if isinstance(error, CREDENTIAL_ERRORS):
            return error
        if not isinstance(error, BaseException) or id(error) in seen:
            break
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return None
