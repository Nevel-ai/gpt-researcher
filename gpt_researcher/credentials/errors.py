"""Credential errors. Messages and attributes never contain credential values."""
import re
import sys

SOURCES = ("openrouter-chat", "openai-chat", "tavily-chat")
# Env name / Vault field of each source (the same name in both places).
SOURCE_FIELDS = {"openrouter-chat": "OPENROUTER_API_KEY", "openai-chat": "OPENAI_API_KEY", "tavily-chat": "TAVILY_API_KEY"}
MAX_VALUE_LENGTH = 4096
# Provider keys travel in HTTP Authorization headers and JSON bodies: only printable ASCII without spaces.
_TRANSPORT_SAFE = re.compile(r"[\x21-\x7e]+")


def transport_safe(value):
    """True for a usable provider key: printable ASCII, no whitespace or control characters, bounded length."""
    return isinstance(value, str) and len(value) <= MAX_VALUE_LENGTH and _TRANSPORT_SAFE.fullmatch(value) is not None


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


def find_in_chain(error, types, limit=32):
    """First exception of `types` reachable through both `__cause__` and `__context__`, breadth-first.

    Both links are followed: an explicit cause can hide the implicit context (for example a budget denial raised
    while a credential error was being handled, or the other way round).
    """
    if not types:
        return None
    queue, seen = [error], set()
    while queue and len(seen) < limit:
        item = queue.pop(0)
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, types):
            return item
        queue.extend((item.__cause__, item.__context__))
    return None


def find_credential_error(error):
    """Return the credential error in an exception chain (SDKs may wrap it)."""
    return find_in_chain(error, CREDENTIAL_ERRORS)


def find_research_budget_error(error):
    """The research budget error anywhere in the chain, if the budget module is loaded (it is whenever one exists).

    Budget classification (exhausted, needs reconciliation, ...) always outranks a credential error: it tells the
    caller whether paid work happened and must not be repeated.
    """
    budget_type = getattr(sys.modules.get("gpt_researcher.utils.budget"), "ResearchBudgetError", None)
    return find_in_chain(error, budget_type) if isinstance(budget_type, type) else None
