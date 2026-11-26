"""Credential errors. Messages and attributes never contain credential values."""
import sys

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
