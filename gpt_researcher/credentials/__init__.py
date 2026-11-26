"""Provider credentials of the Nevel deployment: OpenRouter, OpenAI and Tavily.

`CHAT_CONFIG_CREDENTIAL_SOURCE` (the same switch as Chat's Node processes) selects the source once per process:
- unset or `env`: the original upstream behaviour; every call site reads its env variable as before and Vault is
  never contacted (no client, no thread, no import of the Vault module);
- `vault`: OpenRouter/OpenAI/Tavily keys come only from Vault (`CHAT_CONFIG_VAULT_ENV`, `CHAT_CONFIG_VAULT_ROLE`);
  their env copies are ignored and are never a fallback;
- anything else fails at startup. There is no automatic switching in either direction.

Freshness: every provider client built after a change uses the latest validated key. A client that is already
running (a streaming completion, an HTTP request in flight) finishes with the key it was built with; nothing is
closed or retried. Embeddings check the key at every embedding call and rebuild their client when it changed.
"""
import contextvars
import inspect
import os
import threading
from functools import wraps

from .errors import (
    SOURCES, CredentialConfigError, CredentialPolicyError, CredentialUnavailableError, find_credential_error,
    find_in_chain, find_research_budget_error,
)

MODE_ENV = "CHAT_CONFIG_CREDENTIAL_SOURCE"
ENVIRONMENT_ENV = "CHAT_CONFIG_VAULT_ENV"
ROLE_ENV = "CHAT_CONFIG_VAULT_ROLE"
UNAVAILABLE_MESSAGE = "Research credentials unavailable"

__all__ = [
    "CredentialConfigError", "CredentialPolicyError", "CredentialUnavailableError", "find_credential_error",
    "find_in_chain", "find_research_budget_error",
    "credential_mode", "start", "stop", "vault_mode", "provider_key", "provider_credential", "readiness",
    "require_ready", "reject_credential_headers", "reject_credential_kwargs", "refusal_payload", "start_error",
    "with_research_credentials", "SOURCES",
]

_lock = threading.Lock()
_state = None  # (pid, mode, client)
_run_failures = contextvars.ContextVar("research_credential_failures", default=None)


def credential_mode(environ=None):
    value = (os.environ if environ is None else environ).get(MODE_ENV)
    if value is None or value == "env":
        return "env"
    if value == "vault":
        return "vault"
    raise CredentialConfigError(f"{MODE_ENV} must be vault or env")


def start(environ=None, **client_options):
    """Select the mode for this process and, in Vault mode, start its background reader.

    Call once per worker process after it was spawned (FastAPI lifespan), never at import time.
    """
    global _state
    environ = os.environ if environ is None else environ
    mode = credential_mode(environ)
    client = None
    if mode == "vault":
        from .vault import VaultCredentials
        client = VaultCredentials(environ.get(ENVIRONMENT_ENV), environ.get(ROLE_ENV), **client_options)
    with _lock:
        previous, _state = _state, (os.getpid(), mode, client)
    if previous is not None and previous[2] is not None:
        previous[2].close()
    if client is not None:
        client.start()
    return mode


def stop():
    global _state
    with _lock:
        previous, _state = _state, None
    if previous is not None and previous[2] is not None:
        previous[2].close()


def _active():
    state = _state
    if state is not None and state[0] == os.getpid():
        return state[1], state[2]
    # Not started in this process (library/CLI use or a forked child): the mode still applies, and Vault mode
    # without its own started client has no credentials rather than falling back to env.
    return credential_mode(), None


def vault_mode():
    return _active()[0] == "vault"


def provider_credential(source):
    """Vault mode only: the current accepted credential (value, version, generation)."""
    mode, client = _active()
    if mode != "vault":
        raise CredentialConfigError("Vault credentials are not selected")
    try:
        if client is None:
            raise CredentialUnavailableError(source)
        return client.get(source)
    except CredentialUnavailableError:
        failures = _run_failures.get()
        if failures is not None:
            failures.append(source)
        raise


def provider_key(source, legacy):
    """The key for a new provider client: `legacy()` (unchanged env behaviour) or the current Vault value."""
    if source not in SOURCES:
        raise ValueError("Unknown credential source")
    if not vault_mode():
        return legacy()
    return provider_credential(source).value


def readiness():
    mode, client = _active()
    return {"mode": mode, "ready": mode == "env" or (client is not None and client.ready())}


def require_ready():
    mode, client = _active()
    if mode == "vault":
        for source in SOURCES:
            if client is None:
                raise CredentialUnavailableError(source)
            client.get(source)


_HEADER_MARKERS = ("api_key", "apikey", "api-key", "_key", "secret", "token", "password", "authorization")
_ALLOWED_HEADERS = {"nevel_budget", "nevel_tracking", "retriever", "retrievers"}


def _credential_like(name):
    lowered = str(name).lower()
    return lowered not in _ALLOWED_HEADERS and any(marker in lowered for marker in _HEADER_MARKERS)


def reject_credential_headers(headers):
    """Vault mode: research request headers must not carry provider credentials (env mode keeps upstream overrides)."""
    if vault_mode() and isinstance(headers, dict) and any(_credential_like(name) for name in headers):
        raise CredentialPolicyError()


def reject_credential_kwargs(kwargs):
    """Vault mode: configured client kwargs must not replace the managed key or its auth header."""
    if not vault_mode():
        return
    if any(name in kwargs for name in ("openai_api_key", "api_key")):
        raise CredentialPolicyError()
    default_headers = kwargs.get("default_headers")
    if isinstance(default_headers, dict) and any(str(name).lower() in {"authorization", "api-key", "x-api-key"}
                                                 for name in default_headers):
        raise CredentialPolicyError()


def refusal_payload(error):
    """The sanitized body for a credential error: a fixed message and its code, never source details or values."""
    return {"error": UNAVAILABLE_MESSAGE if isinstance(error, CredentialUnavailableError) else str(error),
            "code": error.code}


def start_error(headers=None):
    """Sanitized refusal for a business entrypoint, or None. Local check only, before any budget or paid call."""
    try:
        reject_credential_headers(headers)
        require_ready()
    except (CredentialPolicyError, CredentialUnavailableError) as error:
        return refusal_payload(error)
    return None


def with_research_credentials(function):
    """Gate a research run and fail it if any provider credential became unavailable during the run.

    Legacy inner code turns many provider failures into empty results; this keeps such a run from looking
    successful. Outermost decorator: runs before budget scoping and any paid call.
    """
    signature = inspect.signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        reject_credential_headers(bound.arguments.get("headers"))
        require_ready()
        failures = []
        token = _run_failures.set(failures)
        try:
            result = await function(*args, **kwargs)
        except Exception as error:
            # Never replace a budget classification: it decides whether paid work needs reconciliation.
            if failures and find_credential_error(error) is None and find_research_budget_error(error) is None:
                raise CredentialUnavailableError(failures[0]) from None
            raise
        finally:
            _run_failures.reset(token)
        if failures:
            raise CredentialUnavailableError(failures[0])
        return result

    return wrapped
