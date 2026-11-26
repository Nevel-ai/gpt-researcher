"""Provider credentials of the Nevel deployment: OpenRouter, OpenAI and Tavily.

`CHAT_CONFIG_CREDENTIAL_SOURCE` (the same switch as Chat's Node processes) selects the source once per process:
- unset or `env`: every call site reads its env variable as before and Vault is never contacted (no client, no thread,
  no import of the Vault module). In a configured Nevel workload (`CHAT_CONFIG_VAULT_ENV` set, which the chart
  always does when the Researcher Vault block is enabled) env mode is the manual mode: readiness and every business
  entrypoint additionally require all three keys to be present and transport-safe. Without that variable the
  upstream behaviour is unchanged;
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
from collections.abc import Mapping
from functools import wraps

from .errors import (
    SOURCE_FIELDS, SOURCES, transport_safe, CredentialConfigError, CredentialPolicyError, CredentialUnavailableError, find_credential_error,
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


def _managed_workload(environ):
    """A configured Nevel Researcher (chart-provided environment) rather than an upstream/local installation."""
    value = environ.get(ENVIRONMENT_ENV)
    if value is None:
        return False
    if value not in ("dev", "prod"):
        raise CredentialConfigError(f"{ENVIRONMENT_ENV} must be dev or prod")
    return True


def start(environ=None, **client_options):
    """Select the mode for this process and, in Vault mode, start its background reader.

    Call once per worker process after it was spawned (FastAPI lifespan), never at import time.
    """
    global _state
    environ = os.environ if environ is None else environ
    mode = credential_mode(environ)
    strict = _managed_workload(environ)
    client = None
    if mode == "vault":
        from .vault import VaultCredentials
        client = VaultCredentials(environ.get(ENVIRONMENT_ENV), environ.get(ROLE_ENV), **client_options)
    with _lock:
        previous, _state = _state, (os.getpid(), mode, client, strict)
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


def _strict_env():
    state = _state
    if state is not None and state[0] == os.getpid() and len(state) > 3:
        return state[3]
    return _managed_workload(os.environ)


def _env_unavailable():
    """Manual env mode of a configured workload: the first source whose env copy is missing or unusable.

    Reads the process env only (no Vault, no polling). Values are checked, never logged or returned.
    """
    for source in SOURCES:
        if not transport_safe(os.environ.get(SOURCE_FIELDS[source])):
            return source
    return None


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
    if mode == "env":
        return {"mode": mode, "ready": not _strict_env() or _env_unavailable() is None}
    return {"mode": mode, "ready": client is not None and client.ready()}


def require_ready():
    mode, client = _active()
    if mode == "env":
        missing = _env_unavailable() if _strict_env() else None
        if missing is not None:
            raise CredentialUnavailableError(missing)  # No automatic switch to Vault either.
        return
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


_AUTH_HEADERS = {"authorization", "api-key", "x-api-key"}


def _auth_header(name):
    if isinstance(name, bytes):
        name = name.decode("latin-1")
    return str(name).strip().lower() in _AUTH_HEADERS


def _rejects_headers(headers):
    """Header names are case-insensitive on the wire; an unreadable header collection fails closed."""
    if headers is None:
        return False
    if not isinstance(headers, Mapping):
        return True
    return any(_auth_header(name) for name in headers)


def reject_credential_kwargs(kwargs):
    """Vault mode: configured client kwargs must not replace the managed key or its auth header.

    The OpenAI SDK lets both client defaults (`default_headers`) and per-request options (`extra_headers`) override
    the bearer header it builds from the key. LangChain sends `model_kwargs` (and moves unknown top-level kwargs such
    as `extra_headers` there) into every request, so all three places are checked. Other headers and options pass.
    """
    if not vault_mode():
        return
    if any(name in kwargs for name in ("openai_api_key", "api_key")):
        raise CredentialPolicyError()
    model_kwargs = kwargs.get("model_kwargs")
    if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
        raise CredentialPolicyError()
    if any(_rejects_headers(headers) for headers in (
            kwargs.get("default_headers"), kwargs.get("extra_headers"), (model_kwargs or {}).get("extra_headers"))):
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
