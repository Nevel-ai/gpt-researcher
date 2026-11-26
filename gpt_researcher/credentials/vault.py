"""Minimal Vault KV v2 reader for the Researcher's provider credentials.

Follows the runtime contract of Chat's `@nevel/secrets` clients without porting the SDK:
- Kubernetes login with the projected ServiceAccount token, reread from the file on every login;
- background polling every 15 s (+/-20% jitter), exponential backoff up to 60 s on failures;
- the last validated document of each source stays usable for 48 h after its last successful read;
- Vault's own 404/deletion and an access denial with a fresh token close a source at once; a denied
  login closes every source; network errors, timeouts, 429/5xx and non-Vault responses keep the cache;
- an invalid new version never replaces the accepted one and does not refresh its age;
- token renew-self at 60-70% of its TTL, a new login on denial, expiry or a capped lease.
Values live only in this process's memory. Nothing is written to env, disk or logs.
"""
import json
import logging
import random
import re
import threading
import time
from urllib.parse import urlsplit

import httpx

from .errors import SOURCES, CredentialConfigError, CredentialUnavailableError

VAULT_ADDRESS = "https://vault.nevel.ai"
TOKEN_FILE = "/var/run/nevel-vault/token"
ENVIRONMENTS = ("dev", "prod")
REFRESH_INTERVAL_S = 15.0
JITTER = 0.2
MAX_RETRY_DELAY_S = 60.0
MAX_STALE_S = 172_800.0
REQUEST_TIMEOUT_S = 5.0
MAX_RESPONSE_BYTES = 1 << 20
MAX_JWT_BYTES = 16_384
MAX_VALUE_LENGTH = 4096

# Existing provider documents shared with Chat; the Researcher reads one field of each.
SOURCE_DOCUMENTS = {
    "openrouter-chat": ("providers/openrouter/{env}/chat", "OPENROUTER_API_KEY"),
    "openai-chat": ("providers/openai/{env}/chat", "OPENAI_API_KEY"),
    "tavily-chat": ("providers/tavily/{env}/chat", "TAVILY_API_KEY"),
}
assert tuple(SOURCE_DOCUMENTS) == SOURCES

_ROLE = re.compile(r"[a-z0-9][a-z0-9-]{0,127}")
_LOG = logging.getLogger("gpt_researcher.credentials")


class VaultError(Exception):
    """Classified Vault failure: code, HTTP status and a short label, never data."""

    def __init__(self, code, status=None, reason=None):
        self.code, self.status, self.reason = code, status, reason
        super().__init__(code + (f" ({reason})" if reason else ""))


class Credential:
    """An accepted value. `generation` changes whenever the accepted value changes."""

    __slots__ = ("value", "version", "generation")

    def __init__(self, value, version, generation):
        self.value, self.version, self.generation = value, version, generation

    def __repr__(self):
        return f"Credential(version={self.version}, generation={self.generation})"


class _Entry:
    __slots__ = ("credential", "fetched")

    def __init__(self, credential, fetched):
        self.credential, self.fetched = credential, fetched


class _Token:
    __slots__ = ("value", "ttl", "renew_at", "expires", "relogin")

    def __init__(self, value, ttl, renew_at, expires):
        self.value, self.ttl, self.renew_at, self.expires, self.relogin = value, ttl, renew_at, expires, False

    def __repr__(self):
        return "_Token(<redacted>)"


def _vault_errors(body):
    return (isinstance(body, dict) and set(body) == {"errors"} and isinstance(body["errors"], list)
            and all(isinstance(item, str) for item in body["errors"]))


def _deleted_version(body):
    data = body.get("data") if isinstance(body, dict) else None
    return (isinstance(data, dict) and data.get("data") is None and isinstance(data.get("metadata"), dict)
            and (bool(data["metadata"].get("deletion_time")) or data["metadata"].get("destroyed") is True))


class VaultCredentials:
    def __init__(self, environment, role, *, address=VAULT_ADDRESS, token_file=TOKEN_FILE, transport=None,
                 clock=time.monotonic, rand=random.random, logger=_LOG):
        if environment not in ENVIRONMENTS:
            raise CredentialConfigError("Researcher Vault environment must be dev or prod")
        if not isinstance(role, str) or not _ROLE.fullmatch(role):
            raise CredentialConfigError("Researcher Vault role is invalid")
        parts = urlsplit(address)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.path not in ("", "/") \
                or parts.query or parts.fragment:
            raise CredentialConfigError("Researcher Vault address must be an HTTPS origin")
        self.environment, self.role = environment, role
        self._mount = f"eks-{environment}"
        self._paths = {name: path.format(env=environment) for name, (path, _) in SOURCE_DOCUMENTS.items()}
        self._token_file = token_file
        self._clock, self._rand, self._log = clock, rand, logger
        # Production: default transport with certificate verification. Proxies from env are ignored so the
        # Vault token never reaches a forward proxy; redirects are never followed.
        self._http = httpx.Client(base_url=address, transport=transport, timeout=REQUEST_TIMEOUT_S,
                                  follow_redirects=False, trust_env=False, verify=True)
        self._lock = threading.Lock()
        self._entries = {}
        self._generation = 0
        self._token = None
        self._fresh_token = False
        self._failures = 0
        self._reported = {}
        self._stop = threading.Event()
        self._thread = None

    def __repr__(self):
        return f"VaultCredentials(environment={self.environment!r}, role={self.role!r}, ready={self.ready()})"

    # --- public, local only -------------------------------------------------------------------------------
    def get(self, source):
        entry = self._entries.get(source)
        if entry is None or self._clock() - entry.fetched >= MAX_STALE_S:
            raise CredentialUnavailableError(source)
        return entry.credential

    def ready(self, sources=SOURCES):
        try:
            for source in sources:
                self.get(source)
        except CredentialUnavailableError:
            return False
        return True

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="researcher-vault", daemon=True)
            self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=REQUEST_TIMEOUT_S * 3)
        with self._lock:
            self._entries, self._token = {}, None
        self._http.close()

    # --- background cycle ---------------------------------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            try:
                self.refresh_once()
            except Exception:  # Never let the refresher die; classified errors are handled inside.
                self._failures += 1
                self._log.error("Research Vault refresh crashed")
            self._stop.wait(self.next_delay())

    def next_delay(self):
        spread = 1 + JITTER * (2 * self._rand() - 1)
        if not self._failures:
            return REFRESH_INTERVAL_S * spread
        return min(MAX_RETRY_DELAY_S, 2.0 ** (self._failures - 1) * spread)

    def refresh_once(self):
        """One login/renew step and one read of every source. Returns True when everything succeeded."""
        self._fresh_token = False
        ok = True
        try:
            self._ensure_token()
        except VaultError as error:
            self._fail(None, error)
            self._failures += 1
            return False
        for source in SOURCES:
            try:
                self._read(source)
                self._recovered(source)
            except VaultError as error:
                ok = False
                self._fail(None if error.code == "AUTH" else source, error)
                if error.code == "AUTH":
                    break
        self._failures = 0 if ok else self._failures + 1
        return ok

    # --- token --------------------------------------------------------------------------------------------
    def _ensure_token(self):
        token, now = self._token, self._clock()
        if token is None or now >= token.expires - REQUEST_TIMEOUT_S or (token.relogin and now >= token.renew_at):
            self._login()
        elif now >= token.renew_at:
            self._renew(token)

    def _schedule(self, lease):
        now = self._clock()
        return now + lease * (0.6 + 0.1 * self._rand()), now + lease

    def _login(self):
        self._token = None
        try:
            with open(self._token_file, "rb") as handle:
                jwt = handle.read(MAX_JWT_BYTES + 1).decode("ascii").strip()
        except (OSError, UnicodeDecodeError):
            raise VaultError("UNAVAILABLE", reason="TOKEN_FILE") from None
        if not jwt or len(jwt) > MAX_JWT_BYTES:
            raise VaultError("UNAVAILABLE", reason="TOKEN_FILE")
        status, body = self._request("POST", f"/v1/auth/{self._mount}/login", json={"role": self.role, "jwt": jwt})
        del jwt
        if status in (400, 401, 403, 404) and _vault_errors(body):
            self._close_all()
            raise VaultError("AUTH", status)
        lease = self._lease(status, body)
        renew_at, expires = self._schedule(lease)
        self._token = _Token(body["auth"]["client_token"], lease, renew_at, expires)
        self._fresh_token = True

    def _renew(self, token):
        try:
            status, body = self._request("POST", "/v1/auth/token/renew-self", token=token.value,
                                         json={"increment": f"{token.ttl}s"})
        except VaultError as error:
            self._fail(None, VaultError(error.code, error.status, "TOKEN_RENEWAL"))
            return
        if status in (401, 403) and _vault_errors(body):
            self._fail(None, VaultError("FORBIDDEN", status, "TOKEN_RENEWAL"))
            self._login()
            return
        try:
            lease = self._lease(status, body)
        except VaultError as error:
            self._fail(None, VaultError(error.code, error.status, "TOKEN_RENEWAL"))
            return
        token.renew_at, token.expires = self._schedule(lease)
        # A lease capped by token_max_ttl: log in again instead of renewing ever shorter tokens.
        token.relogin = lease < token.ttl

    def _lease(self, status, body):
        auth = body.get("auth") if isinstance(body, dict) else None
        if status != 200 or not isinstance(auth, dict) or not isinstance(auth.get("client_token"), str) \
                or not auth["client_token"] or type(auth.get("lease_duration")) is not int or auth["lease_duration"] <= 0:
            raise VaultError("UNAVAILABLE", status, "NOT_VAULT" if status != 200 else "INVALID_RESPONSE")
        return auth["lease_duration"]

    # --- documents ----------------------------------------------------------------------------------------
    def _read(self, source):
        if self._token is None:  # A re-login for an earlier source failed transiently.
            self._login()
        status, body = self._request("GET", f"/v1/kv/data/{self._paths[source]}", token=self._token.value)
        if status in (401, 403) and _vault_errors(body) and not self._fresh_token:
            # 403 also means an expired/revoked token: log in once and repeat before treating it as revocation.
            self._login()
            status, body = self._request("GET", f"/v1/kv/data/{self._paths[source]}", token=self._token.value)
        if status in (401, 403) and _vault_errors(body):
            self._close(source)
            raise VaultError("FORBIDDEN", status)
        if status == 404 and (_vault_errors(body) or _deleted_version(body)):
            self._close(source)
            raise VaultError("NOT_FOUND", status)
        if status != 200:
            raise VaultError("UNAVAILABLE", status, "NOT_VAULT" if 400 <= status < 500 and status != 429 else None)
        value, version = self._validate(source, body)
        with self._lock:
            if self._stop.is_set():  # Closed while this read was in flight: keep nothing.
                return
            previous = self._entries.get(source)
            if previous is not None and previous.credential.value == value:
                credential = previous.credential if previous.credential.version == version \
                    else Credential(value, version, previous.credential.generation)
            else:
                self._generation += 1
                credential = Credential(value, version, self._generation)
            self._entries = {**self._entries, source: _Entry(credential, self._clock())}

    def _validate(self, source, body):
        data = body.get("data") if isinstance(body, dict) else None
        fields = data.get("data") if isinstance(data, dict) else None
        metadata = data.get("metadata") if isinstance(data, dict) else None
        field = SOURCE_DOCUMENTS[source][1]
        value = fields.get(field) if isinstance(fields, dict) else None
        version = metadata.get("version") if isinstance(metadata, dict) else None
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_VALUE_LENGTH or value != value.strip() \
                or type(version) is not int or version < 1:
            raise VaultError("VALIDATION", 200, field)
        return value, version

    def _close(self, source):
        with self._lock:
            if source in self._entries:
                self._entries = {name: entry for name, entry in self._entries.items() if name != source}

    def _close_all(self):
        with self._lock:
            self._entries = {}

    # --- transport ----------------------------------------------------------------------------------------
    def _request(self, method, path, token=None, json=None):
        headers = {"X-Vault-Request": "true"}
        if token:
            headers["X-Vault-Token"] = token
        try:
            with self._http.stream(method, path, headers=headers, json=json) as response:
                declared = response.headers.get("content-length")
                if declared is not None and (not declared.isdigit() or int(declared) > MAX_RESPONSE_BYTES):
                    raise VaultError("UNAVAILABLE", response.status_code, "TOO_LARGE")
                size, chunks = 0, []
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise VaultError("UNAVAILABLE", response.status_code, "TOO_LARGE")
                    chunks.append(chunk)
                status = response.status_code
        except httpx.TimeoutException:
            raise VaultError("UNAVAILABLE", reason="TIMEOUT") from None
        except httpx.HTTPError as error:
            raise VaultError("UNAVAILABLE", reason=type(error).__name__) from None
        try:
            body = json_loads(b"".join(chunks)) if chunks else None
        except ValueError:
            body = None
        return status, body

    # --- reporting ----------------------------------------------------------------------------------------
    def _fail(self, source, error):
        key = source or "token"
        label = (error.code, error.status, error.reason)
        if self._reported.get(key) != label:
            self._reported[key] = label
            self._log.warning("Research Vault refresh failed: source=%s code=%s status=%s reason=%s",
                              key, error.code, error.status, error.reason)

    def _recovered(self, source):
        if self._reported.pop(source, None) is not None:
            self._log.info("Research Vault source recovered: source=%s", source)


def json_loads(data):
    return json.loads(data.decode("utf-8"))
