"""Vault-backed provider credentials, entirely offline: in-memory Vault, providers and budget callbacks."""
import ast
import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace
from typing import Awaitable
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from credentials_fixture import credentials
from test_stream_usage_evidence import module as provider_module
from test_budget_http import module as budget_transport
from gpt_researcher.credentials import vault
from gpt_researcher.credentials.errors import CredentialConfigError, CredentialPolicyError, CredentialUnavailableError

ROOT = Path(__file__).parents[1]
ROLE = "chat-dev-gpt-researcher"
DOCS = {
    "providers/openrouter/dev/chat": ("OPENROUTER_API_KEY", "openrouter-chat"),
    "providers/openai/dev/chat": ("OPENAI_API_KEY", "openai-chat"),
    "providers/tavily/dev/chat": ("TAVILY_API_KEY", "tavily-chat"),
}
SECRET_MARK = "FIXTURE-SECRET"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeVault:
    """Speaks the KV v2 / Kubernetes auth / renew-self API shapes the client relies on."""

    def __init__(self):
        self.values = {path: {field: f"{field.lower()}-{SECRET_MARK}-v1"} for path, (field, _) in DOCS.items()}
        self.versions = {path: 1 for path in DOCS}
        self.overrides = {}
        self.login_denied = False
        self.renew_responses = []
        self.lease = 240
        self.down = False
        self.jwts = []
        self.tokens = set()
        self.calls = []

    def set(self, path, value, version=None):
        field = DOCS[path][0]
        self.values[path] = {field: value}
        self.versions[path] = version or self.versions[path] + 1

    def count(self, suffix):
        return sum(1 for _, path in self.calls if path.endswith(suffix))

    def __call__(self, request):
        assert request.url.scheme == "https" and request.url.host == "vault.nevel.ai"
        self.calls.append((request.method, request.url.path))
        if self.down:
            raise httpx.ConnectError("unreachable")
        path = request.url.path
        if path == "/v1/auth/eks-dev/login":
            body = json.loads(request.content)
            assert body["role"] == ROLE
            self.jwts.append(body["jwt"])
            if self.login_denied:
                return httpx.Response(403, json={"errors": ["permission denied"]})
            token = f"hvs.{SECRET_MARK}.{len(self.jwts)}"
            self.tokens.add(token)
            return httpx.Response(200, json={"auth": {"client_token": token, "lease_duration": self.lease, "renewable": True}})
        token = request.headers.get("x-vault-token")
        if path == "/v1/auth/token/renew-self":
            if self.renew_responses:
                status, payload = self.renew_responses.pop(0)
                return httpx.Response(status, json=payload)
            return httpx.Response(200, json={"auth": {"client_token": token, "lease_duration": self.lease}})
        if path.startswith("/v1/kv/data/"):
            doc = path[len("/v1/kv/data/"):]
            if token not in self.tokens:
                return httpx.Response(403, json={"errors": ["permission denied"]})
            if doc in self.overrides:
                return self.overrides[doc]()
            if doc not in self.values:
                return httpx.Response(404, json={"errors": []})
            return httpx.Response(200, json={"data": {"data": dict(self.values[doc]), "metadata": {
                "version": self.versions[doc], "deletion_time": "", "destroyed": False}}})
        return httpx.Response(404, json={"errors": []})


class VaultFixture:
    def __init__(self, test):
        self.fake = FakeVault()
        self.clock = Clock()
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.token_file = Path(directory.name) / "token"
        self.token_file.write_text(f"sa-jwt-{SECRET_MARK}-1\n")
        self.client = vault.VaultCredentials("dev", ROLE, token_file=str(self.token_file), transport=httpx.MockTransport(self.fake),
                                             clock=self.clock, rand=lambda: 0.5)
        test.addCleanup(self.client.close)

    def install(self, test):
        """Make this client the process' credential source, as `credentials.start()` does in a worker."""
        previous = credentials._state
        credentials._state = (os.getpid(), "vault", self.client)
        test.addCleanup(setattr, credentials, "_state", previous)


def value(field, version):
    return f"{field.lower()}-{SECRET_MARK}-v{version}"


class VaultProtocolTests(unittest.TestCase):
    def setUp(self):
        self.v = VaultFixture(self)
        self.client, self.fake, self.clock = self.v.client, self.v.fake, self.v.clock

    def test_login_reads_exact_documents_and_validates_without_env_or_disk_writes(self):
        before = dict(os.environ)
        self.assertFalse(self.client.ready())
        self.assertTrue(self.client.refresh_once())
        self.assertTrue(self.client.ready())
        self.assertEqual(self.fake.jwts, [f"sa-jwt-{SECRET_MARK}-1"])
        self.assertEqual(sorted(p for m, p in self.fake.calls if m == "GET"),
                         sorted(f"/v1/kv/data/{path}" for path in DOCS))
        self.assertEqual(self.client.get("openrouter-chat").value, value("OPENROUTER_API_KEY", 1))
        self.assertEqual(dict(os.environ), before)
        self.assertNotIn(SECRET_MARK, repr(self.client) + repr(self.client.get("tavily-chat")))

    def test_invalid_configuration_fails_clearly(self):
        for env, role, address in [("staging", ROLE, vault.VAULT_ADDRESS), ("dev", "", vault.VAULT_ADDRESS),
                                   ("dev", "Role With Spaces", vault.VAULT_ADDRESS), ("dev", ROLE, "http://vault.nevel.ai"),
                                   ("dev", ROLE, "https://vault.nevel.ai/v1"), ("dev", ROLE, "https://user@vault.nevel.ai")]:
            with self.subTest(env=env, role=role, address=address), self.assertRaises(CredentialConfigError):
                vault.VaultCredentials(env, role, address=address)

    def test_production_client_verifies_tls_ignores_env_proxies_and_never_follows_redirects(self):
        client = vault.VaultCredentials("prod", "chat-prod-gpt-researcher")
        self.addCleanup(client.close)
        self.assertFalse(client._http.follow_redirects)
        self.assertFalse(client._http.trust_env)
        self.assertEqual(str(client._http.base_url), "https://vault.nevel.ai")
        self.assertEqual(client._mount, "eks-prod")
        self.assertEqual(client._paths["tavily-chat"], "providers/tavily/prod/chat")

    def test_service_account_token_is_reread_on_every_login(self):
        self.client.refresh_once()
        self.v.token_file.write_text(f"sa-jwt-{SECRET_MARK}-2")
        self.fake.tokens.clear()  # The Vault token expired or was revoked: reads get 403 until a new login.
        self.assertTrue(self.client.refresh_once())
        self.assertEqual(self.fake.jwts, [f"sa-jwt-{SECRET_MARK}-1", f"sa-jwt-{SECRET_MARK}-2"])
        self.assertTrue(self.client.ready())

    def test_renewal_schedule_transient_failure_denial_and_capped_lease(self):
        self.client.refresh_once()
        self.clock.now += 240 * 0.65 - 1
        self.client.refresh_once()
        self.assertEqual(self.fake.count("renew-self"), 0)
        self.clock.now += 2
        self.fake.renew_responses = [(503, {"errors": ["sealed"]})]
        self.assertTrue(self.client.refresh_once())  # Reads continue with the valid token.
        self.assertEqual((self.fake.count("renew-self"), len(self.fake.jwts)), (1, 1))
        self.fake.renew_responses = [(403, {"errors": ["permission denied"]})]
        self.assertTrue(self.client.refresh_once())
        self.assertEqual((self.fake.count("renew-self"), len(self.fake.jwts)), (2, 2))  # Denied renewal: new login.
        self.clock.now += 240 * 0.65 + 1
        self.fake.renew_responses = [(200, {"auth": {"client_token": "kept", "lease_duration": 30}})]
        self.client.refresh_once()  # Lease capped by token_max_ttl: no further shrinking renewals.
        self.clock.now += 30 * 0.65 + 1
        self.client.refresh_once()
        self.assertEqual((self.fake.count("renew-self"), len(self.fake.jwts)), (3, 3))

    def test_last_known_good_lasts_48_hours_from_last_success_and_recovers_without_restart(self):
        self.client.refresh_once()
        fetched = self.clock.now
        self.fake.down = True
        self.clock.now = fetched + vault.MAX_STALE_S - 1
        self.assertFalse(self.client.refresh_once())
        self.assertEqual(self.client.get("openai-chat").value, value("OPENAI_API_KEY", 1))
        self.clock.now = fetched + vault.MAX_STALE_S
        self.assertFalse(self.client.ready())
        with self.assertRaises(CredentialUnavailableError):
            self.client.get("openai-chat")
        self.fake.down = False
        self.assertTrue(self.client.refresh_once())
        self.assertTrue(self.client.ready())

    def test_transient_vault_and_non_vault_responses_keep_the_cache(self):
        self.client.refresh_once()
        path = "providers/tavily/dev/chat"
        for response in [lambda: httpx.Response(503, json={"errors": ["Vault is sealed"]}),
                         lambda: httpx.Response(429, json={"errors": ["rate limited"]}),
                         lambda: httpx.Response(403, text="<html>proxy denied</html>"),
                         lambda: httpx.Response(404, json={"message": "gateway"})]:
            self.fake.overrides[path] = response
            self.assertFalse(self.client.refresh_once())
            self.assertTrue(self.client.ready())

    def test_deleted_document_closes_at_once_and_stays_closed_through_an_outage(self):
        self.client.refresh_once()
        path = "providers/openrouter/dev/chat"
        self.fake.overrides[path] = lambda: httpx.Response(404, json={"data": {"data": None, "metadata": {
            "version": 2, "deletion_time": "2026-11-26T00:00:00Z", "destroyed": False}}})
        self.client.refresh_once()
        with self.assertRaises(CredentialUnavailableError):
            self.client.get("openrouter-chat")
        self.assertEqual(self.client.get("tavily-chat").value, value("TAVILY_API_KEY", 1))  # Other sources unaffected.
        self.fake.down = True
        self.client.refresh_once()
        self.assertFalse(self.client.ready())
        self.fake.down = False
        del self.fake.overrides[path]
        self.fake.set(path, value("OPENROUTER_API_KEY", 3), version=3)
        self.client.refresh_once()
        self.assertEqual(self.client.get("openrouter-chat").value, value("OPENROUTER_API_KEY", 3))

    def test_denied_path_relogs_once_and_closes_only_with_a_fresh_token(self):
        self.client.refresh_once()
        path = "providers/openai/dev/chat"
        self.fake.overrides[path] = lambda: httpx.Response(403, json={"errors": ["permission denied"]})
        self.client.refresh_once()
        self.assertEqual(len(self.fake.jwts), 2)  # 403 might be an expired token: one new login, then repeat.
        with self.assertRaises(CredentialUnavailableError):
            self.client.get("openai-chat")
        self.assertTrue(self.client.ready(("openrouter-chat", "tavily-chat")))

    def test_denied_login_closes_every_source(self):
        self.client.refresh_once()
        self.fake.tokens.clear()
        self.fake.login_denied = True
        self.assertFalse(self.client.refresh_once())
        for source in credentials.SOURCES:
            with self.assertRaises(CredentialUnavailableError):
                self.client.get(source)

    def test_invalid_new_version_keeps_the_accepted_one_without_refreshing_its_age(self):
        self.client.refresh_once()
        fetched = self.clock.now
        path = "providers/tavily/dev/chat"
        for broken in ["", " padded ", None, "x" * (vault.MAX_VALUE_LENGTH + 1)]:
            self.fake.values[path] = {"TAVILY_API_KEY": broken}
            self.fake.versions[path] += 1
            self.clock.now += 60
            self.client.refresh_once()
            self.assertEqual(self.client.get("tavily-chat").value, value("TAVILY_API_KEY", 1))
        self.clock.now = fetched + vault.MAX_STALE_S
        with self.assertRaises(CredentialUnavailableError):
            self.client.get("tavily-chat")

    def test_generation_changes_only_with_the_value(self):
        self.client.refresh_once()
        first = self.client.get("openrouter-chat")
        self.fake.versions["providers/openrouter/dev/chat"] += 1  # Same value rewritten as a new version.
        self.client.refresh_once()
        same = self.client.get("openrouter-chat")
        self.assertEqual((same.generation, same.version), (first.generation, 2))
        self.fake.set("providers/openrouter/dev/chat", value("OPENROUTER_API_KEY", 2))
        self.client.refresh_once()
        self.assertNotEqual(self.client.get("openrouter-chat").generation, first.generation)

    def test_response_size_is_bounded_while_streaming(self):
        pulled = []

        class Endless(httpx.SyncByteStream):
            def __iter__(self):
                for index in range(10_000):
                    pulled.append(index)
                    yield b"x" * 65_536

        self.client.refresh_once()
        self.fake.overrides["providers/openai/dev/chat"] = lambda: httpx.Response(200, stream=Endless())
        self.assertFalse(self.client.refresh_once())
        self.assertLessEqual(len(pulled), vault.MAX_RESPONSE_BYTES // 65_536 + 1)
        self.assertTrue(self.client.ready())
        self.fake.overrides["providers/openai/dev/chat"] = lambda: httpx.Response(
            200, headers={"content-length": str(vault.MAX_RESPONSE_BYTES + 1)}, stream=Endless())
        with self.assertRaises(vault.VaultError) as raised:
            self.client._request("GET", "/v1/kv/data/providers/openai/dev/chat", token=self.client._token.value)
        self.assertEqual(raised.exception.reason, "TOO_LARGE")

    def test_polling_jitter_and_backoff(self):
        rand = iter([0.0, 1.0])
        self.client._rand = lambda: next(rand)
        self.assertAlmostEqual(self.client.next_delay(), 12.0)
        self.assertAlmostEqual(self.client.next_delay(), 18.0)
        self.client._rand = lambda: 0.5
        delays = []
        for failures in range(1, 9):
            self.client._failures = failures
            delays.append(self.client.next_delay())
        self.assertEqual(delays, [1, 2, 4, 8, 16, 32, 60, 60])

    def test_logs_and_errors_never_contain_values_tokens_or_jwts(self):
        with self.assertLogs("gpt_researcher.credentials", level="INFO") as logs:
            self.client.refresh_once()
            self.fake.overrides["providers/openai/dev/chat"] = lambda: httpx.Response(403, json={"errors": ["permission denied"]})
            self.client.refresh_once()
            self.fake.down = True
            self.client.refresh_once()
            logging.getLogger("gpt_researcher.credentials").info("marker")
        text = "\n".join(logs.output)
        self.assertIn("code=FORBIDDEN", text)
        self.assertNotIn(SECRET_MARK, text)
        self.assertNotIn("hvs.", text)


class ModeTests(unittest.TestCase):
    def setUp(self):
        previous = credentials._state
        self.addCleanup(setattr, credentials, "_state", previous)
        credentials._state = None

    def test_env_mode_is_the_default_keeps_upstream_reads_and_never_builds_a_vault_client(self):
        with patch.object(vault, "VaultCredentials") as constructor:
            for environ in ({}, {"CHAT_CONFIG_CREDENTIAL_SOURCE": "env", "CHAT_CONFIG_VAULT_ROLE": ROLE}):
                self.assertEqual(credentials.start(environ=environ), "env")
        constructor.assert_not_called()
        legacy = Mock(return_value="env-value")
        self.assertEqual(credentials.provider_key("tavily-chat", legacy), "env-value")
        self.assertEqual(credentials.readiness(), {"mode": "env", "ready": True})
        self.assertIsNone(credentials.start_error({"tavily_api_key": "caller"}))  # Upstream override stays in env mode.

    def test_unknown_mode_fails_and_vault_mode_requires_environment_and_role(self):
        for mode in ("", "VAULT", "sops", "auto"):
            with self.subTest(mode=mode), self.assertRaises(CredentialConfigError):
                credentials.start(environ={"CHAT_CONFIG_CREDENTIAL_SOURCE": mode})
        for environ in ({}, {"CHAT_CONFIG_VAULT_ENV": "staging", "CHAT_CONFIG_VAULT_ROLE": ROLE}, {"CHAT_CONFIG_VAULT_ENV": "dev"}):
            with self.assertRaises(CredentialConfigError):
                credentials.start(environ={"CHAT_CONFIG_CREDENTIAL_SOURCE": "vault", **environ})

    def test_vault_mode_never_falls_back_to_env_copies(self):
        with patch.dict(os.environ, {"CHAT_CONFIG_CREDENTIAL_SOURCE": "vault", "TAVILY_API_KEY": "env-copy"}):
            legacy = Mock(return_value="env-copy")
            with self.assertRaises(CredentialUnavailableError):
                credentials.provider_key("tavily-chat", legacy)
            legacy.assert_not_called()
            self.assertEqual(credentials.readiness(), {"mode": "vault", "ready": False})

    def test_started_worker_reads_in_the_background_and_stops(self):
        v = VaultFixture(self)
        environ = {"CHAT_CONFIG_CREDENTIAL_SOURCE": "vault", "CHAT_CONFIG_VAULT_ENV": "dev", "CHAT_CONFIG_VAULT_ROLE": ROLE}
        self.assertEqual(credentials.start(environ=environ, token_file=str(v.token_file),
                                           transport=httpx.MockTransport(v.fake)), "vault")
        self.addCleanup(credentials.stop)
        client = credentials._state[2]
        client._thread.join(0)
        for _ in range(200):
            if credentials.readiness()["ready"]:
                break
            threading_wait(0.01)
        self.assertEqual(credentials.readiness(), {"mode": "vault", "ready": True})
        credentials.stop()
        self.assertFalse(client._thread.is_alive())
        self.assertFalse(client.ready())  # Closing drops the in-memory values.


def threading_wait(seconds):
    import threading
    threading.Event().wait(seconds)


class RunScopeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.v = VaultFixture(self)
        self.v.install(self)

    async def test_run_is_refused_before_it_starts_without_credentials_or_with_header_credentials(self):
        body = AsyncMock()

        async def research(task, headers=None):
            await body(task)

        run = credentials.with_research_credentials(research)
        with self.assertRaises(CredentialUnavailableError):
            await run("task", headers={})
        self.v.client.refresh_once()
        with self.assertRaises(CredentialPolicyError):
            await run("task", headers={"tavily_api_key": "caller"})
        body.assert_not_called()
        await run("task", headers={"nevel_tracking": {}, "nevel_budget": {"capability": "c"}})
        body.assert_awaited_once()

    async def test_credentials_lost_during_a_run_fail_it_even_when_inner_code_swallowed_the_error(self):
        self.v.client.refresh_once()

        async def research(task, headers=None):
            self.v.fake.overrides["providers/tavily/dev/chat"] = lambda: httpx.Response(404, json={"errors": []})
            self.v.client.refresh_once()
            try:
                await asyncio.to_thread(credentials.provider_key, "tavily-chat", None)  # Context reaches executors.
            except CredentialUnavailableError:
                return "report from empty search results"

        with self.assertRaises(CredentialUnavailableError):
            await credentials.with_research_credentials(research)("task", headers={})


class CallSiteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.v = VaultFixture(self)
        self.v.install(self)
        self.v.client.refresh_once()
        self.no_budget = SimpleNamespace(current_research_budget=SimpleNamespace(get=lambda: None),
                                         ResearchBudgetError=budget_transport.ResearchBudgetError)

    def build(self, provider, **kwargs):
        constructor = Mock()
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": self.no_budget,
                                      "langchain_openai": SimpleNamespace(ChatOpenAI=constructor),
                                      "langchain_core.rate_limiters": SimpleNamespace(InMemoryRateLimiter=Mock())}), \
                patch.object(provider_module, "_check_pkg"), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-copy", "OPENAI_API_KEY": "env-copy"}):
            provider_module.GenericLLMProvider.from_provider(provider, model="gpt-4o-mini", **kwargs)
        return constructor.call_args.kwargs

    def test_each_new_llm_client_takes_the_current_vault_key(self):
        self.assertEqual(self.build("openrouter")["openai_api_key"], value("OPENROUTER_API_KEY", 1))
        self.assertEqual(self.build("openai")["openai_api_key"], value("OPENAI_API_KEY", 1))
        self.v.fake.set("providers/openrouter/dev/chat", value("OPENROUTER_API_KEY", 2))
        self.v.client.refresh_once()
        self.assertEqual(self.build("openrouter")["openai_api_key"], value("OPENROUTER_API_KEY", 2))

    def test_configured_keys_or_auth_headers_cannot_replace_the_managed_key(self):
        for kwargs in ({"openai_api_key": "other"}, {"api_key": "other"}, {"default_headers": {"Authorization": "Bearer x"}}):
            for provider in ("openrouter", "openai"):
                with self.subTest(provider=provider, kwargs=kwargs), self.assertRaises(CredentialPolicyError):
                    self.build(provider, **kwargs)

    def test_env_mode_provider_factory_is_unchanged(self):
        credentials._state = (os.getpid(), "env", None)
        self.assertEqual(self.build("openrouter")["openai_api_key"], "env-copy")
        self.assertNotIn("openai_api_key", self.build("openai"))  # LangChain reads its env as upstream does.

    def load(self, relative, name, extra=None):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        loaded = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": self.no_budget, **(extra or {})}):
            spec.loader.exec_module(loaded)
        return loaded

    def test_tavily_search_resolves_the_key_per_request_and_rejects_header_keys(self):
        posts = []
        requests = SimpleNamespace(post=lambda url, data, headers, timeout: posts.append(json.loads(data)) or
                                   SimpleNamespace(status_code=200, json=lambda: {"results": [{"url": "u", "content": "c"}]}))
        module = self.load("gpt_researcher/retrievers/tavily/tavily_search.py", "tavily_search_fixture", {"requests": requests})
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": self.no_budget}):
            search = module.TavilySearch("query")
            search.search()
            self.v.fake.set("providers/tavily/dev/chat", value("TAVILY_API_KEY", 2))
            self.v.client.refresh_once()
            search.search()
            self.assertEqual([p["api_key"] for p in posts], [value("TAVILY_API_KEY", 1), value("TAVILY_API_KEY", 2)])
            with self.assertRaises(CredentialPolicyError):
                module.TavilySearch("query", headers={"tavily_api_key": "caller"})
            self.v.fake.overrides["providers/tavily/dev/chat"] = lambda: httpx.Response(403, json={"errors": ["denied"]})
            self.v.client.refresh_once()
            with self.assertRaises(CredentialUnavailableError):  # Not swallowed into "no results".
                search.search()
        self.assertEqual(len(posts), 2)

    def test_tavily_extract_builds_its_client_per_extraction_in_vault_mode(self):
        clients = []

        def client(api_key):
            clients.append(api_key)
            return SimpleNamespace(extract=lambda urls: {"failed_results": ["skip"], "results": []})

        spec = importlib.util.spec_from_file_location("extract_pkg.tavily_extract.tavily_extract",
                                                      ROOT / "gpt_researcher/scraper/tavily_extract/tavily_extract.py")
        module = importlib.util.module_from_spec(spec)
        package, sub = ModuleType("extract_pkg"), ModuleType("extract_pkg.tavily_extract")
        package.__path__, sub.__path__ = [], []
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": self.no_budget, "extract_pkg": package,
                                      "extract_pkg.tavily_extract": sub, "bs4": SimpleNamespace(BeautifulSoup=Mock()),
                                      "extract_pkg.utils": SimpleNamespace(get_relevant_images=Mock(), extract_title=Mock())}):
            spec.loader.exec_module(module)
            with patch.dict(sys.modules, {"tavily": SimpleNamespace(TavilyClient=client)}), \
                    patch.dict(os.environ, {"TAVILY_API_KEY": "env-copy"}):
                extract = module.TavilyExtract("https://example.test", session=Mock())
                extract.scrape()
                self.v.fake.set("providers/tavily/dev/chat", value("TAVILY_API_KEY", 2))
                self.v.client.refresh_once()
                extract.scrape()
                self.v.fake.overrides["providers/tavily/dev/chat"] = lambda: httpx.Response(404, json={"errors": []})
                self.v.client.refresh_once()
                with self.assertRaises(CredentialUnavailableError):
                    extract.scrape()
        self.assertEqual(clients, [value("TAVILY_API_KEY", 1), value("TAVILY_API_KEY", 2)])

    async def test_embeddings_rebuild_at_the_call_boundary_after_rotation_and_fail_after_revocation(self):
        built = []

        class FakeEmbeddings:
            def __init__(self, model, openai_api_key, **kwargs):
                built.append((openai_api_key, kwargs))
                self.key = openai_api_key

            def embed_query(self, text):
                return [self.key]

            async def aembed_documents(self, texts):
                return [[self.key] for _ in texts]

        module = self.load("gpt_researcher/memory/embeddings.py", "embeddings_fixture")
        base = type("Embeddings", (), {})
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": self.no_budget,
                                      "langchain_openai": SimpleNamespace(OpenAIEmbeddings=FakeEmbeddings),
                                      "langchain_core.embeddings": SimpleNamespace(Embeddings=base)}):
            embeddings = module.Memory("openai", "text-embedding-3-small", chunk_size=16).get_embeddings()
            self.assertNotIn(SECRET_MARK, repr(embeddings))
            self.assertEqual(embeddings.embed_query("q"), [value("OPENAI_API_KEY", 1)])
            self.assertEqual(embeddings.embed_query("q"), [value("OPENAI_API_KEY", 1)])
            self.assertEqual(len(built), 1)  # Unchanged key: the same client, nothing rebuilt per call.
            self.v.fake.set("providers/openai/dev/chat", value("OPENAI_API_KEY", 2))
            self.v.client.refresh_once()
            self.assertEqual(await embeddings.aembed_documents(["a"]), [[value("OPENAI_API_KEY", 2)]])
            self.assertEqual(built[1][1], {"chunk_size": 16})  # Factory settings (budget clients, chunking) kept.
            self.v.fake.overrides["providers/openai/dev/chat"] = lambda: httpx.Response(404, json={"errors": []})
            self.v.client.refresh_once()
            with self.assertRaises(CredentialUnavailableError):
                embeddings.embed_query("q")
            with self.assertRaises(CredentialPolicyError):
                module.Memory("openai", "text-embedding-3-small", openai_api_key="other")
            with self.assertRaises(CredentialConfigError):
                module.Memory("custom", "local-model")


class LongRunRotationTests(unittest.IsolatedAsyncioTestCase):
    """A streaming completion opened with key v1 outlives a rotation; the next request uses v2, each billed once."""

    async def test_in_flight_stream_completes_on_old_key_while_new_requests_use_the_new_key(self):
        v = VaultFixture(self)
        v.install(self)
        v.client.refresh_once()
        url = "https://openrouter.ai/api/v1/chat/completions"
        provider_requests = []
        release_first = asyncio.Event()

        class Gated(httpx.AsyncByteStream):
            def __init__(self, gated):
                self.gated, self.closed = gated, False

            async def __aiter__(self):
                yield b'data: {"id":"gen-%d","choices":[{"delta":{"content":"part one\\n"}}]}\n\n' % len(provider_requests)
                if self.gated:
                    await release_first.wait()
                yield b'data: {"choices":[{"delta":{"content":"part two\\n"}}],"usage":{"cost":0.01}}\n\ndata: [DONE]\n\n'

            async def aclose(self):
                self.closed = True

        streams = []

        def openrouter(request):
            provider_requests.append(request.headers["authorization"])
            streams.append(Gated(gated=len(provider_requests) == 1))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=streams[-1])

        operations = [Mock(mode="enforce", model_id="openai/gpt-4o-mini", max_output_tokens=200,
                           correlation={"budget_operation_id": f"deep-research:run:{i}", "budget_reservation_id": f"reservation0000{i}"})
                      for i in range(2)]
        run_budget = Mock(mode="enforce")
        run_budget.reserve_model.side_effect = operations
        native = httpx.AsyncClient(transport=httpx.MockTransport(openrouter))
        metered = httpx.AsyncClient(transport=budget_transport.ResearchBudgetTransport(run_budget, native), trust_env=False)
        self.addAsyncCleanup(metered.aclose)
        scope = Mock(mode="enforce")
        scope.http_clients.return_value = {"http_client": Mock(), "http_async_client": metered}

        class FakeChatOpenAI:
            """Minimal SDK stand-in: one streaming HTTP request authenticated with the key it was built with."""

            def __init__(self, openai_api_key, http_async_client, max_retries, **kwargs):
                self.key, self.http, self.max_retries = openai_api_key, http_async_client, max_retries

            async def astream(self, messages, **kwargs):
                async with self.http.stream("POST", url, headers={"Authorization": f"Bearer {self.key}"},
                                            json={"model": "gpt-4o-mini", "messages": messages, "stream": True}) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: {"):
                            payload = json.loads(line[6:])
                            content = payload["choices"][0]["delta"]["content"]
                            yield SimpleNamespace(content=content, usage_metadata={}, response_metadata={})

        budget_module = SimpleNamespace(current_research_budget=SimpleNamespace(get=lambda: scope),
                                        ResearchBudgetError=budget_transport.ResearchBudgetError)
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": budget_module,
                                      "langchain_openai": SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
                                      "langchain_core.rate_limiters": SimpleNamespace(InMemoryRateLimiter=Mock())}), \
                patch.object(provider_module, "_check_pkg"):
            first = provider_module.GenericLLMProvider.from_provider("openrouter", model="gpt-4o-mini", max_retries=3)
            self.assertEqual(first.llm.max_retries, 0)  # Budget enforcement still disables SDK retries.
            socket = Mock(send_json=AsyncMock())
            long_task = asyncio.create_task(first.get_chat_response([{"role": "user", "content": "long"}], stream=True, websocket=socket))
            while len(provider_requests) < 1:
                await asyncio.sleep(0)
            v.fake.set("providers/openrouter/dev/chat", value("OPENROUTER_API_KEY", 2))
            await asyncio.to_thread(v.client.refresh_once)  # Background reader picks up the rotation mid-stream.
            second = provider_module.GenericLLMProvider.from_provider("openrouter", model="gpt-4o-mini")
            self.assertEqual(await second.get_chat_response([{"role": "user", "content": "next"}], stream=True, websocket=socket),
                             "part one\npart two\n")
            self.assertFalse(long_task.done())
            release_first.set()
            self.assertEqual(await long_task, "part one\npart two\n")
        self.assertEqual(provider_requests, [f"Bearer {value('OPENROUTER_API_KEY', 1)}", f"Bearer {value('OPENROUTER_API_KEY', 2)}"])
        self.assertEqual(run_budget.reserve_model.call_count, 2)  # One admission per native request, no duplicate.
        for index, operation in enumerate(operations):
            operation.mark_started.assert_called_once()
            operation.observe.assert_called_once_with(f"gen-{index + 1}")
            operation.finalize.assert_called_once_with("0.01")
        self.assertTrue(all(stream.closed for stream in streams))


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return [namespace[name] for name in names]


class FakeJSONResponse:
    def __init__(self, status_code=200, content=None):
        self.status_code, self.content = status_code, content


class EntrypointTests(unittest.IsolatedAsyncioTestCase):
    """The real handlers refuse business work before budget scoping or any paid call."""

    def setUp(self):
        self.v = VaultFixture(self)
        self.v.install(self)
        self.app = ROOT / "backend/server/app.py"
        self.utils = ROOT / "backend/server/server_utils.py"

    def app_namespace(self, **extra):
        return {"credentials": credentials, "JSONResponse": FakeJSONResponse, "logger": logging.getLogger(__name__),
                "json": json, "time": Mock(time=Mock(return_value=1)),
                # Annotations are evaluated eagerly before Python 3.14.
                "ResearchRequest": object, "BackgroundTasks": object, "ChatRequest": object, "Request": object, **extra}

    async def test_probes_report_only_mode_and_result_and_liveness_is_independent(self):
        ready, health = load_functions(self.app, ["ready", "health"], self.app_namespace())
        self.assertEqual(await health(), {"status": "ok"})
        response = await ready()
        self.assertEqual((response.status_code, response.content), (503, {"credentials": {"mode": "vault", "ready": False}}))
        self.v.client.refresh_once()
        self.assertEqual((await ready()).status_code, 200)

    async def test_http_research_and_chat_entrypoints_refuse_before_work(self):
        agent, write = Mock(), AsyncMock()
        namespace = self.app_namespace(ChatAgentWithMemory=agent, write_report=write, sanitize_filename=lambda name: name)
        refusal, chat, report_chat, generate = load_functions(
            self.app, ["_credential_refusal", "chat", "research_report_chat", "generate_report"], namespace)
        request = Mock(json=AsyncMock(return_value={"report": "r", "messages": []}))
        for response in [await chat(SimpleNamespace(report="r", messages=[])), await report_chat("id", request),
                         await generate(SimpleNamespace(headers=None, task="t", generate_in_background=True), Mock())]:
            self.assertEqual((response.status_code, response.content["code"]), (503, "credentials_unavailable"))
            self.assertNotIn(SECRET_MARK, json.dumps(response.content))
        self.v.client.refresh_once()
        override = await generate(SimpleNamespace(headers={"openai_api_key": "caller"}, task="t", generate_in_background=False), Mock())
        self.assertEqual((override.status_code, override.content["code"]), (400, "credential_override_forbidden"))
        agent.assert_not_called()
        write.assert_not_called()

    async def test_multi_agent_entrypoint_refuses_before_work(self):
        task = AsyncMock()
        execute, = load_functions(self.utils, ["execute_multi_agents"], {
            "credential_start_error": credentials.start_error, "JSONResponse": FakeJSONResponse,
            "run_research_task": task, "stream_output": Mock(), "Any": object})
        response = await execute(Mock(active_connections=[Mock()]))
        self.assertEqual(response.status_code, 503)
        task.assert_not_called()

    async def run_socket(self, data, require_budget=False, verify=None, start=None):
        start = start or AsyncMock()
        namespace = {"asyncio": asyncio, "Awaitable": Awaitable, "logger": logging.getLogger(__name__), "json": json,
                     "traceback": traceback,
                     "verify_budget_start": verify or Mock(), "handle_start_command": start,
                     "credential_start_error": credentials.start_error, "find_credential_error": credentials.find_credential_error,
                     "find_budget_error": lambda error: None}
        communicate, _ = load_functions(self.utils, ["handle_websocket_communication", "_start_headers"], namespace)
        messages = [data]

        async def receive():
            if messages:
                return messages.pop()
            await asyncio.sleep(0.05)
            raise RuntimeError("closed")

        socket = Mock(receive_text=receive, send_json=AsyncMock(), close=AsyncMock())
        await communicate(socket, Mock(), require_budget=require_budget)
        return start, socket

    async def test_socket_start_is_refused_without_credentials_and_keeps_budget_authentication_first(self):
        data = "start " + json.dumps({"task": "t", "report_type": "research_report", "headers": {"nevel_budget": {"capability": "c"}}})
        verify = Mock()
        start, socket = await self.run_socket(data, require_budget=True, verify=verify)
        verify.assert_called_once_with(data)
        start.assert_not_called()
        socket.close.assert_not_called()  # Protocol unchanged: an error message, no policy close.
        socket.send_json.assert_awaited_once_with({"type": "error", "content": "error", "output": "Research credentials unavailable",
                                                   "metadata": {"credentials_error_code": "credentials_unavailable"}})
        denied = Mock(side_effect=budget_transport.ResearchBudgetError("budget_invalid_transition"))
        start, socket = await self.run_socket(data, require_budget=True, verify=denied)
        socket.close.assert_awaited_once_with(code=1008)
        start.assert_not_called()

    async def test_socket_rejects_header_credentials_and_starts_when_ready(self):
        self.v.client.refresh_once()
        start, socket = await self.run_socket("start " + json.dumps({"task": "t", "headers": {"tavily_api_key": "caller"}}))
        start.assert_not_called()
        self.assertEqual(socket.send_json.await_args.args[0]["metadata"], {"credentials_error_code": "credential_override_forbidden"})
        start, socket = await self.run_socket("start " + json.dumps({"task": "t", "headers": {"nevel_tracking": {}}}))
        start.assert_awaited_once()

    async def test_credential_failure_during_a_run_is_reported_sanitized(self):
        self.v.client.refresh_once()
        failing = AsyncMock(side_effect=CredentialUnavailableError("tavily-chat"))
        start, socket = await self.run_socket("start " + json.dumps({"task": "t"}), start=failing)
        start.assert_awaited_once()
        socket.send_json.assert_awaited_once_with({"type": "logs", "content": "error", "output": "Research credentials unavailable",
                                                   "metadata": {"credentials_error_code": "credentials_unavailable"}})

    def test_run_agent_is_gated_before_budget_scoping(self):
        tree = ast.parse((ROOT / "backend/server/websocket_manager.py").read_text())
        run_agent = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_agent")
        self.assertEqual([d.id for d in run_agent.decorator_list], ["with_research_credentials", "with_research_budget"])


if __name__ == "__main__":
    unittest.main()
