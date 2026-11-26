"""The headers the real OpenAI SDK sends for a managed OpenRouter/OpenAI client, through an offline httpx transport.

Requires the pinned `openai` SDK (CI installs it); no provider is contacted. LangChain is not installed here: a thin
adapter forwards the constructor arguments to the SDK the way `ChatOpenAI` does (client `default_headers`, and
`model_kwargs` -- including unknown top-level kwargs such as `extra_headers`, which LangChain moves there -- passed
to every `chat.completions.create` call).
"""
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
import openai

from credentials_fixture import credentials
from test_research_credentials import VaultFixture, budget_transport, provider_module, value
from gpt_researcher.credentials.errors import CredentialPolicyError

COMPLETION = {"id": "gen-1", "object": "chat.completion", "created": 0, "model": "m",
              "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}


class SdkHeaderTests(unittest.TestCase):
    def setUp(self):
        self.v = VaultFixture(self)
        self.v.install(self)
        self.v.client.refresh_once()
        self.sent = []

    def send(self, provider, **kwargs):
        """Build through the real provider factory, then send one completion; return the request the SDK sent."""
        sent = self.sent

        def transport(request):
            sent.append(request)
            return httpx.Response(200, json=COMPLETION)

        class ChatOpenAIAdapter:
            def __init__(self, openai_api_key, openai_api_base=None, default_headers=None, model_kwargs=None,
                         model=None, **rest):
                moved = {name: rest.pop(name) for name in ("extra_headers",) if name in rest}
                self.model, self.model_kwargs = model, {**(model_kwargs or {}), **moved}
                self.client = openai.OpenAI(api_key=openai_api_key, base_url=openai_api_base or "https://api.openai.com/v1",
                                            default_headers=default_headers, max_retries=0,
                                            http_client=httpx.Client(transport=httpx.MockTransport(transport)))

            def invoke(self):
                return self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": "q"}],
                                                           **self.model_kwargs)

        no_budget = SimpleNamespace(current_research_budget=SimpleNamespace(get=lambda: None),
                                    ResearchBudgetError=budget_transport.ResearchBudgetError)
        with patch.dict(sys.modules, {"gpt_researcher.utils.budget": no_budget,
                                      "langchain_openai": SimpleNamespace(ChatOpenAI=ChatOpenAIAdapter),
                                      "langchain_core.rate_limiters": SimpleNamespace(InMemoryRateLimiter=Mock())}), \
                patch.object(provider_module, "_check_pkg"), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-copy", "OPENAI_API_KEY": "env-copy"}):
            provider_module.GenericLLMProvider.from_provider(provider, model="m", **kwargs).llm.invoke()
        return sent[-1]

    def test_sdk_lets_per_request_auth_headers_replace_the_bearer_in_any_case(self):
        """Why the guard exists: in env mode (upstream behaviour kept) such a header reaches the provider.

        The SDK merges header dicts case-sensitively: the exact name replaces its bearer, another case is sent as a
        second Authorization header next to it. Either way a configured credential goes out with the request.
        """
        credentials._state = (os.getpid(), "env", None)
        for name in ("Authorization", "authorization", "AUTHORIZATION"):
            for kwargs in ({"model_kwargs": {"extra_headers": {name: "Bearer other"}}},
                           {"extra_headers": {name: "Bearer other"}}, {"default_headers": {name: "Bearer other"}}):
                with self.subTest(kwargs=kwargs):
                    sent = self.send("openrouter", **kwargs).headers.get_list("authorization")
                    self.assertIn("Bearer other", sent)
                    if name == "Authorization":
                        self.assertEqual(sent, ["Bearer other"])

    def test_vault_mode_refuses_auth_overrides_before_any_request(self):
        for name in ("Authorization", "authorization", "AUTHORIZATION", " Authorization ", "Api-Key", "X-API-KEY"):
            for kwargs in ({"model_kwargs": {"extra_headers": {name: "Bearer other"}}},
                           {"extra_headers": {name: "Bearer other"}}, {"default_headers": {name: "Bearer other"}}):
                for provider in ("openrouter", "openai"):
                    with self.subTest(provider=provider, kwargs=kwargs), self.assertRaises(CredentialPolicyError):
                        self.send(provider, **kwargs)
        self.assertEqual(self.sent, [])

    def test_managed_bearer_is_sent_with_benign_headers_and_follows_rotation(self):
        kwargs = {"default_headers": {"HTTP-Referer": "https://nevel.ai", "X-Title": "Nevel"},
                  "model_kwargs": {"extra_headers": {"X-Request-Tag": "a"}, "top_p": 0.5}}
        for provider, field in (("openrouter", "OPENROUTER_API_KEY"), ("openai", "OPENAI_API_KEY")):
            with self.subTest(provider=provider):
                request = self.send(provider, **kwargs)
                self.assertEqual(request.headers.get_list("authorization"), [f"Bearer {value(field, 1)}"])
                for name, expected in (("http-referer", "https://nevel.ai"), ("x-title", "Nevel"), ("x-request-tag", "a")):
                    self.assertEqual(request.headers[name], expected)
                self.assertIn(b'"top_p":0.5', request.content.replace(b" ", b""))
                # Top-level extra_headers (LangChain moves it into model_kwargs; both at once is a LangChain error).
                request = self.send(provider, extra_headers={"X-Other": "c"})
                self.assertEqual(request.headers.get_list("authorization"), [f"Bearer {value(field, 1)}"])
                self.assertEqual(request.headers["x-other"], "c")
        self.assertEqual(self.sent[0].url.host, "openrouter.ai")
        self.v.fake.set("providers/openrouter/dev/chat", value("OPENROUTER_API_KEY", 2))
        self.v.client.refresh_once()
        self.assertEqual(self.send("openrouter", **kwargs).headers["authorization"],
                         f"Bearer {value('OPENROUTER_API_KEY', 2)}")


if __name__ == "__main__":
    unittest.main()
