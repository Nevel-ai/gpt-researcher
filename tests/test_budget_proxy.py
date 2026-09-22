"""OPENAI_PROXY routing and real SDK construction, without network calls."""
import importlib.util
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from test_budget_http import module, budget_module, package, provider_module

PROXY = "http://proxy.example.test:8080"


class BudgetProxyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        modules = patch.dict(sys.modules, {
            "budget_http_fixture": package,
            "budget_http_fixture.budget_http": module,
            "gpt_researcher.utils.budget": budget_module,
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.run_budget = budget_module.ResearchBudget("nbgt1.fixture." + "s" * 43, "enforce", Mock())
        self.addAsyncCleanup(self.run_budget.aclose)
        token = budget_module.current_research_budget.set(self.run_budget)
        self.addCleanup(budget_module.current_research_budget.reset, token)

    async def test_proxy_pools_are_separate_reused_and_all_closed(self):
        default = self.run_budget.http_clients()
        proxied = self.run_budget.http_clients(proxy=PROXY)
        self.assertIs(proxied["http_client"], self.run_budget.http_clients(proxy=PROXY)["http_client"])
        self.assertIs(default["http_client"], self.run_budget.http_clients(proxy="")["http_client"])
        self.assertIsNot(default["http_client"], proxied["http_client"])
        await self.run_budget.aclose()
        for pair in (default, proxied):
            self.assertTrue(pair["http_client"].is_closed)
            self.assertTrue(pair["http_async_client"].is_closed)

    async def test_close_failure_does_not_leak_other_pools(self):
        first = {"http_client": Mock(), "http_async_client": Mock(aclose=AsyncMock(side_effect=RuntimeError("close")))}
        second = {"http_client": Mock(), "http_async_client": Mock(aclose=AsyncMock())}
        self.run_budget._clients = {None: first, PROXY: second}
        with self.assertRaisesRegex(RuntimeError, "close"):
            await self.run_budget.aclose()
        for pair in (first, second):
            pair["http_client"].close.assert_called_once()
            pair["http_async_client"].aclose.assert_awaited_once()

    async def test_only_inner_clients_receive_proxy(self):
        sync = Mock()
        asynchronous = Mock()
        with patch.object(module.httpx, "Client", return_value=sync) as sync_factory, patch.object(module.httpx, "AsyncClient", return_value=asynchronous) as async_factory:
            self.run_budget.http_clients(proxy=PROXY)
        for factory, transport_type in ((sync_factory, module.ResearchBudgetSyncTransport), (async_factory, module.ResearchBudgetTransport)):
            self.assertEqual(factory.call_args_list[0].kwargs, {"proxy": PROXY, "follow_redirects": False})
            outer = factory.call_args_list[1].kwargs
            self.assertEqual(set(outer), {"transport", "trust_env"})
            self.assertFalse(outer["trust_env"])
            self.assertIsInstance(outer["transport"], transport_type)
        asynchronous.aclose = AsyncMock()

    @unittest.skipUnless(importlib.util.find_spec("langchain_openai"), "Install researcher SDK dependencies")
    async def test_real_chat_and_embedding_factories_honor_proxy_precedence(self):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        path = Path(__file__).parents[1] / "gpt_researcher/memory/embeddings.py"
        spec = importlib.util.spec_from_file_location("proxy_memory_fixture", path)
        memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(memory)
        # No request is made: the real SDK validators alone reproduced the bug.
        with patch.dict(os.environ, {"OPENAI_PROXY": PROXY, "OPENROUTER_API_KEY": "offline-fixture"}, clear=True):
            for override, expected in (({}, PROXY), ({"openai_proxy": "http://explicit.example.test:8080"}, "http://explicit.example.test:8080"), ({"openai_proxy": ""}, ""), ({"openai_proxy": None}, None)):
                with self.subTest(override=override), patch.object(self.run_budget, "http_clients", wraps=self.run_budget.http_clients) as pools:
                    chat = provider_module.GenericLLMProvider.from_provider("openrouter", model="gpt-4o-mini", **override)
                    embedding = memory.Memory("openai", "text-embedding-3-small", api_key="offline-fixture", **override).get_embeddings()
                    self.assertEqual(pools.call_count, 2)
                    for call in pools.call_args_list:
                        self.assertEqual(call.kwargs, {"proxy": expected})
                    self.assertIsInstance(chat.llm, ChatOpenAI)
                    self.assertIsInstance(embedding, OpenAIEmbeddings)
                    for sdk in (chat.llm, embedding):
                        self.assertEqual(sdk.openai_proxy, "")
                        self.assertIsInstance(sdk.http_client._transport, module.ResearchBudgetSyncTransport)
                        self.assertIsInstance(sdk.http_async_client._transport, module.ResearchBudgetTransport)
                        self.assertFalse(sdk.http_client._mounts)
                        self.assertFalse(sdk.http_async_client._mounts)
            # Search/Tavily remains on the default pool despite OPENAI_PROXY.
            self.assertIs(self.run_budget.http_clients()["http_client"], self.run_budget.http_clients(proxy=None)["http_client"])


if __name__ == "__main__":
    unittest.main()
