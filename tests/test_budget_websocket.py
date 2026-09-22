"""Run actual route/command handlers with inert optional server dependencies."""
import ast
import asyncio
import logging
import json
from pathlib import Path
from typing import Awaitable
import unittest
from unittest.mock import AsyncMock, Mock, patch

from test_budget_scope import module, sign, CLAIMS, SECRET

SERVER = Path(__file__).parents[1] / "backend/server"


def load_handler(filename, name, namespace):
    tree = ast.parse((SERVER / filename).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SERVER / filename), "exec"), namespace)
    return namespace[name]


class BudgetWebsocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_requires_protocol_and_enables_mandatory_budget_validation(self):
        manager = Mock(connect=AsyncMock(), disconnect=AsyncMock())
        communicate = AsyncMock()
        endpoint = load_handler("app.py", "budget_websocket_endpoint", {
            "WebSocket": object, "manager": manager, "handle_websocket_communication": communicate,
        })
        socket = Mock(scope={"subprotocols": []}, close=AsyncMock())
        await endpoint(socket)
        socket.close.assert_awaited_once_with(code=1008)
        manager.connect.assert_not_called()
        communicate.assert_not_called()
        socket.scope["subprotocols"] = ["nevel-budget-v1"]
        await endpoint(socket)
        manager.connect.assert_awaited_once_with(socket, subprotocol="nevel-budget-v1")
        communicate.assert_awaited_once_with(socket, manager, require_budget=True)
        manager.disconnect.assert_awaited_once_with(socket)
        # Check the production registration, not just the isolated function.
        tree = ast.parse((SERVER / "app.py").read_text())
        route = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "budget_websocket_endpoint")
        self.assertEqual(ast.literal_eval(route.decorator_list[0].args[0]), "/ws/budget-v1")

    async def test_invalid_or_missing_capability_never_reaches_start_handler(self):
        for private in (None, {}, {"capability": "forged"}, {"capability": sign(CLAIMS), "mode": "shadow"}):
            with self.subTest(private=private):
                start = AsyncMock()
                communicate = load_handler("server_utils.py", "handle_websocket_communication", {
                    "asyncio": asyncio, "Awaitable": Awaitable, "logger": logging.getLogger(__name__),
                    "verify_budget_start": module.verify_budget_start, "handle_start_command": start,
                })
                socket = Mock(receive_text=AsyncMock(return_value="start " + json.dumps({"headers": {"nevel_budget": private}})), send_json=AsyncMock(), close=AsyncMock())
                await communicate(socket, Mock(), require_budget=True)
                start.assert_not_called()
                socket.close.assert_awaited_once_with(code=1008)
                socket.send_json.assert_awaited_once_with({"type": "error", "metadata": {"budget_error_code": "budget_invalid_transition"}})

    async def test_signed_start_validation_preserves_authenticated_mode(self):
        with patch.dict(module.os.environ, {"JWT_SECRET": SECRET}), patch.object(module.time, "time", return_value=1000):
            for mode in ("shadow", "enforce"):
                claims = {**CLAIMS, "mode": mode}
                data = "start " + json.dumps({"headers": {"nevel_budget": {"capability": sign(claims)}}})
                self.assertEqual(module.verify_budget_start(data), claims)
            for data in ("start", "start {", "start []", "start {}", "start null"):
                with self.assertRaises(module.ResearchBudgetError):
                    module.verify_budget_start(data)


if __name__ == "__main__":
    unittest.main()
