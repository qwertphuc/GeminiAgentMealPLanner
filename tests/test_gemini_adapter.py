from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_module.agent import ChatMessage, ToolCall
from ai_module.gemini_adapter import GeminiAdapter, TransportResponse
from ai_module.key_routing import ApiKeyEntry, ApiKeyRouter


class FakeTransport:
    def __init__(self, planned_status_by_key):
        self.planned_status_by_key = dict(planned_status_by_key)
        self.calls = []

    def post_json(self, *, url, headers, payload, timeout_seconds):  # noqa: ANN001
        key = headers["x-goog-api-key"]
        self.calls.append(key)
        status = self.planned_status_by_key.get(key, 200)
        if status == 200:
            return TransportResponse(
                status_code=200,
                body={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "view",
                                            "args": {"resource": "inventory"},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        return TransportResponse(
            status_code=status,
            body={"error": {"message": f"failed on {key}", "code": status}},
        )


class GeminiAdapterTests(unittest.TestCase):
    def test_fallback_when_free_key_rate_limited(self):
        router = ApiKeyRouter(
            [
                ApiKeyEntry(name="free", api_key="k_free", tier="free"),
                ApiKeyEntry(name="low_cost", api_key="k_low", tier="low_cost"),
                ApiKeyEntry(name="standard", api_key="k_std", tier="standard"),
            ]
        )
        transport = FakeTransport({"k_free": 429, "k_low": 200})
        adapter = GeminiAdapter(
            key_router=router,
            model="gemini-3.8-flash",
            transport=transport,
        )

        reply = adapter.complete(
            system_instruction="test",
            conversation=[ChatMessage(role="user", content="Tôi còn gì trong kho?")],
            tools=["view", "add", "update", "delete"],
        )

        self.assertEqual(transport.calls, ["k_free", "k_low"])
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertIsInstance(reply.tool_calls[0], ToolCall)
        self.assertEqual(reply.tool_calls[0].name, "view")
        self.assertEqual(reply.tool_calls[0].arguments["resource"], "inventory")

    def test_fallback_to_standard_when_low_cost_fails(self):
        router = ApiKeyRouter(
            [
                ApiKeyEntry(name="free", api_key="k_free", tier="free"),
                ApiKeyEntry(name="low_cost", api_key="k_low", tier="low_cost"),
                ApiKeyEntry(name="standard", api_key="k_std", tier="standard"),
            ]
        )
        transport = FakeTransport({"k_free": 429, "k_low": 503, "k_std": 200})
        adapter = GeminiAdapter(
            key_router=router,
            model="gemini-3.5-flash",
            transport=transport,
        )

        reply = adapter.complete(
            system_instruction="test",
            conversation=[ChatMessage(role="user", content="Tôi còn gì trong kho?")],
            tools=["view", "add", "update", "delete"],
        )

        self.assertEqual(transport.calls, ["k_free", "k_low", "k_std"])
        self.assertEqual(reply.tool_calls[0].name, "view")


if __name__ == "__main__":
    unittest.main()
