from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_module.agent import (
    AgentStatus,
    ChatMessage,
    ModelReply,
    ToolCall,
    ToolExecutionResult,
    run_agent,
)
from ai_module.key_routing import ApiKeyEntry, ApiKeyRouter


class FakeModelClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system_instruction, conversation, tools):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                tool_calls=(
                    ToolCall(
                        name="add",
                        arguments={
                            "resource": "inventory",
                            "items": [
                                {"name": "trứng", "quantity": 2, "unit": "piece"}
                            ],
                        },
                    ),
                )
            )
        return ModelReply(text="Đã thêm 2 quả trứng vào kho.")


class FakeDispatcher:
    def execute(self, *, tool_name, payload, auth_context):  # noqa: ANN001
        self.last = (tool_name, payload, auth_context)
        return ToolExecutionResult(
            status="ok",
            data={"resource": payload.get("resource"), "ok": True},
            error=None,
        )


class FakeDeleteModelClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system_instruction, conversation, tools):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                tool_calls=(
                    ToolCall(
                        name="delete",
                        arguments={
                            "resource": "shopping_list",
                            "target": {"item_id": "shop_01"},
                        },
                    ),
                )
            )
        return ModelReply(text="Đã xóa mục shopping list.")


class FakeRepairModelClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system_instruction, conversation, tools):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                tool_calls=(
                    ToolCall(
                        name="update",
                        arguments={"resource": "shopping_list", "target": {"item_id": "shop_01"}},
                    ),
                )
            )
        if self.calls == 2:
            return ModelReply(
                tool_calls=(
                    ToolCall(
                        name="update",
                        arguments={
                            "resource": "shopping_list",
                            "target": {"item_id": "shop_01"},
                            "patch": {"purchased": False},
                        },
                    ),
                )
            )
        return ModelReply(text="Đã bỏ tích shop_01.")


class AgentMockTests(unittest.TestCase):
    def test_run_agent_with_fake_input(self):
        model = FakeModelClient()
        dispatcher = FakeDispatcher()
        result = run_agent(
            user_message="Thêm 2 quả trứng vào kho",
            history=[ChatMessage(role="user", content="Tôi cần cập nhật kho")],
            model_client=model,
            tool_dispatcher=dispatcher,
            auth_context={"user_id": "u_001"},
        )
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.steps_used, 1)
        self.assertTrue(result.tool_results)
        self.assertEqual(result.message, "Đã thêm 2 quả trứng vào kho.")

    def test_key_router_prioritizes_free_then_low_cost(self):
        router = ApiKeyRouter(
            [
                ApiKeyEntry(name="standard", api_key="sk-standard", tier="standard"),
                ApiKeyEntry(name="low_cost", api_key="sk-low", tier="low_cost"),
                ApiKeyEntry(name="free", api_key="sk-free", tier="free"),
            ]
        )

        first = router.next_key()
        self.assertEqual(first.name, "free")

        router.block("free")
        second = router.next_key()
        self.assertEqual(second.name, "low_cost")

        router.block("low_cost")
        third = router.next_key()
        self.assertEqual(third.name, "standard")

    def test_delete_passes_through_to_dispatcher(self):
        model = FakeDeleteModelClient()
        dispatcher = FakeDispatcher()
        result = run_agent(
            user_message="Please remove item shop_01",
            history=[],
            model_client=model,
            tool_dispatcher=dispatcher,
        )
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.steps_used, 1)

    def test_invalid_tool_call_can_be_repaired_once(self):
        model = FakeRepairModelClient()
        dispatcher = FakeDispatcher()
        result = run_agent(
            user_message="Bỏ tích cà chua trong shopping list",
            history=[],
            model_client=model,
            tool_dispatcher=dispatcher,
        )
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.steps_used, 1)


if __name__ == "__main__":
    unittest.main()
