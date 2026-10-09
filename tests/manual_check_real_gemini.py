from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ai_module.agent import ToolExecutionResult, run_agent
from ai_module.gemini_adapter import GeminiAdapter
from ai_module.key_routing import ApiKeyRouter, load_api_keys_from_env_file


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


class InMemoryDispatcher:
    def __init__(
        self,
        *,
        user_context: Mapping[str, Any] | None = None,
        inventory: list[dict[str, Any]] | None = None,
        shopping_list: list[dict[str, Any]] | None = None,
    ) -> None:
        self.user_context: dict[str, Any] = dict(user_context or {})
        self.inventory = [dict(item) for item in (inventory or [])]
        self.shopping_list = [dict(item) for item in (shopping_list or [])]
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def execute(
        self,
        *,
        tool_name: str,
        payload: Mapping[str, Any],
        auth_context: Mapping[str, Any] | None,
    ) -> ToolExecutionResult:
        self.calls.append((tool_name, payload))
        resource = payload.get("resource")
        if resource not in {"user_context", "inventory", "shopping_list"}:
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "INVALID_RESOURCE", "message": f"Unknown resource {resource}"},
            )

        if tool_name == "view":
            return ToolExecutionResult(status="ok", data=self._view(resource), error=None)
        if tool_name == "add":
            return self._add(resource=resource, payload=payload)
        if tool_name == "update":
            return self._update(resource=resource, payload=payload)
        if tool_name == "delete":
            return self._delete(resource=resource, payload=payload)

        return ToolExecutionResult(
            status="error",
            data=None,
            error={"code": "INVALID_TOOL", "message": f"Unsupported tool {tool_name}"},
        )

    def _view(self, resource: str) -> dict[str, Any]:
        if resource == "user_context":
            return {"user_context": dict(self.user_context)}
        if resource == "inventory":
            return {"inventory": [dict(item) for item in self.inventory]}
        return {"shopping_list": [dict(item) for item in self.shopping_list]}

    def _normalize_items(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        items = payload.get("items")
        if isinstance(items, list):
            return [dict(row) for row in items if isinstance(row, Mapping)]
        data = payload.get("data")
        if isinstance(data, Mapping):
            return [dict(data)]
        return []

    def _add(self, *, resource: str, payload: Mapping[str, Any]) -> ToolExecutionResult:
        if resource == "user_context":
            data = payload.get("data")
            if not isinstance(data, Mapping):
                items = self._normalize_items(payload)
                data = items[0] if items else {}
            if not isinstance(data, Mapping) or not data:
                return ToolExecutionResult(
                    status="error",
                    data=None,
                    error={"code": "INVALID_INPUT", "message": "Missing user_context data"},
                )
            self._merge_user_context(dict(data))
            return ToolExecutionResult(status="ok", data={"user_context": self.user_context}, error=None)

        items = self._normalize_items(payload)
        if not items:
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "INVALID_INPUT", "message": "Missing items/data"},
            )
        collection = self.inventory if resource == "inventory" else self.shopping_list
        for new_item in items:
            self._merge_collection_item(collection, new_item)
        return ToolExecutionResult(status="ok", data={"items": collection}, error=None)

    def _update(self, *, resource: str, payload: Mapping[str, Any]) -> ToolExecutionResult:
        patch = payload.get("patch")
        if not isinstance(patch, Mapping):
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "INVALID_INPUT", "message": "Missing patch"},
            )

        if resource == "user_context":
            self._merge_user_context(dict(patch))
            return ToolExecutionResult(status="ok", data={"user_context": self.user_context}, error=None)

        target = payload.get("target")
        if not isinstance(target, Mapping):
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "INVALID_INPUT", "message": "Missing target"},
            )
        collection = self.inventory if resource == "inventory" else self.shopping_list
        row = self._find_item(collection, target)
        if row is None:
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "NOT_FOUND", "message": "Target item not found"},
            )
        row.update(dict(patch))
        return ToolExecutionResult(status="ok", data={"item": row}, error=None)

    def _delete(self, *, resource: str, payload: Mapping[str, Any]) -> ToolExecutionResult:
        if resource == "user_context":
            target = payload.get("target")
            if isinstance(target, Mapping):
                for key in target:
                    self.user_context.pop(key, None)
                return ToolExecutionResult(status="ok", data={"user_context": self.user_context}, error=None)
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "INVALID_INPUT", "message": "Missing target"},
            )

        target = payload.get("target")
        if not isinstance(target, Mapping):
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "INVALID_INPUT", "message": "Missing target"},
            )
        collection = self.inventory if resource == "inventory" else self.shopping_list
        row = self._find_item(collection, target)
        if row is None:
            return ToolExecutionResult(
                status="error",
                data=None,
                error={"code": "NOT_FOUND", "message": "Target item not found"},
            )
        collection.remove(row)
        return ToolExecutionResult(status="ok", data={"deleted": row}, error=None)

    def _merge_user_context(self, patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if isinstance(value, list):
                existing = self.user_context.get(key)
                if not isinstance(existing, list):
                    self.user_context[key] = list(value)
                else:
                    for item in value:
                        if item not in existing:
                            existing.append(item)
            else:
                self.user_context[key] = value

    def _merge_collection_item(self, collection: list[dict[str, Any]], new_item: dict[str, Any]) -> None:
        new_name = str(new_item.get("name", "")).strip().lower()
        if not new_name:
            return
        for row in collection:
            row_name = str(row.get("name", "")).strip().lower()
            if row_name == new_name:
                old_qty = _to_float(row.get("quantity"))
                add_qty = _to_float(new_item.get("quantity"))
                if old_qty is not None and add_qty is not None:
                    row["quantity"] = old_qty + add_qty
                for field in ("unit", "purchased"):
                    if field in new_item:
                        row[field] = new_item[field]
                return
        collection.append(dict(new_item))

    def _find_item(self, collection: list[dict[str, Any]], target: Mapping[str, Any]) -> dict[str, Any] | None:
        item_id = target.get("item_id")
        if item_id:
            for row in collection:
                if row.get("item_id") == item_id:
                    return row
        target_name = target.get("name")
        if target_name:
            normalized = str(target_name).strip().lower()
            for row in collection:
                if str(row.get("name", "")).strip().lower() == normalized:
                    return row
        return None


def _build_adapter() -> GeminiAdapter:
    env_path = PROJECT_ROOT / ".env.ai"
    if not env_path.exists():
        raise RuntimeError(".env.ai not found. Create it from .env.ai.example first.")
    keys = load_api_keys_from_env_file(env_path)
    if not keys:
        raise RuntimeError("No API keys found in .env.ai.")
    model = "gemini-3.5-flash-lite"
    return GeminiAdapter(
        key_router=ApiKeyRouter(keys),
        model=model,
        timeout_seconds=45,
    )


def _print_report(title: str, result: Any, dispatcher: InMemoryDispatcher) -> None:
    print("=" * 100)
    print(title)
    print("- status:", result.status.value)
    print("- steps_used:", result.steps_used)
    print("- message:", result.message)
    if result.error is not None:
        print("- error:", {"code": result.error.code, "message": result.error.message})
    print("- tool_calls:")
    for idx, call in enumerate(dispatcher.calls, start=1):
        print(f"  {idx}. {call[0]} -> {json.dumps(call[1], ensure_ascii=False)}")
    print("- final state:")
    print(
        json.dumps(
            {
                "user_context": dispatcher.user_context,
                "inventory": dispatcher.inventory,
                "shopping_list": dispatcher.shopping_list,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_case(
    *,
    title: str,
    adapter: GeminiAdapter,
    dispatcher: InMemoryDispatcher,
    user_message: str,
) -> None:
    try:
        result = run_agent(
            user_message=user_message,
            history=[],
            model_client=adapter,
            tool_dispatcher=dispatcher,
            auth_context={"user_id": "manual_check_user"},
        )
        _print_report(title, result, dispatcher)
    except Exception as exc:  # noqa: BLE001
        print("=" * 100)
        print(title)
        print("- runtime_error:", str(exc))
        print("- tool_calls_before_error:")
        for idx, call in enumerate(dispatcher.calls, start=1):
            print(f"  {idx}. {call[0]} -> {json.dumps(call[1], ensure_ascii=False)}")
        print("- partial_state:")
        print(
            json.dumps(
                {
                    "user_context": dispatcher.user_context,
                    "inventory": dispatcher.inventory,
                    "shopping_list": dispatcher.shopping_list,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    adapter = _build_adapter()

    case1_dispatcher = InMemoryDispatcher(user_context={"preferences": [], "allergies": []})
    _run_case(
        title="CASE 1A: user_context empty -> add eggs preference + milk intolerance",
        adapter=adapter,
        dispatcher=case1_dispatcher,
        user_message="remember that I love having eggs but have milk intolerance.",
    )

    case1b_dispatcher = InMemoryDispatcher(
        user_context={"preferences": ["spicy food"], "allergies": []}
    )
    _run_case(
        title="CASE 1B: user_context existing description -> merge eggs preference + milk intolerance",
        adapter=adapter,
        dispatcher=case1b_dispatcher,
        user_message="remember that I love having eggs but have milk intolerance. and I also enjoy spicy food.",
    )

    case2_dispatcher = InMemoryDispatcher(
        inventory=[
            {"item_id": "inv_1", "name": "tomato", "quantity": 3, "unit": "piece"},
            {"item_id": "inv_2", "name": "onion", "quantity": 2, "unit": "piece"},
            {"item_id": "inv_3", "name": "chicken breast", "quantity": 500, "unit": "g"},
            {"item_id": "inv_4", "name": "rice", "quantity": 1000, "unit": "g"},
        ]
    )
    _run_case(
        title="CASE 2: remove eggs from inventory when eggs are absent",
        adapter=adapter,
        dispatcher=case2_dispatcher,
        user_message="Please remove all eggs and tomatoes from inventory.",
    )

    case3_dispatcher = InMemoryDispatcher(
        shopping_list=[
            {
                "name": "tomato",
                "quantity": 2,
                "unit": "kg",
                "purchased": True,
            }
        ]
    )
    _run_case(
        title="CASE 3: add 0.5kg meat and uncheck tomatoes",
        adapter=adapter,
        dispatcher=case3_dispatcher,
        user_message= "can you help me with the shopping list? First, add half kilogram meat, and then uncheck the purchased status for tomatoes."
      
    )


if __name__ == "__main__":
    main()
