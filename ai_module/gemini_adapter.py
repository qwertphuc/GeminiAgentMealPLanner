from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent import ChatMessage, ModelReply, ToolCall
from .key_routing import ApiKeyEntry, ApiKeyRouter

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_STATUS_CODES = frozenset({401, 403, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    body: Mapping[str, Any]


class HttpTransport(Protocol):
    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        ...


class UrllibTransport:
    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
                if isinstance(parsed, Mapping):
                    return TransportResponse(
                        status_code=response.status,
                        body=dict(parsed),
                    )
                return TransportResponse(status_code=response.status, body={})
        except HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            parsed_body: Mapping[str, Any]
            try:
                parsed = json.loads(body_text) if body_text else {}
                parsed_body = parsed if isinstance(parsed, Mapping) else {}
            except json.JSONDecodeError:
                parsed_body = {"raw_error": body_text}
            return TransportResponse(status_code=exc.code, body=dict(parsed_body))
        except URLError as exc:
            raise GeminiTransportError(f"Network error while calling Gemini: {exc}") from exc


class GeminiTransportError(RuntimeError):
    pass


class GeminiApiError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        body: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}

    @property
    def should_fallback(self) -> bool:
        return self.status_code in RETRYABLE_STATUS_CODES


class GeminiAdapter:
    def __init__(
        self,
        *,
        key_router: ApiKeyRouter,
        model: str,
        timeout_seconds: float = 30.0,
        base_url: str = DEFAULT_BASE_URL,
        transport: HttpTransport | None = None,
    ) -> None:
        self._key_router = key_router
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._transport = transport or UrllibTransport()

    def complete(
        self,
        *,
        system_instruction: str,
        conversation: Sequence[ChatMessage],
        tools: Sequence[str],
    ) -> ModelReply:
        payload = _build_payload(
            system_instruction=system_instruction,
            conversation=conversation,
            tools=tools,
        )
        body = self._request_body_with_fallback(payload)
        return _parse_model_reply(body)

    def scan_ingredients(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> Mapping[str, Any] | Sequence[Any] | str:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"{prompt} "
                                "Return JSON only. Do not wrap with markdown code fences."
                            )
                        },
                        {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                    ],
                }
            ]
        }
        body = self._request_body_with_fallback(payload)
        text = _extract_text_reply(body)
        if text is None:
            raise GeminiApiError(
                status_code=502,
                message="Gemini API returned no text for scan_ingredients.",
                body=body,
            )
        return text

    def generate_recipe_text(
        self,
        *,
        recipe: str,
        user_context: Mapping[str, Any],
        prompt: str,
    ) -> str:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"{prompt}\n\n"
                                "User context (JSON):\n"
                                f"{json.dumps(user_context, ensure_ascii=False)}\n\n"
                                "Recipe source text:\n"
                                f"{recipe}\n\n"
                                "Output plain text only."
                            )
                        }
                    ],
                }
            ]
        }
        body = self._request_body_with_fallback(payload)
        text = _extract_text_reply(body)
        if text is None or not text.strip():
            raise GeminiApiError(
                status_code=502,
                message="Gemini API returned empty text for generate_recipe_text.",
                body=body,
            )
        return text.strip()

    def _request_body_with_fallback(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        attempted_keys: set[str] = set()
        last_error: Exception | None = None

        while True:
            key_entry = self._select_key(attempted_keys)
            if key_entry is None:
                if last_error is not None:
                    raise RuntimeError("All configured Gemini keys failed.") from last_error
                raise RuntimeError("No Gemini API key available.")

            attempted_keys.add(key_entry.name)
            url = f"{self._base_url}/models/{self._model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": key_entry.api_key,
            }

            try:
                response = self._transport.post_json(
                    url=url,
                    headers=headers,
                    payload=payload,
                    timeout_seconds=self._timeout_seconds,
                )
                _ensure_success_or_raise(response)
                return response.body
            except GeminiApiError as exc:
                last_error = exc
                if exc.should_fallback:
                    self._key_router.block(key_entry.name)
                    continue
                raise
            except GeminiTransportError as exc:
                last_error = exc
                self._key_router.block(key_entry.name)
                continue

    def _select_key(self, attempted_keys: set[str]) -> ApiKeyEntry | None:
        while True:
            try:
                candidate = self._key_router.next_key()
            except RuntimeError:
                return None
            if candidate.name in attempted_keys:
                self._key_router.block(candidate.name)
                continue
            return candidate


def _build_payload(
    *,
    system_instruction: str,
    conversation: Sequence[ChatMessage],
    tools: Sequence[str],
) -> dict[str, Any]:
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "SYSTEM RULES (do not expose to user, just follow): "
                        f"{system_instruction}"
                    )
                }
            ],
        }
    ]
    contents.extend(_convert_message(message) for message in conversation)
    return {
        "contents": contents,
        "tools": [{"functionDeclarations": _build_function_declarations(tools)}],
    }


def _build_function_declarations(tools: Sequence[str]) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for name in tools:
        declarations.append(
            {
                "name": name,
                "description": _tool_description(name),
                "parameters": _tool_parameters(name),
            }
        )
    return declarations


def _tool_description(name: str) -> str:
    if name == "view":
        return "Read data from one resource. Input: {resource, filters?}."
    if name == "add":
        return (
            "Add data. Input: {resource, items:[{name, quantity?, unit?, purchased?}]} "
            "or for user_context use {resource:'user_context', data:{...}}. "
            "For user_context prefer canonical keys: preferences, allergies, dietary_restrictions."
        )
    if name == "update":
        return (
            "Update existing data. Must include patch. "
            "For inventory/shopping_list include target (prefer name for normal user requests). "
            "For user_context target is optional and patch should use canonical keys: "
            "preferences, allergies, dietary_restrictions."
        )
    if name == "delete":
        return "Delete existing item/field by target. Input: {resource, target:{item_id or name}}."
    return f"Execute '{name}' operation for meal planner data."


def _tool_parameters(name: str) -> dict[str, Any]:
    base_properties = {
        "resource": {
            "type": "STRING",
            "enum": ["user_context", "inventory", "shopping_list"],
        },
        "filters": {"type": "OBJECT"},
        "target": {
            "type": "OBJECT",
            "properties": {
                "item_id": {"type": "STRING"},
                "name": {"type": "STRING"},
            },
        },
        "patch": {"type": "OBJECT"},
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "quantity": {"type": "NUMBER"},
                    "unit": {"type": "STRING"},
                    "purchased": {"type": "BOOLEAN"},
                },
            },
        },
        "data": {"type": "OBJECT"},
    }

    required_fields = ["resource"]
    if name == "update":
        required_fields = ["resource", "patch"]
    elif name == "delete":
        required_fields = ["resource", "target"]
    elif name == "add":
        required_fields = ["resource"]

    return {
        "type": "OBJECT",
        "required": required_fields,
        "properties": base_properties,
    }


def _convert_message(message: ChatMessage) -> dict[str, Any]:
    role = "model" if message.role == "assistant" else "user"
    if message.role == "tool":
        text = f"Tool result: {message.content}"
    else:
        text = message.content
    return {"role": role, "parts": [{"text": text}]}


def _ensure_success_or_raise(response: TransportResponse) -> None:
    if 200 <= response.status_code < 300:
        return
    message = _extract_error_message(response.body)
    raise GeminiApiError(
        status_code=response.status_code,
        message=message,
        body=response.body,
    )


def _extract_error_message(body: Mapping[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return "Gemini API request failed."


def _parse_model_reply(body: Mapping[str, Any]) -> ModelReply:
    candidates = body.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        raise GeminiApiError(
            status_code=502,
            message="Gemini API returned no candidates.",
            body=body,
        )
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise GeminiApiError(
            status_code=502,
            message="Gemini API candidate format is invalid.",
            body=body,
        )
    content = first.get("content")
    if not isinstance(content, Mapping):
        raise GeminiApiError(
            status_code=502,
            message="Gemini API content format is invalid.",
            body=body,
        )
    parts = content.get("parts")
    if not isinstance(parts, Sequence):
        raise GeminiApiError(
            status_code=502,
            message="Gemini API parts format is invalid.",
            body=body,
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
        function_call = part.get("functionCall")
        if isinstance(function_call, Mapping):
            name = function_call.get("name")
            args = function_call.get("args", {})
            if isinstance(name, str) and name:
                tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments=args if isinstance(args, Mapping) else {},
                    )
                )

    merged_text = "\n".join(text_parts).strip() if text_parts else None
    return ModelReply(text=merged_text, tool_calls=tuple(tool_calls), raw=body)


def _extract_text_reply(body: Mapping[str, Any]) -> str | None:
    reply = _parse_model_reply(body)
    return reply.text
