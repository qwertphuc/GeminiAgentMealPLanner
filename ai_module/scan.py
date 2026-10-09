from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ScanIngredientsError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class ScanModelClient(Protocol):
    def scan_ingredients(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> Mapping[str, Any] | Sequence[Any] | str:
        ...


SCAN_PROMPT = (
    "Return only JSON with shape {\"ingredients\":[{\"name\":\"...\"}]}. "
    "List only clearly recognized food ingredients from the image. "
    "Do not include IDs, quantities, freshness, hidden items, comments, or extra fields."
)


def scan_ingredients(
    *,
    image_bytes: bytes,
    mime_type: str,
    model_client: ScanModelClient,
) -> dict[str, list[dict[str, str]]]:
    _validate_scan_input(image_bytes=image_bytes, mime_type=mime_type)
    raw_output = model_client.scan_ingredients(
        image_bytes=image_bytes,
        mime_type=mime_type,
        prompt=SCAN_PROMPT,
    )
    parsed = _parse_or_repair_once(raw_output)
    ingredients = _normalize_ingredients(parsed)
    return {"ingredients": ingredients}


def _validate_scan_input(*, image_bytes: bytes, mime_type: str) -> None:
    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) == 0:
        raise ScanIngredientsError(
            code="INVALID_INPUT",
            message="image_bytes must be non-empty bytes.",
        )
    if not isinstance(mime_type, str) or not mime_type.strip():
        raise ScanIngredientsError(
            code="INVALID_INPUT",
            message="mime_type must be a non-empty string.",
        )
    if not mime_type.lower().startswith("image/"):
        raise ScanIngredientsError(
            code="INVALID_INPUT",
            message="mime_type must start with image/.",
        )


def _parse_or_repair_once(raw_output: Mapping[str, Any] | Sequence[Any] | str) -> Any:
    parsed: Any = raw_output
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ScanIngredientsError(
                code="OUTPUT_SCHEMA_INVALID",
                message="Model output is not valid JSON.",
            ) from exc
    repaired = _repair_scan_payload(parsed)
    if repaired is None:
        raise ScanIngredientsError(
            code="OUTPUT_SCHEMA_INVALID",
            message="Model output does not match scan schema.",
        )
    return repaired


def _repair_scan_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        if "ingredients" in payload:
            return {"ingredients": payload.get("ingredients")}
        for key in ("items", "ingredient_names", "detected_ingredients"):
            if key in payload:
                return {"ingredients": payload.get(key)}
        return None
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return {"ingredients": payload}
    return None


def _normalize_ingredients(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_items = payload.get("ingredients")
    if raw_items is None:
        return []
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
        raise ScanIngredientsError(
            code="OUTPUT_SCHEMA_INVALID",
            message="'ingredients' must be a list.",
        )

    dedup_seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for item in raw_items:
        name = _extract_name(item)
        if name is None:
            continue
        dedup_key = name.casefold()
        if dedup_key in dedup_seen:
            continue
        dedup_seen.add(dedup_key)
        normalized.append({"name": name})
    return normalized


def _extract_name(item: Any) -> str | None:
    if isinstance(item, str):
        cleaned = item.strip()
        return cleaned or None
    if isinstance(item, Mapping):
        for key in ("name", "ingredient", "item"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None
