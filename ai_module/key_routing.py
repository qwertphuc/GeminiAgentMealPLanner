from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TIER_PRIORITY = {
    "free": 0,
    "low_cost": 1,
    "standard": 2,
}


@dataclass(frozen=True)
class ApiKeyEntry:
    name: str
    api_key: str
    tier: str = "standard"


def parse_env_file(path: str | Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_api_keys_from_env_file(path: str | Path) -> list[ApiKeyEntry]:
    env = parse_env_file(path)
    rows: list[ApiKeyEntry] = []

    for slot in ("FREE", "LOW_COST", "STANDARD"):
        key = env.get(f"GEMINI_API_KEY_{slot}", "")
        if not key:
            continue
        rows.append(
            ApiKeyEntry(
                name=slot.lower(),
                api_key=key,
                tier=slot.lower() if slot != "STANDARD" else "standard",
            )
        )
    return rows


class ApiKeyRouter:
    def __init__(self, keys: Iterable[ApiKeyEntry]):
        self._keys = sorted(
            [k for k in keys if k.api_key],
            key=lambda row: (TIER_PRIORITY.get(row.tier, 99), row.name),
        )
        self._blocked: set[str] = set()

    def next_key(self) -> ApiKeyEntry:
        for entry in self._keys:
            if entry.name not in self._blocked:
                return entry
        raise RuntimeError("No API key available.")

    def block(self, key_name: str) -> None:
        self._blocked.add(key_name)

