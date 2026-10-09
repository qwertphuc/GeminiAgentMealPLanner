from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class GenerateRecipeError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class RecipeModelClient(Protocol):
    def generate_recipe_text(
        self,
        *,
        recipe: str,
        user_context: Mapping[str, Any],
        prompt: str,
    ) -> str:
        ...


RECIPE_PROMPT = (
    "Paraphrase the given recipe text for the user context. "
    "Keep ingredients, quantities, units, servings, steps, order, time and temperature unchanged. "
    "Do not add new cooking facts. Output plain text only."
)


def generate_recipe(
    *,
    recipe: str,
    user_context: Mapping[str, Any],
    model_client: RecipeModelClient,
) -> str:
    if not isinstance(recipe, str):
        raise GenerateRecipeError(
            code="INVALID_INPUT",
            message="recipe must be a string.",
        )
    if not recipe.strip():
        raise GenerateRecipeError(
            code="NO_REFERENCE_RECIPE",
            message="recipe cannot be empty.",
        )
    if not isinstance(user_context, Mapping):
        raise GenerateRecipeError(
            code="INVALID_INPUT",
            message="user_context must be an object.",
        )

    text = model_client.generate_recipe_text(
        recipe=recipe,
        user_context=user_context,
        prompt=RECIPE_PROMPT,
    )
    if not isinstance(text, str) or not text.strip():
        raise GenerateRecipeError(
            code="EMPTY_MODEL_REPLY",
            message="Model returned empty recipe text.",
        )
    return text.strip()
