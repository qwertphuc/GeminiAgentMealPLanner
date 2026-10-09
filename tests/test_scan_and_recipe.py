from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_module.recipe import GenerateRecipeError, generate_recipe
from ai_module.scan import ScanIngredientsError, scan_ingredients


class FakeScanClient:
    def __init__(self, output):
        self.output = output

    def scan_ingredients(self, *, image_bytes, mime_type, prompt):  # noqa: ANN001
        return self.output


class FakeRecipeClient:
    def __init__(self, output: str):
        self.output = output

    def generate_recipe_text(self, *, recipe, user_context, prompt):  # noqa: ANN001
        return self.output


class ScanAndRecipeTests(unittest.TestCase):
    def test_scan_ingredients_deduplicates_and_normalizes(self):
        client = FakeScanClient(
            {
                "ingredients": [
                    {"name": "tomato"},
                    {"ingredient": "egg"},
                    {"name": "Tomato"},
                    "milk",
                    {"item": "egg"},
                ]
            }
        )
        result = scan_ingredients(
            image_bytes=b"fake-image",
            mime_type="image/jpeg",
            model_client=client,
        )
        self.assertEqual(
            result,
            {"ingredients": [{"name": "tomato"}, {"name": "egg"}, {"name": "milk"}]},
        )

    def test_scan_ingredients_allows_empty_list(self):
        client = FakeScanClient({"ingredients": []})
        result = scan_ingredients(
            image_bytes=b"fake-image",
            mime_type="image/png",
            model_client=client,
        )
        self.assertEqual(result, {"ingredients": []})

    def test_scan_ingredients_invalid_input(self):
        client = FakeScanClient({"ingredients": []})
        with self.assertRaises(ScanIngredientsError) as ctx:
            scan_ingredients(
                image_bytes=b"",
                mime_type="image/png",
                model_client=client,
            )
        self.assertEqual(ctx.exception.code, "INVALID_INPUT")

    def test_generate_recipe_success(self):
        client = FakeRecipeClient("Paraphrased recipe text")
        result = generate_recipe(
            recipe="Cook tomato and egg.",
            user_context={"communication_style": "friendly"},
            model_client=client,
        )
        self.assertEqual(result, "Paraphrased recipe text")

    def test_generate_recipe_no_reference_recipe(self):
        client = FakeRecipeClient("unused")
        with self.assertRaises(GenerateRecipeError) as ctx:
            generate_recipe(
                recipe="  ",
                user_context={},
                model_client=client,
            )
        self.assertEqual(ctx.exception.code, "NO_REFERENCE_RECIPE")

    def test_generate_recipe_invalid_user_context(self):
        client = FakeRecipeClient("unused")
        with self.assertRaises(GenerateRecipeError) as ctx:
            generate_recipe(
                recipe="Recipe",
                user_context=[],  # type: ignore[arg-type]
                model_client=client,
            )
        self.assertEqual(ctx.exception.code, "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()
