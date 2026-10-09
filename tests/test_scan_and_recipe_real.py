from pathlib import Path
import os
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_module.gemini_adapter import GeminiAdapter
from ai_module.key_routing import ApiKeyRouter, load_api_keys_from_env_file
from ai_module.recipe import generate_recipe
from ai_module.scan import scan_ingredients


def _guess_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


@unittest.skipUnless(
    os.getenv("RUN_REAL_GEMINI_TESTS") == "1",
    "Set RUN_REAL_GEMINI_TESTS=1 to run real Gemini tests.",
)
class RealScanAndRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env_path = PROJECT_ROOT / ".env.ai"
        if not env_path.exists():
            raise unittest.SkipTest(".env.ai not found; cannot run real Gemini tests.")
        keys = load_api_keys_from_env_file(env_path)
        if not keys:
            raise unittest.SkipTest("No Gemini API keys configured in .env.ai.")
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        cls.adapter = GeminiAdapter(
            key_router=ApiKeyRouter(keys),
            model=model,
            timeout_seconds=45,
        )

    def test_generate_recipe_real(self):
        source_recipe = (
            "Salad cà chua cho 2 người. Nguyên liệu: 200 g cà chua và 1 g muối. "
            "Rửa cà chua, cắt miếng, trộn với muối rồi bày ra đĩa."
        )
        result = generate_recipe(
            recipe=source_recipe,
            user_context={
                "communication_style": "thân thiện, ngắn gọn",
                "cooking_experience": "mới bắt đầu",
                "preferred_language": "vi",
            },
            model_client=self.adapter,
        )
        print("\n[generate_recipe model output]")
        print(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 20)

    def test_scan_ingredients_real(self):
        image_env = os.getenv("REAL_SCAN_IMAGE_PATH")
        if not image_env:
            raise unittest.SkipTest("Set REAL_SCAN_IMAGE_PATH to run real scan_ingredients test.")
        image_path = Path(image_env)
        if not image_path.exists():
            raise unittest.SkipTest(f"REAL_SCAN_IMAGE_PATH not found: {image_path}")
        image_bytes = image_path.read_bytes()
        mime_type = _guess_mime_type(image_path)

        result = scan_ingredients(
            image_bytes=image_bytes,
            mime_type=mime_type,
            model_client=self.adapter,
        )
        print("\n[scan_ingredients model output]")
        print(result)
        self.assertIsInstance(result, dict)
        self.assertIn("ingredients", result)
        self.assertIsInstance(result["ingredients"], list)
        for row in result["ingredients"]:
            self.assertIsInstance(row, dict)
            self.assertIn("name", row)
            self.assertIsInstance(row["name"], str)
            self.assertTrue(row["name"].strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
