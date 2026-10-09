# AI Module (MVP)

## `run_agent`

`run_agent` is implemented in `agent.py` with the following constraints from the plan:

- exactly 4 tools: `view`, `add`, `update`, `delete`
- resource whitelist: `user_context`, `inventory`, `shopping_list`
- per-turn tool-call budget (default: 5)
- backend callback execution through a dispatcher interface
- strict validation for tool name, resource, and required fields (`target`, `patch`, `items`/`data`)
- no false success: assistant only reports final success after model returns a normal text response

### Minimal usage

```python
from ai_module.agent import run_agent

# Provide your own model client and tool dispatcher implementation.
result = run_agent(
    user_message="Thêm 2 quả trứng vào kho",
    history=[],
    model_client=my_model_client,
    tool_dispatcher=my_dispatcher,
    auth_context={"user_id": "u_01"},
)
```

### Return type

`run_agent` returns `AgentFinalResponse` with:

- `status`: `ok` | `needs_clarification` | `error`
- `message`: final assistant text
- `steps_used`: total tool calls executed in this turn
- `tool_results`: tool callback outputs from this turn
- `error`: structured error (`code`, `message`, `retryable`) when applicable

## Reduced-scope functions

### `scan_ingredients`

Implemented in [scan.py](E:/AI/ModunPlanner/ai_module/scan.py):

- input: `image_bytes`, `mime_type`, `model_client`
- output: `{"ingredients":[{"name":"..."}]}`
- behavior:
  - validates image input (`INVALID_INPUT` on invalid bytes/MIME)
  - parses model JSON result
  - accepts only ingredient names (no IDs/quantity inference)
  - deduplicates names in one scan
  - supports empty output list

### `generate_recipe`

Implemented in [recipe.py](E:/AI/ModunPlanner/ai_module/recipe.py):

- input: `recipe` (text), `user_context` (dict-like), `model_client`
- output: paraphrased recipe as plain text
- behavior:
  - `NO_REFERENCE_RECIPE` when `recipe` is empty
  - `INVALID_INPUT` on wrong input types
  - `EMPTY_MODEL_REPLY` when model returns empty text
  - keeps a focused paraphrase prompt and returns text only

  ### Real tests for reduced-scope functions

  Real integration tests are in [test_scan_and_recipe_real.py](E:/AI/ModunPlanner/tests/test_scan_and_recipe_real.py).

  They are opt-in to avoid accidental API spend:

  ```powershell
  $env:RUN_REAL_GEMINI_TESTS="1"
  $env:REAL_SCAN_IMAGE_PATH="E:\path\to\your\ingredient-photo.jpg"
  & .\.venv\Scripts\python.exe .\tests\test_scan_and_recipe_real.py
  ```

  Notes:

  - `generate_recipe` real test runs with only API keys.
  - `scan_ingredients` real test also needs `REAL_SCAN_IMAGE_PATH`.

## API key file and key routing

Use [.env.ai.example](E:/AI/ModunPlanner/.env.ai.example) as a template:

1. Copy it to `.env.ai` (local only).
2. Fill your real Gemini keys.
3. Keep `.env.ai` out of source control.

Routing priority is:

1. `GEMINI_API_KEY_1`
2. `GEMINI_API_KEY_2`
3. `GEMINI_API_KEY_3`

Minimal usage:

```python
from ai_module.key_routing import ApiKeyRouter, load_api_keys_from_env_file

keys = load_api_keys_from_env_file(".env.ai")
router = ApiKeyRouter(keys)

selected = router.next_key()  # free first, then low_cost, then standard
```

## Gemini adapter with auto fallback

[gemini_adapter.py](E:/AI/ModunPlanner/ai_module/gemini_adapter.py) provides `GeminiAdapter`, compatible with `run_agent` `ModelClient`.

```python
from ai_module.agent import run_agent
from ai_module.gemini_adapter import GeminiAdapter
from ai_module.key_routing import ApiKeyRouter, load_api_keys_from_env_file

router = ApiKeyRouter(load_api_keys_from_env_file(".env.ai"))
model_client = GeminiAdapter(
    key_router=router,
    model="gemini-3.5-flash",
)

result = run_agent(
    user_message="Tôi còn gì trong kho?",
    history=[],
    model_client=model_client,
    tool_dispatcher=my_dispatcher,
)
```

Fallback behavior:

- key priority: free → low_cost → standard
- auto fallback on: 401, 403, 429, 500, 502, 503, 504
- failed key is blocked for current router lifecycle

## Real Gemini integration test (opt-in)

Real Gemini integration tests are in [test_real_gemini_integration.py](E:/AI/ModunPlanner/tests/test_real_gemini_integration.py).

They are skipped by default to avoid accidental API spend.

PowerShell example:

```powershell
$env:RUN_REAL_GEMINI_TESTS="1"
& .\.venv\Scripts\python.exe .\tests\test_real_gemini_integration.py
```

Requirements:

- `.env.ai` exists and contains at least one Gemini API key
- optional `GEMINI_MODEL` env var (default: `gemini-2.5-flash`)

## Manual result check (real Gemini, no asserts)

If you want to inspect behavior manually instead of pass/fail assertions, run:

```powershell
& .\.venv\Scripts\python.exe .\tests\manual_check_real_gemini.py
```

This prints, for each scenario:

- model final message and status
- all tool calls made by the agent
- final `user_context`, `inventory`, and `shopping_list` state
