# TDS Project 2 – LLM Quiz Solver

## Overview
This repository hosts the code for an automated quiz solver that receives quiz URLs via a POST API, scrapes each page with a headless browser, parses the instructions with an LLM, orchestrates tool-assisted reasoning, and submits the final answer back to the quiz endpoint. The final deliverable includes a hardened FastAPI service, Playwright-based scraper, task-specific tools, and deployment assets.

## Current Status (Phases 2–9.5)
Phase 2 (**Web Server & API Endpoint**) is complete. The FastAPI app (`app.main:app`) exposes `/solve` with strict schema validation, secret verification, and structured logging, plus `/healthz` for diagnostics. Full-stack automation (browser + LLM orchestration) lands in Phases 3–8, so only lightweight smoke tests are run at this stage until the downstream components are integrated. Phase 3 (Headless Browser Integration) is fully implemented via `app/browser.py`, which bundles a Playwright context manager, the `ScrapeResult` dataclass, and helpers such as `scrape_quiz_page`/`scrape_quiz_page_sync` to render quiz pages, extract resources, and surface submission endpoints for downstream modules. Phase 4 (LLM Integration & Task Parsing) introduces `app/llm.py`, an async AIPipe-powered client plus `parse_task`, enabling rendered HTML to be summarized into structured task metadata that feeds the upcoming solver pipeline. Phase 5 (Tool Implementation) adds `app/downloader.py` for constrained, checksum-aware resource fetching, `app/data_tools.py` for PDF/text/image ingestion utilities, a subprocess-backed `app/executor.py` sandbox for safe Python evaluation, and a centralized `app/tools.py` registry that exposes these helpers to the solver agent, along with a growing suite of pytest scenarios that lock in this functionality before the solver agent consumes it. Phase 6 (Solver Agent) layers an LLM-driven planner/executor loop (`app/agent.py`) on top of the registry, wiring in richer analytics helpers (table saving, dataframe summaries, chart generation, PDF/image aliases) plus guardrails for tool-calling JSON so the orchestrator can reason about each quiz step programmatically. Phase 7 (Answer Submission) adds `app/submission.py`, which standardizes payload construction, guards against duplicate attempts, and records reasoning metadata for downstream audit. Phase 8 (Orchestration & Flow Control) delivers `app/orchestrator.py` and the `QuizOrchestrator`, enforcing total/attempt budgets, chaining follow-up URLs, and exposing rich attempt telemetry for observability. Phase 9 (Advanced Tooling) rounds out the stack with `app/visualization.py`, `app/vision.py`, and `app/api_integration.py`, plus their registered tools, so the agent can render charts, triage images, and fetch third-party APIs directly during a quiz run. Phase 9.5 (Vision Escalation) integrates NVIDIA's Nemotron Nano 2 VL multimodal model for advanced image/video/document reasoning via `app/vision_llm.py`, automatically escalating from lightweight heuristics when entropy or region-score thresholds are exceeded, providing OCR, chart analysis, and document understanding capabilities through the `vision_reasoner` and `reason_over_image` tools.

## Tech Stack
- **Runtime:** Python 3.11+
- **API Framework:** FastAPI + Uvicorn
- **Browser Automation:** Playwright
- **Data Processing:** Pandas, NumPy, PyPDF2, pdfplumber, Pillow, OpenPyXL
- **Visualization:** Matplotlib, Seaborn, Plotly + Kaleido (for PNG export)
- **Vision & Multimodal:** Nemotron Nano 2 VL via AIPipe (Phase 9.5)
- **LLM Client:** OpenAI SDK (+ Tenacity) routed through AIPipe, defaulting to `x-ai/grok-4.1-fast:free` with an escalated fallback to `moonshotai/kimi-k2:free`

## Getting Started
1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd TDS-Proj2
   ```
2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   playwright install  # Downloads browser binaries
   ```
4. **Validate environment setup (recommended)**
   ```bash
   python scripts/check_env.py
   # Or automatically fix issues:
   python scripts/check_env.py --fix
   ```
   This script verifies:
   - Python version (>= 3.11)
   - Required dependencies are installed
   - Playwright browsers are available
   - `.env` file exists
   - All required environment variables are set

5. **Create an `.env` file**
   ```
   EMAIL=your_email@example.com
   SECRET=your_secret_string
   OPENAI_API_KEY=sk-...
   AIPIPE_API_KEY=paipipe_xxx   # required for Phase 4 parsing and Phase 9.5 vision
   AIPIPE_ENDPOINT=https://aipipe.org/openrouter/v1  # optional override
   AIPIPE_GEMINIV1BETA_ENDPOINT=https://aipipe.org/geminiv1beta  # optional override
   PLAYWRIGHT_BROWSERS_PATH=0  # optional: forces local cache
   LLM_DEFAULT_MODEL=x-ai/grok-4.1-fast:free  # optional: override primary parser model
   LLM_FALLBACK_MODEL=moonshotai/kimi-k2:free  # optional: override escalation model
   
   # Phase 9.5 Vision & Multimodal Settings (optional)
   VISION_MODEL_NAME=nvidia/nemotron-nano-12b-v2-vl:free
   VISION_ENTROPY_TRIGGER=4.2  # escalation threshold for image complexity
   VISION_REGION_SCORE_THRESHOLD=0.18  # escalation threshold for text regions
   VISION_MAX_FRAMES=24  # frame budget for video/multi-image reasoning
```
   > Use `.env.example` as a template and ensure your populated `.env` stays untracked (already handled by `.gitignore`).

   Optional Playwright overrides (set these only when you need non-default scraping behavior):

   - `PLAYWRIGHT_HEADLESS=false` — open a visible browser window for debugging.
   - `PLAYWRIGHT_WAIT_SELECTOR=.task-body` — wait for a custom CSS selector before extracting content (set to blank/None to skip explicit waits).
   - `PLAYWRIGHT_NAVIGATION_TIMEOUT_SECONDS=30` — increase navigation patience for slower quiz hosts.
   - `PLAYWRIGHT_WAIT_TIMEOUT_SECONDS=30` — extend the selector wait budget for heavy client-side rendering.
   - `PLAYWRIGHT_VIEWPORT_WIDTH=1440` / `PLAYWRIGHT_VIEWPORT_HEIGHT=900` — resize the virtual browser when layout-sensitive tasks demand it.
   
   Optional LLM overrides:
   
   - `LLM_DEFAULT_MODEL=x-ai/grok-4.1-fast:free` — change the primary parser model routed through AIPipe.
   - `LLM_FALLBACK_MODEL=moonshotai/kimi-k2:free` — change the higher-capability backup model used when the primary model fails.

## Project Structure (evolving)
```
TDS-Proj2/
├── Question.md        # Assignment brief
├── TASKS.md           # Detailed project plan & checklist
├── README.md          # Project overview
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
├── .env.example       # Template for local secrets (copy to .env locally)
├── venv/              # Local virtual environment (ignored)
└── .gitignore
```

## Running the API locally
1. Start the server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
2. Test your endpoint:
   ```bash
   # Test against official demo endpoint (recommended)
   python scripts/test_demo_endpoint.py
   
   # Or test with local test server
   python scripts/run_test_server.py --port 8888  # Terminal 1
   python scripts/test_pipeline.py --url http://localhost:8888/quiz-834  # Terminal 2
   
   # Or send a direct curl request
   curl -X POST http://localhost:8000/solve \
     -H "Content-Type: application/json" \
     -d '{"email":"test@test.com","secret":"YOUR_SECRET","url":"https://tds-llm-analysis.s-anand.net/demo"}'
   ```

For comprehensive testing instructions, [scripts/README.md](scripts/README.md).

## API Surface (Phase 2)
- `POST /solve`: accepts `email`, `secret`, and `url`, rejects malformed JSON (400) or bad secrets (403), and currently returns an authenticated stub response while downstream solvers are under construction.
- `GET /healthz`: reports service health plus the active LLM routing target (default `x-ai/grok-4.1-fast:free` via AIPipe) for observability.
- `GET /`: simple pointer directing callers to `/solve` and `/healthz`.

*Testing note:* only smoke tests are executed right now. LLM access is funneled through AIPipe with `x-ai/grok-4.1-fast:free` as the primary parser and `moonshotai/kimi-k2:free` as the fallback; future upgrades will add `gemini-2.5-pro` via the google-genai client once available.

## Browser Scraper Usage (Phase 3)
1. **Install Playwright browsers** (once per machine):
   ```bash
   playwright install
   ```
2. **Invoke the async helper** inside any coroutine to fetch a rendered quiz page:
   ```python
   from app.browser import scrape_quiz_page

   result = await scrape_quiz_page(
       "https://tds-llm-analysis.s-anand.net/demo",
   )
   print(result.text)
   print(result.resource_links)
   print(result.submission_url)
   ```
3. **Use the sync wrapper** when you are outside an event loop (e.g., ad-hoc scripts):
   ```python
   from app.browser import scrape_quiz_page_sync

   result = scrape_quiz_page_sync("https://example.com/quiz")
   ```
4. **Adjust behavior via env vars** documented above (headless toggle, selector overrides, custom timeouts). These flow into `BrowserRuntimeConfig`, so the scraper automatically honors your settings without code changes.

## LLM Task Parser Usage (Phase 4)
1. Ensure your `.env` includes `AIPIPE_API_KEY` (and optionally endpoint overrides if you need a sandbox gateway).
2. Import the parser and feed it rendered HTML/text from the scraper:
   ```python
   from app.browser import scrape_quiz_page
   from app.llm import parse_task

   async def inspect_quiz(url: str):
       scrape_result = await scrape_quiz_page(url)
       task = await parse_task(scrape_result.html)
       print(task.to_dict())
   ```
3. Reuse the same `AIPipeClient` instance when parsing multiple pages to benefit from connection pooling and the built-in rate limiter (pass it through the `client=` argument).

## Solver Agent & Tool Orchestration (Phase 6)
1. Instantiate `LLMSolverAgent` with an optional `AIPipeClient`, the TaskParseResult returned by `parse_task`, and a workspace directory for downloaded artifacts. The agent enforces a JSON-only schema (`thought`, `action`, `input`) so each LLM turn is inspectable and tool calls remain auditable.
2. The centralized registry (`app/tools.py`) now exposes additional helpers required by the planner: `save_table`, `describe_dataframe`, `value_counts`, `read_csv` / `read_excel` aliases, `write_text`, `read_pdf`, `encode_image`, and `create_chart`, alongside the existing downloader/data/executor functions. Every helper is awaitable, which keeps the agent’s think-act-observe loop uniform.
3. Tool catalogs can be generated via `build_tool_catalog()` to present curated signatures to the planner, while the `tests/test_agent.py` suite stubs planner responses to validate tool invocation, final-answer emission, and catalog contents without making real LLM calls.

## Submission & Orchestration (Phases 7–8)
1. `app/submission.py` exposes `submit_answer`, which packages answers (numbers, text, JSON, or base64) with the configured email/secret, forwards optional reasoning metadata, and parses the response into a structured payload used by the orchestrator and tests in `tests/test_submission.py`.
2. `app/orchestrator.py` coordinates scraper → parser → agent → submit loops with total/attempt timeouts, workspace management, and chaining, while `tests/test_orchestrator.py` captures multiple scenarios (fallback submission URLs, timeout handling, success flow).
3. The orchestration layer is wired into FastAPI via the Phase 8 entry path, and heavy sample pipelines live in `tests/test_sample_pipeline.py` for environments that have Playwright browsers and AIPipe credentials available.

## Advanced Tooling & Integrations (Phase 9)
1. Visualization helpers in `app/visualization.py` provide `ChartArtifact`, Matplotlib/Seaborn rendering via `render_matplotlib_chart`, Plotly-based exports through `render_plotly_chart`, and the high-level `create_chart_from_records` helper that powers the agent’s `create_chart`/`render_chart` tool; coverage lives in `tests/test_visualization.py`, and Plotly PNG export depends on Kaleido (already pinned in `requirements.txt`) so the tooling works offline in CI.
2. Vision utilities under `app/vision.py` supply `analyze_image`, which computes brightness/contrast/entropy, dominant colors, and text-like regions using lightweight Pillow/PyPDF2 primitives, with regression tests in `tests/test_vision.py`.
3. HTTP API integrations in `app/api_integration.py` add `AsyncAPIClient`, `call_json_api`, `extract_instruction_headers`, and `parse_instruction_text`, ensuring consistent request construction, response parsing, and instruction parsing; see `tests/test_api_integration.py` for usage patterns.
4. The central registry (`app/tools.py`) now surfaces `create_chart`, `analyze_image`, `call_json_api`, `parse_instruction_headers`, and `parse_instruction_text` so the Phase 6 agent can invoke the new capabilities without bespoke glue code.
5. **Phase 9.5 (Vision Escalation)** adds `app/vision_llm.py` with `NemotronVisionClient` and the `reason_over_media` helper, enabling the agent to escalate from fast on-device heuristics to NVIDIA's Nemotron Nano 2 VL (12B multimodal reasoning model) when image entropy or text-region scores exceed configured thresholds; two new tools (`vision_reasoner`, `reason_over_image`) are registered in the tool catalog, and comprehensive tests in `tests/test_vision_llm.py` and updated `tests/test_vision.py` validate escalation behavior, hint construction, and API payload formatting without requiring live credentials in CI.

## Development Workflow
1. Track progress via `TASKS.md`, checking off items as they are completed.
2. Keep secrets confined to `.env` and environment variables (see `.env.example` for template).
3. Implement modules in the order outlined (API → Browser → Tools → Agent → Orchestrator).
4. Add automated tests in tandem with new modules to avoid regressions.

## Testing & Quality
- Unit and integration tests will live under a forthcoming `tests/` directory.
- Use `pytest` (to be added) for automated validation.
- Playwright scripts should include graceful teardown to prevent orphaned browsers.

## Deployment Roadmap
1. Implement FastAPI app and supporting modules (Phases 2–8).
2. Containerize with Docker (Phase 11) and deploy to a managed platform (Render/Railway/etc.).
3. Configure monitoring/logging before the evaluation window.

## License
This project is licensed under the MIT License. See `LICENSE` for the full text.

---

For detailed milestones and acceptance criteria, refer to `TASKS.md`.
