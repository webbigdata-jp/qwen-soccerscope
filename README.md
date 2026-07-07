# qwen-soccerscope

Demo repository for the [Qwen Cloud Hackathon](https://qwencloud-hackathon.devpost.com/).

This is a port of the original GCP(Gemini) + Cloud Run + MongoDB Atlas product,
[webbigdata-jp/soccerscope](https://github.com/webbigdata-jp/soccerscope),
a multilingual RAG AI agent for World Cup 2026, to a Qwen(DashScope) + Alibaba Cloud Function Compute 3.0 + MongoDB Atlas stack.

Live demo: http://qwen-soccer.tubesaku.com/

> The requirements/implementation of the "porting agent" (the automated GCP → Alibaba/Qwen
> migration agent) itself will be documented in a separate repository. This repository is the
> **output** of that migration — the demo application.

---

## 1. Overview

SoccerScope is an app for searching soccer-related YouTube videos, analyzing their comments,
and generating reports. This repository replaces the following components with their
Qwen / Alibaba Cloud equivalents.

| Component | Original (Gemini version) | This repo (Qwen version) |
|---|---|---|
| LLM (agent core) | Gemini | Qwen (`LiteLlm(model="dashscope/qwen-plus")`, etc., via ADK) |
| Embedding | Gemini | Qwen (`text-embedding-v4`, DashScope OpenAI-compatible API) |
| Comment sentiment analysis | Gemini structured output | Qwen (`qwen-plus` family, called directly via the OpenAI SDK) |
| Infrastructure | Cloud Run | Alibaba Cloud Function Compute 3.0 (Custom Runtime) |
| Database | MongoDB Atlas (`soccertube`) | MongoDB Atlas (`qwen-soccertube`, separate from the Gemini production DB) |

The agent framework itself is still Google ADK (Apache 2.0), used as-is. Swapping both the
model and the infrastructure for Alibaba/Qwen equivalents without modifying the framework
also serves as a demonstration of how portable it is.

---

## 2. Directory structure

```
qwen-soccerscope/
├── app/                      # Demo app (deployed to Alibaba Cloud Function Compose)
│   ├── main.py                  # FastAPI entry point
│   ├── soccer_agent/             # Google ADK agent (LLM = Qwen)
│   ├── static/                   # Frontend static files
│   ├── requirements.txt          # For deployment (pip -t target install)
│   └── deploy.sh                 # Build & zip script
│
├── pipeline/                 # Data pipeline (video embedding, comment sentiment analysis)
│   ├── 1_embed_videos.py
│   ├── 2_load_to_mongo.py
│   ├── 3_analyze_comments.py
│   ├── 4_load_comment_analysis.py
│   ├── pyproject.toml            # Pipeline-only dependencies (managed with uv)
│   └── data/                     # Per-date intermediate files and checkpoint output
│
└── README.md
```

`pipeline/` only **reads** the intermediate files produced by the original Gemini-side data
pipeline (a separate repository, `soccer/`, expected to be cloned at the same directory level
as this repo). It never writes back into the Gemini pipeline's files.

---

## 3. Data pipeline (`pipeline/`)

### 3-1. Overall flow

```
../soccer/                                  ← Gemini pipeline (separate repo, read-only)
  phase2 → phase3 → phase7 → phase4
  generates data/<date>/ daily

pipeline/                                   ← This repository (new)
  phase7 → 1_embed_videos.py → 2_load_to_mongo.py        (videos + vectors + index)
  phase4 → 3_analyze_comments.py → 4_load_comment_analysis.py (sentiment analysis → $set)

  input:  ../../soccer/data/<date>/phase7_with_buzz_score_*.json
          ../../soccer/data/<date>/phase4_comments_*.json
  output: ./data/<date>/videos_embedded_*.json
          ./data/<date>/comment_analysis_*.json

  DB: qwen-soccertube (set via SOCCER_DB_NAME; separate from the Gemini production DB, soccertube)
```

- Every script takes an optional date argument (`YYYYMMDD`, defaults to today), so backfilling
  is supported.
- The Gemini version's `archive_run_files()` (which moved the Gemini pipeline's own intermediate
  files around) is **not included** here, in keeping with the policy of never touching the
  Gemini pipeline's files.

### 3-2. Script responsibilities

| Script | Input | Output | Role |
|---|---|---|---|
| `1_embed_videos.py` | `../../soccer/data/<date>/phase7_with_buzz_score_*.json` | `./data/<date>/videos_embedded_*.json` | Embeds video metadata with Qwen (`text-embedding-v4`) |
| `2_load_to_mongo.py` | `./data/<date>/videos_embedded_*.json` | MongoDB `videos` collection | Loads vectors/buzz_score etc. into MongoDB, sets up the vector search index |
| `3_analyze_comments.py` | `../../soccer/data/<date>/phase4_comments_*.json` | `./data/<date>/comment_analysis_*.json` | Runs sentiment analysis and generates quote candidates with Qwen (`qwen-plus` family) |
| `4_load_comment_analysis.py` | `./data/<date>/comment_analysis_*.json` | MongoDB `videos.comment_analysis` (`$set`) | Loads analysis results, deletes videos judged unrelated to soccer |

### 3-3. Constraints and workarounds for Qwen (DashScope) structured output

Unlike Gemini, which lets you enforce a schema simply by passing `response_schema=PydanticModel`,
Qwen (DashScope) has no schema-enforcement mechanism. `response_format={"type": "json_object"}`
only guarantees syntactically valid JSON — it does not guarantee field names, types, or structure.

`3_analyze_comments.py` works around this with a two-step approach:

1. Write the schema out explicitly, as text, in the prompt
2. Validate the returned JSON against a local Pydantic model (`pydantic.BaseModel`); retry
   generation on mismatch

Other DashScope-specific quirks:

- If the message doesn't contain the word "json" (case-insensitive) anywhere, `json_object`
  mode returns a 400 error
- Using `json_object` mode with a model that has thinking enabled by default also returns a
  400 error; disable it explicitly via the OpenAI SDK with `extra_body={"enable_thinking": False}`
- Don't set `max_tokens` when using structured output (per official guidance — truncation risks
  producing broken JSON)

Default thinking behavior by model (easy to mix up, worth double-checking):

| Model | Thinking default |
|---|---|
| qwen-plus / qwen-flash | Off |
| qwen3-max | Off (not to be confused with qwen3.7-max) |
| qwen3.7-plus / qwen3.7-max | On (hybrid thinking) |
| qwen3.5 / qwen3.6 family | On by default |

### 3-4. Fault tolerance and throughput (`3_analyze_comments.py`)

- **Timeouts**: the OpenAI SDK's default (600s) is too long, so `timeout=60.0` is set explicitly
  (tunable via `QWEN_TIMEOUT_SECONDS`). Timeouts and connection errors are retried with the same
  exponential backoff as 429s.
- **Retry classification** (recorded as `reason`):
  - `content_moderation`: rejected by content moderation. Retrying the same input is pointless,
    so it's abandoned immediately.
  - `rate_limit_exhausted`: 429 / timeout / connection errors that didn't clear within the retry limit
  - `schema_retry_exhausted`: JSON syntax/schema mismatches that didn't clear within the retry limit
  - `api_error:...`: everything else
- **Parallelism**: `ThreadPoolExecutor` with a default concurrency of 5 (tunable via
  `QWEN_MAX_WORKERS`), tuned in practice by watching the 429 rate.
- **Checkpointing**: progress is saved incrementally to `./data/<date>/_analyze_checkpoint.json`
  after every completed item. On restart, already-processed `video_id`s are skipped. Once all
  items are done, the checkpoint file is deleted and the final `comment_analysis_<timestamp>.json`
  is written. The leading underscore in the checkpoint filename is intentional, to avoid
  colliding with the `comment_analysis_*.json` glob pattern. On interruption (e.g. Ctrl+C),
  in-flight tasks are allowed to finish while unstarted tasks are cancelled.

### 3-5. Environment variables (`pipeline/.env`)

| Variable | Purpose |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `DASHSCOPE_API_KEY` | Qwen (DashScope) API key |
| `SOCCER_DB_NAME` | Target database name (default: `qwen-soccertube`) |
| `QWEN_CHAT_MODEL` | Chat model used by `3_analyze_comments.py` (e.g. `qwen-plus`, `qwen3.7-plus`) |
| `QWEN_TIMEOUT_SECONDS` | API timeout in seconds (default: 60) |
| `QWEN_MAX_WORKERS` | Concurrency (default: 5) |

Actual values are never committed. `.env` is gitignored.

---

## 4. App overview (`app/`)

- FastAPI (`main.py`) + a Google ADK agent (`soccer_agent/`)
- The LLM itself runs on Qwen (e.g. `dashscope/qwen-plus`) via
  `google.adk.models.lite_llm.LiteLlm`. There is no remaining Gemini dependency except types.
- Video search (`search_videos`) talks to pymongo directly; find/count/etc. go through the
  official MongoDB MCP server (`mongodb-mcp-server`), invoked as a direct `node` process
  (no `npx`; `node_modules` is bundled ahead of time at build time)
- `deploy.sh` builds and zips the app for deployment to Alibaba Cloud Function Compute 3.0
  (Custom Runtime, Python 3.12)

Deployment gotchas (Node.js layer PATH setup, why `npx` was dropped, Cloudflare custom domain
setup, etc.) and further architecture details will be covered separately in an upcoming
technical blog post and in the porting-agent repository. This README only covers the overview.

### 4-1. Environment variables (set in the Function Compose console; `.env` is not bundled into the zip)

| Variable | Purpose |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `DASHSCOPE_API_KEY` | Qwen (DashScope) API key |
| `SOCCER_DB_NAME` | Target database name (`qwen-soccertube`) |

`app/soccer_agent/.env` (if present) is a separate file from the top-level environment
variables and can be used for agent-specific model settings (e.g. `qwen3.7-max`).

---

## 5. Setup

### 5-1. Pipeline (`pipeline/`)

```bash
cd pipeline
uv sync
cp .env.example .env   # set your environment variables (create .env.example yourself)

# example: process data for 2026-07-05
uv run 1_embed_videos.py 20260705
uv run 2_load_to_mongo.py 20260705
uv run 3_analyze_comments.py 20260705
uv run 4_load_comment_analysis.py 20260705
```

### 5-2. App (`app/`)

```bash
cd app
# .env is configured in the FC console, so this only installs dependencies and builds
./deploy.sh
# upload the resulting code.zip via the Alibaba Cloud Function Compute 3.0 console
```

---

## 6. Acknowledgments

- Data source: [tubesaku.com](https://tubesaku.com/)
- [Qwen Cloud Hackathon](https://qwencloud-hackathon.devpost.com/) / Alibaba Cloud / DashScope
- [Google ADK](https://github.com/google/adk-python) (Apache 2.0) — used as a model- and
  infra-agnostic agent framework
- [MongoDB Atlas](https://www.mongodb.com/atlas) / the official [MongoDB MCP Server](https://github.com/mongodb-js/mongodb-mcp-server)
- [Cloudflare](https://www.cloudflare.com/) — custom domain and HTTPS

## 7. License

Apache License 2.0
