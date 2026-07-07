"""
SoccerScope — Web backend (FastAPI)

Backend for the custom Web UI.
  - Runs the existing ADK agent (soccer_agent.agent.root_agent) through the ADK Runner.
  - Reflects {query, format, lang} received from the frontend in the output format
    (report / sns / webpage) and output language (ja / en) by injecting delivery
    instructions into the prompt, without modifying the agent itself.
  - Serves the static frontend (static/index.html) from the same origin.

The agent (agent.py) is unchanged. It has no write operations, and reads are
performed inside the agent through search_videos -> official MongoDB MCP,
preserving the MCP integration requirement.

Local run:
    uvicorn main:app --host 0.0.0.0 --port 8080
Cloud Run:
    Dockerfile included (Python + Node 22). See README.md.
"""

import os
import uuid

# --- Load .env before importing the agent, because agent.py reads MONGODB_URI
#     at import time. Harmless on Cloud Run even when .env is absent. ---
try:
    from dotenv import load_dotenv

    _here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_here, ".env"))
    load_dotenv(os.path.join(_here, "soccer_agent", ".env"))
except Exception:  # noqa: BLE001
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from soccer_agent.agent import root_agent  # noqa: E402  (import after loading .env)

# --- Temporary debug endpoint: standalone MCP startup test -------------------
# Temporary endpoint unrelated to production features, used to verify whether
# npx mongodb-mcp-server can actually start on FC with only the official
# Node.js 20 public layer attached.
# Remove this whole block together with /api/generate-related debug code after verification.
import shutil
import time
import traceback
import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from soccer_agent.agent import _mcp_server_params, search_videos  # noqa: E402

APP_NAME = "soccerscope"

# 2) Rate limiting (per IP, in memory)
limiter = Limiter(key_func=get_remote_address)

# 1) Query length limit (characters)
QUERY_MAX_LEN = 500

# Build the Runner / Session only once at startup (reuse root_agent)
_session_service = InMemorySessionService()
_runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=_session_service)


# --- Convert frontend choices into delivery instructions for the agent --------
# Explicitly select the article/SNS/HTML generation flow that already exists in
# the agent INSTRUCTION. Do not modify the agent itself.
FORMAT_DIRECTIVES = {
    # Report: Markdown article with country sections and an overall synthesis
    "report": (
        "OUTPUT FORMAT = REPORT. Produce a complete Markdown article exactly as "
        "described in your COMPOSING ARTICLES flow: a punchy title and short lead, "
        "one section per country (country name + flag, a 1-2 sentence buzz summary, "
        "the thumbnail as a Markdown image, and a [watch] link), sentiment where "
        "available, and a closing insightful synthesis. "
        "MANDATORY: every single video section MUST include both a "
        "`![title](thumbnail_url)` image line and a `[▶ Watch video](url)` link line "
        "using the real thumbnail_url/url values from the tool results — never omit "
        "them. Run the SELF-CHECK described in your instructions before answering. "
        "Output Markdown only — do NOT wrap it in a code block, do NOT output raw HTML."
    ),
    # SNS: 2-3 X post drafts
    "sns": (
        "OUTPUT FORMAT = SNS POSTS. Output 2-3 short, ready-to-post social/X drafts "
        "based on the buzzing videos. Each draft: punchy, 1-2 relevant hashtags, and "
        "exactly one video link. Separate each draft with a blank line and prefix it "
        "with its number (1. / 2. / 3.). Do not add an article or extra commentary "
        "around the drafts — output the posts only."
    ),
    # Web page: return the same Markdown article as a report; the frontend styles it as a page
    "webpage": (
        "OUTPUT FORMAT = WEB FEATURE PAGE. Produce a complete, shareable Markdown "
        "feature article as in your COMPOSING ARTICLES flow (title + lead, one section "
        "per country with flag + thumbnail image + watch link + sentiment, and a strong "
        "closing overall synthesis). Make it engaging and presentation-ready. "
        "MANDATORY: every single video section MUST include both a "
        "`![title](thumbnail_url)` image line and a `[▶ Watch video](url)` link line "
        "using the real thumbnail_url/url values from the tool results — never omit "
        "them. Run the SELF-CHECK described in your instructions before answering. "
        "Output Markdown only — do NOT output raw HTML or a code block."
    ),
}

LANG_DIRECTIVES = {
    "ja": "LANGUAGE = JAPANESE. Write the entire output in natural Japanese.",
    "zh": (
        "LANGUAGE = CHINESE (Simplified). Write the entire output in natural "
        "Simplified Chinese suitable for a mainland Chinese audience. Use Chinese "
        "country/team names. Translate any Japanese titles/quotes, but keep "
        "original video titles recognizable."
    ),
    "en": (
        "LANGUAGE = ENGLISH. Write the entire output in natural English suitable for "
        "an international (US) audience. Use English country names. Translate any "
        "Japanese titles/quotes, but keep original video titles recognizable."
    ),
}


def _build_prompt(query: str, fmt: str, lang: str) -> str:
    fmt_d = FORMAT_DIRECTIVES.get(fmt, FORMAT_DIRECTIVES["report"])
    lang_d = LANG_DIRECTIVES.get(lang, LANG_DIRECTIVES["ja"])
    return (
        f"{query.strip()}\n\n"
        f"--- DELIVERY INSTRUCTIONS (follow strictly) ---\n"
        f"{fmt_d}\n{lang_d}\n"
        f"Use ONLY data returned by your tools (search_videos / find / count). "
        f"Never invent videos, stats, or quotes."
    )


async def _run_agent(prompt: str) -> str:
    """Run the agent with one session per request and return the final response text."""
    user_id = "web"
    session_id = uuid.uuid4().hex
    await _session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    chunks: list[str] = []
    async for event in _runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        # Aggregate only the final response text and ignore intermediate tool-call events
        if event.is_final_response() and getattr(event, "content", None):
            for part in (event.content.parts or []):
                if getattr(part, "text", None):
                    chunks.append(part.text)
    return "".join(chunks).strip()


# --- API ---------------------------------------------------------------------
class GenerateRequest(BaseModel):
    query: str
    format: str = "report"   # report | sns | webpage
    lang: str = "ja"         # ja | zh | en


app = FastAPI(title="SoccerScope")

# 2) Attach slowapi to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # In production, this can be restricted to the submitted URL origin
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "agent": root_agent.name}


@app.get("/debug/mcp-test")
async def debug_mcp_test(token: str = ""):
    """
    Temporary debug endpoint to verify whether the bundled mongodb-mcp-server,
    launched directly with node, can actually start in this FC runtime.
    Remove this after verification.

    If anyone could call this endpoint, they could start MCP subprocesses at
    will, which would waste cold-start time and cost and expose the internal
    tool list. Therefore, return 404 when ?token=... does not match the
    DEBUG_TOKEN environment variable, hiding the endpoint itself.
    """
    expected = os.environ.get("DEBUG_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(status_code=404, detail="not found")

    log: list[str] = []
    log.append(f"PATH={os.environ.get('PATH')}")

    node_path = shutil.which("node")
    log.append(f"node -> {node_path}")
    if node_path is None:
        return {
            "ok": False,
            "log": log,
            "hint": "node was not found. Check whether the official Node.js 20 public "
            "layer is attached and whether the layer version/runtime is compatible.",
        }

    # --- v4: Check whether the bundled node_modules/mongodb-mcp-server exists ----
    # Since npx was removed and the server bundled into code.zip at build time is
    # now run directly with node, first check whether node_modules was included
    # in the zip at all.
    from soccer_agent.agent import _MONGODB_MCP_ENTRY  # noqa: E402

    log.append(f"mongodb-mcp-server entry -> {_MONGODB_MCP_ENTRY}")
    if not os.path.exists(_MONGODB_MCP_ENTRY):
        return {
            "ok": False,
            "log": log,
            "hint": "The bundled mongodb-mcp-server was not found. "
            "Check that `npm install --prefix build mongodb-mcp-server` was "
            "run at build time and that node_modules/ was included in code.zip.",
        }

    # --- Pre-check: run node <entry> --dryRun directly as a raw subprocess and
    # inspect startup settings and enabled tools from raw stdout before they are
    # wrapped by the mcp SDK anyio TaskGroup. Since npx is no longer involved,
    # this should complete within a few seconds. If it is still slow or hangs,
    # suspect the MongoDB connection itself rather than npm.
    t_pre = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            "node",
            _MONGODB_MCP_ENTRY,
            "--readOnly",
            env={
                **os.environ,
                "MDB_MCP_CONNECTION_STRING": os.environ.get("MONGODB_URI", ""),
                "MDB_MCP_TELEMETRY": "disabled",
                "MDB_MCP_DRY_RUN": "true",
                "HOME": "/tmp",
                "MDB_MCP_LOG_PATH": "/tmp/mongodb-mcp-logs",
                "MDB_MCP_EXPORTS_PATH": "/tmp/mongodb-mcp-exports",
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
            pre_elapsed = time.monotonic() - t_pre
            log.append(
                f"[pre-check] node <entry> --readOnly (MDB_MCP_DRY_RUN=true) "
                f"(exit={proc.returncode}, {pre_elapsed:.1f}s)"
            )
            if stdout:
                log.append(f"[pre-check stdout] {stdout.decode(errors='replace')[:2000]}")
            if stderr:
                log.append(f"[pre-check stderr] {stderr.decode(errors='replace')[:2000]}")
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            pre_elapsed = time.monotonic() - t_pre
            log.append(
                f"[pre-check] TIMEOUT after {pre_elapsed:.1f}s — if this still hangs "
                "after removing npx, suspect the MongoDB Atlas connection itself "
                "(network/IP allowlist/VPC settings)"
            )
    except Exception as e:  # noqa: BLE001
        log.append(f"[pre-check] failed to start the subprocess itself: {type(e).__name__}: {e}")

    # --- Main check: establish an MCP session through the mcp SDK and call list_tools() ---
    def _flatten_exceptions(exc: BaseException):
        """Recursively flatten ExceptionGroup/TaskGroup contents to extract the real exceptions."""
        subs = getattr(exc, "exceptions", None)
        if subs:
            for s in subs:
                yield from _flatten_exceptions(s)
        else:
            yield exc

    async def _run_mcp_check() -> list[str]:
        # Bug fix: this used to `return tool_names` here, but when return is inside
        # an `async with` block and an exception occurs during cleanup (__aexit__)
        # after the value has been finalized, Python lets that cleanup exception
        # override the return value and propagate to the caller. Store the fetched
        # result in an outer variable, then swallow cleanup-only exceptions here.
        got: dict[str, list[str] | None] = {"tools": None}
        try:
            async with stdio_client(_mcp_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    elapsed = time.monotonic() - t0
                    log.append(f"initialize OK ({elapsed:.1f}s)")

                    tools_result = await session.list_tools()
                    tool_names = sorted(t.name for t in tools_result.tools)
                    log.append(f"tools ({len(tool_names)}): {tool_names}")
                    got["tools"] = tool_names

                    expected_tools = {
                        "find", "count", "list-collections", "collection-schema",
                    }
                    missing = expected_tools - set(tool_names)
                    if missing:
                        log.append(f"warning: missing tool names: {sorted(missing)}")
        except Exception as e:  # noqa: BLE001
            if got["tools"] is not None:
                # The data we need has already been fetched, so ignore cleanup errors during disconnect.
                log.append(
                    f"(warning: an exception occurred during session cleanup, but "
                    f"tool-list retrieval itself already succeeded, so it is ignored: "
                    f"{type(e).__name__}: {e})"
                )
                return got["tools"]
            raise
        return got["tools"]  # type: ignore[return-value]

    t0 = time.monotonic()
    try:
        # Without an explicit limit here, a hang would be force-killed by the FC timeout
        # and no logs would be returned, which is the issue observed here.
        tool_names = await asyncio.wait_for(_run_mcp_check(), timeout=30)
        return {"ok": True, "tools": tool_names, "log": log}
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        log.append(
            f"TIMEOUT after {elapsed:.1f}s — MCP session establishment itself is likely "
            "hanging, for example because package retrieval through npm is stuck"
        )
        return {"ok": False, "log": log}
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        log.append(f"ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
        for i, sub in enumerate(_flatten_exceptions(e)):
            log.append(f"  cause[{i}]: {type(sub).__name__}: {sub}")
        log.append("--- traceback ---")
        log.append(traceback.format_exc())
        return {"ok": False, "log": log}


@app.get("/debug/search-test")
async def debug_search_test(token: str = "", q: str = "buzzing football video", country: str = ""):
    """
    Temporary debug endpoint that calls search_videos() directly, without going
    through the Qwen agent, to inspect raw data and verify whether documents
    stored in MongoDB Atlas actually contain url / thumbnail_url.
    Remove this after verification.

    Background: reports were missing [watch] links or thumbnail images. This
    checks only the data layer, without involving the LLM, to distinguish between
    (a) missing or empty url/thumbnail_url in the data itself and (b) data exists
    but Qwen does not follow the instruction to include it in the output.
    """
    expected = os.environ.get("DEBUG_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(status_code=404, detail="not found")

    try:
        result = await search_videos(query_text=q, country=country, limit=5, buzz_only=False)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    videos = result.get("videos", [])
    summary = [
        {
            "video_id": v.get("video_id"),
            "title": v.get("title"),
            "url": v.get("url"),
            "thumbnail_url": v.get("thumbnail_url"),
            "has_url": bool(v.get("url")),
            "has_thumbnail_url": bool(v.get("thumbnail_url")),
        }
        for v in videos
    ]
    n_missing_url = sum(1 for v in videos if not v.get("url"))
    n_missing_thumb = sum(1 for v in videos if not v.get("thumbnail_url"))

    return {
        "ok": "error" not in result,
        "count": result.get("count"),
        "error": result.get("error"),
        "n_missing_url": n_missing_url,
        "n_missing_thumbnail_url": n_missing_thumb,
        "hint": (
            "If any document has an empty url/thumbnail_url, the issue is in the data "
            "layer (embedding pipeline). If all values are present but the report "
            "has no links, the issue is Qwen instruction following; address it by "
            "strengthening the prompt."
            if (n_missing_url or n_missing_thumb) or videos
            else None
        ),
        "videos": summary,
    }


@app.post("/api/generate")
@limiter.limit("3/minute")          # 2) Up to 3 requests per minute per IP
async def generate(req: GenerateRequest, request: Request):
    # 1) Query length check
    if len(req.query) > QUERY_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"query too long (max {QUERY_MAX_LEN} chars, got {len(req.query)})",
        )
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    if req.format not in FORMAT_DIRECTIVES:
        raise HTTPException(status_code=400, detail=f"unknown format: {req.format}")
    if req.lang not in LANG_DIRECTIVES:
        raise HTTPException(status_code=400, detail=f"unknown lang: {req.lang}")

    prompt = _build_prompt(req.query, req.format, req.lang)
    try:
        content = await _run_agent(prompt)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"agent error: {e}")

    if not content:
        raise HTTPException(status_code=502, detail="agent returned empty output")

    return {"format": req.format, "lang": req.lang, "content": content}


# Static frontend. Mount last so the API routes above take precedence.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    # timeout_keep_alive: Function Compute 3.0 custom container requirements say
    # to enable keep-alive and set the request timeout to at least 15 minutes
    # (900 seconds), so set it explicitly. With the default 5 seconds, requests
    # through FC may be disconnected midway.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        timeout_keep_alive=900,
    )
