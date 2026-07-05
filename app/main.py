"""
SoccerScope — Web backend (FastAPI)

独自Web UI のためのバックエンド。
  - 既存の ADK エージェント（soccer_agent.agent.root_agent）を ADK Runner で実行する。
  - フロントから受け取る {query, format, lang} を、エージェント本体を改変せずに
    「プロンプトへ指示を注入」する形で出力形式（report / sns / webpage）と
    出力言語（ja / en）に反映する。
  - 同一オリジンで静的フロント（static/index.html）も配信する。

エージェント(agent.py)は無改変。書き込み系は持たず、読み出しは agent 内の
search_videos → 公式MongoDB MCP 経由（MCP統合要件を維持）。

ローカル実行:
    uvicorn main:app --host 0.0.0.0 --port 8080
Cloud Run:
    Dockerfile 同梱（Python + Node 22）。README.md 参照。
"""

import os
import uuid

# --- .env を読み込む（agent.py はインポート時に MONGODB_URI を参照するので、
#     エージェント取り込みより前に読み込む。Cloud Run では .env が無くても無害）---
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

from soccer_agent.agent import root_agent  # noqa: E402  (.env 読み込み後にimport)

# --- 【一時デバッグ用】MCP単体起動テスト -------------------------------------
# FC上でNode.js 20公式公共層をアタッチしただけで npx mongodb-mcp-server が
# 実際に起動できるかを確認するための、本番機能とは無関係な一時エンドポイント。
# 確認が終わったら / api/generate 等と一緒にこのブロックごと削除すること。
import shutil
import time
import traceback
import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from soccer_agent.agent import _mcp_server_params, search_videos  # noqa: E402

APP_NAME = "soccerscope"

# ② レートリミット（IP別、インメモリ）
limiter = Limiter(key_func=get_remote_address)

# ① クエリ長上限（文字数）
QUERY_MAX_LEN = 500

# Runner / Session は起動時に一度だけ構築（root_agent は使い回す）
_session_service = InMemorySessionService()
_runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=_session_service)


# --- フロントから来る選択肢を、エージェントへの「指示文」に変換する ----------
# エージェントの INSTRUCTION 側に既にある記事/SNS/HTML生成フローを、ここから
# 明示的に呼び分ける。エージェント本体は触らない。
FORMAT_DIRECTIVES = {
    # レポート: マークダウン記事（国別セクション＋総合コメント）
    "report": (
        "OUTPUT FORMAT = REPORT. Produce a complete Markdown article exactly as "
        "described in your COMPOSING ARTICLES flow: a punchy title and short lead, "
        "one section per country (country name + flag, a 1-2 sentence buzz summary, "
        "the thumbnail as a Markdown image, and a [watch] link), sentiment where "
        "available, and a closing insightful synthesis (総合コメント). "
        "MANDATORY: every single video section MUST include both a "
        "`![title](thumbnail_url)` image line and a `[▶ 動画を見る](url)` link line "
        "using the real thumbnail_url/url values from the tool results — never omit "
        "them. Run the SELF-CHECK described in your instructions before answering. "
        "Output Markdown only — do NOT wrap it in a code block, do NOT output raw HTML."
    ),
    # SNS: X投稿ドラフト 2-3本
    "sns": (
        "OUTPUT FORMAT = SNS POSTS. Output 2-3 short, ready-to-post social/X drafts "
        "based on the buzzing videos. Each draft: punchy, 1-2 relevant hashtags, and "
        "exactly one video link. Separate each draft with a blank line and prefix it "
        "with its number (1. / 2. / 3.). Do not add an article or extra commentary "
        "around the drafts — output the posts only."
    ),
    # Webページ: レポートと同じマークダウン記事を返す（見た目はフロントで“ページ風”に整える）
    "webpage": (
        "OUTPUT FORMAT = WEB FEATURE PAGE. Produce a complete, shareable Markdown "
        "feature article as in your COMPOSING ARTICLES flow (title + lead, one section "
        "per country with flag + thumbnail image + watch link + sentiment, and a strong "
        "closing 総合コメント). Make it engaging and presentation-ready. "
        "MANDATORY: every single video section MUST include both a "
        "`![title](thumbnail_url)` image line and a `[▶ 動画を見る](url)` link line "
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
    """1リクエスト = 1セッションでエージェントを実行し、最終応答テキストを返す。"""
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
        # 最終応答のテキストのみ集約（途中のツール呼び出しイベントは無視）
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

# ② slowapi をアプリに接続
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 本番は提出URLのオリジンに絞ってよい
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "agent": root_agent.name}


@app.get("/debug/mcp-test")
async def debug_mcp_test(token: str = ""):
    """
    【一時デバッグ用】mongodb-mcp-server（バンドル済み、node直接実行）が
    このFC実行環境で実際に起動できるか確認するためのエンドポイント。
    確認が終わったら削除すること。

    誰でも叩けるとMCPサブプロセスを勝手に起動されてしまう（コールドスタート時間
    やコストの無駄・内部ツール一覧の露出につながる）ため、環境変数DEBUG_TOKENと
    一致する ?token=... が無い場合は404を返す（存在自体を隠す）。
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
            "hint": "nodeが見つかりません。Node.js 20公式公共層がアタッチ"
            "されているか、層のバージョン/互換ランタイムを確認してください。",
        }

    # --- 【v4】バンドル済みnode_modules/mongodb-mcp-serverの存在チェック -------
    # npxを廃止し、ビルド時にcode.zipへバンドルしたものをnode直接実行する方式に
    # 変更したため、まず「そもそもzipにnode_modulesが入っているか」を確認する。
    from soccer_agent.agent import _MONGODB_MCP_ENTRY  # noqa: E402

    log.append(f"mongodb-mcp-server entry -> {_MONGODB_MCP_ENTRY}")
    if not os.path.exists(_MONGODB_MCP_ENTRY):
        return {
            "ok": False,
            "log": log,
            "hint": "バンドルされているはずのmongodb-mcp-serverが見つかりません。"
            "ビルド時に `npm install --prefix build mongodb-mcp-server` を"
            "実行し、code.zipにnode_modules/を含めたか確認してください。",
        }

    # --- 事前チェック: node <entry> --dryRun を素のサブプロセスとして直接実行し、
    # 起動時の設定・有効ツール一覧を、mcp SDKのanyio TaskGroupに包まれる前の
    # 生の標準出力で確認する。npxを介さないため、ここは数秒で完了するはず
    # （もし依然として遅い/固まるなら、npmではなくMongoDB接続自体を疑う）。
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
                f"[pre-check] TIMEOUT after {pre_elapsed:.1f}s — npxを廃止した"
                "後もここが詰まるなら、MongoDB Atlasへの接続自体（ネットワーク/"
                "IP allowlist/VPC設定）を疑う"
            )
    except Exception as e:  # noqa: BLE001
        log.append(f"[pre-check] サブプロセス起動自体に失敗: {type(e).__name__}: {e}")

    # --- 本チェック: mcp SDK経由でMCPセッションを確立してlist_tools() -----------
    def _flatten_exceptions(exc: BaseException):
        """ExceptionGroup/TaskGroupの中身を再帰的に展開して本当の例外を取り出す。"""
        subs = getattr(exc, "exceptions", None)
        if subs:
            for s in subs:
                yield from _flatten_exceptions(s)
        else:
            yield exc

    async def _run_mcp_check() -> list[str]:
        # 【バグ修正】以前はここで`return tool_names`していたが、returnが
        # `async with`ブロックの中にあると、値を確定させた後の後片付け(__aexit__)
        # で例外が起きた場合、その後片付け中の例外がreturnを上書きして呼び出し元に
        # 伝播してしまう(Pythonの仕様)。取得済みの結果を outer 変数に退避し、
        # 後片付け専用の例外はここで飲み込むようにする。
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
                        log.append(f"注意: 見つからないツール名: {sorted(missing)}")
        except Exception as e:  # noqa: BLE001
            if got["tools"] is not None:
                # 欲しいデータはもう取れているので、切断時の後片付けエラーは無視する。
                log.append(
                    f"(注意: セッション終了時の後片付けで例外が発生しましたが、"
                    f"ツール一覧の取得自体は成功済みのため無視します: "
                    f"{type(e).__name__}: {e})"
                )
                return got["tools"]
            raise
        return got["tools"]  # type: ignore[return-value]

    t0 = time.monotonic()
    try:
        # ここに明示的な上限を付けないと、ハングした場合にFCのタイムアウトで
        # 強制killされ、ログが一切返らない（今回発生した事象）。
        tool_names = await asyncio.wait_for(_run_mcp_check(), timeout=30)
        return {"ok": True, "tools": tool_names, "log": log}
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        log.append(
            f"TIMEOUT after {elapsed:.1f}s — MCPセッションの確立自体がハングして"
            "いる可能性が高い（npm経由の取得が止まっている等）"
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
    【一時デバッグ用】search_videos()をエージェント（Qwen）を経由せず直接呼び出し、
    MongoDB Atlasに実際に保存されているドキュメントに url / thumbnail_url が
    含まれているかを生データで確認するためのエンドポイント。
    確認が終わったら削除すること。

    背景: レポート出力に動画への[watch]リンクやサムネイル画像が含まれない問題が
    報告された。原因が (a) データ側にそもそもurl/thumbnail_urlが無い/空、
    (b) データはあるがQwenが指示に従わず出力に含めていない、のどちらかを
    切り分けるために、LLMを介さずデータ層だけを直接見る。
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
            "url/thumbnail_urlが空のドキュメントがある場合はデータ層(埋め込み"
            "パイプライン)の問題。全部埋まっているのにレポートにリンクが無いなら"
            "Qwen側の指示追従の問題（プロンプト強化で対応）。"
            if (n_missing_url or n_missing_thumb) or videos
            else None
        ),
        "videos": summary,
    }


@app.post("/api/generate")
@limiter.limit("3/minute")          # ② IP別 1分間に3リクエストまで
async def generate(req: GenerateRequest, request: Request):
    # ① クエリ長チェック
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


# 静的フロント（最後にマウント：上の API ルートが優先される）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    # timeout_keep_alive: Function Compute 3.0 のカスタムコンテナ要件で
    # 「keep-aliveを有効にし、リクエストタイムアウトを15分(900秒)以上にする」
    # とあるため明示的に設定。デフォルト(5秒)のままだとFC経由のリクエストが
    # 途中で切断される可能性がある。
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        timeout_keep_alive=900,
    )
