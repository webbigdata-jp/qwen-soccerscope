#!/usr/bin/env python3
"""
Phase5 ステージ1: コメント感情分析 + サッカー関連性判定 → ローカル保存
【Qwen Cloud移植版】

../../soccer/data/<date>/phase4_comments_*.json の各動画について、タイトル・
説明文・コメント群を Qwen(DashScope, OpenAI互換API) で分析し、
サッカー関連性(is_soccer_related)・感情比率・ポジ/ネガのテーマ(ja/en)・
引用候補(ja/en訳付き) を生成して ./data/<date>/comment_analysis_<timestamp>.json
に保存する。

【Gemini版との差分（重要）】
  DashScopeのOpenAI互換APIには、Geminiの response_schema=PydanticModel の
  ような「スキーマ強制」機能が無い。サポートされるのは
  response_format={"type": "json_object"} という、構文的に正しいJSONで
  あることだけを保証する「JSONモード」のみ（フィールド名・型・ネスト構造は
  保証されない）。そのため、以下の2段構えにしている:
    1. プロンプト(system_instruction)にスキーマをテキストで明示的に書き下す
       （Alibaba公式ドキュメント推奨のパターン）
    2. 返ってきたJSONを自前でこちらのPydanticモデルにバリデーションし、
       不一致ならリトライする（Gemini版には無かった処理）
  また、DashScopeはプロンプトに"json"という単語が含まれていないと
  response_format指定時に400エラーになる制約があるため、system_instruction
  に明示的に含めている。thinkingモードとjson_objectモードは併用不可のため、
  thinking関連のパラメータは一切指定しない（Gemini版のthinking_configは削除）。
  max_tokens系のハードキャップも、Alibaba公式が「構造化出力使用時は設定するな
  （途中で切れてJSON破損のリスク）」と明記しているため設定しない。

【日付ベースのI/O】
  入力: ../../soccer/data/<date>/phase4_comments_*.json （Geminiフロー側、読むだけ）
  出力: ./data/<date>/comment_analysis_<timestamp>.json
  <date> は省略時は今日(YYYYMMDD)。バックフィル時は引数で指定
  （例: python 3_analyze_comments.py 20260704）。

【耐障害性・速度に関する追加対応】
  1. タイムアウト明示 + タイムアウトも429と同じ指数バックオフでリトライ
     （openai公式SDKはtimeout未指定だとデフォルト600秒・自動リトライ込みで
     1件が最悪30分近く固まり得るため、明示的に短くしている）。
  2. 途中経過を ./data/<date>/_analyze_checkpoint.json に逐次保存し、
     中断後の再実行時は処理済みvideo_idをスキップして再開する。
  3. ThreadPoolExecutorによる並列実行（デフォルト同時実行数5、
     環境変数QWEN_MAX_WORKERSで調整可）。逐次実行+固定sleepをやめ、
     並列数の上限自体をレート制御として使う。

【2段構成の前半】DashScope APIを叩くのはこのスクリプトだけ。
MongoDB投入(ステージ2)をやり直してもAPIを再消費しない。

事前準備:
    pip install openai pydantic python-dotenv
    export DASHSCOPE_API_KEY='...'

実行:
    python 3_analyze_comments.py [YYYYMMDD]
    （省略時は今日の日付で ../../soccer/data/<今日>/ を読みに行く）
"""

import os
import sys
import glob
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError
from openai import OpenAI

MODEL = os.environ.get("QWEN_CHAT_MODEL", "qwen-plus")
FIX_MODEL = os.environ.get("QWEN_FIX_MODEL", "qwen-flash")  # JSON構文修復専用（Alibaba公式推奨パターン）
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
TIMEOUT_SECONDS = float(os.environ.get("QWEN_TIMEOUT_SECONDS", "60"))
# openai公式SDKはデフォルト600秒・自動リトライ込みで1件が最悪30分近く固まり得るため、
# 明示的に短くする。60秒あれば通常のqwen-plus応答には十分すぎるくらい余裕がある。

MAX_WORKERS = int(os.environ.get("QWEN_MAX_WORKERS", "5"))
# 並列実行数。DashScopeの一般的なレート上限(有償枠でqwen-plusはRPM 600程度が目安)を
# 踏まえた安全側の初期値。429が頻発するようなら下げる、余裕があれば上げる。

MAX_COMMENTS = 100                    # 1動画あたりQwenに渡す上限（いいね順）
MAX_QUOTES = 3                        # 引用候補の最大数
MAX_TEAMS = 8                         # 言及チームの最大数
TEMPERATURE = 0.3                     # 分析タスクなので低めで安定寄り

MAX_RATE_LIMIT_RETRIES = 4            # 429/タイムアウト時の指数バックオフ回数
MAX_SCHEMA_RETRIES = 2                # こちらのPydanticモデルと不一致だった場合の再生成回数
                                       # （429リトライとは別カウンタ。無限リトライでの課金膨張を防ぐ）

CHECKPOINT_FILENAME = "_analyze_checkpoint.json"
# 先頭に"_"を付け、comment_analysis_*.json / phase4_comments_*.json 等の
# globパターンと衝突しない名前にしている（4_load_comment_analysis.py側の
# 「最新のcomment_analysis_*.jsonを選ぶ」ロジックに影響を与えないため）。

SCRIPT_DIR = Path(__file__).parent

from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / '.env')


# ---- 構造化出力スキーマ（引き継ぎ書の comment_analysis に準拠。Gemini版と同一）----
class Sentiment(BaseModel):
    positive: float   # 比率(%)。3つの合計が約100になる想定
    negative: float
    neutral: float


class Theme(BaseModel):
    theme_ja: str
    theme_en: str
    mention_count: int


class QuotableComment(BaseModel):
    original: str
    translated_ja: str
    translated_en: str
    author: str
    likes: int
    original_language: str


class TeamMention(BaseModel):
    team: str          # 集計のため英語の代表チーム名で正規化（例: "Argentina", "Japan"）
    sentiment: str     # そのチームへの全体論調: "positive" / "neutral" / "negative"
    mention_count: int  # 言及したと思われるコメント数の概算


class CommentAnalysis(BaseModel):
    is_soccer_related: bool
    relevance_reason: str
    sentiment: Sentiment
    positive_themes: list[Theme]
    negative_themes: list[Theme]
    quotable_comments: list[QuotableComment]
    mentioned_teams: list[TeamMention]


# DashScopeは response_format 指定時、プロンプト中に"json"という単語(大小文字問わず)が
# 含まれていないと400エラーになる制約がある。そのため明示的に含めている。
SYSTEM_INSTRUCTION = (
    "あなたはサッカー(FIFAワールドカップ)関連のYouTube動画コメントを分析する専門家です。"
    "与えられた多言語のコメント群を分析し、視聴者の感情・話題・引用に値する声を構造化して返します。"
    "\n\n"
    "必ず有効なJSON形式のみで応答してください。前置き・説明文・マークダウンのコードブロック"
    "(```json ... ```などの記法)は一切付けず、JSONオブジェクト単体を出力してください。"
    "出力するJSONオブジェクトは、以下の構造・型に厳密に従ってください:\n"
    "{\n"
    '  "is_soccer_related": true または false (真偽値),\n'
    '  "relevance_reason": "判定理由(日本語1文程度、文字列)",\n'
    '  "sentiment": {"positive": 数値(%), "negative": 数値(%), "neutral": 数値(%)},\n'
    '  "positive_themes": [{"theme_ja": "文字列", "theme_en": "文字列", '
    '"mention_count": 整数}, ...],\n'
    '  "negative_themes": [{"theme_ja": "文字列", "theme_en": "文字列", '
    '"mention_count": 整数}, ...],\n'
    '  "quotable_comments": [{"original": "文字列", "translated_ja": "文字列", '
    '"translated_en": "文字列", "author": "文字列", "likes": 整数, '
    '"original_language": "言語コード文字列"}, ...],\n'
    '  "mentioned_teams": [{"team": "英語の代表チーム名", '
    '"sentiment": "positive"か"neutral"か"negative"のいずれか, '
    '"mention_count": 整数}, ...]\n'
    "}\n"
    "全てのフィールドは省略せず必ず含めてください（該当が無い場合は空配列 [] '"
    "を使い、フィールド自体を欠落させないでください）。"
)


def make_client() -> OpenAI:
    """DashScope用OpenAIクライアントを生成する。timeoutを明示指定（デフォルト60秒）。

    openai公式SDKはtimeout未指定だとデフォルト600秒・自動リトライ込みで
    1件が最悪30分近く固まり得るため、明示的に短くしている。
    """
    return OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=DASHSCOPE_BASE_URL,
        timeout=TIMEOUT_SECONDS,
    )


def find_phase4_path(date_dir: Path) -> str:
    hits = sorted(glob.glob(str(date_dir / "phase4_comments_*.json")))
    if not hits:
        print(f"ERROR: {date_dir}/phase4_comments_*.json が見つかりません。", file=sys.stderr)
        print("日付指定が正しいか、Geminiフロー側(phase2→phase3→phase7→phase4)が"
              "その日付で実行済みか確認してください。", file=sys.stderr)
        sys.exit(1)
    return hits[-1]


def build_prompt(meta: dict, comments: list) -> str:
    """動画メタ + コメント一覧から分析プロンプトを組む。（Gemini版と同一ロジック）

    meta['countries'] は (b) 多対多方式で、動画が出現した全国のリスト。
    1動画が複数国の検索結果に出現し得るため、国名・言語は複数列挙する。

    is_soccer_related の判定はタイトル・説明文を主な根拠とする（コメントが
    0件/取得失敗の動画でも判定できるようにするため）。コメントがあれば
    判定の補助材料として使ってよい。
    """
    lines = []
    for c in comments:
        author = c.get("author_display_name", "")
        likes = c.get("like_count", 0)
        text = (c.get("text_original") or "").replace("\n", " ").strip()
        lines.append(f"[likes={likes}] {author}: {text}")
    comments_block = "\n".join(lines) if lines else "(コメントなし、または取得不可)"

    countries = meta.get("countries", []) or []
    country_names_ja = [c.get("country_name_ja", "") for c in countries if c.get("country_name_ja")]
    primary_langs = sorted({c.get("primary_lang", "") for c in countries if c.get("primary_lang")})
    countries_str = "、".join(country_names_ja) if country_names_ja else "不明"
    langs_str = "、".join(primary_langs) if primary_langs else "不明"

    description = (meta.get("description") or "").strip()
    description_block = description[:500] if description else "(説明文なし)"

    return (
        f"# 動画情報\n"
        f"出現国(この動画が話題になっている国、複数の場合あり): {countries_str}\n"
        f"主要言語(出現国の言語、複数の場合あり): {langs_str}\n"
        f"タイトル: {meta.get('title')}\n"
        f"説明文(先頭500字): {description_block}\n\n"
        f"# コメント({len(comments)}件、いいね数の多い順)\n"
        f"{comments_block}\n\n"
        f"# 指示\n"
        f"上記の動画情報・コメントを分析し、以下を生成してください:\n"
        f"0. is_soccer_related: この動画がサッカー（FIFAワールドカップ含む）に"
        f"関連する内容かどうかを true/false で判定してください。タイトル・説明文を"
        f"主な根拠にし、コメントがあれば補助的に使ってください。"
        f"検索キーワードに「World Cup」が含まれていても、クリケット・バスケットボール・"
        f"バレーボール等の他競技の「World Cup」を冠した大会である場合は false としてください"
        f"（例: ICC Women's T20 World Cup はクリケットなので false）。"
        f"判断に迷う場合や情報が不十分な場合は true（除外しすぎない方を優先）としてください。"
        f"relevance_reason にはその判定理由を日本語1文程度で書いてください。\n"
        f"以降の1〜4は is_soccer_related が false の場合でも形式上埋めてください"
        f"（コメントが無ければ sentiment は positive=0,negative=0,neutral=100、"
        f"各リストは空配列で構いません）。\n"
        f"1. sentiment: ポジティブ/ネガティブ/中立の比率(%)。合計が約100になるように。\n"
        f"2. positive_themes / negative_themes: 主要な話題を、theme_ja(日本語)・"
        f"theme_en(英語)・mention_count(言及したと思われるコメント数の概算)で。各最大5件。\n"
        f"3. quotable_comments: 記事に引用して映える、いいね数が多く印象的なコメントを最大{MAX_QUOTES}件。"
        f"original(原文そのまま)・translated_ja(日本語訳)・translated_en(英語訳)・"
        f"author(投稿者名)・likes(入力のlikes値)・original_language(原文の言語コード)。\n"
        f"4. mentioned_teams: コメントで言及されている『代表チーム』を最大{MAX_TEAMS}件。"
        f"team は後段で言語をまたいで集計するため、必ず英語の代表チーム名/国名で正規化する"
        f"(例: 'Argentina','Brazil','Japan','Morocco'。'日本'や'🇲🇽'やクラブ名ではなく代表名に寄せる)。"
        f"sentiment はそのチームに対する全体の論調を 'positive'/'neutral'/'negative' のいずれかで。"
        f"mention_count はそのチームに言及したと思われるコメント数の概算。"
        f"代表チームへの言及が無ければ空配列で良い。\n"
        f"コメントが少ない/分析困難な場合は、可能な範囲で返してください。"
    )


def _is_retryable_transient_error(msg: str) -> bool:
    """429・タイムアウト・接続エラーなど「もう一度試せば直る可能性がある」エラーか判定する。

    openai公式SDKはタイムアウト時に openai.APITimeoutError を投げ、メッセージには
    "Request timed out." のような文言が入る（実際に今回のハングで観測した文言）。
    接続エラー(httpx.ConnectError等)も同様に一時的な問題である可能性が高いので
    まとめてリトライ対象にする。
    """
    lowered = msg.lower()
    return (
        "429" in msg
        or "resource_exhausted" in lowered
        or "throttling" in lowered
        or "timed out" in lowered
        or "timeout" in lowered
        or "connection" in lowered
    )


def _is_content_moderation_error(msg: str) -> bool:
    """DashScope側のコンテンツモデレーション拒否かどうかを判定する。

    観測されたエラー文言に基づくヒューリスティックな判定
    （例: "Input data may contain inappropriate content" / code=DataInspectionFailed）。
    厳密な判定ではないが、429と違い「同じ入力をリトライしても結果は変わらない」
    という運用上重要な性質があるため、無駄なリトライを避けるために区別する。
    """
    lowered = msg.lower()
    return (
        "inappropriate content" in lowered
        or "datainspectionfailed" in lowered
        or "data_inspection_failed" in lowered
        or "content_filter" in lowered
    )


def _call_json(client: OpenAI, model: str, system: str, user: str):
    """response_format=json_objectでchat.completions.createを呼ぶ薄いラッパー。

    enable_thinking=False を extra_body 経由で明示指定している。qwen-plus/qwen3-max等は
    デフォルトでthinking無効だが、qwen3.7-plus等のhybrid thinkingモデル(デフォルトthinking
    有効)を QWEN_CHAT_MODEL に指定した場合、thinking有効のままだと
    response_format=json_object と併用できず400エラーになる（Alibaba公式ドキュメントに
    明記）。enable_thinking はOpenAI標準パラメータではないため、OpenAI SDK経由では
    extra_body に入れて渡す必要がある（同じくAlibaba公式ドキュメントの記載通り）。
    thinkingがデフォルト無効なモデルに対しては無害（明示的にFalseにするだけ）。
    """
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
        extra_body={"enable_thinking": False},
    )


def _repair_json(client: OpenAI, broken_text: str) -> str:
    """JSON構文エラー時、軽量モデルに修復を依頼する（Alibaba公式推奨パターン）。

    FIX_MODEL(デフォルトqwen-flash)はデフォルトでthinking無効だが、環境変数で
    別モデルに差し替えられた場合の保険として、こちらも明示的にFalseにしておく。
    """
    resp = client.chat.completions.create(
        model=FIX_MODEL,
        messages=[
            {"role": "system", "content": "あなたはJSON形式の専門家です。"
                                            "ユーザーが渡す壊れたJSON文字列を、"
                                            "有効なJSON形式に修復してください。"
                                            "JSONオブジェクト単体のみを出力し、"
                                            "説明文やコードブロック記法は付けないでください。"},
            {"role": "user", "content": broken_text},
        ],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content


def analyze_with_retry(client: OpenAI, prompt: str,
                        max_rate_retries: int = MAX_RATE_LIMIT_RETRIES,
                        max_schema_retries: int = MAX_SCHEMA_RETRIES):
    """1動画を分析。429時は指数バックオフ、スキーマ不一致時は再生成。

    戻り値: (CommentAnalysis or None, reason or None)
      成功時は (CommentAnalysis, None)。
      失敗時は (None, reason) で、reason は以下のいずれか:
        - "content_moderation"     : DashScope側のコンテンツモデレーション拒否
                                      (同じ入力のリトライは無意味なため即座に諦める)
        - "rate_limit_exhausted"   : 429/タイムアウト/接続エラーが指定回数リトライしても
                                      解消しなかった
        - "schema_retry_exhausted" : JSON構文/スキーマ不一致が指定回数リトライしても解消しなかった
        - "api_error:<詳細>"       : 上記以外のAPIエラー

    429・タイムアウト・接続エラー(通信の問題)とスキーマリトライ(モデル出力の質の問題)は
    性質が違うので別カウンタにしている。
    """
    schema_attempt = 0
    while schema_attempt <= max_schema_retries:
        raw = None
        for rate_attempt in range(max_rate_retries):
            try:
                resp = _call_json(client, MODEL, SYSTEM_INSTRUCTION, prompt)
                raw = resp.choices[0].message.content
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if _is_content_moderation_error(msg):
                    print(f"  WARNING: コンテンツモデレーション拒否: {msg[:160]}", file=sys.stderr)
                    return None, "content_moderation"
                if _is_retryable_transient_error(msg) and rate_attempt < max_rate_retries - 1:
                    wait = 5 * (2 ** rate_attempt)
                    print(f"  一時的なエラーの可能性({msg[:60]})。{wait}秒待機してリトライ... "
                          f"({rate_attempt + 1}/{max_rate_retries})")
                    time.sleep(wait)
                elif _is_retryable_transient_error(msg):
                    print(f"  ERROR: 分析失敗(リトライ上限): {msg[:160]}", file=sys.stderr)
                    return None, "rate_limit_exhausted"
                else:
                    print(f"  ERROR: 分析失敗: {msg[:160]}", file=sys.stderr)
                    return None, f"api_error:{msg[:100]}"

        if raw is None:
            return None, "rate_limit_exhausted"

        # JSON構文パース
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("  WARNING: JSON構文エラー。修復モデルで修復を試みます...", file=sys.stderr)
            try:
                fixed = _repair_json(client, raw)
                data = json.loads(fixed)
            except Exception as e:  # noqa: BLE001
                schema_attempt += 1
                print(f"  WARNING: JSON修復も失敗(試行{schema_attempt}/{max_schema_retries}): "
                      f"{str(e)[:160]}", file=sys.stderr)
                continue

        # こちらのPydanticモデルへバリデーション（DashScopeはスキーマを強制しないため必須）
        try:
            return CommentAnalysis(**data), None
        except ValidationError as e:
            schema_attempt += 1
            print(f"  WARNING: スキーマ不一致(試行{schema_attempt}/{max_schema_retries}): "
                  f"{str(e)[:300]}", file=sys.stderr)
            continue

    print("  ERROR: スキーマ検証のリトライ上限に到達。スキップします。", file=sys.stderr)
    return None, "schema_retry_exhausted"


def resolve_dirs(date_str: str):
    """日付文字列から入力ディレクトリ(Geminiフロー側, 読取専用)と
    出力ディレクトリ(qwen_soccer側)を決める。"""
    input_dir = (SCRIPT_DIR / ".." / ".." / "soccer" / "data" / date_str).resolve()
    output_dir = (SCRIPT_DIR / "data" / date_str).resolve()
    return input_dir, output_dir


def load_checkpoint(checkpoint_path: Path) -> dict:
    """既存のチェックポイントを読み込む。無ければ空の状態を返す。"""
    if not checkpoint_path.exists():
        return {"analyses": {}, "skipped": []}
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("analyses", {})
        data.setdefault("skipped", [])
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: チェックポイント読み込み失敗、破棄して最初からやり直します: {e}",
              file=sys.stderr)
        return {"analyses": {}, "skipped": []}


def save_checkpoint(checkpoint_path: Path, state: dict) -> None:
    """チェックポイントを保存する（一時ファイル→リネームでの原子的な書き込み）。"""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    tmp_path.replace(checkpoint_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase5 ステージ1: コメント感情分析(Qwen版)")
    parser.add_argument("date", nargs="?", default=None,
                         help="対象日付(YYYYMMDD)。省略時は今日")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY が未設定です。", file=sys.stderr)
        return 1

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    input_dir, output_dir = resolve_dirs(date_str)

    path = find_phase4_path(input_dir)
    print(f"対象日付: {date_str}")
    print(f"入力ファイル: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_video = data.get("comments_by_video", {})
    if not by_video:
        print("ERROR: comments_by_video が空です。", file=sys.stderr)
        return 1
    print(f"対象動画: {len(by_video)}件 / 並列実行数: {MAX_WORKERS} / タイムアウト: {TIMEOUT_SECONDS}秒")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    state = load_checkpoint(checkpoint_path)
    analyses = state["analyses"]
    skipped = state["skipped"]
    already_done = set(analyses.keys()) | {s["video_id"] for s in skipped}
    if already_done:
        print(f"  チェックポイントを検出: {len(already_done)}件は処理済みとしてスキップします。")

    client = make_client()
    lock = threading.Lock()
    items = list(by_video.items())
    total = len(items)

    def process_one(idx: int, vid: str, meta: dict):
        """1動画分の分析を行い、結果をチェックポイントへ反映する（複数スレッドから呼ばれる）。

        Exception全般をここで捕まえて必ずチェックポイントに反映する。
        1件の予期せぬバグ・異常系で他の完了済み分析結果が失われないようにするため。
        （本物のCtrl+C(KeyboardInterrupt)やSystemExitはExceptionのサブクラスではない
        ため、ここでは捕まえず正常にプロセス全体を中断できる。）
        """
        try:
            if not (meta.get("title") or "").strip():
                with lock:
                    skipped.append({"video_id": vid, "reason": "no_title_no_judgeable_info"})
                    save_checkpoint(checkpoint_path, state)
                print(f"[{idx}/{total}] {vid} スキップ (タイトルも無く判定材料が無い)")
                return

            comments = meta.get("comments", []) or []
            top = sorted(comments, key=lambda c: c.get("like_count", 0), reverse=True)[:MAX_COMMENTS]
            prompt = build_prompt(meta, top)

            result, fail_reason = analyze_with_retry(client, prompt)
            with lock:
                if result is None:
                    skipped.append({"video_id": vid, "reason": fail_reason or "analysis_failed"})
                else:
                    rec = result.model_dump()
                    rec["analyzed_at"] = datetime.now(timezone.utc).isoformat()
                    rec["total_analyzed"] = len(top)
                    analyses[vid] = rec
                save_checkpoint(checkpoint_path, state)

            status = "OK" if result is not None else f"失敗({fail_reason})"
            print(f"[{idx}/{total}] {vid} 完了: {status}")
        except Exception as e:  # noqa: BLE001
            # 想定外のバグ等でもここで打ち止めにし、他の動画の結果を道連れにしない
            with lock:
                skipped.append({"video_id": vid, "reason": f"unexpected_error:{str(e)[:150]}"})
                save_checkpoint(checkpoint_path, state)
            print(f"[{idx}/{total}] {vid} 完了: 予期せぬエラー({str(e)[:150]})", file=sys.stderr)

    targets = [(i, vid, meta) for i, (vid, meta) in enumerate(items, 1) if vid not in already_done]
    print(f"  今回処理する動画: {len(targets)}件（スキップ済み{len(already_done)}件を除く）")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, idx, vid, meta) for idx, vid, meta in targets]
        try:
            for fut in as_completed(futures):
                fut.result()  # 例外があればここで再送出される
        except KeyboardInterrupt:
            # デフォルトのshutdown(wait=True)は未着手タスクもキューが空になるまで
            # 処理し続けてしまい、Ctrl+Cへの反応が遅い。cancel_futures=Trueで
            # 「未着手」のタスクだけキャンセルし、実行中のタスクの完了は待つ
            # （実行中タスクは中断せず、チェックポイントへの反映を確実にするため）。
            print("\n中断を検知しました。実行中のタスクの完了を待ち、未着手分はキャンセルします...",
                  file=sys.stderr)
            executor.shutdown(wait=True, cancel_futures=True)
            raise

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"comment_analysis_{ts}.json"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_file": os.path.basename(path),
        "model": MODEL,
        "total_videos_analyzed": len(analyses),
        "total_skipped": len(skipped),
        "skipped": skipped,
        "analyses": analyses,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"\n分析完了: 成功{len(analyses)}件 / スキップ{len(skipped)}件")
    if skipped:
        from collections import Counter
        reason_counts = Counter(s["reason"] for s in skipped)
        print("  内訳:")
        for reason, cnt in reason_counts.most_common():
            print(f"    {reason}: {cnt}件")
    print(f"保存しました: {out_path}")

    # 全件書き出しに成功したのでチェックポイントは削除
    # （残しておくと、翌日以降に同じ日付で誤って再実行した際、意図せず前回結果を
    #  引き継いでしまう恐れがあるため）
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError:
        pass

    print("次はステージ2でこれを videos.comment_analysis に投入します。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
