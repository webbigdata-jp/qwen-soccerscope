#!/usr/bin/env python3
"""
Phase5 Stage 1: Comment sentiment analysis + soccer relevance judgment -> local save
[Qwen Cloud migration version]

For each video in ../../soccer/data/<date>/phase4_comments_*.json, analyze the
title, description, and comments with Qwen (DashScope, OpenAI-compatible API),
generate soccer relevance (is_soccer_related), sentiment ratios, positive/negative
themes (ja/en), and quotable comments (with ja/en translations), then save them to
./data/<date>/comment_analysis_<timestamp>.json.

[Important differences from the Gemini version]
  DashScope's OpenAI-compatible API does not have a "schema enforcement" feature
  like Gemini's response_schema=PydanticModel. It only supports JSON mode via
  response_format={"type": "json_object"}, which guarantees only syntactically
  valid JSON (field names, types, and nested structure are not guaranteed). This
  script therefore uses the following two-layer approach:
    1. Explicitly write out the schema in text in the prompt (system_instruction)
       (the pattern recommended by Alibaba's official documentation).
    2. Validate the returned JSON with our own Pydantic model and retry if it does
       not match (a process that the Gemini version did not need).
  Also, DashScope returns a 400 error when response_format is specified unless the
  prompt contains the word "json", so it is included explicitly in system_instruction.
  Thinking mode and json_object mode cannot be used together, so no thinking-related
  parameters are specified (Gemini's thinking_config was removed). Hard caps such as
  max_tokens are also not set because Alibaba's official documentation says not to
  set them when using structured output (to avoid truncation that can break JSON).

[Date-based I/O]
  Input: ../../soccer/data/<date>/phase4_comments_*.json (Gemini flow side, read-only)
  Output: ./data/<date>/comment_analysis_<timestamp>.json
  If <date> is omitted, today's date (YYYYMMDD) is used. For backfills, pass it as an argument
  (example: python 3_analyze_comments.py 20260704).

[Additional reliability and speed handling]
  1. Explicit timeout + retry timeouts with the same exponential backoff as 429s
     (the official openai SDK defaults to 600 seconds when timeout is unspecified,
     and with automatic retries one item can hang for nearly 30 minutes in the worst case;
     therefore, this is set shorter explicitly).
  2. Save progress incrementally to ./data/<date>/_analyze_checkpoint.json, and on
     rerun after interruption, skip already processed video_id values and resume.
  3. Parallel execution with ThreadPoolExecutor (default concurrency 5, adjustable
     with environment variable QWEN_MAX_WORKERS). Instead of sequential execution +
     fixed sleeps, the concurrency limit itself is used as rate control.

[First half of the two-stage flow] This is the only script that calls the DashScope API.
Re-running the MongoDB load step (stage 2) will not consume API usage again.

Prerequisites:
    pip install openai pydantic python-dotenv
    export DASHSCOPE_API_KEY='...'

Run:
    python 3_analyze_comments.py [YYYYMMDD]
    (when omitted, it reads ../../soccer/data/<today>/ using today's date)
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
FIX_MODEL = os.environ.get("QWEN_FIX_MODEL", "qwen-flash")  # Dedicated to JSON syntax repair (Alibaba's recommended pattern).
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
TIMEOUT_SECONDS = float(os.environ.get("QWEN_TIMEOUT_SECONDS", "60"))
# The official openai SDK defaults to 600 seconds when timeout is unspecified, and with
# automatic retries one item can hang for nearly 30 minutes in the worst case, so set it
# shorter explicitly. 60 seconds is more than enough for normal qwen-plus responses.

MAX_WORKERS = int(os.environ.get("QWEN_MAX_WORKERS", "5"))
# Parallel worker count. The initial value is conservative based on typical DashScope
# rate limits (roughly RPM 600 for qwen-plus on paid quotas). Lower it if 429s are
# frequent, or raise it if there is headroom.

MAX_COMMENTS = 100                    # Maximum comments to send to Qwen per video (sorted by likes).
MAX_QUOTES = 3                        # Maximum number of quotable comments.
MAX_TEAMS = 8                         # Maximum number of mentioned teams.
TEMPERATURE = 0.3                     # Low and stable for an analysis task.

MAX_RATE_LIMIT_RETRIES = 4            # Number of exponential-backoff retries for 429/timeouts.
MAX_SCHEMA_RETRIES = 2                # Number of regenerations when the result does not match our Pydantic model.
                                       # Separate from 429 retries to avoid unlimited retries and growing API costs.

CHECKPOINT_FILENAME = "_analyze_checkpoint.json"
# Prefix with "_" so the name does not collide with glob patterns such as
# comment_analysis_*.json / phase4_comments_*.json, and so it does not affect the
# "select the latest comment_analysis_*.json" logic in 4_load_comment_analysis.py.

SCRIPT_DIR = Path(__file__).parent

from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / '.env')


# ---- Structured output schema (follows comment_analysis in the handoff document; same as the Gemini version) ----
class Sentiment(BaseModel):
    positive: float   # Ratio (%). The three values are expected to total roughly 100.
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
    team: str          # Normalized to an English national team name for aggregation (examples: "Argentina", "Japan").
    sentiment: str     # Overall tone toward that team: "positive" / "neutral" / "negative".
    mention_count: int  # Approximate number of comments that appear to mention the team.


class CommentAnalysis(BaseModel):
    is_soccer_related: bool
    relevance_reason: str
    sentiment: Sentiment
    positive_themes: list[Theme]
    negative_themes: list[Theme]
    quotable_comments: list[QuotableComment]
    mentioned_teams: list[TeamMention]


# When response_format is specified, DashScope returns a 400 error unless the prompt contains
# the word "json" (case-insensitive). Include it explicitly here.
SYSTEM_INSTRUCTION = (
    "You are an expert analyst of YouTube video comments related to soccer (including the FIFA World Cup). "
    "Analyze the provided multilingual comments and return structured information about viewer sentiment, topics, and quotable voices."
    "\n\n"
    "Respond only with valid JSON. Do not add a preface, explanatory text, or Markdown code blocks "
    "such as ```json ... ```. Output a single JSON object only. "
    "The JSON object you output must strictly follow this structure and these types:\n"
    "{\n"
    '  "is_soccer_related": true or false (boolean),\n'
    '  "relevance_reason": "reason for the judgment (about one sentence in Japanese, string)",\n'
    '  "sentiment": {"positive": number (%), "negative": number (%), "neutral": number (%)},\n'
    '  "positive_themes": [{"theme_ja": "string", "theme_en": "string", '
    '"mention_count": integer}, ...],\n'
    '  "negative_themes": [{"theme_ja": "string", "theme_en": "string", '
    '"mention_count": integer}, ...],\n'
    '  "quotable_comments": [{"original": "string", "translated_ja": "string", '
    '"translated_en": "string", "author": "string", "likes": integer, '
    '"original_language": "language-code string"}, ...],\n'
    '  "mentioned_teams": [{"team": "English national team name", '
    '"sentiment": one of "positive", "neutral", or "negative", '
    '"mention_count": integer}, ...]\n'
    "}\n"
    "Include every field without omission. If nothing applies, use an empty array [] and do not drop the field."
)


def make_client() -> OpenAI:
    """Create an OpenAI client for DashScope with an explicit timeout (default 60 seconds).

    The official openai SDK defaults to 600 seconds when timeout is unspecified, and with
    automatic retries one item can hang for nearly 30 minutes in the worst case, so set it
    shorter explicitly.
    """
    return OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=DASHSCOPE_BASE_URL,
        timeout=TIMEOUT_SECONDS,
    )


def find_phase4_path(date_dir: Path) -> str:
    hits = sorted(glob.glob(str(date_dir / "phase4_comments_*.json")))
    if not hits:
        print(f"ERROR: {date_dir}/phase4_comments_*.json was not found.", file=sys.stderr)
        print("Check that the date argument is correct and that the Gemini flow "
              "(phase2 -> phase3 -> phase7 -> phase4) has already run for that date.", file=sys.stderr)
        sys.exit(1)
    return hits[-1]


def build_prompt(meta: dict, comments: list) -> str:
    """Build the analysis prompt from video metadata + a comment list (same logic as Gemini version).

    meta['countries'] uses the many-to-many approach (b), listing all countries where the video appeared.
    Because one video can appear in search results for multiple countries, list multiple countries and languages.

    The is_soccer_related judgment should mainly be based on title and description so that videos with
    zero comments or failed comment retrieval can still be judged. Comments may be used as supporting evidence
    if available.
    """
    lines = []
    for c in comments:
        author = c.get("author_display_name", "")
        likes = c.get("like_count", 0)
        text = (c.get("text_original") or "").replace("\n", " ").strip()
        lines.append(f"[likes={likes}] {author}: {text}")
    comments_block = "\n".join(lines) if lines else "(No comments, or comments could not be retrieved)"

    countries = meta.get("countries", []) or []
    country_names_ja = [c.get("country_name_ja", "") for c in countries if c.get("country_name_ja")]
    primary_langs = sorted({c.get("primary_lang", "") for c in countries if c.get("primary_lang")})
    countries_str = ", ".join(country_names_ja) if country_names_ja else "Unknown"
    langs_str = ", ".join(primary_langs) if primary_langs else "Unknown"

    description = (meta.get("description") or "").strip()
    description_block = description[:500] if description else "(No description)"

    return (
        f"# Video information\n"
        f"Countries where this video is trending (may include multiple countries): {countries_str}\n"
        f"Primary languages of those countries (may include multiple languages): {langs_str}\n"
        f"Title: {meta.get('title')}\n"
        f"Description (first 500 characters): {description_block}\n\n"
        f"# Comments ({len(comments)} comments, sorted by likes descending)\n"
        f"{comments_block}\n\n"
        f"# Instructions\n"
        f"Analyze the video information and comments above and generate the following:\n"
        f"0. is_soccer_related: Judge true/false for whether this video is related to soccer "
        f"(including the FIFA World Cup). Use the title and description as the main evidence, "
        f"and use comments as supporting evidence if available. Even if the search keyword contains "
        f"'World Cup', set false when the video is about another sport's World Cup, such as cricket, "
        f"basketball, or volleyball (example: ICC Women's T20 World Cup is cricket, so false). "
        f"If you are unsure or information is insufficient, choose true to avoid over-exclusion. "
        f"Write relevance_reason as about one sentence in Japanese explaining the judgment.\n"
        f"For items 1 through 4 below, fill the fields for format consistency even when "
        f"is_soccer_related is false. If there are no comments, sentiment may be "
        f"positive=0, negative=0, neutral=100, and the lists may be empty arrays.\n"
        f"1. sentiment: Positive/negative/neutral ratios (%), totaling roughly 100.\n"
        f"2. positive_themes / negative_themes: Major topics with theme_ja (Japanese), "
        f"theme_en (English), and mention_count (approximate number of comments that mention it). "
        f"Maximum 5 items each.\n"
        f"3. quotable_comments: Up to {MAX_QUOTES} comments that are impressive, highly liked, and good for article quotes. "
        f"Include original (verbatim), translated_ja (Japanese translation), translated_en (English translation), "
        f"author (poster name), likes (input likes value), and original_language (language code of the original).\n"
        f"4. mentioned_teams: Up to {MAX_TEAMS} national teams mentioned in comments. "
        f"Because team will be aggregated across languages downstream, always normalize it to an English national team name/country name "
        f"(examples: 'Argentina', 'Brazil', 'Japan', 'Morocco'; use national team names, not local-language names, flags, or club names). "
        f"For sentiment, use one of 'positive'/'neutral'/'negative' for the overall tone toward that team. "
        f"mention_count is the approximate number of comments that appear to mention that team. "
        f"If no national teams are mentioned, use an empty array.\n"
        f"If there are few comments or the analysis is difficult, return the best possible result."
    )


def _is_retryable_transient_error(msg: str) -> bool:
    """Determine whether the error is likely transient and worth retrying, such as 429, timeout, or connection error.

    On timeout, the official openai SDK raises openai.APITimeoutError and the message contains text such as
    "Request timed out." (the text observed in the actual hang). Connection errors such as httpx.ConnectError
    are also likely to be temporary, so they are treated as retryable as well.
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
    """Determine whether this is a content moderation rejection by DashScope.

    This is a heuristic based on observed error text (for example,
    "Input data may contain inappropriate content" / code=DataInspectionFailed). It is not a strict
    classifier, but unlike 429, retrying the same input will not change the result, which matters
    operationally. Distinguish it to avoid wasted retries.
    """
    lowered = msg.lower()
    return (
        "inappropriate content" in lowered
        or "datainspectionfailed" in lowered
        or "data_inspection_failed" in lowered
        or "content_filter" in lowered
    )


def _call_json(client: OpenAI, model: str, system: str, user: str):
    """Thin wrapper around chat.completions.create with response_format=json_object.

    enable_thinking=False is explicitly passed via extra_body. qwen-plus/qwen3-max and similar models
    default to thinking disabled, but if a hybrid thinking model such as qwen3.7-plus (thinking enabled
    by default) is specified in QWEN_CHAT_MODEL, leaving thinking enabled makes it incompatible with
    response_format=json_object and causes a 400 error (as stated in Alibaba's official documentation).
    Because enable_thinking is not a standard OpenAI parameter, it must be passed through extra_body when
    using the OpenAI SDK (also as described in Alibaba's official documentation). For models where thinking
    is disabled by default, this is harmless; it only explicitly sets False.
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
    """Ask a lightweight model to repair JSON syntax errors (Alibaba's recommended pattern).

    FIX_MODEL (default qwen-flash) has thinking disabled by default, but as a safeguard when it is
    replaced with another model via environment variable, explicitly set it to False here as well.
    """
    resp = client.chat.completions.create(
        model=FIX_MODEL,
        messages=[
            {"role": "system", "content": "You are a JSON formatting expert. "
                                            "Repair the broken JSON string provided by the user into valid JSON. "
                                            "Output only a single JSON object, with no explanatory text or code block syntax."},
            {"role": "user", "content": broken_text},
        ],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content


def analyze_with_retry(client: OpenAI, prompt: str,
                        max_rate_retries: int = MAX_RATE_LIMIT_RETRIES,
                        max_schema_retries: int = MAX_SCHEMA_RETRIES):
    """Analyze one video. Use exponential backoff for 429s and regenerate on schema mismatch.

    Return value: (CommentAnalysis or None, reason or None)
      On success: (CommentAnalysis, None).
      On failure: (None, reason), where reason is one of:
        - "content_moderation"     : Content moderation rejection by DashScope
                                      (retrying the same input is meaningless, so give up immediately)
        - "rate_limit_exhausted"   : 429/timeout/connection error did not resolve after the retry limit
        - "schema_retry_exhausted" : JSON syntax/schema mismatch did not resolve after the retry limit
        - "api_error:<details>"    : Any other API error

    429/timeouts/connection errors (communication issues) and schema retries (model output quality issues)
    have different characteristics, so they use separate counters.
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
                    print(f"  WARNING: Content moderation rejected the request: {msg[:160]}", file=sys.stderr)
                    return None, "content_moderation"
                if _is_retryable_transient_error(msg) and rate_attempt < max_rate_retries - 1:
                    wait = 5 * (2 ** rate_attempt)
                    print(f"  Possible transient error ({msg[:60]}). Waiting {wait} seconds before retrying... "
                          f"({rate_attempt + 1}/{max_rate_retries})")
                    time.sleep(wait)
                elif _is_retryable_transient_error(msg):
                    print(f"  ERROR: Analysis failed (retry limit reached): {msg[:160]}", file=sys.stderr)
                    return None, "rate_limit_exhausted"
                else:
                    print(f"  ERROR: Analysis failed: {msg[:160]}", file=sys.stderr)
                    return None, f"api_error:{msg[:100]}"

        if raw is None:
            return None, "rate_limit_exhausted"

        # Parse JSON syntax.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("  WARNING: JSON syntax error. Trying to repair with the repair model...", file=sys.stderr)
            try:
                fixed = _repair_json(client, raw)
                data = json.loads(fixed)
            except Exception as e:  # noqa: BLE001
                schema_attempt += 1
                print(f"  WARNING: JSON repair also failed (attempt {schema_attempt}/{max_schema_retries}): "
                      f"{str(e)[:160]}", file=sys.stderr)
                continue

        # Validate against our Pydantic model (required because DashScope does not enforce the schema).
        try:
            return CommentAnalysis(**data), None
        except ValidationError as e:
            schema_attempt += 1
            print(f"  WARNING: Schema mismatch (attempt {schema_attempt}/{max_schema_retries}): "
                  f"{str(e)[:300]}", file=sys.stderr)
            continue

    print("  ERROR: Reached the schema validation retry limit. Skipping.", file=sys.stderr)
    return None, "schema_retry_exhausted"


def resolve_dirs(date_str: str):
    """Resolve the input directory (Gemini flow side, read-only) and output directory
    (qwen_soccer side) from a date string."""
    input_dir = (SCRIPT_DIR / ".." / ".." / "soccer" / "data" / date_str).resolve()
    output_dir = (SCRIPT_DIR / "data" / date_str).resolve()
    return input_dir, output_dir


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load an existing checkpoint, or return an empty state if it does not exist."""
    if not checkpoint_path.exists():
        return {"analyses": {}, "skipped": []}
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("analyses", {})
        data.setdefault("skipped", [])
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to read checkpoint; discarding it and starting over: {e}",
              file=sys.stderr)
        return {"analyses": {}, "skipped": []}


def save_checkpoint(checkpoint_path: Path, state: dict) -> None:
    """Save a checkpoint (atomic write via temporary file -> rename)."""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    tmp_path.replace(checkpoint_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase5 Stage 1: Comment sentiment analysis (Qwen version)")
    parser.add_argument("date", nargs="?", default=None,
                         help="Target date (YYYYMMDD). Defaults to today if omitted")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 1

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    input_dir, output_dir = resolve_dirs(date_str)

    path = find_phase4_path(input_dir)
    print(f"Target date: {date_str}")
    print(f"Input file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_video = data.get("comments_by_video", {})
    if not by_video:
        print("ERROR: comments_by_video is empty.", file=sys.stderr)
        return 1
    print(f"Target videos: {len(by_video)} / workers: {MAX_WORKERS} / timeout: {TIMEOUT_SECONDS} seconds")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    state = load_checkpoint(checkpoint_path)
    analyses = state["analyses"]
    skipped = state["skipped"]
    already_done = set(analyses.keys()) | {s["video_id"] for s in skipped}
    if already_done:
        print(f"  Checkpoint detected: skipping {len(already_done)} already processed items.")

    client = make_client()
    lock = threading.Lock()
    items = list(by_video.items())
    total = len(items)

    def process_one(idx: int, vid: str, meta: dict):
        """Analyze one video and write the result back to the checkpoint (called from multiple threads).

        Catch general Exceptions here and always reflect them in the checkpoint, so that one unexpected
        bug or abnormal case does not cause other completed analysis results to be lost. Real Ctrl+C
        (KeyboardInterrupt) and SystemExit are not subclasses of Exception, so they are not caught here
        and can correctly interrupt the whole process.
        """
        try:
            if not (meta.get("title") or "").strip():
                with lock:
                    skipped.append({"video_id": vid, "reason": "no_title_no_judgeable_info"})
                    save_checkpoint(checkpoint_path, state)
                print(f"[{idx}/{total}] {vid} skipped (no title and no judgeable information)")
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

            status = "OK" if result is not None else f"failed({fail_reason})"
            print(f"[{idx}/{total}] {vid} complete: {status}")
        except Exception as e:  # noqa: BLE001
            # Stop this item here even on unexpected bugs, and do not drag down other video results.
            with lock:
                skipped.append({"video_id": vid, "reason": f"unexpected_error:{str(e)[:150]}"})
                save_checkpoint(checkpoint_path, state)
            print(f"[{idx}/{total}] {vid} complete: unexpected error ({str(e)[:150]})", file=sys.stderr)

    targets = [(i, vid, meta) for i, (vid, meta) in enumerate(items, 1) if vid not in already_done]
    print(f"  Videos to process in this run: {len(targets)} (excluding {len(already_done)} already skipped/processed items)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_one, idx, vid, meta) for idx, vid, meta in targets]
        try:
            for fut in as_completed(futures):
                fut.result()  # Re-raise exceptions here if any occurred.
        except KeyboardInterrupt:
            # The default shutdown(wait=True) continues processing queued tasks until the queue is empty,
            # making Ctrl+C slow to respond. Use cancel_futures=True to cancel only tasks that have not
            # started, while still waiting for running tasks to finish so checkpoint updates are preserved.
            print("\nInterrupt detected. Waiting for running tasks to finish and canceling tasks that have not started...",
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

    print(f"\nAnalysis complete: {len(analyses)} succeeded / {len(skipped)} skipped")
    if skipped:
        from collections import Counter
        reason_counts = Counter(s["reason"] for s in skipped)
        print("  Breakdown:")
        for reason, cnt in reason_counts.most_common():
            print(f"    {reason}: {cnt}")
    print(f"Saved: {out_path}")

    # Delete the checkpoint after successfully writing all results.
    # Leaving it behind could accidentally carry over previous results when the same date is rerun later.
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError:
        pass

    print("Next, load this into videos.comment_analysis with stage 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
