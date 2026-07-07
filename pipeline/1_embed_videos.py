#!/usr/bin/env python3
"""
Stage 1: Embed video metadata and save it locally [Qwen Cloud migration version]

For all videos in phase7, build embedding text, generate 768-dimensional embeddings
with Qwen text-embedding-v4 (DashScope OpenAI-compatible API, normalized), and save
the original metadata + embedding to videos_embedded_<timestamp>.json.

[First half of the two-stage flow] This is the only script that calls the
DashScope API. Re-running the MongoDB load step (stage 2: load_to_mongo.py) will
not consume API usage unless this script is run again.

[Differences from the Gemini version]
  - Model: gemini-embedding-001 -> text-embedding-v4 (DashScope)
  - The per-request item limit changes from Gemini (roughly 20 items) to Qwen
    text-embedding-v4's documented "batch size 10", so CHUNK_SIZE is changed to 10.
  - The asymmetric-pair setting equivalent to task_type(RETRIEVAL_DOCUMENT) is not
    available in the OpenAI-compatible API, so it is not used (future support is
    possible by using text_type="document" in the DashScope-specific SDK).
  - If you want to separate the output DB, this script does not need to change.
    Set the environment variable SOCCER_DB_NAME=qwen-soccertube when running the
    next stage, 2_load_to_mongo.py / 4_load_comment_analysis.py.

[Date-based I/O]
  Input: ../../soccer/data/<date>/phase7_with_buzz_score_*.json (Gemini flow side, read-only)
  Output: ./data/<date>/videos_embedded_<timestamp>.json
  If <date> is omitted, today's date (YYYYMMDD) is used. For backfills, pass it as an argument
  (example: python 1_embed_videos.py 20260704).

Prerequisites:
    pip install openai numpy
    export DASHSCOPE_API_KEY='...'

Run:
    python 1_embed_videos.py [YYYYMMDD]
    (when omitted, it reads ../../soccer/data/<today>/ using today's date)
"""

import os
import sys
import glob
import json
import time
import argparse
from datetime import datetime

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

EMBED_MODEL = os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = 768            # Cannot be changed later; must match the existing Vector Search index.
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
DESC_MAX_CHARS = 500       # Use the first 500 chars of description to save input tokens; the topic usually appears near the beginning.
CHUNK_SIZE = 10            # Batch limit for text-embedding-v4 (official docs: 10 items/request).
SLEEP_BETWEEN_CHUNKS = 1.0 # Short wait between chunks (seconds).

from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')


def resolve_dirs(date_str: str):
    """Resolve the input directory (Gemini flow side, read-only) and output directory
    (qwen_soccer side) from a date string."""
    input_dir = (SCRIPT_DIR / ".." / "soccer" / "data" / date_str).resolve()
    output_dir = (SCRIPT_DIR / "data" / date_str).resolve()
    return input_dir, output_dir


def find_phase7_path(date_dir: Path) -> str:
    hits = sorted(glob.glob(str(date_dir / "phase7_with_buzz_score_*.json")))
    if not hits:
        print(f"ERROR: {date_dir}/phase7_with_buzz_score_*.json was not found.", file=sys.stderr)
        print("Check that the date argument is correct and that the Gemini flow "
              "(phase2 -> phase3 -> phase7) has already run for that date.", file=sys.stderr)
        sys.exit(1)
    return hits[-1]


def build_embed_text(v: dict) -> str:
    """Embedding target: title + description (first 500 chars) + country names (en, multiple).
    Tags are often empty, so they are not used.

    Since phase3 now has a many-to-many countries array (+reach), list the
    country_name_en values for all countries that appeared, comma-separated
    (rank ascending = original appearance order).
    """
    title = v.get("title", "") or ""
    desc = (v.get("description") or "")[:DESC_MAX_CHARS]
    countries = v.get("countries", []) or []
    country_names = [c.get("country_name_en", "") for c in countries if c.get("country_name_en")]
    country = ", ".join(country_names)
    return f"{title}\n{desc}\n{country}".strip()


def normalize(vec) -> list:
    """Manually L2-normalize because 768-dimensional vectors are returned unnormalized (required)."""
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Zero vector (the embedding target text may be empty)")
    return (arr / norm).tolist()


def embed_chunk_with_retry(client: OpenAI, texts: list, max_retries: int = 4) -> list:
    """Embed one chunk (up to 10 items). Retry with exponential backoff on rate limits (429)."""
    for attempt in range(max_retries):
        try:
            result = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                dimensions=EMBED_DIM,
            )
            # DashScope is expected to guarantee index order as in the OpenAI-compatible spec,
            # but sort by index before extracting just in case.
            ordered = sorted(result.data, key=lambda e: e.index)
            return [normalize(e.embedding) for e in ordered]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "Throttling" in msg) and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)  # 5,10,20,40 seconds
                print(f"  Possible rate limit. Waiting {wait} seconds before retrying... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Reached the embedding retry limit.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: Embed video metadata (Qwen version)")
    parser.add_argument("date", nargs="?", default=None,
                         help="Target date (YYYYMMDD). Defaults to today if omitted")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 1

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    input_dir, output_dir = resolve_dirs(date_str)

    path = find_phase7_path(input_dir)
    print(f"Target date: {date_str}")
    print(f"Input file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos", [])
    if not videos:
        print("ERROR: videos is empty.", file=sys.stderr)
        return 1
    print(f"Target videos: {len(videos)}")

    # Build embedding text for all items (detect and warn about empty text).
    texts = []
    for v in videos:
        t = build_embed_text(v)
        if not t:
            print(f"  WARNING: embedding text is empty for video_id={v.get('video_id')}.", file=sys.stderr)
        texts.append(t)

    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=DASHSCOPE_BASE_URL,
    )

    # Split into chunks and embed.
    all_vecs = []
    n_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\nEmbedding in {n_chunks} chunks of {CHUNK_SIZE} items each (768 dimensions, normalized, {EMBED_MODEL}).")
    for i in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[i:i + CHUNK_SIZE]
        idx = i // CHUNK_SIZE + 1
        print(f"  Embedding chunk {idx}/{n_chunks} ({len(chunk)} items)...")
        vecs = embed_chunk_with_retry(client, chunk)
        if len(vecs) != len(chunk):
            print(f"ERROR: Number of returned vectors ({len(vecs)}) does not match number of inputs ({len(chunk)}).", file=sys.stderr)
            return 1
        all_vecs.extend(vecs)
        if idx < n_chunks:
            time.sleep(SLEEP_BETWEEN_CHUNKS)

    assert len(all_vecs) == len(videos), "Total vector count does not match video count"
    print(f"\nEmbedding complete: {len(all_vecs)} items / {len(all_vecs[0])} dimensions each")

    # Merge original metadata + embedding and save (stage 2 does not reread phase7).
    out_videos = []
    for v, vec, t in zip(videos, all_vecs, texts):
        rec = dict(v)               # Keep the entire original phase7 record.
        rec["embedding"] = vec
        rec["_embed_text"] = t      # For debugging (what was embedded). Stage 2 can ignore it.
        out_videos.append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"videos_embedded_{ts}.json"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_file": os.path.basename(path),
        "embed_model": EMBED_MODEL,
        "embed_provider": "dashscope",
        "embed_dim": EMBED_DIM,
        "normalized": True,
        "total": len(out_videos),
        "videos": out_videos,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print("Next, load this file into MongoDB with stage 2 (load_to_mongo.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
