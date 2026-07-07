#!/usr/bin/env python3
"""
Stage 2: Load embedded data into MongoDB and create the Vector Search index [Qwen Cloud migration version]

Read videos_embedded_<timestamp>.json produced by stage 1 (1_embed_videos.py), and
upsert it into the videos collection in the qwen-soccertube DB using video_id as
the key. After loading, create the Vector Search index (video_semantic_index) and
wait until it becomes queryable.

[Second half of the two-stage flow] This does not call the Gemini API at all.
It does not consume the free quota no matter how many times it is run. Data is
upserted by video_id, so the operation is idempotent (reruns do not create duplicates).

[Date-based I/O]
  Input: ./data/<date>/videos_embedded_*.json (only outputs from our own
  1_embed_videos.py are used; ../soccer/ is not referenced)
  If <date> is omitted, today's date (YYYYMMDD) is used. For backfills, pass it as an argument
  (example: python 2_load_to_mongo.py 20260704).

Prerequisites:
    pip install "pymongo[srv]"
    export MONGODB_URI='mongodb+srv://<user>:<password>@xxxx.mongodb.net/'
    export SOCCER_DB_NAME='qwen-soccertube'   # Keep this as-is if it is already set in .env

Run:
    python 2_load_to_mongo.py [YYYYMMDD]
    (when omitted, it reads ./data/<today>/ using today's date)
"""

import os
import sys
import glob
import json
import time
import argparse
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel
from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError, BulkWriteError
from pymongo import ReplaceOne
from dotenv import load_dotenv
from pathlib import Path

COLL_NAME = "videos"                  # Production collection.
INDEX_NAME = "video_semantic_index"
EMBED_DIM = 768
DROP_FIELDS = ("_embed_text",)        # Debug fields that should not be inserted into production.

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')

# The default DB for the Qwen migration version is qwen-soccertube (separate from Gemini production soccertube).
DB_NAME = os.environ.get("SOCCER_DB_NAME", "qwen-soccertube")
COLL_NAME = os.environ.get("SOCCER_COLL_NAME", "videos")
INDEX_NAME = os.environ.get("SOCCER_INDEX_NAME", "video_semantic_index")


def resolve_dirs(date_str: str) -> Path:
    """Resolve the I/O directory (qwen_soccer side, only our own outputs) from a date string."""
    return (SCRIPT_DIR / "data" / date_str).resolve()


def find_input_path(date_dir: Path) -> str:
    hits = sorted(glob.glob(str(date_dir / "videos_embedded_*.json")))
    if not hits:
        print(f"ERROR: {date_dir}/videos_embedded_*.json was not found.", file=sys.stderr)
        print("Run 1_embed_videos.py for that date first.", file=sys.stderr)
        sys.exit(1)
    return hits[-1]


def parse_dt(s):
    """Convert 'YYYY-MM-DDTHH:MM:SSZ' to a tz-aware datetime (so date-range filters work)."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def to_doc(v: dict) -> dict:
    """Format a record for insertion: convert published_at to datetime and remove debug fields."""
    doc = {k: val for k, val in v.items() if k not in DROP_FIELDS}
    if "published_at" in doc:
        doc["published_at"] = parse_dt(doc.get("published_at"))
    doc["ingested_at"] = datetime.now(timezone.utc)
    return doc


def ensure_index(coll) -> bool:
    """Create the Vector Search index if it does not exist and wait until it is queryable.
    If creation is not possible, print JSON that can be pasted into the UI."""
    existing = list(coll.list_search_indexes(INDEX_NAME))
    if existing:
        print(f"  Index '{INDEX_NAME}' already exists.")
    else:
        definition = {
            "fields": [
                {"type": "vector", "path": "embedding",
                 "numDimensions": EMBED_DIM, "similarity": "cosine"},
                # Note: countries is an array of objects, and Atlas Vector Search filter-type
                # indexes cannot directly index fields inside arrays of objects (a constraint of
                # vectorSearch-type indexes). Therefore, use country_codes, a simple string array
                # generated as a duplicate in phase3, specifically for filtering.
                {"type": "filter", "path": "country_codes"},
                {"type": "filter", "path": "published_at"},
                {"type": "filter", "path": "is_buzz"},
                {"type": "filter", "path": "category"},
            ]
        }
        model = SearchIndexModel(definition=definition, name=INDEX_NAME, type="vectorSearch")
        try:
            coll.create_search_indexes([model])
            print(f"  Requested creation of index '{INDEX_NAME}' (asynchronous).")
        except OperationFailure as e:
            print(f"\nERROR: Failed to create the index from code: {e}", file=sys.stderr)
            print("-> Paste the following into Atlas UI > Atlas Search > Create Index > JSON Editor:",
                  file=sys.stderr)
            print(json.dumps({"name": INDEX_NAME, "type": "vectorSearch", "definition": definition},
                             ensure_ascii=False, indent=2), file=sys.stderr)
            return False

    print("  Waiting until the index becomes queryable (this may take several minutes)...")
    deadline = time.time() + 300
    while time.time() < deadline:
        info = list(coll.list_search_indexes(INDEX_NAME))
        if info and info[0].get("queryable"):
            print("  Index queryable=True.")
            return True
        time.sleep(5)
    print("ERROR: The index did not become queryable within the time limit.", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2: Load into MongoDB (Qwen version)")
    parser.add_argument("date", nargs="?", default=None,
                         help="Target date (YYYYMMDD). Defaults to today if omitted")
    args = parser.parse_args()

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("ERROR: MONGODB_URI is not set.", file=sys.stderr)
        return 1

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    date_dir = resolve_dirs(date_str)

    path = find_input_path(date_dir)
    print(f"Target date: {date_str} / DB: {DB_NAME}")
    print(f"Input file: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos", [])
    if not videos:
        print("ERROR: videos is empty.", file=sys.stderr)
        return 1

    # Verify the dimension count just in case (check for mismatch with stage 1).
    dim = len(videos[0].get("embedding", []))
    if dim != EMBED_DIM:
        print(f"ERROR: embedding dimension is {dim}; expected {EMBED_DIM}.", file=sys.stderr)
        return 1
    print(f"Load target: {len(videos)} items / {dim} dimensions")

    # Connect to MongoDB.
    try:
        mclient = MongoClient(uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
        mclient.admin.command("ping")
    except (ConnectionFailure, OperationFailure, ConfigurationError) as e:
        print(f"ERROR: MongoDB connection failed: {e}", file=sys.stderr)
        return 1
    coll = mclient[DB_NAME][COLL_NAME]

    # Upsert by video_id (idempotent). Load in a batch with bulk_write.
    print("\n[1/2] Loading into the videos collection (upsert by video_id)...")
    ops = []
    for v in videos:
        if not v.get("video_id"):
            print("  WARNING: Skipping a record without video_id.", file=sys.stderr)
            continue
        ops.append(ReplaceOne({"video_id": v["video_id"]}, to_doc(v), upsert=True))
    try:
        res = coll.bulk_write(ops, ordered=False)
    except BulkWriteError as e:
        print(f"ERROR: Problem during bulk insert: {e.details}", file=sys.stderr)
        return 1
    print(f"  OK — upserted={res.upserted_count}, modified={res.modified_count}, "
          f"collection total={coll.count_documents({})}")

    # Create the index.
    print("\n[2/2] Checking/creating the Vector Search index...")
    if not ensure_index(coll):
        return 1

    mclient.close()
    print(f"\nStage 2 complete. {len(videos)} embedded items have been loaded into the videos collection and are searchable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
