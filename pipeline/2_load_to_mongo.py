#!/usr/bin/env python3
"""
ステージ2: embed済みデータをMongoDBに投入し、Vector Searchインデックスを作成 【Qwen Cloud移植版】

ステージ1(1_embed_videos.py)が出力した videos_embedded_<timestamp>.json を読み、
qwen-soccertube DBの videos コレクションに upsert（video_idキー）で投入する。
投入後、Vector Searchインデックス(video_semantic_index)を作成し queryable
になるまで待つ。

【2段構成の後半】Gemini APIは一切叩かない。何度実行しても無料枠を消費しない。
データは video_id で upsert するので冪等（再実行で重複しない）。

【日付ベースのI/O】
  入力: ./data/<date>/videos_embedded_*.json （自分たちの1_embed_videos.pyの
  出力のみを見る。../soccer/側は参照しない）
  <date> は省略時は今日(YYYYMMDD)。バックフィル時は引数で指定
  （例: python 2_load_to_mongo.py 20260704）。

事前準備:
    pip install "pymongo[srv]"
    export MONGODB_URI='mongodb+srv://<user>:<password>@xxxx.mongodb.net/'
    export SOCCER_DB_NAME='qwen-soccertube'   # .envに設定済みならこのままでOK

実行:
    python 2_load_to_mongo.py [YYYYMMDD]
    （省略時は今日の日付で ./data/<今日>/ を読みに行く）
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

COLL_NAME = "videos"                  # 本番コレクション
INDEX_NAME = "video_semantic_index"
EMBED_DIM = 768
DROP_FIELDS = ("_embed_text",)        # 本番に入れないデバッグ用フィールド

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')

# Qwen移植版のデフォルトDBはqwen-soccertube（Gemini本番soccertubeとは別DB）
DB_NAME = os.environ.get("SOCCER_DB_NAME", "qwen-soccertube")
COLL_NAME = os.environ.get("SOCCER_COLL_NAME", "videos")
INDEX_NAME = os.environ.get("SOCCER_INDEX_NAME", "video_semantic_index")


def resolve_dirs(date_str: str) -> Path:
    """日付文字列から入出力ディレクトリ(qwen_soccer側、自分たちの出力のみ)を決める。"""
    return (SCRIPT_DIR / "data" / date_str).resolve()


def find_input_path(date_dir: Path) -> str:
    hits = sorted(glob.glob(str(date_dir / "videos_embedded_*.json")))
    if not hits:
        print(f"ERROR: {date_dir}/videos_embedded_*.json が見つかりません。", file=sys.stderr)
        print("先に 1_embed_videos.py をその日付で実行してください。", file=sys.stderr)
        sys.exit(1)
    return hits[-1]


def parse_dt(s):
    """ 'YYYY-MM-DDTHH:MM:SSZ' → tz-aware datetime（filterで日付範囲を効かせるため）。"""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def to_doc(v: dict) -> dict:
    """投入用ドキュメントへ整形。published_atをdatetime化、デバッグ用フィールド除去。"""
    doc = {k: val for k, val in v.items() if k not in DROP_FIELDS}
    if "published_at" in doc:
        doc["published_at"] = parse_dt(doc.get("published_at"))
    doc["ingested_at"] = datetime.now(timezone.utc)
    return doc


def ensure_index(coll) -> bool:
    """Vector Searchインデックスが無ければ作成し queryable まで待つ。
    作成不可ならUI貼り付け用JSONを出力。"""
    existing = list(coll.list_search_indexes(INDEX_NAME))
    if existing:
        print(f"  インデックス '{INDEX_NAME}' は既に存在します。")
    else:
        definition = {
            "fields": [
                {"type": "vector", "path": "embedding",
                 "numDimensions": EMBED_DIM, "similarity": "cosine"},
                # 注意: countries はオブジェクトの配列であり、Atlas Vector Searchの
                # filter type インデックスはオブジェクト配列内のフィールドを直接
                # インデックスできない（vectorSearchタイプ索引の制約）。そのため
                # フィルタ専用に country_codes（単純な文字列配列, phase3で複製生成）を使う。
                {"type": "filter", "path": "country_codes"},
                {"type": "filter", "path": "published_at"},
                {"type": "filter", "path": "is_buzz"},
                {"type": "filter", "path": "category"},
            ]
        }
        model = SearchIndexModel(definition=definition, name=INDEX_NAME, type="vectorSearch")
        try:
            coll.create_search_indexes([model])
            print(f"  インデックス '{INDEX_NAME}' の作成を要求しました（非同期）。")
        except OperationFailure as e:
            print(f"\nERROR: コードからのインデックス作成に失敗: {e}", file=sys.stderr)
            print("→ Atlas UI の Atlas Search > Create Index > JSON Editor で以下を貼ってください:",
                  file=sys.stderr)
            print(json.dumps({"name": INDEX_NAME, "type": "vectorSearch", "definition": definition},
                             ensure_ascii=False, indent=2), file=sys.stderr)
            return False

    print("  queryable になるまで待機中（数分かかることがあります）...")
    deadline = time.time() + 300
    while time.time() < deadline:
        info = list(coll.list_search_indexes(INDEX_NAME))
        if info and info[0].get("queryable"):
            print("  インデックス queryable=True。")
            return True
        time.sleep(5)
    print("ERROR: インデックスが時間内に queryable になりませんでした。", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="ステージ2: MongoDB投入(Qwen版)")
    parser.add_argument("date", nargs="?", default=None,
                         help="対象日付(YYYYMMDD)。省略時は今日")
    args = parser.parse_args()

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("ERROR: MONGODB_URI が未設定です。", file=sys.stderr)
        return 1

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    date_dir = resolve_dirs(date_str)

    path = find_input_path(date_dir)
    print(f"対象日付: {date_str} / DB: {DB_NAME}")
    print(f"入力ファイル: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos", [])
    if not videos:
        print("ERROR: videos が空です。", file=sys.stderr)
        return 1

    # 念のため次元数を検証（ステージ1と食い違っていないか）
    dim = len(videos[0].get("embedding", []))
    if dim != EMBED_DIM:
        print(f"ERROR: embedding次元が{dim}。期待値{EMBED_DIM}と不一致。", file=sys.stderr)
        return 1
    print(f"投入対象: {len(videos)}件 / {dim}次元")

    # MongoDB接続
    try:
        mclient = MongoClient(uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
        mclient.admin.command("ping")
    except (ConnectionFailure, OperationFailure, ConfigurationError) as e:
        print(f"ERROR: MongoDB接続失敗: {e}", file=sys.stderr)
        return 1
    coll = mclient[DB_NAME][COLL_NAME]

    # video_id で upsert（冪等）。bulk_writeでまとめて投入
    print("\n[1/2] videosコレクションへ投入中（video_idでupsert）...")
    ops = []
    for v in videos:
        if not v.get("video_id"):
            print("  WARNING: video_idが無いレコードをスキップ", file=sys.stderr)
            continue
        ops.append(ReplaceOne({"video_id": v["video_id"]}, to_doc(v), upsert=True))
    try:
        res = coll.bulk_write(ops, ordered=False)
    except BulkWriteError as e:
        print(f"ERROR: 一括投入で問題: {e.details}", file=sys.stderr)
        return 1
    print(f"  OK — upserted={res.upserted_count}, modified={res.modified_count}, "
          f"コレクション総数={coll.count_documents({})}")

    # インデックス作成
    print("\n[2/2] Vector Searchインデックスの確認/作成...")
    if not ensure_index(coll):
        return 1

    mclient.close()
    print(f"\nステージ2完了。videosコレクションにembed済み{len(videos)}件が投入され、検索可能です。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

