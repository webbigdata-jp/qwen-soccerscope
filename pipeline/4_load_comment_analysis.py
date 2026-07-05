#!/usr/bin/env python3
"""
Phase5 ステージ2: コメント分析結果を videos.comment_analysis に投入 【Qwen Cloud移植版】

ステージ1(3_analyze_comments.py)が出力した comment_analysis_<timestamp>.json を
読み、qwen-soccertube DBの videos コレクションの各ドキュメントに
comment_analysis フィールドを $set で追加する。embedding・メタデータなど
既存フィールドは一切触らない。

is_soccer_related が false と判定された動画は、サッカーと無関係な動画
（例: search.list の "World Cup" 系クエリに誤ってヒットしたクリケット/
バスケットボール/バレーボール等の動画）とみなし、$setではなく
videosコレクションから完全に削除する。is_soccer_related フィールドが
存在しない（3_analyze_comments.py の旧バージョンで生成された）分析結果は
従来通り扱い、削除しない（後方互換）。

【2段構成の後半】Gemini APIは叩かない。video_idでマッチするので冪等。

【日付ベースのI/O】
  入力: ./data/<date>/comment_analysis_*.json （自分たちの3_analyze_comments.py
  の出力のみを見る。../../soccer/側は参照しない）
  <date> は省略時は今日(YYYYMMDD)。バックフィル時は引数で指定
  （例: python 4_load_comment_analysis.py 20260704）。

【Gemini版からの変更点】
  Gemini版にあった archive_run_files()（../../soccer/data/ 配下のGeminiフロー
  自身の中間ファイルを動かす処理）は撤去した。データパイプライン(Geminiフロー)
  には触れない方針のため、qwen_soccer側は自分が生成したファイルを最初から
  ./data/<date>/ に置くだけで、退避（移動）処理自体が不要になっている。

事前準備:
    pip install "pymongo[srv]"
    export MONGODB_URI='mongodb+srv://<user>:<password>@xxxx.mongodb.net/'
    export SOCCER_DB_NAME='qwen-soccertube'   # .envに設定済みならこのままでOK

実行:
    python 4_load_comment_analysis.py [YYYYMMDD]
    （省略時は今日の日付で ./data/<今日>/ を読みに行く）
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime, timezone

from pymongo import MongoClient, UpdateOne, DeleteOne
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError, BulkWriteError

from dotenv import load_dotenv
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')

# Qwen移植版のデフォルトDBはqwen-soccertube（Gemini本番soccertubeとは別DB）
DB_NAME = os.environ.get("SOCCER_DB_NAME", "qwen-soccertube")
COLL_NAME = os.environ.get("SOCCER_COLL_NAME", "videos")


def resolve_dirs(date_str: str) -> Path:
    """日付文字列から入出力ディレクトリ(qwen_soccer側、自分たちの出力のみ)を決める。"""
    return (SCRIPT_DIR / "data" / date_str).resolve()


def find_input_path(date_dir: Path) -> str:
    hits = sorted(glob.glob(str(date_dir / "comment_analysis_*.json")))
    if not hits:
        print(f"ERROR: {date_dir}/comment_analysis_*.json が見つかりません。", file=sys.stderr)
        print("先に 3_analyze_comments.py をその日付で実行してください。", file=sys.stderr)
        sys.exit(1)
    return hits[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase5 ステージ2: コメント分析結果投入(Qwen版)")
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
    analyses = data.get("analyses", {})
    if not analyses:
        print("ERROR: analyses が空です。", file=sys.stderr)
        return 1
    print(f"投入対象: {len(analyses)}件の分析結果")

    try:
        mclient = MongoClient(uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
        mclient.admin.command("ping")
    except (ConnectionFailure, OperationFailure, ConfigurationError) as e:
        print(f"ERROR: MongoDB接続失敗: {e}", file=sys.stderr)
        return 1
    coll = mclient[DB_NAME][COLL_NAME]

    # video_id で該当ドキュメントに comment_analysis を $set。
    # is_soccer_related == False の動画はvideosコレクションから削除する
    # （フィールドが存在しない旧フォーマットの分析結果は従来通りupdate扱い、
    #  後方互換のため削除しない）。
    to_update = {}
    to_delete_ids = []
    for vid, analysis in analyses.items():
        is_soccer_related = analysis.get("is_soccer_related")
        if is_soccer_related is False:
            to_delete_ids.append(vid)
        else:
            to_update[vid] = analysis

    print(f"\n内訳: 更新対象={len(to_update)}件 / "
          f"サッカー非関連のため削除対象={len(to_delete_ids)}件")
    if to_delete_ids:
        print("  削除対象 video_id: " + ", ".join(to_delete_ids))

    now = datetime.now(timezone.utc)

    if to_update:
        print("\nvideos.comment_analysis を更新中（$set, video_idマッチ）...")
        ops = [
            UpdateOne(
                {"video_id": vid},
                {"$set": {"comment_analysis": analysis, "last_analyzed": now}},
            )
            for vid, analysis in to_update.items()
        ]
        try:
            res = coll.bulk_write(ops, ordered=False)
        except BulkWriteError as e:
            print(f"ERROR: 一括更新で問題: {e.details}", file=sys.stderr)
            return 1

        matched = res.matched_count
        modified = res.modified_count
        print(f"  matched={matched}, modified={modified}")

        # video_idが videos に存在せずマッチしなかったものを洗い出す（通常は0のはず）
        if matched < len(to_update):
            existing_ids = set(coll.distinct("video_id"))
            unmatched = [vid for vid in to_update if vid not in existing_ids]
            print(f"  WARNING: videosに存在せずマッチしなかったvideo_id {len(unmatched)}件: {unmatched}",
                  file=sys.stderr)
    else:
        print("\n更新対象が0件のため $set はスキップします。")

    if to_delete_ids:
        print("\nサッカー非関連動画をvideosコレクションから削除中...")
        delete_ops = [DeleteOne({"video_id": vid}) for vid in to_delete_ids]
        try:
            del_res = coll.bulk_write(delete_ops, ordered=False)
        except BulkWriteError as e:
            print(f"ERROR: 一括削除で問題: {e.details}", file=sys.stderr)
            return 1
        print(f"  deleted_count={del_res.deleted_count}")
        if del_res.deleted_count < len(to_delete_ids):
            print(f"  WARNING: 削除対象{len(to_delete_ids)}件に対し実削除"
                  f"{del_res.deleted_count}件（既に存在しなかった可能性）", file=sys.stderr)

    # 検証: comment_analysis を持つドキュメント数
    with_analysis = coll.count_documents({"comment_analysis": {"$exists": True}})
    total = coll.count_documents({})
    print(f"\n検証: comment_analysis 保有 {with_analysis} / 全 {total} 件")

    mclient.close()
    print(f"\nステージ2完了。videosの{len(to_update)}件に感情分析結果を追加、"
          f"{len(to_delete_ids)}件をサッカー非関連として削除しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
