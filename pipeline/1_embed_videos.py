#!/usr/bin/env python3
"""
ステージ1: 動画メタデータをembedしてローカル保存 【Qwen Cloud移植版】

phase7のvideos全件について embedテキストを作り、Qwen text-embedding-v4 で
768次元embed（DashScope OpenAI互換API・正規化）を生成し、元メタデータ +
embedding を videos_embedded_<timestamp>.json に保存する。

【2段構成の前半】DashScope APIを叩くのはこのスクリプトだけ。
MongoDB投入(ステージ2 load_to_mongo.py)をやり直しても、ここを再実行しない限り
APIを消費しない。

【Gemini版との差分】
  - モデル: gemini-embedding-001 → text-embedding-v4 (DashScope)
  - 1リクエストあたりの件数上限が Gemini(20件目安) → Qwen text-embedding-v4は
    公式マニュアル記載で「バッチサイズ10」のため CHUNK_SIZE を 10 に変更。
  - task_type(RETRIEVAL_DOCUMENT)相当の非対称ペア指定はOpenAI互換APIに無いため
    未使用（DashScope独自SDK限定のtext_type="document"を使えば将来対応可）。
  - 出力先DBを分離する場合は、このスクリプト自体は変更不要。次段の
    2_load_to_mongo.py / 4_load_comment_analysis.py を実行する際に
    環境変数 SOCCER_DB_NAME=qwen-soccertube を設定すること。

【日付ベースのI/O】
  入力: ../../soccer/data/<date>/phase7_with_buzz_score_*.json （Geminiフロー側、読むだけ）
  出力: ./data/<date>/videos_embedded_<timestamp>.json
  <date> は省略時は今日(YYYYMMDD)。バックフィル時は引数で指定
  （例: python 1_embed_videos.py 20260704）。

事前準備:
    pip install openai numpy
    export DASHSCOPE_API_KEY='...'

実行:
    python 1_embed_videos.py [YYYYMMDD]
    （省略時は今日の日付で ../../soccer/data/<今日>/ を読みに行く）
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
EMBED_DIM = 768            # 後から変更不可。既存Vector Searchインデックスと一致させる
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
DESC_MAX_CHARS = 500       # descriptionは先頭500字（入力トークン節約・主題は冒頭に出る）
CHUNK_SIZE = 10            # text-embedding-v4 のバッチ上限(公式マニュアル: 10件/リクエスト)
SLEEP_BETWEEN_CHUNKS = 1.0 # チャンク間の軽い待機（秒）

from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / '.env')


def resolve_dirs(date_str: str):
    """日付文字列から入力ディレクトリ(Geminiフロー側, 読取専用)と
    出力ディレクトリ(qwen_soccer側)を決める。"""
    input_dir = (SCRIPT_DIR / ".." / "soccer" / "data" / date_str).resolve()
    output_dir = (SCRIPT_DIR / "data" / date_str).resolve()
    return input_dir, output_dir


def find_phase7_path(date_dir: Path) -> str:
    hits = sorted(glob.glob(str(date_dir / "phase7_with_buzz_score_*.json")))
    if not hits:
        print(f"ERROR: {date_dir}/phase7_with_buzz_score_*.json が見つかりません。", file=sys.stderr)
        print("日付指定が正しいか、Geminiフロー側(phase2→phase3→phase7)が"
              "その日付で実行済みか確認してください。", file=sys.stderr)
        sys.exit(1)
    return hits[-1]


def build_embed_text(v: dict) -> str:
    """embed対象: title + description(先頭500字) + 国名(en, 複数列挙)。tagsは空が多いので不使用。

    phase3が (b) 多対多方式で countries 配列(+reach) を持つようになったため、
    出現した全国の country_name_en をカンマ区切りで列挙する（rank昇順=出現順のまま）。
    """
    title = v.get("title", "") or ""
    desc = (v.get("description") or "")[:DESC_MAX_CHARS]
    countries = v.get("countries", []) or []
    country_names = [c.get("country_name_en", "") for c in countries if c.get("country_name_en")]
    country = ", ".join(country_names)
    return f"{title}\n{desc}\n{country}".strip()


def normalize(vec) -> list:
    """768次元は非正規化で返るため手動L2正規化（必須）。"""
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("ゼロベクトル（embed対象テキストが空の可能性）")
    return (arr / norm).tolist()


def embed_chunk_with_retry(client: OpenAI, texts: list, max_retries: int = 4) -> list:
    """1チャンク(最大10件)をembed。レート制限(429)時は指数バックオフでリトライ。"""
    for attempt in range(max_retries):
        try:
            result = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                dimensions=EMBED_DIM,
            )
            # DashScopeもOpenAI互換仕様通りindex順を保証して返す想定だが、
            # 念のためindexでソートしてから取り出す。
            ordered = sorted(result.data, key=lambda e: e.index)
            return [normalize(e.embedding) for e in ordered]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "Throttling" in msg) and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)  # 5,10,20,40秒
                print(f"  レート制限の可能性。{wait}秒待機してリトライします... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("embedリトライ上限に達しました。")


def main() -> int:
    parser = argparse.ArgumentParser(description="ステージ1: 動画メタデータのembed(Qwen版)")
    parser.add_argument("date", nargs="?", default=None,
                         help="対象日付(YYYYMMDD)。省略時は今日")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY が未設定です。", file=sys.stderr)
        return 1

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    input_dir, output_dir = resolve_dirs(date_str)

    path = find_phase7_path(input_dir)
    print(f"対象日付: {date_str}")
    print(f"入力ファイル: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos", [])
    if not videos:
        print("ERROR: videos が空です。", file=sys.stderr)
        return 1
    print(f"対象動画: {len(videos)}件")

    # embedテキストを全件分作る（空テキストは検出して警告）
    texts = []
    for v in videos:
        t = build_embed_text(v)
        if not t:
            print(f"  WARNING: video_id={v.get('video_id')} のembedテキストが空です。", file=sys.stderr)
        texts.append(t)

    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=DASHSCOPE_BASE_URL,
    )

    # チャンク分割してembed
    all_vecs = []
    n_chunks = (len(texts) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"\n{CHUNK_SIZE}件ずつ {n_chunks}チャンクでembedします（768次元・正規化・{EMBED_MODEL}）。")
    for i in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[i:i + CHUNK_SIZE]
        idx = i // CHUNK_SIZE + 1
        print(f"  チャンク {idx}/{n_chunks}（{len(chunk)}件）をembed中...")
        vecs = embed_chunk_with_retry(client, chunk)
        if len(vecs) != len(chunk):
            print(f"ERROR: 返却ベクトル数({len(vecs)})が入力数({len(chunk)})と不一致。", file=sys.stderr)
            return 1
        all_vecs.extend(vecs)
        if idx < n_chunks:
            time.sleep(SLEEP_BETWEEN_CHUNKS)

    assert len(all_vecs) == len(videos), "ベクトル総数と動画数が不一致"
    print(f"\nembed完了: {len(all_vecs)}件 / 各{len(all_vecs[0])}次元")

    # 元メタデータ + embedding をマージして保存（ステージ2はphase7を読み直さない）
    out_videos = []
    for v, vec, t in zip(videos, all_vecs, texts):
        rec = dict(v)               # phase7の元データを丸ごと保持
        rec["embedding"] = vec
        rec["_embed_text"] = t      # デバッグ用（何をembedしたか）。ステージ2で無視してよい
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
    print(f"\n保存しました: {out_path}")
    print("次はステージ2 (load_to_mongo.py) でこのファイルをMongoDBに投入します。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

