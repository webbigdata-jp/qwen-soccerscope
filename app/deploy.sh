#!/bin/bash

# 古いbuildと作業用venvを完全に削除（重要：使い回すと壊れる。後述）
deactivate
rm -rf build .venv-fc

# FCのランタイム(Python 3.12)に合わせ、pip同梱のvenvを作る
uv venv .venv-fc --python 3.12 --seed
source .venv-fc/bin/activate

# ビルド用ディレクトリに必要なファイルだけコピー
mkdir build
cp main.py build/
cp -r soccer_agent build/
cp -r static build/
rm -rf build/soccer_agent/__pycache__

# npx タイアウト問題
npm install --prefix build mongodb-mcp-server
rm -rf build/node_modules/@oven


# build/直下に依存関係をインストール（venv内の"本物のpip"を使う。uv pipではない）
cd build
pip install -t . -r ../requirements.txt

# zip化（コードパッケージのルートディレクトリで実行するのがコツ）
zip -rq -y ../code.zip ./
cd ..
deactivate

