#!/bin/bash
# Chạy chatbot với Python đúng (có đủ packages)
PYTHON=/usr/local/bin/python3
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"

# Kiểm tra FAISS index — tự động ingest nếu chưa có
if [ ! -f "vectorstore/faiss_index/index.faiss" ]; then
    echo "FAISS index chưa có. Đang chạy ingest..."
    "$PYTHON" src/ingest.py
fi

"$PYTHON" app.py "$@"
