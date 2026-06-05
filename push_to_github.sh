#!/bin/bash
# Script push chatbot-tuyen-sinh len GitHub
# Chay: bash push_to_github.sh <GITHUB_TOKEN>
set -e

TOKEN="$1"
if [ -z "$TOKEN" ]; then
  echo "Usage: bash push_to_github.sh <GITHUB_TOKEN>"
  exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

git config user.email "congluanvl@gmail.com"
git config user.name "Luanntc2"

git remote remove origin 2>/dev/null || true
git remote add origin "https://Luanntc2:${TOKEN}@github.com/Luanntc2/chatbot-tuyen-sinh.git"
git push -u origin main

echo "Push thanh cong! Xem tai: https://github.com/Luanntc2/chatbot-tuyen-sinh"
