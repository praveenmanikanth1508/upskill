#!/usr/bin/env bash
set -euo pipefail

# Easy workflow for GitHub Pages/static mode:
# 1) refresh interview bank JSON
# 2) package static site bundle into ./site
#
# Usage:
#   bash scripts/refresh_static_site.sh [window] [max_items_per_source] [max_questions_per_source]
# Example:
#   bash scripts/refresh_static_site.sh 6m 50 60

WINDOW="${1:-6m}"
MAX_ITEMS="${2:-50}"
MAX_QUESTIONS="${3:-60}"

python3 scripts/build_knowledge_bank.py \
  --config config/sources.json \
  --output knowledge_bank/interview_bank.json \
  --window "${WINDOW}" \
  --max-items-per-source "${MAX_ITEMS}" \
  --max-questions-per-source "${MAX_QUESTIONS}" \
  --verbose

python3 scripts/export_static_site.py \
  --web-dir web \
  --bank knowledge_bank/interview_bank.json \
  --output-dir site

echo "Static site refreshed successfully."
echo "Open local bundle: site/index.html"
