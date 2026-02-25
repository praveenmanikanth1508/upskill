# Data Engineering Interview Prep Hub (Mid-Level, USA)

This project is an MVP knowledge bank for **Data Engineering interview preparation** focused on:

- **Experience band**: 3-6 years
- **Market**: USA
- **Format**: company-wise interview questions, topic tags, and answer hints

The idea is similar to a content aggregator, but specialized for interview prep so you can avoid noise during job hunt.

---

## Why this exists

Job search can get chaotic. Useful interview prep material is scattered across blogs, communities, newsletters, and forum posts.

This project helps you:

1. Pull content from multiple sources (RSS + HTML pages)
2. Extract likely interview questions with simple heuristics
3. Attach company/topic tags
4. Build one local searchable knowledge bank

---

## Quick start

### 1) Build the knowledge bank

```bash
python3 scripts/build_knowledge_bank.py \
  --config config/sources.json \
  --output knowledge_bank/interview_bank.json \
  --days-back 3650 \
  --max-items-per-source 50
```

### 2) Query by company/topic/keyword

```bash
python3 scripts/query_knowledge_bank.py \
  --bank knowledge_bank/interview_bank.json \
  --company uber \
  --tag spark \
  --keyword "slowly changing dimension" \
  --limit 20
```

---

## Project structure

```text
config/
  sources.json                 # Source definitions and default tags
knowledge_bank/
  sample_bank.json             # Example output format
  manual_entry_template.json   # Template for adding your own interview logs
scripts/
  build_knowledge_bank.py      # Ingestion + extraction pipeline
  query_knowledge_bank.py      # CLI search on generated bank
```

---

## Source config format

Edit `config/sources.json` and add as many sources as you want.

Supported source types:

- `rss`: parses RSS/Atom entries
- `html`: fetches a page and extracts likely questions from visible text

Each source can include:

- `name`
- `type` (`rss` or `html`)
- `url`
- `company` (or `"multiple"`)
- `region` (for your case use `"usa"`)
- `role_level` (for your case use `"mid-level"`)
- `base_tags` (for example: `["sql", "spark", "python"]`)

---

## Notes on legality and quality

- Only collect from publicly available sources.
- Respect website terms, robots.txt, and rate limits.
- Keep original source URLs in your bank.
- Treat generated answer hints as a starting point, not final truth.

---

## Practical workflow for your job hunt

1. Add 10-20 reliable sources by company/topic.
2. Run build script daily or weekly.
3. Query by company + topic.
4. Review and improve answer hints manually.
5. Track weak areas (for example: Spark tuning, CDC, data modeling, orchestration).

This turns interview prep into a focused, low-noise system.
