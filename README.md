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

### 0) Easiest path (static mode, no backend required)

```bash
bash scripts/refresh_static_site.sh 6m 50 60
```

This does both:

1. Rebuild `knowledge_bank/interview_bank.json`
2. Export static bundle into `site/` for GitHub Pages

---

### 1) First load (latest 2 years by default)

```bash
python3 scripts/build_knowledge_bank.py \
  --config config/sources.json \
  --output knowledge_bank/interview_bank.json \
  --window 2y \
  --max-items-per-source 50
```

Other window presets:

- `--window 1w` (weekly)
- `--window 1m` (monthly)
- `--window 6m`
- `--window 2y`

### 2) Incremental load (new items since last run)

```bash
python3 scripts/build_knowledge_bank.py \
  --config config/sources.json \
  --output knowledge_bank/interview_bank.json \
  --incremental \
  --window 6m \
  --incremental-partition company \
  --max-items-per-source 50
```

Notes:

- `--incremental` uses `knowledge_bank/.build_state.json` to remember the last successful run.
- In incremental mode, old items are merged with newly fetched items, then deduplicated.
- `--incremental-partition company` tracks checkpoints per company key (for example: `company:uber`).
- `--window` (or `--days-back`) still acts as a safety cap.
- Some community sites (for example Reddit RSS) may block automated fetches in certain environments.
  If a source is blocked, keep it as manual notes via `knowledge_bank/manual_entry_template.json`.

### 3) Query by company/topic/keyword

```bash
python3 scripts/query_knowledge_bank.py \
  --bank knowledge_bank/interview_bank.json \
  --company uber \
  --tag spark \
  --keyword "slowly changing dimension" \
  --limit 20
```

### 4) Run backend server (API + UI + SQLite)

```bash
python3 scripts/web_app.py --host 127.0.0.1 --port 8080 --bootstrap-sync
```

Open: `http://127.0.0.1:8080`

Phase 2 backend API additions:

- `POST /api/load/async` -> async load jobs
- `GET /api/jobs` and `GET /api/jobs/{job_id}` -> job tracking
- `GET /api/runs` -> load history
- `GET /api/scheduler` -> scheduler state
- `POST /api/scheduler/start` / `POST /api/scheduler/stop` -> timed incremental runs

Example: start timed scheduler (every 6 hours)

```bash
curl -s -X POST "http://127.0.0.1:8080/api/scheduler/start" \
  -H "Content-Type: application/json" \
  -d '{"interval_minutes":360,"mode":"incremental","window":"6m","incremental_partition":"company"}'
```

### 5) Deploy backend (Render or Railway free tier)

This repo includes platform configs:

- `render.yaml`
- `railway.json`
- `Procfile`

#### Render

1. Create a new **Web Service** from this GitHub repo.
2. Render auto-detects `render.yaml`.
3. Deploy.
4. Copy your backend URL (example: `https://upskill-interview-backend.onrender.com`).

#### Railway

1. Create a new Railway project from this GitHub repo.
2. Railway uses `railway.json`/`Procfile` start command.
3. Deploy and copy public URL.

#### Connect GitHub Pages UI to hosted backend

On your Pages site (`https://praveenmanikanth1508.github.io/upskill/`):

1. In **Backend API Connection**, paste backend URL.
2. Click **Connect API**.
3. UI switches from static fallback to live API mode.

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
  web_app.py                   # Backend API + SQLite + local UI server
  export_static_site.py        # Static bundle export for GitHub Pages
  refresh_static_site.sh       # One-command static refresh
web/
  index.html                   # UI shell
  app.js                       # UI behavior
  styles.css                   # UI styling
render.yaml                    # Render deployment config
railway.json                   # Railway deployment config
Procfile                       # Procfile start command
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

## Scheduling options

The loader supports both styles:

- **On-demand**: run the command manually whenever you want.
- **Timed/Scheduled**: run incremental command via cron (or any scheduler).

Example daily cron (UTC 06:00):

```bash
0 6 * * * cd /path/to/repo && python3 scripts/build_knowledge_bank.py --config config/sources.json --output knowledge_bank/interview_bank.json --incremental --window 6m --incremental-partition company >> knowledge_bank/build.log 2>&1
```

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
