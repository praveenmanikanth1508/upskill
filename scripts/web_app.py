#!/usr/bin/env python3
"""
Web UI backend for the Data Engineering interview prep hub.

Features:
  - Serves a minimal frontend UI
  - Provides JSON API for loading/querying data
  - Stores interview bank records in SQLite (free and easy for testing)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "config/sources.json"
DEFAULT_BANK = ROOT_DIR / "knowledge_bank/interview_bank.json"
DEFAULT_DB = ROOT_DIR / "knowledge_bank/interview_bank.db"
DEFAULT_UI_DIR = ROOT_DIR / "web"
BUILD_SCRIPT = ROOT_DIR / "scripts/build_knowledge_bank.py"

MAX_LIMIT = 200
DEFAULT_LIMIT = 25

MIME_BY_SUFFIX = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
}

STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}


@dataclass
class RuntimeConfig:
    config_path: Path
    bank_path: Path
    db_path: Path
    ui_dir: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_positive_int(raw: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


def open_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS interview_items (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer_hint TEXT,
            company TEXT NOT NULL,
            region TEXT,
            role_level TEXT,
            tags_json TEXT NOT NULL,
            source_name TEXT,
            source_type TEXT,
            source_url TEXT,
            entry_title TEXT,
            entry_url TEXT,
            published_at TEXT,
            collected_at TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_items_company ON interview_items (company);
        CREATE INDEX IF NOT EXISTS idx_items_collected ON interview_items (collected_at);
        CREATE INDEX IF NOT EXISTS idx_items_published ON interview_items (published_at);

        CREATE TABLE IF NOT EXISTS load_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            window TEXT,
            incremental_partition TEXT,
            item_count INTEGER NOT NULL,
            inserted_count INTEGER NOT NULL,
            updated_count INTEGER NOT NULL,
            output_path TEXT NOT NULL
        );
        """
    )


def normalize_tags(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    tags = {str(tag).strip().lower() for tag in raw if str(tag).strip()}
    return sorted(tags)


def read_bank_payload(bank_path: Path) -> dict[str, Any]:
    if not bank_path.exists():
        raise FileNotFoundError(f"Bank file not found: {bank_path}")
    with bank_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Invalid bank payload: expected JSON object.")
    return payload


def sync_bank_to_db(
    bank_path: Path,
    db_path: Path,
    mode: str,
    window: str | None,
    incremental_partition: str | None,
) -> dict[str, Any]:
    payload = read_bank_payload(bank_path)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Invalid bank payload: 'items' must be a list.")

    inserted = 0
    updated = 0
    skipped = 0
    now = utc_now_iso()

    with open_connection(db_path) as conn:
        ensure_schema(conn)
        for item in items:
            if not isinstance(item, dict):
                skipped += 1
                continue

            item_id = str(item.get("id", "")).strip()
            question = str(item.get("question", "")).strip()
            if not item_id or not question:
                skipped += 1
                continue

            source = item.get("source", {})
            if not isinstance(source, dict):
                source = {}

            tags = normalize_tags(item.get("tags", []))
            tags_json = json.dumps(tags, ensure_ascii=True)
            company = str(item.get("company", "multiple")).strip().lower() or "multiple"
            region = str(item.get("region", "usa")).strip().lower() or "usa"
            role_level = str(item.get("role_level", "mid-level")).strip().lower() or "mid-level"

            already_exists = (
                conn.execute(
                    "SELECT 1 FROM interview_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
                is not None
            )

            conn.execute(
                """
                INSERT INTO interview_items (
                    id, question, answer_hint, company, region, role_level, tags_json,
                    source_name, source_type, source_url, entry_title, entry_url,
                    published_at, collected_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    question=excluded.question,
                    answer_hint=excluded.answer_hint,
                    company=excluded.company,
                    region=excluded.region,
                    role_level=excluded.role_level,
                    tags_json=excluded.tags_json,
                    source_name=excluded.source_name,
                    source_type=excluded.source_type,
                    source_url=excluded.source_url,
                    entry_title=excluded.entry_title,
                    entry_url=excluded.entry_url,
                    published_at=excluded.published_at,
                    collected_at=excluded.collected_at,
                    updated_at=excluded.updated_at
                """,
                (
                    item_id,
                    question,
                    str(item.get("answer_hint", "")).strip(),
                    company,
                    region,
                    role_level,
                    tags_json,
                    str(source.get("source_name", "")).strip(),
                    str(source.get("source_type", "")).strip(),
                    str(source.get("source_url", "")).strip(),
                    str(source.get("entry_title", "")).strip(),
                    str(source.get("entry_url", "")).strip(),
                    source.get("published_at"),
                    item.get("collected_at"),
                    now,
                ),
            )
            if already_exists:
                updated += 1
            else:
                inserted += 1

        conn.execute(
            """
            INSERT INTO load_runs (
                run_at, mode, window, incremental_partition,
                item_count, inserted_count, updated_count, output_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                mode,
                window,
                incremental_partition,
                len(items),
                inserted,
                updated,
                str(bank_path),
            ),
        )
        conn.commit()

    return {
        "item_count": len(items),
        "inserted_count": inserted,
        "updated_count": updated,
        "skipped_count": skipped,
        "db_path": str(db_path),
    }


def build_where_clause(company: str | None, tags: list[str], keyword: str | None) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []

    if company and company.lower() != "all":
        clauses.append("company = ?")
        params.append(company.lower())

    keyword_text = (keyword or "").strip()
    if keyword_text:
        clauses.append("(question LIKE ? OR answer_hint LIKE ?)")
        like_value = f"%{keyword_text}%"
        params.extend([like_value, like_value])

    for tag in tags:
        tag_value = tag.strip().lower()
        if not tag_value or tag_value == "all":
            continue
        clauses.append("tags_json LIKE ?")
        params.append(f'%"{tag_value}"%')

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def row_to_item(row: sqlite3.Row) -> dict[str, Any]:
    try:
        tags = json.loads(row["tags_json"]) if row["tags_json"] else []
    except json.JSONDecodeError:
        tags = []

    return {
        "id": row["id"],
        "question": row["question"],
        "answer_hint": row["answer_hint"],
        "company": row["company"],
        "region": row["region"],
        "role_level": row["role_level"],
        "tags": tags,
        "source": {
            "source_name": row["source_name"],
            "source_type": row["source_type"],
            "source_url": row["source_url"],
            "entry_title": row["entry_title"],
            "entry_url": row["entry_url"],
            "published_at": row["published_at"],
        },
        "collected_at": row["collected_at"],
        "updated_at": row["updated_at"],
    }


def query_items(
    db_path: Path,
    company: str | None,
    tags: list[str],
    keyword: str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    with open_connection(db_path) as conn:
        ensure_schema(conn)
        where, params = build_where_clause(company, tags, keyword)
        total = conn.execute(f"SELECT COUNT(*) FROM interview_items{where}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT *
            FROM interview_items
            {where}
            ORDER BY COALESCE(published_at, collected_at) DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()

    items = [row_to_item(row) for row in rows]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def compute_filters(db_path: Path, top_tag_count: int = 40) -> dict[str, Any]:
    with open_connection(db_path) as conn:
        ensure_schema(conn)
        company_rows = conn.execute(
            """
            SELECT company, COUNT(*) AS n
            FROM interview_items
            GROUP BY company
            ORDER BY n DESC, company ASC
            """
        ).fetchall()
        tags_rows = conn.execute("SELECT tags_json FROM interview_items").fetchall()

    tag_counter: Counter[str] = Counter()
    for row in tags_rows:
        raw = row["tags_json"] or "[]"
        try:
            tags = json.loads(raw)
        except json.JSONDecodeError:
            tags = []
        for tag in tags:
            cleaned = str(tag).strip().lower()
            if cleaned:
                tag_counter[cleaned] += 1

    return {
        "companies": [{"value": row["company"], "count": row["n"]} for row in company_rows],
        "tags": [{"value": tag, "count": count} for tag, count in tag_counter.most_common(top_tag_count)],
    }


def compute_stats(db_path: Path) -> dict[str, Any]:
    with open_connection(db_path) as conn:
        ensure_schema(conn)
        total_items = conn.execute("SELECT COUNT(*) FROM interview_items").fetchone()[0]
        distinct_companies = conn.execute("SELECT COUNT(DISTINCT company) FROM interview_items").fetchone()[0]
        latest_collected = conn.execute("SELECT MAX(collected_at) FROM interview_items").fetchone()[0]
        latest_published = conn.execute("SELECT MAX(published_at) FROM interview_items").fetchone()[0]
        last_run = conn.execute(
            """
            SELECT run_at, mode, window, incremental_partition, item_count, inserted_count, updated_count
            FROM load_runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    payload: dict[str, Any] = {
        "total_items": total_items,
        "distinct_companies": distinct_companies,
        "latest_collected_at": latest_collected,
        "latest_published_at": latest_published,
        "last_run": None,
    }
    if last_run:
        payload["last_run"] = dict(last_run)
    return payload


def run_loader(
    config_path: Path,
    output_path: Path,
    mode: str,
    window: str,
    incremental_partition: str,
    max_items_per_source: int,
    max_questions_per_source: int,
    timeout: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--window",
        window,
        "--max-items-per-source",
        str(max_items_per_source),
        "--max-questions-per-source",
        str(max_questions_per_source),
        "--timeout",
        str(timeout),
        "--verbose",
    ]
    if mode == "incremental":
        command.extend(["--incremental", "--incremental-partition", incremental_partition])

    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


class InterviewServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        runtime_config: RuntimeConfig,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.runtime_config = runtime_config


class InterviewHandler(BaseHTTPRequestHandler):
    server: InterviewServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    @property
    def cfg(self) -> RuntimeConfig:
        return self.server.runtime_config

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        content_length = parse_positive_int(raw_length, fallback=0, minimum=0, maximum=1_000_000)
        if content_length <= 0:
            return {}
        body = self.rfile.read(content_length)
        try:
            parsed = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _serve_static(self, route_path: str) -> None:
        relative = STATIC_ROUTES.get(route_path)
        if not relative:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        file_path = self.cfg.ui_dir / relative
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        content = file_path.read_bytes()
        content_type = MIME_BY_SUFFIX.get(file_path.suffix, "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_post(parsed)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle_api_get(self, parsed: Any) -> None:
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "time": utc_now_iso(),
                    "db_path": str(self.cfg.db_path),
                },
            )
            return

        if path == "/api/items":
            company = (query.get("company", [None])[0] or None)
            tags = [value for value in query.get("tag", []) if value.strip()]
            if not tags:
                combined_tags = (query.get("tags", [""])[0] or "").strip()
                if combined_tags:
                    tags = [part.strip() for part in combined_tags.split(",") if part.strip()]
            keyword = (query.get("keyword", [None])[0] or None)
            limit = parse_positive_int(query.get("limit", [DEFAULT_LIMIT])[0], DEFAULT_LIMIT, 1, MAX_LIMIT)
            offset = parse_positive_int(query.get("offset", [0])[0], 0, 0, 1_000_000)

            payload = query_items(
                db_path=self.cfg.db_path,
                company=company,
                tags=tags,
                keyword=keyword,
                limit=limit,
                offset=offset,
            )
            self._send_json(HTTPStatus.OK, payload)
            return

        if path == "/api/filters":
            payload = compute_filters(self.cfg.db_path)
            self._send_json(HTTPStatus.OK, payload)
            return

        if path == "/api/stats":
            payload = compute_stats(self.cfg.db_path)
            self._send_json(HTTPStatus.OK, payload)
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle_api_post(self, parsed: Any) -> None:
        path = parsed.path
        body = self._read_json_body()

        if path == "/api/sync":
            try:
                result = sync_bank_to_db(
                    bank_path=self.cfg.bank_path,
                    db_path=self.cfg.db_path,
                    mode="sync-only",
                    window=None,
                    incremental_partition=None,
                )
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "message": "Bank synced to SQLite successfully.",
                    "sync_result": result,
                    "stats": compute_stats(self.cfg.db_path),
                },
            )
            return

        if path == "/api/load":
            mode = str(body.get("mode", "incremental")).strip().lower()
            if mode not in {"full", "incremental"}:
                self._send_error(HTTPStatus.BAD_REQUEST, "mode must be 'full' or 'incremental'")
                return

            window = str(body.get("window", "2y")).strip().lower() or "2y"
            incremental_partition = str(body.get("incremental_partition", "company")).strip().lower() or "company"
            if incremental_partition not in {"global", "company"}:
                self._send_error(
                    HTTPStatus.BAD_REQUEST,
                    "incremental_partition must be 'global' or 'company'",
                )
                return

            max_items_per_source = parse_positive_int(
                body.get("max_items_per_source", 50),
                fallback=50,
                minimum=1,
                maximum=500,
            )
            max_questions_per_source = parse_positive_int(
                body.get("max_questions_per_source", 60),
                fallback=60,
                minimum=1,
                maximum=500,
            )
            timeout = parse_positive_int(body.get("timeout", 20), fallback=20, minimum=5, maximum=120)

            load_result = run_loader(
                config_path=self.cfg.config_path,
                output_path=self.cfg.bank_path,
                mode=mode,
                window=window,
                incremental_partition=incremental_partition,
                max_items_per_source=max_items_per_source,
                max_questions_per_source=max_questions_per_source,
                timeout=timeout,
            )

            if load_result["return_code"] != 0:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": "Build loader failed.",
                        "load_result": load_result,
                    },
                )
                return

            try:
                sync_result = sync_bank_to_db(
                    bank_path=self.cfg.bank_path,
                    db_path=self.cfg.db_path,
                    mode=mode,
                    window=window,
                    incremental_partition=incremental_partition,
                )
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": f"Loader succeeded but DB sync failed: {exc}",
                        "load_result": load_result,
                    },
                )
                return

            self._send_json(
                HTTPStatus.OK,
                {
                    "message": "Load completed and DB synced successfully.",
                    "load_result": load_result,
                    "sync_result": sync_result,
                    "stats": compute_stats(self.cfg.db_path),
                },
            )
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run web UI backend for interview prep hub.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server.")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to sources config.")
    parser.add_argument("--bank", default=str(DEFAULT_BANK), help="Path to generated interview bank JSON.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database file.")
    parser.add_argument("--ui-dir", default=str(DEFAULT_UI_DIR), help="Directory for static UI files.")
    parser.add_argument(
        "--bootstrap-sync",
        action="store_true",
        help="Sync existing bank JSON into SQLite before server starts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    runtime_config = RuntimeConfig(
        config_path=Path(args.config),
        bank_path=Path(args.bank),
        db_path=Path(args.db),
        ui_dir=Path(args.ui_dir),
    )

    if not runtime_config.ui_dir.exists():
        print(f"UI directory not found: {runtime_config.ui_dir}", file=sys.stderr)
        return 2
    if not BUILD_SCRIPT.exists():
        print(f"Build script not found: {BUILD_SCRIPT}", file=sys.stderr)
        return 2

    if args.bootstrap_sync:
        try:
            result = sync_bank_to_db(
                bank_path=runtime_config.bank_path,
                db_path=runtime_config.db_path,
                mode="bootstrap-sync",
                window=None,
                incremental_partition=None,
            )
            print(
                f"Bootstrap sync complete: {result['item_count']} items (inserted={result['inserted_count']}, updated={result['updated_count']})"
            )
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Bootstrap sync failed: {exc}", file=sys.stderr)
            return 2

    server = InterviewServer((args.host, args.port), InterviewHandler, runtime_config=runtime_config)
    print(f"Serving Interview Prep UI on http://{args.host}:{args.port}")
    print(f"DB: {runtime_config.db_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
