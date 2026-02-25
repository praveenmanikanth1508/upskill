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
import copy
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
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
DEFAULT_SCHEDULE_MODE = "incremental"
DEFAULT_SCHEDULE_WINDOW = "6m"
DEFAULT_SCHEDULE_PARTITION = "company"

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


def env_port(default: int = 8080) -> int:
    raw = os.getenv("PORT", str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def env_host(default: str = "127.0.0.1") -> str:
    return os.getenv("HOST", default).strip() or default


def parse_positive_int(raw: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


class LoadExecutionError(RuntimeError):
    def __init__(self, message: str, load_result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.load_result = load_result


def normalize_load_request(body: dict[str, Any]) -> dict[str, Any]:
    mode = str(body.get("mode", "incremental")).strip().lower()
    if mode not in {"full", "incremental"}:
        raise ValueError("mode must be 'full' or 'incremental'")

    window = str(body.get("window", "2y")).strip().lower() or "2y"
    incremental_partition = str(body.get("incremental_partition", "company")).strip().lower() or "company"
    if incremental_partition not in {"global", "company"}:
        raise ValueError("incremental_partition must be 'global' or 'company'")

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

    return {
        "mode": mode,
        "window": window,
        "incremental_partition": incremental_partition,
        "max_items_per_source": max_items_per_source,
        "max_questions_per_source": max_questions_per_source,
        "timeout": timeout,
    }


class JobStore:
    """In-memory job store for async load runs."""

    def __init__(self, max_jobs: int = 200) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._max_jobs = max_jobs

    def _trim_locked(self) -> None:
        while len(self._order) > self._max_jobs:
            oldest = self._order.pop(0)
            self._jobs.pop(oldest, None)

    def create(self, load_request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            job_id = uuid.uuid4().hex[:12]
            payload = {
                "id": job_id,
                "status": "queued",
                "created_at": utc_now_iso(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "request": copy.deepcopy(load_request),
                "load_result": None,
                "sync_result": None,
                "stats": None,
            }
            self._jobs[job_id] = payload
            self._order.append(job_id)
            self._trim_locked()
            return copy.deepcopy(payload)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return None
            current.update(fields)
            return copy.deepcopy(current)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            current = self._jobs.get(job_id)
            return copy.deepcopy(current) if current else None

    def list(self, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            total = len(self._order)
            ordered_ids = list(reversed(self._order))
            selected_ids = ordered_ids[offset : offset + limit]
            jobs = [copy.deepcopy(self._jobs[job_id]) for job_id in selected_ids if job_id in self._jobs]
        return {"jobs": jobs, "total": total, "limit": limit, "offset": offset}


class LoadScheduler:
    """Periodic loader scheduler for incremental/full refresh."""

    def __init__(self, run_callback: Any) -> None:
        self._run_callback = run_callback
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._state: dict[str, Any] = {
            "enabled": False,
            "interval_minutes": None,
            "request": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_status": None,
            "last_error": None,
            "next_run_at": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def start(self, interval_minutes: int, load_request: dict[str, Any]) -> dict[str, Any]:
        interval = max(1, interval_minutes)
        with self._lock:
            if self._state["enabled"]:
                raise RuntimeError("Scheduler is already running.")
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._state.update(
                {
                    "enabled": True,
                    "interval_minutes": interval,
                    "request": copy.deepcopy(load_request),
                    "last_started_at": None,
                    "last_finished_at": None,
                    "last_status": None,
                    "last_error": None,
                    "next_run_at": utc_now_iso(),
                }
            )
            self._thread = threading.Thread(target=self._loop, name="load-scheduler", daemon=True)
            self._thread.start()
            return copy.deepcopy(self._state)

    def stop(self) -> dict[str, Any]:
        thread: threading.Thread | None
        stop_event: threading.Event | None
        with self._lock:
            if not self._state["enabled"]:
                return copy.deepcopy(self._state)
            thread = self._thread
            stop_event = self._stop_event
            self._state["enabled"] = False
            self._state["next_run_at"] = None
            self._thread = None
            self._stop_event = None

        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=2)
        return self.snapshot()

    def _loop(self) -> None:
        while True:
            with self._lock:
                enabled = bool(self._state.get("enabled"))
                interval_minutes = int(self._state.get("interval_minutes") or 1)
                load_request = copy.deepcopy(self._state.get("request") or {})
                stop_event = self._stop_event
            if not enabled or stop_event is None or stop_event.is_set():
                return

            with self._lock:
                self._state["last_started_at"] = utc_now_iso()
                self._state["last_status"] = "running"
                self._state["last_error"] = None

            try:
                self._run_callback(load_request)
                with self._lock:
                    self._state["last_status"] = "succeeded"
            except Exception as exc:  # pragma: no cover - defensive runtime path
                with self._lock:
                    self._state["last_status"] = "failed"
                    self._state["last_error"] = str(exc)
            finally:
                with self._lock:
                    self._state["last_finished_at"] = utc_now_iso()
                    next_run_ts = time.time() + interval_minutes * 60
                    self._state["next_run_at"] = datetime.fromtimestamp(next_run_ts, timezone.utc).isoformat()

            if stop_event.wait(interval_minutes * 60):
                return


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


def list_load_runs(db_path: Path, limit: int, offset: int) -> dict[str, Any]:
    with open_connection(db_path) as conn:
        ensure_schema(conn)
        total = conn.execute("SELECT COUNT(*) FROM load_runs").fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, run_at, mode, window, incremental_partition, item_count, inserted_count, updated_count, output_path
            FROM load_runs
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return {
        "runs": [dict(row) for row in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


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


def execute_load_and_sync(runtime_config: RuntimeConfig, load_request: dict[str, Any]) -> dict[str, Any]:
    load_result = run_loader(
        config_path=runtime_config.config_path,
        output_path=runtime_config.bank_path,
        mode=load_request["mode"],
        window=load_request["window"],
        incremental_partition=load_request["incremental_partition"],
        max_items_per_source=load_request["max_items_per_source"],
        max_questions_per_source=load_request["max_questions_per_source"],
        timeout=load_request["timeout"],
    )

    if load_result["return_code"] != 0:
        raise LoadExecutionError("Build loader failed.", load_result=load_result)

    sync_result = sync_bank_to_db(
        bank_path=runtime_config.bank_path,
        db_path=runtime_config.db_path,
        mode=load_request["mode"],
        window=load_request["window"],
        incremental_partition=load_request["incremental_partition"],
    )
    stats = compute_stats(runtime_config.db_path)
    return {
        "message": "Load completed and DB synced successfully.",
        "load_result": load_result,
        "sync_result": sync_result,
        "stats": stats,
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
        self.load_lock = threading.Lock()
        self.job_store = JobStore(max_jobs=250)
        self.scheduler = LoadScheduler(self._run_scheduled_load_once)

    def execute_load(self, load_request: dict[str, Any]) -> dict[str, Any]:
        with self.load_lock:
            return execute_load_and_sync(self.runtime_config, load_request)

    def start_async_job(self, load_request: dict[str, Any]) -> dict[str, Any]:
        job = self.job_store.create(load_request)
        thread = threading.Thread(
            target=self._async_job_runner,
            args=(job["id"], load_request),
            name=f"load-job-{job['id']}",
            daemon=True,
        )
        thread.start()
        return job

    def _async_job_runner(self, job_id: str, load_request: dict[str, Any]) -> None:
        self.job_store.update(job_id, status="running", started_at=utc_now_iso(), error=None)
        try:
            result = self.execute_load(load_request)
            self.job_store.update(
                job_id,
                status="succeeded",
                finished_at=utc_now_iso(),
                load_result=result["load_result"],
                sync_result=result["sync_result"],
                stats=result["stats"],
            )
        except LoadExecutionError as exc:
            self.job_store.update(
                job_id,
                status="failed",
                finished_at=utc_now_iso(),
                error=str(exc),
                load_result=exc.load_result,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self.job_store.update(
                job_id,
                status="failed",
                finished_at=utc_now_iso(),
                error=str(exc),
            )

    def _run_scheduled_load_once(self, load_request: dict[str, Any]) -> None:
        self.execute_load(load_request)


class InterviewHandler(BaseHTTPRequestHandler):
    server: InterviewServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

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

        if path == "/api/jobs":
            limit = parse_positive_int(query.get("limit", [25])[0], 25, 1, 200)
            offset = parse_positive_int(query.get("offset", [0])[0], 0, 0, 1_000_000)
            payload = self.server.job_store.list(limit=limit, offset=offset)
            self._send_json(HTTPStatus.OK, payload)
            return

        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip()
            if not job_id:
                self._send_error(HTTPStatus.BAD_REQUEST, "Missing job id.")
                return
            payload = self.server.job_store.get(job_id)
            if payload is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Job not found.")
                return
            self._send_json(HTTPStatus.OK, payload)
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

        if path == "/api/runs":
            limit = parse_positive_int(query.get("limit", [25])[0], 25, 1, 200)
            offset = parse_positive_int(query.get("offset", [0])[0], 0, 0, 1_000_000)
            payload = list_load_runs(self.cfg.db_path, limit=limit, offset=offset)
            self._send_json(HTTPStatus.OK, payload)
            return

        if path == "/api/scheduler":
            self._send_json(HTTPStatus.OK, self.server.scheduler.snapshot())
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
            try:
                load_request = normalize_load_request(body)
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return

            try:
                payload = self.server.execute_load(load_request)
            except LoadExecutionError as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": str(exc),
                        "load_result": exc.load_result,
                    },
                )
                return
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": f"Loader succeeded but DB sync failed: {exc}",
                    },
                )
                return

            self._send_json(HTTPStatus.OK, payload)
            return

        if path == "/api/load/async":
            try:
                load_request = normalize_load_request(body)
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return

            job = self.server.start_async_job(load_request)
            self._send_json(
                HTTPStatus.ACCEPTED,
                {
                    "message": "Async load job accepted.",
                    "job_id": job["id"],
                    "job": job,
                },
            )
            return

        if path == "/api/scheduler/start":
            interval_minutes = parse_positive_int(
                body.get("interval_minutes", 360),
                fallback=360,
                minimum=1,
                maximum=10_080,
            )
            scheduler_body = {
                "mode": body.get("mode", DEFAULT_SCHEDULE_MODE),
                "window": body.get("window", DEFAULT_SCHEDULE_WINDOW),
                "incremental_partition": body.get("incremental_partition", DEFAULT_SCHEDULE_PARTITION),
                "max_items_per_source": body.get("max_items_per_source", 50),
                "max_questions_per_source": body.get("max_questions_per_source", 60),
                "timeout": body.get("timeout", 20),
            }
            try:
                load_request = normalize_load_request(scheduler_body)
                snapshot = self.server.scheduler.start(interval_minutes=interval_minutes, load_request=load_request)
            except (ValueError, RuntimeError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return

            self._send_json(
                HTTPStatus.OK,
                {
                    "message": "Scheduler started.",
                    "scheduler": snapshot,
                },
            )
            return

        if path == "/api/scheduler/stop":
            snapshot = self.server.scheduler.stop()
            self._send_json(
                HTTPStatus.OK,
                {
                    "message": "Scheduler stopped.",
                    "scheduler": snapshot,
                },
            )
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run web UI backend for interview prep hub.")
    parser.add_argument("--host", default=env_host(), help="Host to bind server.")
    parser.add_argument("--port", type=int, default=env_port(), help="Port to bind server.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to sources config.")
    parser.add_argument("--bank", default=str(DEFAULT_BANK), help="Path to generated interview bank JSON.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database file.")
    parser.add_argument("--ui-dir", default=str(DEFAULT_UI_DIR), help="Directory for static UI files.")
    parser.add_argument(
        "--bootstrap-sync",
        action="store_true",
        help="Sync existing bank JSON into SQLite before server starts.",
    )
    parser.add_argument(
        "--schedule-interval-minutes",
        type=int,
        default=0,
        help="If > 0, start periodic loader scheduler with this interval.",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=["full", "incremental"],
        default=DEFAULT_SCHEDULE_MODE,
        help="Mode used for auto scheduler runs.",
    )
    parser.add_argument(
        "--schedule-window",
        default=DEFAULT_SCHEDULE_WINDOW,
        help="Window value used for auto scheduler runs (e.g., 1w, 1m, 6m, 2y).",
    )
    parser.add_argument(
        "--schedule-partition",
        choices=["global", "company"],
        default=DEFAULT_SCHEDULE_PARTITION,
        help="Incremental partition for auto scheduler runs.",
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
    if args.schedule_interval_minutes > 0:
        scheduler_body = {
            "mode": args.schedule_mode,
            "window": args.schedule_window,
            "incremental_partition": args.schedule_partition,
        }
        try:
            scheduler_request = normalize_load_request(scheduler_body)
            server.scheduler.start(
                interval_minutes=args.schedule_interval_minutes,
                load_request=scheduler_request,
            )
            print(
                "Scheduler started: "
                f"every {args.schedule_interval_minutes} minute(s), mode={args.schedule_mode}, window={args.schedule_window}, partition={args.schedule_partition}"
            )
        except (ValueError, RuntimeError) as exc:
            print(f"Failed to start scheduler: {exc}", file=sys.stderr)
            return 2

    print(f"Serving Interview Prep UI on http://{args.host}:{args.port}")
    print(f"DB: {runtime_config.db_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.scheduler.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
