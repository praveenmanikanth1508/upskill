#!/usr/bin/env python3
"""
Build a Data Engineering interview knowledge bank from public sources.

The script intentionally uses only Python standard library modules.
It supports:
  - RSS/Atom sources
  - HTML sources (heuristic question extraction from visible page text)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


QUESTION_STARTERS = (
    "how ",
    "what ",
    "why ",
    "when ",
    "where ",
    "which ",
    "explain ",
    "describe ",
    "tell me ",
    "walk me through ",
    "can you ",
    "could you ",
    "design ",
    "implement ",
    "build ",
    "optimize ",
    "write ",
)

TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "sql": (r"\bsql\b", r"\bjoin\b", r"\bwindow function\b", r"\bcte\b", r"\bgroup by\b"),
    "python": (r"\bpython\b", r"\bpandas\b", r"\bpyspark\b", r"\bpytest\b"),
    "spark": (r"\bspark\b", r"\bpartition\b", r"\bshuffle\b", r"\bexecutor\b", r"\bdatabricks\b"),
    "airflow": (r"\bairflow\b", r"\bdag\b", r"\bscheduler\b"),
    "dbt": (r"\bdbt\b", r"\bmodel\b", r"\bmacro\b"),
    "streaming": (r"\bkafka\b", r"\bstream\b", r"\bflink\b", r"\bkinesis\b"),
    "warehousing": (r"\bwarehouse\b", r"\bsnowflake\b", r"\bbigquery\b", r"\bredshift\b"),
    "data-modeling": (r"\bdata model\b", r"\bschema\b", r"\bscd\b", r"\bnormaliz", r"\bdenormaliz"),
    "etl": (r"\betl\b", r"\belt\b", r"\bingestion\b", r"\bpipeline\b"),
    "orchestration": (r"\borchestr", r"\bdependency\b", r"\bretry\b", r"\bidempotent\b"),
    "system-design": (r"\bdesign\b", r"\bscal", r"\btrade[- ]off\b", r"\barchitecture\b"),
    "behavioral": (r"\bconflict\b", r"\bmistake\b", r"\bownership\b", r"\bdeadline\b"),
}


class VisibleTextExtractor(HTMLParser):
    """Collect visible text from selected HTML tags."""

    TARGET_TAGS = {"title", "h1", "h2", "h3", "h4", "p", "li", "blockquote"}
    IGNORED_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._target_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth == 0 and tag in self.TARGET_TAGS:
            self._target_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.IGNORED_TAGS and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if self._ignored_depth == 0 and tag in self.TARGET_TAGS and self._target_depth > 0:
            self._target_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0 or self._target_depth == 0:
            return
        text = normalize_text(data)
        if text:
            self.chunks.append(text)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_tags(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", value)
    return normalize_text(unescape(cleaned))


def looks_like_question(text: str) -> bool:
    candidate = normalize_text(text)
    if len(candidate) < 15 or len(candidate) > 220:
        return False

    lowered = candidate.lower()
    if "?" in candidate:
        return True
    if lowered.startswith("q:"):
        return True
    return any(lowered.startswith(prefix) for prefix in QUESTION_STARTERS)


def extract_question_candidates(text: str, max_questions: int) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []

    raw_lines = re.split(r"[\r\n]+", text)
    for line in raw_lines:
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        line = re.sub(r"^\s*q:\s*", "", line, flags=re.IGNORECASE)
        line = normalize_text(line)
        if not line:
            continue

        fragments = re.split(r"(?<=[.?!])\s+", line)
        for fragment in fragments:
            fragment = normalize_text(fragment)
            if not fragment:
                continue
            if looks_like_question(fragment):
                cleaned = fragment.rstrip(" .")
                lowered = cleaned.lower()
                if lowered not in seen:
                    seen.add(lowered)
                    results.append(cleaned)
                    if len(results) >= max_questions:
                        return results
    return results


def infer_tags(text: str) -> set[str]:
    tags: set[str] = set()
    lowered = text.lower()
    for tag, patterns in TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                tags.add(tag)
                break
    return tags


def answer_hint_for(question: str, tags: set[str]) -> str:
    if "sql" in tags:
        return "Explain table assumptions first, then walk through SQL step-by-step and discuss performance trade-offs."
    if "system-design" in tags:
        return "Structure answer as requirements, architecture, data model, scale bottlenecks, and failure handling."
    if "spark" in tags or "streaming" in tags:
        return "Discuss partitioning, state management, backfills, monitoring, and cost/performance trade-offs."
    if "behavioral" in tags:
        return "Use STAR format with concrete metrics, your specific decisions, and post-mortem learnings."
    return "Clarify assumptions, provide a concrete approach, and mention trade-offs and operational considerations."


def source_base_tags(source: dict[str, Any]) -> set[str]:
    raw = source.get("base_tags", [])
    if not isinstance(raw, list):
        return set()
    return {str(tag).lower() for tag in raw if str(tag).strip()}


def make_item_id(source_name: str, entry_url: str, question: str) -> str:
    payload = f"{source_name}|{entry_url}|{question.lower()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass

    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def fetch_url(url: str, timeout: int) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "de-interview-prep-hub/0.1 (+https://example.local)",
            "Accept": "text/html,application/xml,application/rss+xml,application/atom+xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
        return body.decode(charset, errors="replace")


def parse_rss_or_atom(xml_text: str, max_items: int) -> list[dict[str, str | None]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    entries: list[dict[str, str | None]] = []

    rss_items = root.findall("./channel/item")
    for node in rss_items[:max_items]:
        title = normalize_text(node.findtext("title") or "")
        link = normalize_text(node.findtext("link") or "")
        description = node.findtext("description") or ""
        summary = strip_tags(description)
        pub = node.findtext("pubDate") or node.findtext("dc:date")
        entries.append(
            {
                "title": title or None,
                "link": link or None,
                "summary": summary or None,
                "published": pub,
            }
        )

    if entries:
        return entries

    atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
    atom_entries = root.findall(".//atom:entry", atom_ns)
    for node in atom_entries[:max_items]:
        title = normalize_text(node.findtext("atom:title", default="", namespaces=atom_ns))
        summary = normalize_text(
            strip_tags(
                node.findtext("atom:summary", default="", namespaces=atom_ns)
                or node.findtext("atom:content", default="", namespaces=atom_ns)
            )
        )
        link_url: str | None = None
        for link_node in node.findall("atom:link", atom_ns):
            href = (link_node.attrib.get("href") or "").strip()
            rel = (link_node.attrib.get("rel") or "alternate").strip()
            if href and rel in {"alternate", ""}:
                link_url = href
                break
        pub = node.findtext("atom:published", default="", namespaces=atom_ns) or node.findtext(
            "atom:updated",
            default="",
            namespaces=atom_ns,
        )
        entries.append(
            {
                "title": title or None,
                "link": link_url,
                "summary": summary or None,
                "published": pub or None,
            }
        )

    return entries


def collect_from_rss(
    source: dict[str, Any],
    xml_text: str,
    now_utc: datetime,
    cutoff: datetime,
    max_items_per_source: int,
    max_questions_per_entry: int,
) -> list[dict[str, Any]]:
    source_name = source.get("name", source.get("url", "unknown-source"))
    base_tags = source_base_tags(source)
    company = str(source.get("company", "multiple")).lower()
    region = str(source.get("region", "usa")).lower()
    role_level = str(source.get("role_level", "mid-level")).lower()

    items: list[dict[str, Any]] = []
    feed_entries = parse_rss_or_atom(xml_text, max_items=max_items_per_source)
    for entry in feed_entries:
        published_dt = parse_dt(entry.get("published"))
        if published_dt is not None and published_dt < cutoff:
            continue

        entry_title = entry.get("title") or ""
        entry_summary = entry.get("summary") or ""
        entry_link = entry.get("link") or source.get("url", "")
        candidate_blob = f"{entry_title}\n{entry_summary}"

        questions = extract_question_candidates(candidate_blob, max_questions_per_entry)
        if not questions and looks_like_question(entry_title):
            questions = [normalize_text(entry_title)]

        for question in questions:
            inferred_tags = infer_tags(question)
            all_tags = set(base_tags) | inferred_tags | {role_level, region}
            item = {
                "id": make_item_id(source_name, entry_link, question),
                "question": question,
                "answer_hint": answer_hint_for(question, all_tags),
                "company": company,
                "region": region,
                "role_level": role_level,
                "tags": sorted(all_tags),
                "source": {
                    "source_name": source_name,
                    "source_type": "rss",
                    "source_url": source.get("url", ""),
                    "entry_title": entry_title,
                    "entry_url": entry_link,
                    "published_at": published_dt.isoformat() if published_dt else entry.get("published"),
                },
                "collected_at": now_utc.isoformat(),
            }
            items.append(item)

    return items


def collect_from_html(
    source: dict[str, Any],
    html_text: str,
    now_utc: datetime,
    max_questions_per_source: int,
) -> list[dict[str, Any]]:
    source_name = source.get("name", source.get("url", "unknown-source"))
    base_tags = source_base_tags(source)
    company = str(source.get("company", "multiple")).lower()
    region = str(source.get("region", "usa")).lower()
    role_level = str(source.get("role_level", "mid-level")).lower()

    extractor = VisibleTextExtractor()
    extractor.feed(html_text)
    text = "\n".join(extractor.chunks)
    questions = extract_question_candidates(text, max_questions=max_questions_per_source)

    items: list[dict[str, Any]] = []
    for question in questions:
        inferred_tags = infer_tags(question)
        all_tags = set(base_tags) | inferred_tags | {role_level, region}
        item = {
            "id": make_item_id(source_name, source.get("url", ""), question),
            "question": question,
            "answer_hint": answer_hint_for(question, all_tags),
            "company": company,
            "region": region,
            "role_level": role_level,
            "tags": sorted(all_tags),
            "source": {
                "source_name": source_name,
                "source_type": "html",
                "source_url": source.get("url", ""),
                "entry_title": source_name,
                "entry_url": source.get("url", ""),
                "published_at": None,
            },
            "collected_at": now_utc.isoformat(),
        }
        items.append(item)
    return items


def load_config(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("config.sources must be a list")
    return sources


def load_existing_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []

    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and "id" in item]


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(state, dict):
        return {}
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def resolve_cutoff(
    now_utc: datetime,
    days_back: int,
    incremental: bool,
    state: dict[str, Any],
) -> tuple[datetime, str]:
    default_cutoff = now_utc - timedelta(days=days_back)
    if not incremental:
        return default_cutoff, "days_back_window"

    last_run_raw = state.get("last_successful_run")
    last_run = parse_dt(str(last_run_raw)) if last_run_raw else None
    if not last_run:
        return default_cutoff, "incremental_no_state_fallback"

    cutoff = max(default_cutoff, last_run)
    if cutoff == last_run:
        return cutoff, "incremental_since_last_successful_run"
    return cutoff, "incremental_capped_by_days_back"


def deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique[item["id"]] = item
    return list(unique.values())


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def metadata_for(
    items: list[dict[str, Any]],
    source_count: int,
    now_utc: datetime,
    load_mode: str,
    cutoff: datetime,
    cutoff_reason: str,
) -> dict[str, Any]:
    company_counts = Counter(item.get("company", "unknown") for item in items)
    tag_counts = Counter(tag for item in items for tag in item.get("tags", []))

    return {
        "generated_at": now_utc.isoformat(),
        "load_mode": load_mode,
        "cutoff_utc": cutoff.isoformat(),
        "cutoff_reason": cutoff_reason,
        "source_count": source_count,
        "item_count": len(items),
        "companies": dict(company_counts.most_common()),
        "top_tags": dict(tag_counts.most_common(20)),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Data Engineering interview knowledge bank.")
    parser.add_argument("--config", default="config/sources.json", help="Path to source config JSON.")
    parser.add_argument(
        "--output",
        default="knowledge_bank/interview_bank.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=730,
        help="Look-back window in days (default 730 = 2 years).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: fetch items since last successful run (state-based).",
    )
    parser.add_argument(
        "--state-file",
        default="knowledge_bank/.build_state.json",
        help="Path to incremental state file.",
    )
    parser.add_argument(
        "--no-state-update",
        action="store_true",
        help="Do not write state file after successful run.",
    )
    parser.add_argument(
        "--max-items-per-source",
        type=int,
        default=50,
        help="Max feed entries fetched per source.",
    )
    parser.add_argument(
        "--max-questions-per-entry",
        type=int,
        default=4,
        help="Max question candidates extracted from each RSS entry.",
    )
    parser.add_argument(
        "--max-questions-per-source",
        type=int,
        default=60,
        help="Max question candidates extracted for HTML source.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="Network timeout in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Print per-source diagnostics.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    output_path = Path(args.output)
    state_path = Path(args.state_file)

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        sources = load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 2

    now_utc = datetime.now(timezone.utc)
    state = load_state(state_path) if args.incremental else {}
    cutoff, cutoff_reason = resolve_cutoff(
        now_utc=now_utc,
        days_back=args.days_back,
        incremental=args.incremental,
        state=state,
    )
    load_mode = "incremental" if args.incremental else "full"
    all_items: list[dict[str, Any]] = []

    if args.verbose:
        print(
            f"[info] mode={load_mode} cutoff={cutoff.isoformat()} reason={cutoff_reason}",
            file=sys.stderr,
        )

    for source in sources:
        source_name = source.get("name", source.get("url", "unknown-source"))
        source_type = str(source.get("type", "rss")).lower()
        source_url = source.get("url", "")
        if not source_url:
            if args.verbose:
                print(f"[skip] {source_name}: missing url", file=sys.stderr)
            continue

        try:
            body = fetch_url(source_url, timeout=args.timeout)
            if source_type == "rss":
                source_items = collect_from_rss(
                    source=source,
                    xml_text=body,
                    now_utc=now_utc,
                    cutoff=cutoff,
                    max_items_per_source=args.max_items_per_source,
                    max_questions_per_entry=args.max_questions_per_entry,
                )
            elif source_type == "html":
                source_items = collect_from_html(
                    source=source,
                    html_text=body,
                    now_utc=now_utc,
                    max_questions_per_source=args.max_questions_per_source,
                )
            else:
                print(f"[skip] {source_name}: unsupported source type '{source_type}'", file=sys.stderr)
                continue

            all_items.extend(source_items)
            if args.verbose:
                print(f"[ok] {source_name}: extracted {len(source_items)} items")
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            print(f"[warn] {source_name}: failed to fetch/parse ({exc})", file=sys.stderr)

    items_for_dedupe = list(all_items)
    if args.incremental:
        existing_items = load_existing_items(output_path)
        if args.verbose:
            print(
                f"[info] merged {len(existing_items)} existing items from {output_path}",
                file=sys.stderr,
            )
        items_for_dedupe = existing_items + all_items

    deduped_items = deduplicate(items_for_dedupe)
    deduped_items.sort(key=lambda item: (item.get("company", ""), item.get("question", "")))

    payload = {
        "metadata": metadata_for(
            deduped_items,
            len(sources),
            now_utc,
            load_mode=load_mode,
            cutoff=cutoff,
            cutoff_reason=cutoff_reason,
        ),
        "items": deduped_items,
    }
    write_output(output_path, payload)

    if not args.no_state_update:
        write_state(
            state_path,
            {
                "last_successful_run": now_utc.isoformat(),
                "last_run_mode": load_mode,
                "last_output_path": str(output_path),
                "last_item_count": len(deduped_items),
                "last_sources_count": len(sources),
            },
        )
        if args.verbose:
            print(f"[info] updated state file: {state_path}", file=sys.stderr)

    print(
        f"Generated knowledge bank at {output_path} with {len(deduped_items)} items from {len(sources)} sources ({load_mode} mode)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
