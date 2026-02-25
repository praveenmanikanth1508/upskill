#!/usr/bin/env python3
"""Search the generated interview knowledge bank JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query interview knowledge bank.")
    parser.add_argument("--bank", required=True, help="Path to bank JSON file.")
    parser.add_argument("--company", help="Filter by company (case-insensitive).")
    parser.add_argument("--tag", action="append", default=[], help="Filter by tag (repeatable).")
    parser.add_argument("--keyword", help="Filter by question keyword.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of results.")
    return parser.parse_args(argv)


def load_bank(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def matches(item: dict[str, Any], company: str | None, tags: list[str], keyword: str | None) -> bool:
    if company:
        if str(item.get("company", "")).lower() != company.lower():
            return False

    item_tags = {str(tag).lower() for tag in item.get("tags", [])}
    if tags:
        for tag in tags:
            if tag.lower() not in item_tags:
                return False

    if keyword:
        question = str(item.get("question", "")).lower()
        if keyword.lower() not in question:
            return False

    return True


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bank_path = Path(args.bank)
    if not bank_path.exists():
        print(f"Bank file not found: {bank_path}", file=sys.stderr)
        return 2

    try:
        payload = load_bank(bank_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read bank: {exc}", file=sys.stderr)
        return 2

    items = payload.get("items", [])
    if not isinstance(items, list):
        print("Invalid bank format: 'items' must be a list.", file=sys.stderr)
        return 2

    selected: list[dict[str, Any]] = []
    for item in items:
        if matches(item, company=args.company, tags=args.tag, keyword=args.keyword):
            selected.append(item)
            if len(selected) >= args.limit:
                break

    print(f"Found {len(selected)} result(s)")
    print("-" * 80)
    for idx, item in enumerate(selected, start=1):
        source = item.get("source", {})
        print(f"{idx}. {item.get('question', '')}")
        print(f"   company: {item.get('company', 'unknown')}")
        print(f"   tags: {', '.join(item.get('tags', []))}")
        print(f"   source: {source.get('entry_url', source.get('source_url', ''))}")
        print(f"   hint: {item.get('answer_hint', '')}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
