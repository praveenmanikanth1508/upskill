#!/usr/bin/env python3
"""Export a static bundle for GitHub Pages deployment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static site bundle from web assets and bank JSON.")
    parser.add_argument("--web-dir", default="web", help="Directory containing static web assets.")
    parser.add_argument("--bank", default="knowledge_bank/interview_bank.json", help="Path to bank JSON.")
    parser.add_argument("--output-dir", default="site", help="Output directory for static site artifact.")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Bank payload must be a JSON object.")
    if not isinstance(payload.get("items", []), list):
        raise ValueError("Bank payload must contain an 'items' list.")
    return payload


def copy_web_assets(web_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ("index.html", "styles.css", "app.js"):
        source = web_dir / file_name
        if not source.exists():
            raise FileNotFoundError(f"Required web file not found: {source}")
        shutil.copy2(source, output_dir / file_name)


def write_bank_data(payload: dict[str, Any], output_dir: Path) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "interview_bank.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return output_path


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    web_dir = Path(args.web_dir)
    bank_path = Path(args.bank)
    output_dir = Path(args.output_dir)

    if not web_dir.exists():
        print(f"Web directory not found: {web_dir}", file=sys.stderr)
        return 2
    if not bank_path.exists():
        print(f"Bank JSON not found: {bank_path}", file=sys.stderr)
        return 2

    try:
        payload = load_json(bank_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to parse bank JSON: {exc}", file=sys.stderr)
        return 2

    if output_dir.exists():
        shutil.rmtree(output_dir)

    copy_web_assets(web_dir, output_dir)
    written_path = write_bank_data(payload, output_dir)
    print(f"Static bundle generated at: {output_dir}")
    print(f"Data file: {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
