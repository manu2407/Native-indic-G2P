#!/usr/bin/env python3
"""Clean Hindi text into a deterministic, checksummed JSONL dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO


def open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.replace("\ufeff", "").split())


def devanagari_purity(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    devanagari = sum("\u0900" <= char <= "\u097f" for char in letters)
    return devanagari / len(letters)


def is_devanagari_letter_or_mark(char: str) -> bool:
    return "\u0900" <= char <= "\u097f" and unicodedata.category(char)[0] in "LM"


def has_only_devanagari_letters_or_marks(text: str) -> bool:
    return all(
        is_devanagari_letter_or_mark(char) or unicodedata.category(char)[0] not in "LM" for char in text
    )


def iter_text(path: Path, file_format: str, text_key: str) -> Iterable[str]:
    if file_format == "auto":
        suffixes = path.suffixes[:-1] if path.suffix == ".gz" else path.suffixes
        file_format = "jsonl" if suffixes and suffixes[-1] == ".jsonl" else "plain"

    with open_text(path) as source:
        for line_number, line in enumerate(source, 1):
            if file_format == "plain":
                yield line
                continue
            try:
                record = json.loads(line)
                text = record[text_key]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record: {error}") from error
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: {text_key!r} must be a string")
            yield text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_corpus(
    input_path: Path,
    output_dir: Path,
    *,
    file_format: str = "auto",
    text_key: str = "text",
    min_chars: int = 2,
    max_chars: int = 500,
    min_purity: float = 0.8,
    max_records: int | None = None,
    strict_devanagari: bool = False,
) -> dict:
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    data_path = output_dir / "data.jsonl"
    counts: Counter[str] = Counter()
    seen: set[str] = set()

    with data_path.open("w", encoding="utf-8") as output:
        for raw_text in iter_text(input_path, file_format, text_key):
            counts["read"] += 1
            text = normalize(raw_text)
            reason = None
            if len(text) < min_chars:
                reason = "too_short"
            elif len(text) > max_chars:
                reason = "too_long"
            elif devanagari_purity(text) < min_purity:
                reason = "low_purity"
            elif strict_devanagari and not has_only_devanagari_letters_or_marks(text):
                reason = "non_devanagari_letters_or_marks"
            elif text in seen:
                reason = "duplicate"
            if reason:
                counts[reason] += 1
                continue

            seen.add(text)
            stable_id = "HI-C-" + hashlib.sha256(text.encode()).hexdigest()[:20].upper()
            output.write(json.dumps({"id": stable_id, "text": text}, ensure_ascii=False) + "\n")
            counts["kept"] += 1
            if max_records is not None and counts["kept"] >= max_records:
                break

    manifest = {
        "version": "hindi_clean_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path.resolve()),
        "source_sha256": sha256(input_path),
        "parameters": {
            "format": file_format,
            "text_key": text_key,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_purity": min_purity,
            "max_records": max_records,
            "strict_devanagari": strict_devanagari,
        },
        "counts": dict(sorted(counts.items())),
        "files": {"data.jsonl": {"sha256": sha256(data_path), "records": counts["kept"]}},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 text/JSONL file, optionally gzip-compressed")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--format", choices=("auto", "plain", "jsonl"), default="auto")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--min-purity", type=float, default=0.8)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--strict-devanagari", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_purity <= 1:
        parser.error("--min-purity must be between 0 and 1")
    if args.min_chars < 0 or args.max_chars < args.min_chars:
        parser.error("character limits are invalid")
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = clean_corpus(
        arguments.input,
        arguments.output,
        file_format=arguments.format,
        text_key=arguments.text_key,
        min_chars=arguments.min_chars,
        max_chars=arguments.max_chars,
        min_purity=arguments.min_purity,
        max_records=arguments.max_records,
        strict_devanagari=arguments.strict_devanagari,
    )
    print(json.dumps(result["counts"], ensure_ascii=False))
