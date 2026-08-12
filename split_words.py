#!/usr/bin/env python3
"""Create deterministic, disjoint train/dev/blind Hindi word lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from prepare_data import sha256


def words(text: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum() or unicodedata.category(char).startswith("M"):
            current.append(char)
        elif current:
            result.append("".join(current))
            current.clear()
    if current:
        result.append("".join(current))
    return result


def is_devanagari_word(word: str) -> bool:
    return bool(word) and all(
        "\u0900" <= char <= "\u097f" and unicodedata.category(char)[0] in "LM" for char in word
    )


def split_for(word: str, seed: str, train_percent: int, dev_percent: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}\0{word}".encode()).digest()[:8], "big") % 100
    if value < train_percent:
        return "train"
    if value < train_percent + dev_percent:
        return "dev"
    return "blind"


def split_word_types(
    input_path: Path,
    output_dir: Path,
    *,
    seed: str = "BLIND-HI-G2P-v1",
    train_percent: int = 90,
    dev_percent: int = 5,
    strict_devanagari: bool = False,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if train_percent <= 0 or dev_percent <= 0 or train_percent + dev_percent >= 100:
        raise ValueError("split percentages must leave a non-empty blind split")

    frequencies: Counter[str] = Counter()
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
                extracted = words(record["text"])
                frequencies.update(word for word in extracted if not strict_devanagari or is_devanagari_word(word))
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{input_path}:{line_number}: invalid clean record: {error}") from error

    output_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    file_metadata: dict[str, dict] = {}
    for split in ("train", "dev", "blind"):
        split_dir = output_dir / split
        split_dir.mkdir()
        split_path = split_dir / "words.jsonl"
        with split_path.open("w", encoding="utf-8") as output:
            for word in sorted(frequencies):
                if split_for(word, seed, train_percent, dev_percent) != split:
                    continue
                output.write(json.dumps({"word": word, "frequency": frequencies[word]}, ensure_ascii=False) + "\n")
                counts[split] += 1
        relative_path = str(split_path.relative_to(output_dir))
        file_metadata[relative_path] = {"sha256": sha256(split_path), "word_types": counts[split]}

    manifest = {
        "version": "blind_hi_g2p_v1",
        "source": str(input_path.resolve()),
        "source_sha256": sha256(input_path),
        "seed": seed,
        "percentages": {
            "train": train_percent,
            "dev": dev_percent,
            "blind": 100 - train_percent - dev_percent,
        },
        "strict_devanagari": strict_devanagari,
        "word_type_counts": dict(counts),
        "files": file_metadata,
        # ponytail: challenge strata belong after the labeler exposes linguistic features.
        "challenge_strata": "pending",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="clean data.jsonl")
    parser.add_argument("output", type=Path, help="new split directory")
    parser.add_argument("--seed", default="BLIND-HI-G2P-v1")
    parser.add_argument("--train-percent", type=int, default=90)
    parser.add_argument("--dev-percent", type=int, default=5)
    parser.add_argument("--strict-devanagari", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = split_word_types(
        arguments.input,
        arguments.output,
        seed=arguments.seed,
        train_percent=arguments.train_percent,
        dev_percent=arguments.dev_percent,
        strict_devanagari=arguments.strict_devanagari,
    )
    print(json.dumps(result["word_type_counts"], ensure_ascii=False))
