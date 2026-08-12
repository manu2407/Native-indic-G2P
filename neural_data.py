#!/usr/bin/env python3
"""Shared, frozen data contract for neural Hindi G2P experiments."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path


INVALID_TARGET_CHARACTERS = frozenset("X34^~")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(record: dict, *, strict_devanagari: bool = False) -> bool:
    text = record["text"]
    target = record["ps"] + record["source_phoneme"]
    return (
        bool(text)
        and all(unicodedata.category(char)[0] in "LM" for char in text)
        and (not strict_devanagari or all("\u0900" <= char <= "\u097f" for char in text))
        and not (
        set(target) & INVALID_TARGET_CHARACTERS
        )
    )


def target(record: dict) -> str:
    syllables = record["syllables"]
    parts = []
    for syllable in syllables:
        marker = syllable["marker"]
        boundary = "σ" + marker if marker is not None else "" if len(syllables) == 1 else "."
        parts.append(boundary + ("'" if syllable["stressed"] else "") + syllable["phonemes"])
    return "".join(parts)


def load_records(path: Path, *, strict_devanagari: bool = False) -> list[dict]:
    return [
        record
        for line in path.read_text(encoding="utf-8").splitlines()
        if eligible(record := json.loads(line), strict_devanagari=strict_devanagari)
    ]


def load_training(paths: list[Path], *, strict_devanagari: bool = False) -> list[dict]:
    unique: dict[str, dict] = {}
    for path in paths:
        for record in load_records(path, strict_devanagari=strict_devanagari):
            previous = unique.get(record["text"])
            if previous and target(previous) != target(record):
                raise ValueError(f"conflicting neural targets for {record['text']!r}")
            unique.setdefault(record["text"], record)
    return list(unique.values())


def word_hash(records: list[dict]) -> str:
    payload = "".join(record["text"] + "\n" for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strata(record: dict) -> tuple[str, ...]:
    result = ["all"]
    if len(record["text"]) >= 10:
        result.append("long")
    if "्" in record["text"]:
        result.append("conjunct")
    if any(char in record["text"] for char in "ंँ"):
        result.append("nasal")
    if "़" in record["text"]:
        result.append("nukta")
    if len(record["syllables"]) >= 4:
        result.append("four_plus_syllables")
    return tuple(result)


def verify_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[3]
    training_paths = [root / source["path"] for source in manifest["training_sources"]]
    for source, source_path in zip(manifest["training_sources"], training_paths):
        if sha256(source_path) != source["sha256"]:
            raise ValueError(f"training source hash mismatch: {source_path}")
    strict_devanagari = manifest.get("strict_devanagari", False)
    training = load_training(training_paths, strict_devanagari=strict_devanagari)
    if len(training) != manifest["training"]["records"] or word_hash(training) != manifest["training"]["words_sha256"]:
        raise ValueError("training partition does not match sealed manifest")
    training_words = {record["text"] for record in training}
    for split in ("dev", "blind"):
        source = manifest[split]
        source_path = root / source["path"]
        if sha256(source_path) != source["sha256"]:
            raise ValueError(f"{split} source hash mismatch: {source_path}")
        records = load_records(source_path, strict_devanagari=strict_devanagari)
        if len(records) != source["records"] or word_hash(records) != source["words_sha256"]:
            raise ValueError(f"{split} partition does not match sealed manifest")
        if training_words & {record["text"] for record in records}:
            raise ValueError(f"training overlaps sealed {split} partition")
        counts = {name: 0 for name in source["strata"]}
        for record in records:
            for name in strata(record):
                counts[name] += 1
        if counts != source["strata"]:
            raise ValueError(f"{split} strata do not match sealed manifest")
    return manifest


if __name__ == "__main__":
    verified = verify_manifest(Path("datasets/validation/neural_g2p_eval_v1/manifest.json"))
    print(json.dumps({split: verified[split]["records"] for split in ("training", "dev", "blind")}, indent=2))
