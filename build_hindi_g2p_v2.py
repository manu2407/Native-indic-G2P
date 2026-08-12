#!/usr/bin/env python3
"""Build a strict-Devanagari, checksummed neural G2P dataset from one corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from label_pilot import label_pilot
from neural_data import load_records, load_training, strata, word_hash
from prepare_data import clean_corpus, sha256
from split_words import split_word_types


PROJECT_ROOT = Path(__file__).resolve().parent


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def split_metadata(path: Path) -> dict:
    records = load_records(path, strict_devanagari=True)
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(strata(record))
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "records": len(records),
        "words_sha256": word_hash(records),
        "strata": dict(counts),
    }


def build(arguments: argparse.Namespace) -> dict:
    name = arguments.name
    processed = PROJECT_ROOT / "datasets" / "processed" / f"{name}_clean"
    split = PROJECT_ROOT / "datasets" / "validation" / f"{name}_split"
    labels = PROJECT_ROOT / "datasets" / "tier_a" / name
    neural = PROJECT_ROOT / "datasets" / "validation" / f"neural_{name}"
    if any(path.exists() for path in (processed, split, labels, neural)):
        raise FileExistsError(f"v2 output already exists for {name!r}; choose a new --name")

    clean_corpus(
        arguments.input,
        processed,
        min_purity=1.0,
        max_records=arguments.max_records,
        strict_devanagari=True,
    )
    split_word_types(
        processed / "data.jsonl",
        split,
        seed=arguments.seed,
        train_percent=arguments.train_percent,
        dev_percent=arguments.dev_percent,
        strict_devanagari=True,
    )
    label_pilot(split, labels, arguments.upstream)

    train_path = labels / "train" / "labels.jsonl"
    training = load_training([train_path], strict_devanagari=True)
    manifest = {
        "version": "neural-g2p-eval-v2",
        "status": "sealed-before-neural-training",
        "eligibility": "nonempty Devanagari letters/marks; target excludes X, 3, 4, ^, and ~",
        "strict_devanagari": True,
        "target": "one compact PS sequence with realized-stress apostrophes after syllable markers",
        "training_sources": [{"path": relative(train_path), "sha256": sha256(train_path)}],
        "training": {"records": len(training), "words_sha256": word_hash(training)},
        "dev": split_metadata(labels / "dev" / "labels.jsonl"),
        "blind": split_metadata(labels / "blind" / "labels.jsonl"),
        "source": {
            "clean_manifest": relative(processed / "manifest.json"),
            "split_manifest": relative(split / "split_manifest.json"),
            "label_run_manifest": relative(labels / "run_manifest.json"),
        },
    }
    neural.mkdir(parents=True)
    manifest_path = neural / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(training) < arguments.minimum_training_words:
        raise RuntimeError(
            f"only {len(training):,} accepted training words; need {arguments.minimum_training_words:,}. "
            "Keep this dataset for provenance, then collect more source text under a new --name."
        )
    return {"manifest": relative(manifest_path), "training_words": len(training)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="licensed UTF-8 Hindi corpus text or JSONL")
    parser.add_argument("--upstream", type=Path, required=True, help="pinned Hindi-word-prosody checkout")
    parser.add_argument("--name", default="hindi_g2p_v2_1m")
    parser.add_argument("--max-records", type=int, help="accepted source-sentence ceiling")
    parser.add_argument("--minimum-training-words", type=int, default=1_000_000)
    parser.add_argument("--seed", default="NATIVE-INDIC-G2P-V2")
    parser.add_argument("--train-percent", type=int, default=90)
    parser.add_argument("--dev-percent", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.minimum_training_words <= 0:
        raise SystemExit("--minimum-training-words must be positive")
    print(json.dumps(build(arguments), ensure_ascii=False))
