#!/usr/bin/env python3
"""Label deterministic pilot splits and record the complete provenance chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from label_words import EXPECTED_UPSTREAM_REVISION, label_word, load_upstream
from prepare_data import sha256


SPLITS = ("train", "dev", "blind")
PROJECT_FILES = ("label_pilot.py", "label_words.py", "parse_prosody.py")
SPLIT_INPUTS = {f"{split}/words.jsonl" for split in SPLITS}


def label_pilot(split_root: Path, output_root: Path, upstream: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    split_manifest = split_root / "split_manifest.json"
    if not split_manifest.is_file():
        raise FileNotFoundError(f"missing split manifest: {split_manifest}")
    split_manifest_bytes = split_manifest.read_bytes()
    split_manifest_sha256 = hashlib.sha256(split_manifest_bytes).hexdigest()
    split_metadata = json.loads(split_manifest_bytes.decode("utf-8"))
    if set(split_metadata.get("files", {})) != SPLIT_INPUTS:
        raise ValueError("split manifest must contain exactly the train/dev/blind word files")
    inputs = {}
    validated_words = {}
    for split in SPLITS:
        relative = f"{split}/words.jsonl"
        source = split_root / relative
        metadata = split_metadata["files"][relative]
        source_bytes = source.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if metadata.get("sha256") != source_sha256:
            raise ValueError(f"split file does not match its manifest: {relative}")
        records = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()]
        expected_count = split_metadata["word_type_counts"].get(split)
        if len(records) != expected_count or len(records) != metadata.get("word_types"):
            raise ValueError(f"split word count does not match its manifest: {split}")
        words = [record["word"] for record in records]
        if not all(isinstance(word, str) and word for word in words):
            raise ValueError(f"{source}: every word must be a non-empty string")
        inputs[relative] = {"sha256": source_sha256, "word_types": len(words)}
        validated_words[split] = words
    revision = EXPECTED_UPSTREAM_REVISION
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_root.name + "-", dir=output_root.parent))
    files = {}
    try:
        with load_upstream(upstream) as phoneme:
            for split in SPLITS:
                words = validated_words[split]
                destination = temporary / split
                destination.mkdir()
                words_path = destination / "words.txt"
                labels_path = destination / "labels.jsonl"
                rejects_path = destination / "rejects.jsonl"
                words_path.write_text("".join(word + "\n" for word in words), encoding="utf-8")
                with labels_path.open("w", encoding="utf-8") as labels, rejects_path.open(
                    "w", encoding="utf-8"
                ) as rejects:
                    for line_number, word in enumerate(words, 1):
                        try:
                            labels.write(json.dumps(label_word(word, phoneme), ensure_ascii=False) + "\n")
                        except Exception as error:
                            rejects.write(
                                json.dumps({"line": line_number, "text": word, "error": str(error)}, ensure_ascii=False)
                                + "\n"
                            )
                for path in (words_path, labels_path, rejects_path):
                    files[str(path.relative_to(temporary))] = {"sha256": sha256(path), "bytes": path.stat().st_size}

        manifest = {
            "version": "hindi_g2p_pilot_run_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code": {
                name: {"sha256": sha256(Path(__file__).with_name(name))}
                for name in PROJECT_FILES
            },
            "runtime": {
                "implementation": sys.implementation.name,
                "version_info": list(sys.version_info[:3]),
                "python": sys.version,
            },
            "upstream": {"path": str(upstream.resolve()), "revision": revision, "clean": True},
            "split_manifest": {"path": str(split_manifest.resolve()), "sha256": split_manifest_sha256},
            "inputs": inputs,
            "files": files,
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = label_pilot(arguments.split_root, arguments.output_root, arguments.upstream)
    print(json.dumps({"revision": result["upstream"]["revision"], "files": len(result["files"])}))
