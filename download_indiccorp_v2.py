#!/usr/bin/env python3
"""Download one newline-aligned, pinned IndicCorpV2 Hindi byte range."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

from prepare_data import sha256


DEFAULT_URL = (
    "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/"
    "cd4a6c99b522321a22aa50c798f0199f691eb746/data/hi-1.txt"
)
DEFAULT_REVISION = "cd4a6c99b522321a22aa50c798f0199f691eb746"


def download_range(url: str, output_dir: Path, start_byte: int, byte_count: int) -> dict:
    if start_byte < 0 or byte_count <= 0:
        raise ValueError("start byte must be non-negative and byte count must be positive")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    end_byte = start_byte + byte_count - 1
    destination = output_dir / "hi-1.txt"
    request = urllib.request.Request(url, headers={"Range": f"bytes={start_byte}-{end_byte}"})
    try:
        with urllib.request.urlopen(request) as response, tempfile.NamedTemporaryFile(
            dir=output_dir, prefix="range-", delete=False
        ) as temporary:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise RuntimeError(f"server did not honor byte range: HTTP {status}")
            while block := response.read(1024 * 1024):
                temporary.write(block)
            temporary_path = Path(temporary.name)
    except Exception:
        for path in output_dir.iterdir():
            path.unlink()
        output_dir.rmdir()
        raise

    with temporary_path.open(encoding="utf-8", errors="ignore") as source, destination.open("w", encoding="utf-8") as output:
        if start_byte:
            source.readline()
        for line in source:
            if line.endswith("\n"):
                output.write(line)
    temporary_path.unlink()
    manifest = {
        "dataset": "ai4bharat/IndicCorpV2",
        "dataset_revision": DEFAULT_REVISION,
        "license": "CC0-1.0",
        "source_file": "data/hi-1.txt",
        "source_url": url,
        "requested_byte_range": f"{start_byte}-{end_byte}",
        "local_file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "line_alignment": "first and final partial lines removed",
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for source text and manifest")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--start-byte", type=int, default=0)
    parser.add_argument("--byte-count", type=int, default=1_500_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(download_range(arguments.url, arguments.output, arguments.start_byte, arguments.byte_count)))
