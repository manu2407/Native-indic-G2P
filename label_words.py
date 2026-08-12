#!/usr/bin/env python3
"""Run a user-supplied Hindi-word-prosody checkout headlessly on a word list."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

from parse_prosody import apply_source_stress, parse_prosody


FIELDS = (
    "Hindi Input",
    "IPA Equivalent",
    "Underlying Phonemic Form",
    "I-Level Syllabification",
    "Prosodic Label(PLSB)",
    "Phoneme Level(IPA)",
    "Phoneme Level(ASCII)",
)
UPSTREAM_MODULES = (
    "IPAEquv",
    "Labelchang1",
    "Labelchang2",
    "Syll_label",
    "Syllabification",
    "countvc",
    "phoneme",
)
LEGACY_NUKTA = {
    "क़": "क़",
    "ख़": "ख़",
    "ग़": "ग़",
    "ज़": "ज़",
    "ड़": "ड़",
    "ढ़": "ढ़",
    "फ़": "फ़",
}
EXPECTED_UPSTREAM_REVISION = "ac01da28ff2801a6a7e94829efbfe14b1c8179dd"


class Entry:
    """The three Tkinter Entry methods used by the upstream algorithm."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def delete(self, *_: object) -> None:
        self.value = ""

    def insert(self, _: object, value: str) -> None:
        self.value = value


def convert_source(source: Path, destination: Path) -> None:
    required = [source / f"{module}.py" for module in UPSTREAM_MODULES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"upstream checkout is missing: {', '.join(missing)}")

    destination.mkdir()
    for path in required:
        shutil.copy2(path, destination / path.name)
    try:
        subprocess.run(
            [sys.executable, "-m", "lib2to3", "-w", "-n", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Python 2 conversion failed: {error.stderr.strip()}") from error

    removed_imports = {
        "import tkinter.filedialog",
        "import tkinter.font",
        "import xlrd",
        "import xlsxwriter",
    }
    for path in destination.glob("*.py"):
        lines = []
        for line in path.read_text(encoding="utf-8").expandtabs(8).splitlines():
            stripped = line.strip()
            if stripped == "from tkinter import *":
                lines.append("END = 'end'")
            elif stripped in removed_imports or "setdefaultencoding" in stripped or stripped == "importlib.reload(sys)":
                continue
            else:
                lines.append(line)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_upstream(upstream: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot verify upstream checkout: {error}") from error
    if revision != EXPECTED_UPSTREAM_REVISION:
        raise RuntimeError(f"upstream revision must be {EXPECTED_UPSTREAM_REVISION}, got {revision}")
    if status:
        raise RuntimeError("upstream checkout must be clean")
    return revision


@contextmanager
def load_upstream(upstream: Path) -> Iterable[Callable[[dict[str, Entry]], None]]:
    verify_upstream(upstream)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import lib2to3  # noqa: F401  # Removed in Python 3.13.
    except ImportError as error:
        raise RuntimeError("headless conversion requires Python 3.12 or older") from error

    with tempfile.TemporaryDirectory(prefix="hindi-word-prosody-") as directory:
        converted = Path(directory) / "src"
        convert_source(upstream / "src", converted)
        sys.path.insert(0, str(converted))
        try:
            for module in UPSTREAM_MODULES:
                sys.modules.pop(module, None)
            yield importlib.import_module("phoneme").Phoneme
        finally:
            sys.path.remove(str(converted))
            for module in UPSTREAM_MODULES:
                sys.modules.pop(module, None)


def label_word(word: str, phoneme: Callable[[dict[str, Entry]], None]) -> dict:
    word = unicodedata.normalize("NFC", word.strip())
    if not word or any(char.isspace() for char in word):
        raise ValueError("input must contain exactly one non-empty word per line")

    entries = {field: Entry() for field in FIELDS}
    entries["Hindi Input"].value = legacy_nukta(word)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        phoneme(entries)

    ps = entries["Prosodic Label(PLSB)"].get().strip()
    source_phoneme = unicodedata.normalize("NFC", entries["Phoneme Level(IPA)"].get().strip())
    parsed = parse_prosody(ps)
    apply_source_stress(parsed, source_phoneme)
    return {"text": word, **parsed, "source_phoneme": source_phoneme}


def legacy_nukta(word: str) -> str:
    for decomposed, legacy in LEGACY_NUKTA.items():
        word = word.replace(decomposed, legacy)
    return word


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 file containing one Hindi word per line")
    parser.add_argument("--upstream", type=Path, required=True, help="Hindi-word-prosody checkout")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    rejected = 0
    with load_upstream(arguments.upstream) as upstream_phoneme, arguments.input.open(encoding="utf-8") as words:
        for line_number, word in enumerate(words, 1):
            try:
                print(json.dumps(label_word(word, upstream_phoneme), ensure_ascii=False))
            except Exception as error:
                rejected += 1
                print(
                    json.dumps({"line": line_number, "text": word.strip(), "error": str(error)}, ensure_ascii=False),
                    file=sys.stderr,
                )
    if rejected:
        raise SystemExit(1)
