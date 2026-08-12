#!/usr/bin/env python3
"""Parse Hindi-word-prosody PLSB/PS output into JSON records."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


MARKERS = {
    "ʷ": {"weight": "light", "stressed": False},
    "ʰ": {"weight": "heavy", "stressed": None},
    "ˢʰ": {"weight": "heavy", "stressed": True},
}
UNMARKED = {"marker": None, "weight": None, "stressed": None, "extrametrical": False}
MARKER_PATTERN = re.compile(r"σ(ˢʰ|ʷ|ʰ)")


def normalize_grapheme(value: str) -> str:
    return unicodedata.normalize("NFC", "".join(char for char in value.strip() if unicodedata.category(char) != "Cf"))


def parse_prosody(ps: str) -> dict:
    ps = unicodedata.normalize("NFC", ps.strip())
    if not ps:
        raise ValueError("prosody string is empty")

    matches = list(MARKER_PATTERN.finditer(ps))
    recognized = MARKER_PATTERN.sub("", ps)
    if "σ" in recognized:
        raise ValueError(f"unknown syllable marker in {ps!r}")
    if not matches:
        parts = ps.split(".")
        if any(not part for part in parts):
            raise ValueError(f"empty dot-delimited syllable in {ps!r}")
        return {
            "phoneme_string": ps,
            "syllables": [{"phonemes": part, **UNMARKED} for part in parts],
            "ps": ps,
        }
    if matches[0].start() != 0:
        raise ValueError(f"phonemes precede the first syllable marker in {ps!r}")

    syllables = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(ps)
        body = ps[match.end() : end]
        if not body:
            raise ValueError(f"empty syllable in {ps!r}")
        parts = body.split(".")
        if not parts[0] and index == 0:
            raise ValueError(f"empty dot-delimited syllable in {ps!r}")
        if not parts[0]:
            parts = parts[1:]
        if not parts or not parts[0] or any(not part for part in parts[1:-1]):
            raise ValueError(f"empty dot-delimited syllable in {ps!r}")
        marker = match.group(1)
        syllables.append(
            {
                "phonemes": parts[0],
                "marker": marker,
                **MARKERS[marker],
                "extrametrical": False,
            }
        )
        syllables.extend({"phonemes": part, **UNMARKED} for part in parts[1:] if part)

    syllables[-1]["extrametrical"] = syllables[-1]["marker"] == "ʰ"
    return {"phoneme_string": MARKER_PATTERN.sub("", ps), "syllables": syllables, "ps": ps}


def apply_source_stress(parsed: dict, source_phoneme: str) -> str:
    source = unicodedata.normalize("NFC", source_phoneme.strip())
    stress_positions = {index - source[:index].count("'") for index, char in enumerate(source) if char == "'"}
    if source.replace("'", "") != parsed["phoneme_string"]:
        raise ValueError("source Phoneme does not match marker-free PLSB/PS")
    stress_positions = {
        position + 1 if parsed["phoneme_string"][position : position + 1] == "." else position
        for position in stress_positions
    }

    boundary = 0
    boundaries = []
    for syllable in parsed["syllables"]:
        boundaries.append(boundary)
        boundary += len(syllable["phonemes"])
        if boundary < len(parsed["phoneme_string"]) and parsed["phoneme_string"][boundary] == ".":
            boundary += 1
    if not stress_positions <= set(boundaries):
        raise ValueError("source stress apostrophe is not on a syllable boundary")
    for boundary, syllable in zip(boundaries, parsed["syllables"]):
        if syllable["marker"] is None and boundary not in stress_positions:
            syllable["stressed"] = None
        else:
            syllable["stressed"] = boundary in stress_positions
    return source


def parse_lexicon(path: Path, rejects: list[dict] | None = None) -> Iterable[dict]:
    try:
        root = ET.fromstring(f"<lexicon>{path.read_text(encoding='utf-8')}</lexicon>")
    except ET.ParseError as error:
        raise ValueError(f"invalid lexicon XML fragments in {path}: {error}") from error

    for index, lexeme in enumerate(root.findall("lexeme"), 1):
        grapheme = lexeme.findtext("Grapheme")
        ps = lexeme.findtext("PLSB")
        if ps is None:
            ps = lexeme.findtext("PS")
        text = normalize_grapheme(grapheme or "")
        if not text or ps is None:
            error = f"lexeme {index} must contain non-empty Grapheme and PLSB/PS"
            if rejects is None:
                raise ValueError(error)
            rejects.append({"lexeme": index, "text": grapheme, "ps": ps, "error": error})
            continue
        try:
            parsed = parse_prosody(ps)
            source_phoneme = lexeme.findtext("Phoneme")
            if source_phoneme is not None:
                source_phoneme = apply_source_stress(parsed, source_phoneme)
            yield {"text": text, **parsed, "source_phoneme": source_phoneme}
        except ValueError as error:
            if rejects is None:
                raise ValueError(f"lexeme {index} ({text}): {error}") from error
            rejects.append({"lexeme": index, "text": text, "ps": ps, "error": str(error)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PLS XML fragment file")
    return parser.parse_args()


if __name__ == "__main__":
    rejected: list[dict] = []
    for record in parse_lexicon(parse_args().input, rejected):
        print(json.dumps(record, ensure_ascii=False))
    for rejection in rejected:
        print(json.dumps(rejection, ensure_ascii=False), file=sys.stderr)
    if rejected:
        raise SystemExit(1)
