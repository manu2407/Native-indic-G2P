#!/usr/bin/env python3
"""Exact structured metrics shared by native and neural Hindi G2P."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from neural_data import strata
from parse_prosody import MARKER_PATTERN, apply_source_stress, parse_prosody


FIELDS = ("phonemes", "marker", "weight", "stressed", "extrametrical")


def decode_target(value: str) -> dict:
    ps = value.replace("'", "")
    source = MARKER_PATTERN.sub("", value)
    parsed = parse_prosody(ps)
    parsed["source_phoneme"] = apply_source_stress(parsed, source)
    return parsed


def _metrics(target: dict, prediction: dict) -> dict[str, bool]:
    target_syllables = target["syllables"]
    predicted_syllables = prediction["syllables"]
    phoneme = prediction["phoneme_string"].replace(".", "") == target["phoneme_string"].replace(".", "")
    syllables = [item["phonemes"] for item in predicted_syllables] == [item["phonemes"] for item in target_syllables]
    stress = [item["stressed"] for item in predicted_syllables] == [item["stressed"] for item in target_syllables]
    schema = [tuple(item[field] for field in FIELDS) for item in predicted_syllables] == [
        tuple(item[field] for field in FIELDS) for item in target_syllables
    ]
    return {"phoneme": phoneme, "syllables": syllables, "stress": stress, "schema": schema, "all": phoneme and schema}


def evaluate(records: list[dict], predictor: Callable[[str], dict]) -> dict:
    counts = defaultdict(lambda: defaultdict(int))
    examples = []
    for record in records:
        names = strata(record)
        for name in names:
            counts[name]["records"] += 1
        try:
            prediction = predictor(record["text"])
            matched = _metrics(record, prediction)
        except (KeyError, TypeError, ValueError) as error:
            for name in names:
                counts[name]["rejected"] += 1
            if len(examples) < 20:
                examples.append({"text": record["text"], "error": str(error)})
            continue
        for name in names:
            for metric, value in matched.items():
                counts[name][metric] += value
        if not matched["all"] and len(examples) < 20:
            examples.append({"text": record["text"], "target": record["ps"], "prediction": prediction["ps"]})
    report = {}
    for name, values in counts.items():
        denominator = values["records"]
        report[name] = {
            "counts": dict(values),
            "rates_percent": {
                metric: round(100 * values[metric] / denominator, 3)
                for metric in ("phoneme", "syllables", "stress", "schema", "all", "rejected")
            },
        }
    return {"strata": report, "mismatch_examples": examples}
