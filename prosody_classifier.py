"""Runtime features and artifact validation for learned Hindi prosody."""

from __future__ import annotations

import json
import hashlib
import inspect
import math
from functools import lru_cache
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("models") / "prosody_perceptron_v1.json"
FEATURE_SCHEMA = "prosody-context-v2-independent-markers"
EXPECTED_MODEL_SHA256 = "a1e59e1a20b1cc876f92308734ba85aa7e0eaf9bf36771cb77ebc9aa44a6aa3c"


def features(word: str, syllables: list[dict], index: int) -> tuple[str, ...]:
    current = syllables[index]
    text = current["phonemes"]
    result = [
        "bias",
        f"position={min(index, 6)}",
        f"position_right={min(len(syllables) - index - 1, 6)}",
        f"syllable_count={min(len(syllables), 8)}",
        f"weight={current['weight']}",
        f"shape={text}",
        f"length={min(len(text), 10)}",
        f"weight_pattern={''.join((item['weight'] or '?')[0] for item in syllables)}",
    ]
    for width in range(1, 5):
        result.extend(
            (
                f"word_prefix{width}={word[:width]}",
                f"word_suffix{width}={word[-width:]}",
                f"syllable_prefix{width}={text[:width]}",
                f"syllable_suffix{width}={text[-width:]}",
            )
        )
    for offset in range(-2, 3):
        position = index + offset
        if 0 <= position < len(syllables):
            neighbour = syllables[position]
            result.extend(
                (
                    f"s{offset}={neighbour['phonemes']}",
                    f"w{offset}={neighbour['weight']}",
                )
            )
        else:
            result.append(f"s{offset}={'<' if position < 0 else '>'}")
    return tuple(result)


def feature_contract_sha256() -> str:
    return hashlib.sha256(inspect.getsource(features).encode("utf-8")).hexdigest()


def predict(model: dict, word: str, syllables: list[dict], index: int) -> tuple[str, bool]:
    items = features(word, syllables, index)
    weight_score = sum(model["weight_weights"].get(item, 0.0) for item in items)
    marker_score = sum(model["marker_weights"].get(item, 0.0) for item in items)
    stress_score = sum(model["stress_weights"].get(item, 0.0) for item in items)
    marker = "ʷ" if weight_score < 0.0 else "ˢʰ" if marker_score >= 0.0 else "ʰ"
    stressed = False if marker == "ʷ" else True if marker == "ˢʰ" else stress_score >= 0.0
    return marker, stressed


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    try:
        serialized = path.read_bytes()
        model = json.loads(serialized)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load prosody model {path}: {error}") from error
    if path == MODEL_PATH and hashlib.sha256(serialized).hexdigest() != EXPECTED_MODEL_SHA256:
        raise ValueError(f"prosody model integrity check failed: {path}")
    if not isinstance(model, dict) or model.get("format") != "hindi-prosody-perceptron-v1":
        raise ValueError(f"unsupported prosody model format in {path}")
    if model.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(f"prosody model feature schema does not match runtime: {path}")
    if model.get("feature_contract_sha256") != feature_contract_sha256():
        raise ValueError(f"prosody model feature contract does not match runtime: {path}")
    training = model.get("training")
    pipeline = model.get("pipeline")
    if (
        not isinstance(training, dict)
        or training.get("records") != 67110
        or training.get("aligned_records") != 57447
        or training.get("epochs") != 4
        or training.get("seed") != 11
        or not isinstance(training.get("sources"), list)
        or len(training["sources"]) != 2
        or not isinstance(pipeline, dict)
        or pipeline.get("schwa_model_sha256") != "a6bfa53fe966e5c98102801d36197917d8a45c3500e0af32e233daebebd68751"
        or pipeline.get("native_g2p_sha256")
        != hashlib.sha256(Path(__file__).with_name("native_g2p.py").read_bytes()).hexdigest()
    ):
        raise ValueError(f"prosody model has invalid training provenance: {path}")
    for name in ("weight_weights", "marker_weights", "stress_weights"):
        weights = model.get(name)
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"prosody model has no {name}: {path}")
        if any(
            not isinstance(item, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for item, value in weights.items()
        ):
            raise ValueError(f"prosody model contains invalid {name}: {path}")
    return model


def score(weights: dict[str, float], word: str, syllables: list[dict], index: int) -> float:
    return sum(weights.get(item, 0.0) for item in features(word, syllables, index))
