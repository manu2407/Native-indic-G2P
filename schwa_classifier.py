"""Runtime feature extraction for the contextual schwa model."""

from __future__ import annotations

import json
import hashlib
import inspect
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


MODEL_PATH = Path(__file__).with_name("models") / "schwa_perceptron_v1.json"
FEATURE_SCHEMA = "context-v3-orthographic-position"
EXPECTED_MODEL_SHA256 = "a6bfa53fe966e5c98102801d36197917d8a45c3500e0af32e233daebebd68751"


def features(word: str, phones: tuple[Any, ...], index: int) -> tuple[str, ...]:
    """Describe one inherent schwa using local segment and word-edge context."""

    result = ["bias"]
    source_index = phones[index].source_index
    graphemes = {}
    for offset in range(-4, 5):
        position = source_index + offset
        grapheme = word[position] if 0 <= position < len(word) else ("<" if position < 0 else ">")
        graphemes[offset] = grapheme
        result.append(f"g{offset}={grapheme}")
    for radius in range(1, 5):
        result.append(
            f"gwindow{radius}="
            + "/".join(graphemes[offset] for offset in range(-radius, radius + 1))
        )
    context = {}
    for offset in range(-4, 5):
        if offset == 0:
            continue
        position = index + offset
        token = phones[position].text if 0 <= position < len(phones) else ("<" if position < 0 else ">")
        context[offset] = token
        result.append(f"p{offset}={token}")
    for radius in range(1, 5):
        window = "/".join(context[offset] for offset in range(-radius, radius + 1) if offset)
        result.append(f"window{radius}={window}")
    result.extend(
        (
            f"left_pair={context[-2]}/{context[-1]}",
            f"right_pair={context[1]}/{context[2]}",
            f"across={context[-1]}/{context[1]}",
        )
    )
    for width in range(1, 5):
        result.append(f"prefix{width}={word[:width]}")
        result.append(f"suffix{width}={word[-width:]}")
    schwas = [position for position, phone in enumerate(phones) if phone.inherent]
    ordinal = schwas.index(index)
    result.extend(
        (
            f"phone_from_left={min(index, 8)}",
            f"phone_from_right={min(len(phones) - index - 1, 8)}",
            f"schwa_from_left={min(ordinal, 5)}",
            f"schwa_from_right={min(len(schwas) - ordinal - 1, 5)}",
            f"schwa_count={min(len(schwas), 6)}",
            f"word_length={min(len(word), 12)}",
            f"edge_context={min(ordinal, 3)}/{context[-1]}/{context[1]}/{min(len(schwas) - ordinal - 1, 3)}",
        )
    )
    for width in range(1, 4):
        result.append(f"suffix_context{width}={word[-width:]}/{context[-1]}/{context[1]}")
    for side, position in (("left", index - 1), ("right", index + 1)):
        if 0 <= position < len(phones):
            phone = phones[position]
            result.extend(
                (
                    f"{side}_kind={phone.kind}",
                    f"{side}_group={phone.group or '-'}",
                    f"{side}_place={phone.place or '-'}",
                    f"{side}_joined={phone.joined}",
                )
            )
    return tuple(result)


def feature_contract_sha256() -> str:
    return hashlib.sha256(inspect.getsource(features).encode("utf-8")).hexdigest()


def predict_retained(weights: dict[str, float], word: str, phones: tuple[Any, ...]) -> dict[int, bool]:
    candidates = [index for index, phone in enumerate(phones) if phone.inherent]
    scores = {index: score(weights, word, phones, index) for index in candidates}
    retained = {index: value >= 0.0 for index, value in scores.items()}
    if scores and not any(
        phone.kind == "vowel" and (phone.text != "ə" or retained.get(index, True))
        for index, phone in enumerate(phones)
    ):
        retained[max(scores, key=scores.get)] = True
    return retained


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    try:
        serialized = path.read_bytes()
        model = json.loads(serialized)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load schwa model {path}: {error}") from error
    if path == MODEL_PATH and hashlib.sha256(serialized).hexdigest() != EXPECTED_MODEL_SHA256:
        raise ValueError(f"schwa model integrity check failed: {path}")
    if not isinstance(model, dict) or model.get("format") != "hindi-schwa-perceptron-v1":
        raise ValueError(f"unsupported schwa model format in {path}")
    if model.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(f"schwa model feature schema does not match runtime: {path}")
    if model.get("feature_contract_sha256") != feature_contract_sha256():
        raise ValueError(f"schwa model feature contract does not match runtime: {path}")
    training = model.get("training")
    if (
        not isinstance(training, dict)
        or training.get("records") != 67110
        or training.get("epochs") != 12
        or training.get("seed") != 7
        or not isinstance(training.get("sources"), list)
        or len(training["sources"]) != 2
    ):
        raise ValueError(f"schwa model has invalid training provenance: {path}")
    weights = model.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"schwa model has no weights: {path}")
    if any(
        not isinstance(item, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for item, value in weights.items()
    ):
        raise ValueError(f"schwa model contains invalid weights: {path}")
    return model


def score(weights: dict[str, float], word: str, phones: tuple[Any, ...], index: int) -> float:
    return sum(weights.get(item, 0.0) for item in features(word, phones, index))
