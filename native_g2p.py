#!/usr/bin/env python3
"""Experimental clean-room Python 3 Hindi word-prosody labeler.

The processing stages follow Roy (2017), with compact learned classifiers for
contextual schwa and prosody compatibility. It has no dependency on the legacy
Hindi-word-prosody source tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

from prosody_classifier import load_model as load_prosody_model
from prosody_classifier import predict as predict_prosody
from schwa_classifier import load_model as load_schwa_model
from schwa_classifier import predict_retained


VOWELS = {
    "अ": ("ə", False),
    "आ": ("a:", True),
    "इ": ("i", False),
    "ई": ("i:", True),
    "उ": ("u", False),
    "ऊ": ("u:", True),
    "ऋ": ("ri", False),
    "ए": ("e:", True),
    "ऐ": ("æ:", True),
    "ओ": ("o:", True),
    "औ": ("ᴔ:", True),
    "ऑ": ("ɔ:", True),
}
MATRAS = {
    "ा": ("a:", True),
    "ि": ("i", False),
    "ी": ("i:", True),
    "ु": ("u", False),
    "ू": ("u:", True),
    "ृ": ("ri", False),
    "े": ("e:", True),
    "ै": ("æ:", True),
    "ो": ("o:", True),
    "ौ": ("ᴔ:", True),
    "ॅ": ("æ", False),
    "ॉ": ("ɔ:", True),
}


@dataclass(frozen=True)
class Consonant:
    ipa: str
    group: str = "other"
    place: str = "dental"


CONSONANTS = {
    "क": Consonant("k", "stop", "velar"),
    "ख": Consonant("kʰ", "stop", "velar"),
    "ग": Consonant("g", "stop", "velar"),
    # Compatibility alphabet follows the reference lexicon, including its
    # historical IPA code-point choices.
    "घ": Consonant("ɡʱ", "stop", "velar"),
    "ङ": Consonant("ŋ", place="velar"),
    "च": Consonant("tʃ", "stop", "palatal"),
    "छ": Consonant("tʃʰ", "stop", "palatal"),
    "ज": Consonant("dʒ", "stop", "palatal"),
    "झ": Consonant("dʒʱ", "stop", "palatal"),
    "ञ": Consonant("ɲ", place="palatal"),
    "ट": Consonant("ʈ", "stop", "retroflex"),
    "ठ": Consonant("ʈʰ", "stop", "retroflex"),
    "ड": Consonant("ɖ", "stop", "retroflex"),
    "ढ": Consonant("ɖʱ", "stop", "retroflex"),
    "ण": Consonant("ɳ", place="retroflex"),
    "त": Consonant("t", "stop", "dental"),
    "थ": Consonant("tʰ", "stop", "dental"),
    "द": Consonant("d", "stop", "dental"),
    "ध": Consonant("dʱ", "stop", "dental"),
    "न": Consonant("n", place="dental"),
    "प": Consonant("p", "stop", "labial"),
    "फ": Consonant("pʰ", "stop", "labial"),
    "ब": Consonant("b", "stop", "labial"),
    "भ": Consonant("bʱ", "stop", "labial"),
    "म": Consonant("m", place="labial"),
    "य": Consonant("j", "semivowel"),
    "र": Consonant("r", "semivowel"),
    "ल": Consonant("l", "semivowel"),
    "व": Consonant("ʋ", "semivowel", "labial"),
    "श": Consonant("ʃ", place="palatal"),
    "ष": Consonant("ȿ", place="retroflex"),
    "स": Consonant("s"),
    "ह": Consonant("h", place="glottal"),
    "ळ": Consonant("ɭ", "semivowel", "retroflex"),
    "क़": Consonant("q", "stop", "velar"),
    "ख़": Consonant("x", place="velar"),
    "ग़": Consonant("ɣ", "stop", "velar"),
    "ज़": Consonant("z", place="dental"),
    "ड़": Consonant("ɽ", "semivowel", "retroflex"),
    "ढ़": Consonant("ɽʱ", "semivowel", "retroflex"),
    "फ़": Consonant("f", place="labial"),
}
NUKTA_FORMS = {
    "क": "क़",
    "ख": "ख़",
    "ग": "ग़",
    "ज": "ज़",
    "ड": "ड़",
    "ढ": "ढ़",
    "फ": "फ़",
}
NASAL_FOR_PLACE = {
    "velar": "ŋ",
    "palatal": "n",
    # The compatibility phone alphabet uses underspecified /n/ for anusvara
    # before palatal and retroflex stops; explicit ण remains /ɳ/.
    "retroflex": "n",
    "dental": "n",
    "labial": "m",
    "glottal": "n",
}
NASALIZED_VOWELS = {
    "ə": "ə̃",
    "a:": "ã:",
    "i": "ĩ",
    "i:": "ĩ:",
    "u": "ũ",
    "u:": "ũ:",
    "e:": "ẽ:",
    "æ:": "æ̃:",
    "o:": "õ:",
    "ᴔ:": "ᴔ̃:",
    "ɔ:": "ɔ̃:",
}
VIRAMA = "्"
NUKTA = "़"
NASAL_MARKS = {"ं", "ँ"}
IGNORABLES = {"\u200c", "\u200d"}


@dataclass(frozen=True)
class Phone:
    text: str
    kind: str
    long: bool = False
    group: str = ""
    place: str = ""
    joined: bool = False
    inherent: bool = False
    source_index: int = -1


@dataclass(frozen=True)
class Syllable:
    phones: tuple[Phone, ...]

    @property
    def text(self) -> str:
        return "".join(phone.text for phone in self.phones)

    @property
    def nucleus(self) -> int:
        return next(index for index, phone in enumerate(self.phones) if phone.kind == "vowel")

    @property
    def moras(self) -> int:
        nucleus = self.nucleus
        vowel = self.phones[nucleus]
        codas = sum(phone.kind in {"consonant", "nasal"} for phone in self.phones[nucleus + 1 :])
        return (2 if vowel.long else 1) + codas


def _has_following_grapheme(word: str, start: int) -> bool:
    return any(char not in IGNORABLES for char in word[start:])


def normalize_word(word: str) -> str:
    return unicodedata.normalize("NFC", "".join(char for char in word.strip() if unicodedata.category(char) != "Cf"))


def underlying_phonemes(word: str) -> list[Phone]:
    """Map one Devanagari word to its underlying phoneme sequence."""

    word = normalize_word(word)
    if not word or any(char.isspace() for char in word):
        raise ValueError("input must contain exactly one non-empty word per line")

    phones: list[Phone] = []
    index = 0
    while index < len(word):
        char = word[index]
        if char in IGNORABLES:
            index += 1
            continue
        if char in VOWELS:
            text, long = VOWELS[char]
            phones.append(Phone(text, "vowel", long, source_index=index))
            index += 1
            continue
        if char in NASAL_MARKS:
            phones.append(Phone("X", "nasal", source_index=index))
            index += 1
            continue
        if char == "ः":
            phones.append(Phone("h", "consonant", place="glottal", source_index=index))
            index += 1
            continue

        consonant_index = index
        consonant_char = char
        if index + 1 < len(word) and word[index + 1] == NUKTA:
            consonant_char = NUKTA_FORMS.get(char, "")
            if not consonant_char:
                raise ValueError(f"unsupported nukta consonant {char + NUKTA!r}")
            index += 1
        consonant = CONSONANTS.get(consonant_char)
        if consonant is None:
            raise ValueError(f"unsupported character {char!r}")

        phones.append(
            Phone(
                consonant.ipa,
                "consonant",
                group=consonant.group,
                place=consonant.place,
                source_index=consonant_index,
            )
        )
        index += 1
        while index < len(word) and word[index] in IGNORABLES:
            index += 1
        if index < len(word) and word[index] == VIRAMA:
            phones[-1] = replace(phones[-1], joined=True)
            index += 1
        elif index < len(word) and word[index] in MATRAS:
            text, long = MATRAS[word[index]]
            phones.append(Phone(text, "vowel", long, source_index=index))
            index += 1
        elif _has_following_grapheme(word, index):
            phones.append(Phone("ə", "vowel", inherent=True, source_index=consonant_index))

    terminal = len(phones) - 1
    while terminal >= 0 and phones[terminal].kind == "nasal":
        terminal -= 1
    if terminal >= 0 and phones[terminal].kind == "vowel" and phones[terminal].text in {"i", "u"}:
        phones[terminal] = replace(phones[terminal], text=phones[terminal].text + ":", long=True)
    if not any(phone.kind == "vowel" for phone in phones):
        raise ValueError("word has no pronounceable vowel")
    return phones


def resolve_nasals(phones: list[Phone]) -> list[Phone]:
    """Resolve anusvara/anunasika using post-resyllabification mora weights."""

    result = list(phones)
    while any(phone.kind == "nasal" for phone in result):
        syllables = syllabify(result)
        starts = []
        position = 0
        for syllable in syllables:
            starts.append(position)
            position += len(syllable.phones)

        index = next(index for index, phone in enumerate(result) if phone.kind == "nasal")
        syllable_index = max(index_ for index_, start in enumerate(starts) if start <= index)
        syllable = syllables[syllable_index]
        local_index = index - starts[syllable_index]
        previous_vowel = next(
            (position for position in range(index - 1, -1, -1) if result[position].kind == "vowel"),
            None,
        )
        if previous_vowel is None:
            raise ValueError("nasal mark has no preceding vowel")
        following = result[index + 1] if index + 1 < len(result) and result[index + 1].kind == "consonant" else None
        final = syllable_index == len(syllables) - 1
        before_palatal_affricate = following is not None and following.text in {"tʃ", "tʃʰ", "dʒ", "dʒʱ"}
        nasalize = (
            following is None
            or final and (local_index == len(syllable.phones) - 1 or before_palatal_affricate)
        ) or (
            not final and syllable.moras > syllables[syllable_index + 1].moras
        )
        if nasalize:
            vowel = result[previous_vowel]
            result[previous_vowel] = replace(
                vowel,
                text=NASALIZED_VOWELS.get(vowel.text, vowel.text + "̃"),
            )
            result.pop(index)
        elif following is not None:
            result[index] = Phone(
                NASAL_FOR_PLACE[following.place],
                "consonant",
                place=following.place,
                source_index=result[index].source_index,
            )
    if any(phone.kind == "nasal" or phone.text == "X" for phone in result):
        raise ValueError("unresolved nasal placeholder")
    return result


def syllabify(phones: list[Phone]) -> list[Syllable]:
    """Apply the paper's maximal-onset syllabification constraints."""

    nuclei = [index for index, phone in enumerate(phones) if phone.kind == "vowel"]
    if not nuclei:
        raise ValueError("phoneme sequence has no vowel nucleus")
    starts = [0]
    for left, right in zip(nuclei, nuclei[1:]):
        cluster = [index for index in range(left + 1, right) if phones[index].kind == "consonant"]
        if not cluster:
            starts.append(right)
            continue
        onset_size = 1
        if (
            len(cluster) >= 2
            and phones[cluster[-2]].joined
            and phones[cluster[-1]].group == "semivowel"
            and phones[cluster[-2]].group == "stop"
        ):
            onset_size = 2
        starts.append(cluster[-onset_size])
    ends = starts[1:] + [len(phones)]
    return [Syllable(tuple(phones[start:end])) for start, end in zip(starts, ends)]


def delete_foot_schwas(phones: list[Phone]) -> list[Phone]:
    """Delete a weak schwa on the right edge of each left-to-right binary foot."""

    syllables = syllabify(phones)
    delete_ids: set[int] = set()
    phone_id = 0
    ids_by_syllable: list[list[int]] = []
    for syllable in syllables:
        ids = list(range(phone_id, phone_id + len(syllable.phones)))
        ids_by_syllable.append(ids)
        phone_id += len(ids)
    for index in range(1, len(syllables), 2):
        syllable = syllables[index]
        nucleus = syllable.nucleus
        if syllable.moras == 1 and syllable.phones[nucleus].inherent:
            delete_ids.add(ids_by_syllable[index][nucleus])
    return [phone for index, phone in enumerate(phones) if index not in delete_ids]


def delete_contextual_schwas(word: str, phones: list[Phone]) -> list[Phone]:
    """Apply the trained context model, retaining the rule path for nasals."""

    if any(phone.kind == "nasal" for phone in phones):
        return delete_foot_schwas(phones)
    frozen = tuple(phones)
    if not any(phone.inherent for phone in frozen):
        return phones
    weights = load_schwa_model()["weights"]
    retained = predict_retained(weights, word, frozen)
    return [
        phone
        for index, phone in enumerate(frozen)
        if not phone.inherent or retained[index]
    ]


def _stress(syllables: list[Syllable]) -> list[bool]:
    if len(syllables) == 1:
        return [False]
    stressed = []
    for index, syllable in enumerate(syllables):
        is_final = index == len(syllables) - 1
        stressed.append(
            syllable.moras >= 3
            or (syllable.moras >= 2 and not is_final)
        )
    return stressed


def label_word(word: str, *, use_prosody_model: bool = True) -> dict:
    """Return the frozen structured schema for one Hindi word."""

    text = normalize_word(word)
    phones = underlying_phonemes(text)
    phones = delete_contextual_schwas(text, phones)
    phones = resolve_nasals(phones)
    syllables = syllabify(phones)
    stress = _stress(syllables)
    markers: list[str | None]
    if len(syllables) == 1:
        markers = [None]
    elif use_prosody_model:
        model = load_prosody_model()
        prosody_syllables = [
            {
                "phonemes": syllable.text,
                "weight": "heavy" if syllable.moras >= 2 else "light",
            }
            for syllable in syllables
        ]
        predictions = [
            predict_prosody(model, text, prosody_syllables, index)
            for index in range(len(syllables))
        ]
        markers = [marker for marker, _ in predictions]
        stress = [stressed for _, stressed in predictions]
    else:
        markers = ["ʰ" if syllable.moras >= 2 else "ʷ" for syllable in syllables]

    structured = []
    ps_parts = []
    source_parts = []
    for index, (syllable, stressed, marker) in enumerate(zip(syllables, stress, markers)):
        if len(syllables) == 1:
            weight = None
            realized_stress = None
            extrametrical = False
            ps_parts.append(syllable.text)
        else:
            weight = "light" if marker == "ʷ" else "heavy"
            realized_stress = stressed
            extrametrical = index == len(syllables) - 1 and marker == "ʰ"
            ps_parts.append("σ" + marker + syllable.text)
        source_parts.append(("'" if stressed else "") + syllable.text)
        structured.append(
            {
                "phonemes": syllable.text,
                "marker": marker,
                "weight": weight,
                "stressed": realized_stress,
                "extrametrical": extrametrical,
            }
        )

    return {
        "text": text,
        "phoneme_string": "".join(syllable.text for syllable in syllables),
        "syllables": structured,
        "ps": "".join(ps_parts),
        "source_phoneme": "".join(source_parts),
        "engine": "native-clean-room-v0.4",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 file containing one Hindi word per line")
    return parser.parse_args()


if __name__ == "__main__":
    rejected = 0
    with parse_args().input.open(encoding="utf-8") as words:
        for line_number, word in enumerate(words, 1):
            try:
                print(json.dumps(label_word(word), ensure_ascii=False))
            except ValueError as error:
                rejected += 1
                print(
                    json.dumps({"line": line_number, "text": word.strip(), "error": str(error)}, ensure_ascii=False),
                    file=sys.stderr,
                )
    if rejected:
        raise SystemExit(1)
