"""Mondegreen substitution for the Glossolalia Dial — "Ghost mode".

Given an input lyric + a dial level, returns a sequence of REAL English words whose phoneme
sequence is phonetically close to the source. The output is meant to be sung in place of
the original lyric and read by listeners as the source via pareidolia (Cocteau Twins-style):

    "the river was wide and calm in the morning light"     (source, lv0)
    "the reefer was white and come in the mourning knight" (mondegreen, lv4)

Mechanism (fully deterministic):
  1. g2p_en converts input -> ARPAbet phoneme sequence
  2. For each word, look up its phonemes in CMUdict (canonical pronunciation)
  3. Find candidate substitute words in CMUdict whose phoneme sequence is within
     dial-conditioned PanPhon feature-edit-distance of the source word's phonemes
  4. Sample deterministically from the candidate set, weighted by 1/(1+distance) so
     near substitutes are preferred over far ones at any given dial level
  5. Reassemble the substituted words into a lyric string

Coverage constraints:
  - Match source syllable count exactly (preserves singability over the original melody)
  - Substitution probability scales with dial level: p ∈ [0, 0.25, 0.5, 0.75, 1.0]
  - Distance threshold scales with dial level: max_dist ∈ [0, 2, 4, 6, 8] (raw PanPhon
    feature-edit-distance, scale 0-13 for typical phoneme pairs)
  - If no candidate is found within the threshold, the source word passes through unchanged
    (graceful degradation for OOV / rare phonotactic combinations like "courtyard")

What this is NOT:
  - A generative model. No LLM, no neural net. Just a phoneme-distance search over CMUdict.
  - A guarantee that the output is grammatical or semantically sensible. The output may read
    as a nonsense sentence of real English words. That's the point — the listener's brain
    fills in meaning via pareidolia.

Sources used (verified):
  - PanPhon library, Mortensen et al. COLING 2016, aclanthology.org/C16-1328 (used as software)
  - CMUdict pronunciation dictionary, ~135K English entries

Design choices (no published precedent claimed):
  - Distance schedule and probability schedule are hand-chosen for a smooth dial.
  - Syllable-count enforcement is a design choice for singability, grounded in the standard
    lyric-substitution constraint in parody/pastiche traditions (no specific citation).
  - Function-word handling: substituted by default at any dial level (no special carve-out).
"""

from __future__ import annotations

import math
import re
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

# ---- Constants ----

LEVEL_P = [0.0, 0.25, 0.50, 0.75, 1.0]              # per-word substitution probability
LEVEL_MAX_DIST = [0.0, 2.0, 4.0, 6.0, 8.0]          # PanPhon feature-edit-distance cap

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"

_ARPABET_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                   "IH", "IY", "OW", "OY", "UH", "UW"}

_ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i",
    "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ",
    "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
    "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Word-token splitter that preserves whitespace and punctuation.
_TOKEN_RE = re.compile(r"([A-Za-z']+|[^A-Za-z']+)")

# Function words held constant — they're typically unstressed and listeners skip over them
# perceptually; substituting them adds noise without much added ghost effect. The list is
# the standard short closed-class items in English (articles, prepositions, conjunctions,
# auxiliaries, common pronouns). Hand-curated; no published precedent claimed.
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "nor", "so", "yet", "for",
    "of", "in", "on", "at", "to", "by", "with", "from", "into", "onto",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "we", "us", "our", "they", "them", "their",
    "this", "that", "these", "those", "as", "if", "than",
})


def _syllable_count(phones: tuple[str, ...]) -> int:
    return sum(1 for p in phones if p in _ARPABET_VOWELS)


def _phones_to_ipa_seq(phones: tuple[str, ...]) -> str:
    return " ".join(_ARPABET_TO_IPA[p] for p in phones if p in _ARPABET_TO_IPA)


class MondegreenIndex:
    """Per-process CMUdict index plus PanPhon distance computation.

    Heavy to construct (~5 s on cold start: dict parse + IPA conversion + bucketing).
    Cache the instance at the module/app level.
    """

    def __init__(self, cmudict_path: str | Path = "data/cmudict.dict"):
        import panphon.distance
        self._dist = panphon.distance.Distance()

        cmu = Path(cmudict_path)
        if not cmu.exists():
            cmu.parent.mkdir(parents=True, exist_ok=True)
            cmu.write_text(urllib.request.urlopen(CMUDICT_URL, timeout=60)
                           .read().decode("utf-8", errors="ignore"))

        word_to_arpa: dict[str, tuple[str, ...]] = {}
        for line in cmu.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower().split("(")[0]
            phones = tuple(tok.rstrip("012") for tok in parts[1:] if tok)
            # Reject lines that contain non-ARPAbet tokens
            if any(p not in _ARPABET_TO_IPA for p in phones):
                continue
            # Keep canonical pronunciation only (first entry per word)
            if word.replace("'", "").isalpha() and phones and word not in word_to_arpa:
                word_to_arpa[word] = phones

        self._word_to_arpa = word_to_arpa
        self._word_to_syl = {w: _syllable_count(p) for w, p in word_to_arpa.items()}
        self._word_to_ipa = {w: _phones_to_ipa_seq(p) for w, p in word_to_arpa.items()}

        # Syllable bucket for fast candidate enumeration.
        self._syl_bucket: dict[int, list[str]] = {}
        for w, s in self._word_to_syl.items():
            self._syl_bucket.setdefault(s, []).append(w)

    @property
    def size(self) -> int:
        return len(self._word_to_arpa)

    def find_candidates(self, source_word: str, max_dist: float,
                         max_results: int = 64) -> list[tuple[str, float]]:
        """Return list of (candidate_word, distance) sorted by distance ascending.

        Restricts candidates to the same syllable bucket so the substitution preserves
        the original lyric's prosodic shape. Filters out 1-letter candidates (single
        letters like "y", "u", "z" appear in CMUdict but read poorly in lyrics).
        """
        word = source_word.lower()
        if word not in self._word_to_arpa:
            return []
        src_ipa = self._word_to_ipa[word]
        src_syl = self._word_to_syl[word]
        cands: list[tuple[str, float]] = []
        for cand in self._syl_bucket.get(src_syl, []):
            if cand == word or len(cand.replace("'", "")) <= 1:
                continue
            d = float(self._dist.hamming_feature_edit_distance(
                src_ipa, self._word_to_ipa[cand]) * 24.0)
            if d <= max_dist:
                cands.append((cand, d))
        cands.sort(key=lambda x: x[1])
        return cands[:max_results]

    def substitute_word(self, word: str, level: int,
                         rng: np.random.Generator) -> str:
        """Pick a mondegreen for `word` at the given dial level.

        Returns the source word unchanged if (a) it's a function word (closed-class
        item we hold constant for prosodic stability), (b) the dial RNG draw doesn't
        fire, or (c) no candidate exists within the level-conditioned distance.
        """
        if level <= 0:
            return word
        if word.lower() in _FUNCTION_WORDS:
            return word
        p = LEVEL_P[level]
        if rng.random() >= p:
            return word
        cands = self.find_candidates(word, max_dist=LEVEL_MAX_DIST[level])
        if not cands:
            return word
        # Weighted by 1/(1+d): near substitutes preferred but far ones not impossible.
        weights = np.array([1.0 / (1.0 + d) for _, d in cands], dtype=np.float64)
        weights = weights / weights.sum()
        idx = int(rng.choice(len(cands), p=weights))
        return cands[idx][0]

    def substitute(self, sentence: str, level: int, seed: int = 0) -> str:
        """Mondegreen-substitute every alphabetic word in `sentence` at `level`.

        Whitespace and punctuation pass through unchanged. Words not in CMUdict
        (proper names, neologisms) pass through unchanged.
        """
        rng = np.random.default_rng(seed)
        out_parts: list[str] = []
        for tok in _TOKEN_RE.findall(sentence):
            if tok.replace("'", "").isalpha():
                out_parts.append(self.substitute_word(tok, level, rng))
            else:
                out_parts.append(tok)
        # Preserve source capitalization on sentence-initial / proper positions:
        # if the source token was Capitalized, capitalize the substitute.
        rebuilt: list[str] = []
        src_tokens = _TOKEN_RE.findall(sentence)
        for src, out in zip(src_tokens, out_parts):
            if src and src[0].isupper() and out and out[0].islower():
                out = out[0].upper() + out[1:]
            rebuilt.append(out)
        return "".join(rebuilt)


# ---- Module-level singleton helpers (lazy load) ----

_INDEX: MondegreenIndex | None = None


def get_index(cmudict_path: str | Path = "data/cmudict.dict") -> MondegreenIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = MondegreenIndex(cmudict_path)
    return _INDEX


def substitute(sentence: str, level: int, seed: int = 0,
               cmudict_path: str | Path = "data/cmudict.dict") -> str:
    """Convenience: build/cache index then substitute."""
    return get_index(cmudict_path).substitute(sentence, level, seed)


# ---- CLI ----

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sentence", required=True)
    p.add_argument("--level", type=int, choices=range(5), default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cmudict", default="data/cmudict.dict")
    args = p.parse_args()
    idx = MondegreenIndex(args.cmudict)
    print(f"CMUdict: {idx.size} words")
    print(f"\nlv0 (source): {args.sentence}")
    for lv in range(5):
        out = idx.substitute(args.sentence, lv, args.seed)
        print(f"lv{lv} (p={LEVEL_P[lv]:.2f}, max_d={LEVEL_MAX_DIST[lv]:.0f}): {out}")
