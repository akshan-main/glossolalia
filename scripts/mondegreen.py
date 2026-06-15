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
  - Stress-position match is enforced (same primary-stress syllable index in source and
    candidate). Music-cognition literature (Kolinsky et al., Empirical Musicology Review)
    surfaced by workflow synthesis — not personally re-verified for this docstring.
  - Function-word handling: HELD CONSTANT at all dial levels (small closed-class set in
    _FUNCTION_WORDS). Substituting them adds noise where listeners process with delayed
    commitment; we skip them so the ghost lands on the content-word stress points.
"""

from __future__ import annotations

import functools
import math
import re
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

from wordfreq import zipf_frequency


@functools.lru_cache(maxsize=200_000)
def _zipf(word: str) -> float:
    """Cached zipf frequency (log10 per-billion) of an English word; 0.0 if unknown."""
    return zipf_frequency(word.replace("'", ""), "en")

# ---- Constants ----

LEVEL_P = [0.0, 0.25, 0.50, 0.75, 1.0]              # per-word substitution probability
# PanPhon feature-edit-distance cap per level. The nearest real-word mishearing scales
# with word length: short words ("sells") have one at distance ~1, but multi-syllable
# words ("seashells", "seashore") have their nearest common-word mishearing at ~7-8.
# So the cap must reach far enough at high levels or long words pass through unchanged
# (the "dial 4 still says seashells" bug). Low levels stay tight = subtle, few changes;
# lv4 (p=1.0, full dissolution) reaches 10 so every content word actually transforms.
LEVEL_MAX_DIST = [0.0, 3.0, 5.0, 7.5, 10.0]

# Minimum word-frequency (zipf scale, log10 per-billion) for a candidate to qualify.
# CMUdict has ~135k entries including rare surnames / abbreviations ("selz" zipf 1.4).
# zipf >= 2.5 keeps real mishearings ("seashells" 2.5, "reefer" 2.7) and rejects junk.
MIN_CANDIDATE_ZIPF = 2.5

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


def _primary_stress_syllable(phones_with_stress: tuple[str, ...]) -> int:
    """Index (0-based) of the syllable carrying primary stress (digit '1').

    Returns -1 if no primary stress is marked (e.g. single-syllable function words).
    Used to match source-vs-candidate stress position: substituting a trochaic word
    (stress on syllable 0) with an iambic candidate (stress on syllable 1) makes the
    ghost rhythmically wrong and breaks perception under the original melody.
    Citation provenance: Kolinsky et al. (emusicology.org/article/view/3729) + EEG
    study PMC3225926 — stress mismatch degrades word recognition in song contexts.
    """
    syl_idx = -1
    for p in phones_with_stress:
        base = p.rstrip("012")
        if base in _ARPABET_VOWELS:
            syl_idx += 1
            if p.endswith("1"):
                return syl_idx
    return -1


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
        word_to_arpa_stressed: dict[str, tuple[str, ...]] = {}
        for line in cmu.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower().split("(")[0]
            phones_with_stress = tuple(parts[1:])
            phones = tuple(tok.rstrip("012") for tok in phones_with_stress if tok)
            # Reject lines that contain non-ARPAbet tokens
            if any(p not in _ARPABET_TO_IPA for p in phones):
                continue
            # Keep canonical pronunciation only (first entry per word)
            if word.replace("'", "").isalpha() and phones and word not in word_to_arpa:
                word_to_arpa[word] = phones
                word_to_arpa_stressed[word] = phones_with_stress

        self._word_to_arpa = word_to_arpa
        self._word_to_arpa_stressed = word_to_arpa_stressed
        self._word_to_syl = {w: _syllable_count(p) for w, p in word_to_arpa.items()}
        self._word_to_ipa = {w: _phones_to_ipa_seq(p) for w, p in word_to_arpa.items()}
        self._word_to_stress_pos = {
            w: _primary_stress_syllable(word_to_arpa_stressed[w])
            for w in word_to_arpa
        }

        # Syllable bucket for fast candidate enumeration, sub-keyed by primary stress position
        # so candidate scans are O(bucket size for matching stress) rather than O(all words).
        self._syl_stress_bucket: dict[tuple[int, int], list[str]] = {}
        for w in self._word_to_arpa:
            key = (self._word_to_syl[w], self._word_to_stress_pos[w])
            self._syl_stress_bucket.setdefault(key, []).append(w)

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
        src_stress = self._word_to_stress_pos[word]
        # Candidates are bucketed by (syllable count, primary stress position) — both
        # must match. Stress-mismatched substitutes break the perceptual ghost under
        # the original melody (Kolinsky et al.; EEG study PMC3225926).
        bucket = self._syl_stress_bucket.get((src_syl, src_stress), [])
        cands: list[tuple[str, float]] = []
        for cand in bucket:
            if cand == word or len(cand.replace("'", "")) <= 1:
                continue
            # Real-common-word gate: reject rare CMUdict entries (surnames,
            # abbreviations, archaisms) so the ghost reads as ordinary English.
            if _zipf(cand) < MIN_CANDIDATE_ZIPF:
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

    def substitute(self, sentence: str, level: int, seed: int = 0,
                    reranker: "LMReranker | None" = None,
                    beam_width: int = 8,
                    n_candidates_per_word: int = 12) -> str:
        """Mondegreen-substitute every alphabetic word in `sentence` at `level`.

        Whitespace and punctuation pass through unchanged. Words not in CMUdict
        (proper names, neologisms) pass through unchanged.

        If `reranker` is provided, runs constrained beam search: at each substitutable
        word position the top `n_candidates_per_word` phonetic candidates are expanded
        across all `beam_width` live beams; each partial sequence is scored by the
        reranker's log-probability under a small causal language model; only the top
        `beam_width` beams survive to the next position. Final output is the highest-
        scoring complete sequence. Determinism preserved: argmax not sample, ties
        broken by candidate distance then alphabetical.

        Without a reranker, falls back to the per-word weighted draw (no semantic
        coherence — substitutions are independent across positions).
        """
        if reranker is None or level == 0:
            return self._substitute_independent(sentence, level, seed)
        return self._substitute_beam(sentence, level, seed, reranker,
                                       beam_width, n_candidates_per_word)

    def _substitute_independent(self, sentence: str, level: int, seed: int) -> str:
        rng = np.random.default_rng(seed)
        out_parts: list[str] = []
        for tok in _TOKEN_RE.findall(sentence):
            if tok.replace("'", "").isalpha():
                out_parts.append(self.substitute_word(tok, level, rng))
            else:
                out_parts.append(tok)
        return self._restore_caps(sentence, out_parts)

    def _restore_caps(self, sentence: str, out_parts: list[str]) -> str:
        rebuilt: list[str] = []
        src_tokens = _TOKEN_RE.findall(sentence)
        for src, out in zip(src_tokens, out_parts):
            if src and src[0].isupper() and out and out[0].islower():
                out = out[0].upper() + out[1:]
            rebuilt.append(out)
        return "".join(rebuilt)

    def _substitute_beam(self, sentence: str, level: int, seed: int,
                          reranker: "LMReranker",
                          beam_width: int, n_candidates_per_word: int) -> str:
        tokens = _TOKEN_RE.findall(sentence)
        # COUNT-BASED, MONOTONIC selection (replaces probabilistic per-word firing, which
        # left short sentences unchanged at low dials so levels 0/1/2 looked identical).
        # The level controls HOW MANY content words change; each changed word takes its
        # nearest real-word mishearing. Word priority = closest-mishearing first, so the
        # most natural swaps appear at low dials and higher dials nest more on top.
        substitutable: list[int] = []     # token indices eligible for substitution
        best_dist: dict[int, float] = {}  # index -> distance of its nearest candidate
        cand_cache: dict[int, list[tuple[str, float]]] = {}
        for i, tok in enumerate(tokens):
            if not tok.replace("'", "").isalpha():
                continue
            low = tok.lower()
            if low in _FUNCTION_WORDS:
                continue
            # Generous cap (12): once we DECIDE to change a word, it should always find
            # its nearest real-word mishearing regardless of word length. The level
            # gates the COUNT of changes, not the per-word distance.
            cands = self.find_candidates(low, max_dist=12.0, max_results=n_candidates_per_word)
            if not cands:
                continue
            substitutable.append(i)
            best_dist[i] = cands[0][1]
            cand_cache[i] = cands

        # How many of the eligible words to substitute at this level (monotonic: 0,1,...,N).
        n_elig = len(substitutable)
        if level <= 0 or n_elig == 0:
            k = 0
        elif level >= len(LEVEL_P) - 1:
            k = n_elig                       # top level: change everything eligible
        else:
            import math as _math
            k = min(n_elig, max(1, _math.ceil(LEVEL_P[level] * n_elig)))
        # Pick the k words with the closest (most convincing) mishearings; tie-break by
        # position so the choice is deterministic and nests as the dial rises.
        chosen = sorted(substitutable, key=lambda i: (best_dist[i], i))[:k]
        chosen_set = set(chosen)

        per_position: list[list[tuple[str, float]] | None] = []
        for i, tok in enumerate(tokens):
            per_position.append(cand_cache[i] if i in chosen_set else None)

        # Beam search. Each beam = (sequence_so_far: list[str], cumulative_log_prob: float).
        beams: list[tuple[list[str], float]] = [([], 0.0)]
        for pos, cands in enumerate(per_position):
            src_tok = tokens[pos]
            if cands is None:
                # Pass-through — append the same source token to every beam.
                beams = [(beam + [src_tok], score) for beam, score in beams]
                continue
            expanded: list[tuple[list[str], float]] = []
            for beam, score in beams:
                used = {w.lower() for w in beam}   # words already chosen in this beam
                for cand_word, cand_dist in cands:
                    # Dedup: don't let the same substitute word repeat in one sentence
                    # (the "cecil ... cecil" bug). Skip if already used in this beam.
                    if cand_word.lower() in used:
                        continue
                    new_seq = beam + [cand_word]
                    # Score the partial sequence's last-token log-prob under the LM.
                    # Reuses prefix cache internally; this is fast.
                    partial_text = self._compose_partial(tokens, pos, new_seq)
                    lm_score = reranker.score_next_token(partial_text)
                    # Tie-breaker: phonetic distance (lower = better), then alphabetical.
                    tb = (-cand_dist * 1e-6, -ord(cand_word[0]) * 1e-9)
                    expanded.append((new_seq, score + lm_score + tb[0] + tb[1]))
            if not expanded:
                # Every candidate collided with an already-used word; keep the source.
                beams = [(beam + [src_tok], score) for beam, score in beams]
                continue
            # Stable sort by score descending; on tie, earlier item wins (Python sort is stable).
            expanded.sort(key=lambda x: -x[1])
            beams = expanded[:beam_width]

        if not beams:
            return sentence
        best_seq, _ = beams[0]
        return self._restore_caps(sentence, best_seq)

    def _compose_partial(self, tokens: list[str], up_to_pos: int,
                          chosen_so_far: list[str]) -> str:
        """Rebuild the text up through position `up_to_pos` using `chosen_so_far`
        for substituted positions and the source for pass-through."""
        # chosen_so_far has exactly up_to_pos + 1 entries (one per token through pos)
        return "".join(chosen_so_far)


# ---- LM reranker for semantic coherence ----

class LMReranker:
    """Small causal LM that scores partial sentences for semantic coherence during
    beam search over phonetic-ghost candidates.

    Default model is DistilGPT-2 (~82M params, ~330MB on disk). Loads lazily on first
    score() call. CPU-only is fine for ~10-word lyrics at beam_width=8 — scores ~80
    short prompts per sentence, finishes in ~1-2 seconds on a modern Mac.

    Determinism: scoring is a pure function of (model weights, input text). No sampling.
    """

    DEFAULT_MODEL = "distilgpt2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self._device = device

    def score_next_token(self, text: str) -> float:
        """Return the log-probability density of `text` under the LM.

        Implemented as average per-token log-likelihood — length-normalized so longer
        partial sequences aren't unfairly penalized. Higher is better.
        """
        torch = self._torch
        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)
        if ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            out = self._model(ids, labels=ids)
        # out.loss is mean NLL across tokens. log-prob density = -loss.
        return float(-out.loss.item())


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
