"""Phoneme-level corruption for the Glossolalia Dial.

Given a sentence + a dial level (0..4), returns the same sentence's phoneme sequence with
every phoneme drawn from a Boltzmann distribution over the 39 ARPAbet phonemes:

    q(y | x, level) ∝ exp( -D_panphon(x, y) / T(level) ) * bias_weight(y)

where D_panphon is the precomputed feature-edit-distance matrix from data/phoneme_lm.npz
(PanPhon library, Mortensen et al. COLING 2016, values verified empirically: P/B=1, S/SH=2,
P/M=3, P/ZH=7, K/N=8, AA/P=11). T(level) is a temperature schedule:

    T(level) = 0.5 * exp(2.5 * p_level)
    -> T(0)=0.50 (only Hamming<=1 neighbors get weight; near-identity)
    -> T(2)=1.75 (distance-3 neighbors come into play)
    -> T(4)=6.09 (full range opens; bias_weight steers the attractor)

The temperature schedule is a design choice, exponential ramp so early dial departures
move only to near-identical phonemes (P->B, S->SH) and the dial only fully opens at the top.
No published precedent for this exact schedule; chosen by feel.

bias_weight is the per-phoneme importance multiplier from the active preset (`dreamy`,
`sigur-ros`, `fraser`). The composition is multiplicative reweighting, not a formal
product-of-experts (which would require both terms to be exp(-energy)). Hand-tuned values.

Stress markers and syllable count are preserved by 1-for-1 substitution. At levels 3-4 we
additionally apply CV cluster simplification: consonant-consonant onset runs collapse to a
single consonant. This is grounded in the documented 95.7% CV-structure preference in
real glossolalia (Link & Tomaschek 2024 PMC10916350; Samarin 1973 Language and Speech).

Outputs four views of the corrupted phonemes:
  - ARPAbet  (with stress digits)           , for training labels
  - IPA      (no stress)                    , for F5-TTS phoneme input (if model accepts it)
  - pseudo   (lowercase English orthography), the in-distribution TTS input we feed F5-TTS
  - display  (UPPER-stressed, hyphen-syllab), for the Gradio UI readout

p_level: { 0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0 }
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

LEVEL_P = [0.0, 0.25, 0.50, 0.75, 1.0]


def temperature(level: int) -> float:
    """T(level) = 0.5 * exp(2.5 * p_level). Design choice. See module docstring."""
    return 0.5 * math.exp(2.5 * LEVEL_P[level])

VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}

ARPABET_TO_IPA = {
    "AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ","AW":"aʊ","AY":"aɪ","EH":"ɛ","ER":"ɜɹ","EY":"eɪ",
    "IH":"ɪ","IY":"i","OW":"oʊ","OY":"ɔɪ","UH":"ʊ","UW":"u",
    "B":"b","CH":"tʃ","D":"d","DH":"ð","F":"f","G":"ɡ","HH":"h","JH":"dʒ","K":"k","L":"l",
    "M":"m","N":"n","NG":"ŋ","P":"p","R":"ɹ","S":"s","SH":"ʃ","T":"t","TH":"θ","V":"v",
    "W":"w","Y":"j","Z":"z","ZH":"ʒ",
}

ARPABET_TO_SPELLING = {
    "AA":"ah","AE":"a","AH":"uh","AO":"aw","AW":"ow","AY":"i","EH":"e","ER":"er","EY":"ay",
    "IH":"i","IY":"ee","OW":"oh","OY":"oi","UH":"oo","UW":"oo",
    "B":"b","CH":"ch","D":"d","DH":"th","F":"f","G":"g","HH":"h","JH":"j","K":"k","L":"l",
    "M":"m","N":"n","NG":"ng","P":"p","R":"r","S":"s","SH":"sh","T":"t","TH":"th","V":"v",
    "W":"w","Y":"y","Z":"z","ZH":"zh",
}


def load_lm(path):
    d = np.load(path, allow_pickle=True)
    out = {
        "phonemes": list(d["phonemes"]),
        "vowel_mask": d["vowel_mask"],
        "unigram": d["unigram"],
        "bigram": d["bigram"],
    }
    # v6 keys added by build_phoneme_lm.py: PanPhon distance matrix + per-phoneme bias weights.
    # Old LMs without these fall back to bigram-conditional sampling (legacy code path).
    if "dist_matrix" in d.files:
        out["dist_matrix"] = d["dist_matrix"]
    if "bias_weights" in d.files:
        out["bias_weights"] = d["bias_weights"]
    return out


_G2P = None


def g2p_tokens(sentence: str):
    """Returns the raw g2p_en token stream (interleaved phonemes + spaces/punctuation)."""
    global _G2P
    if _G2P is None:
        # g2p_en uses NLTK's pos_tag which (since NLTK 3.9) wants the *_eng suffixed taggers,
        # but g2p_en's own bootstrap still references the legacy names. Pre-fetch both quietly.
        import nltk
        for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
        from g2p_en import G2p
        _G2P = G2p()
    return [t for t in _G2P(sentence) if t != ""]


def corrupt(tokens, level: int, lm, rng):
    """Boltzmann substitution kernel + CV cluster simplification at high levels.

    Each ARPAbet phoneme x is replaced by a draw y ~ q(y|x, level) where
        q(y|x, level) ∝ exp(-D[x,y] / T(level)) * bias_weight(y)
    using D = panphon feature-edit-distance matrix (raw count, 0-48) and T(level) =
    0.5 * exp(2.5 * p_level). At level=0, T=0.5 -> only distance-0 (self) gets meaningful
    weight, so the lyric stays nearly intact. At level=4, T=6.09 -> the distribution spreads
    and bias_weight steers toward the dreamy attractor.

    The legacy bigram path (old LM without dist_matrix) is preserved for backward compat.
    """
    phonemes = lm["phonemes"]
    idx = {ph: i for i, ph in enumerate(phonemes)}

    use_boltzmann = "dist_matrix" in lm and "bias_weights" in lm
    if use_boltzmann:
        D = lm["dist_matrix"]
        bw = lm["bias_weights"]
        T = temperature(level)
        # Precompute per-source distributions so we don't redo softmax per token.
        # logits[i, j] = -D[i,j]/T + log(bw[j])
        logits = -D / T + np.log(np.clip(bw, 1e-12, None))[None, :]
        logits = logits - logits.max(axis=1, keepdims=True)
        Q = np.exp(logits)
        Q = Q / Q.sum(axis=1, keepdims=True)
    else:
        # Legacy: per-class bigram fallback (kept for old LMs)
        vmask = lm["vowel_mask"]
        uni = lm["unigram"]
        bi = lm["bigram"]
        p_legacy = LEVEL_P[level]

    out = []
    prev_i = None
    for tok in tokens:
        base = tok.rstrip("012")
        stress = tok[len(base):]
        if base not in idx:
            out.append(tok)
            continue
        i = idx[base]
        if use_boltzmann:
            # Boltzmann draw at this level. At level=0 this is almost always self.
            new_i = int(rng.choice(len(phonemes), p=Q[i]))
            new_base = phonemes[new_i]
        else:
            if rng.random() < p_legacy:
                class_mask = vmask if base in VOWELS else (~vmask)
                dist = bi[prev_i] if prev_i is not None else uni
                d = dist * class_mask
                if d.sum() == 0:
                    d = uni * class_mask
                d = d / d.sum()
                new_i = int(rng.choice(len(phonemes), p=d))
                new_base = phonemes[new_i]
            else:
                new_i = i
                new_base = base
        out.append(new_base + stress)
        prev_i = new_i
    if use_boltzmann and level >= 3:
        out = _simplify_clusters(out)
    return out


def _simplify_clusters(tokens):
    """Collapse CC onset runs to single onset at levels 3-4.

    A CC onset run is two consecutive ARPAbet consonants between a word break and a vowel.
    We drop the second consonant. CV preference is documented in real glossolalia
    (Link & Tomaschek 2024 PMC10916350, 95.7% CV; Samarin 1973, open-syllable preference).
    """
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        base = tok.rstrip("012")
        # Detect: previous emitted is a non-phoneme (word break) AND current+next are both
        # consonants AND the one AFTER next is a vowel, collapse to single onset.
        prev_is_break = (len(out) == 0) or (not out[-1].rstrip("012").isalpha()) or \
                        (out[-1].rstrip("012") not in (set(VOWELS) | _CONSONANTS))
        if prev_is_break and base in _CONSONANTS and i + 1 < n:
            nxt = tokens[i + 1].rstrip("012")
            if nxt in _CONSONANTS and i + 2 < n:
                nxt2 = tokens[i + 2].rstrip("012")
                if nxt2 in VOWELS:
                    # Drop tokens[i+1], keep the first onset only.
                    out.append(tok)
                    out.append(tokens[i + 2])
                    i += 3
                    continue
        out.append(tok)
        i += 1
    return out


_CONSONANTS = {"B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","P","R","S","SH",
               "T","TH","V","W","Y","Z","ZH"}


def to_ipa(tokens):
    parts = []
    for tok in tokens:
        base = tok.rstrip("012")
        parts.append(ARPABET_TO_IPA.get(base, tok))
    return "".join(parts)


def to_spelling(tokens):
    """Lowercase pseudo-English orthography. THE input we feed to F5-TTS at training and
    inference time, empirically in-distribution per F5-TTS issue #362 (owner SWivid confirms
    'current base models are using characters rather than phonemes')."""
    parts = []
    for tok in tokens:
        base = tok.rstrip("012")
        parts.append(ARPABET_TO_SPELLING.get(base, tok if not base.isalpha() else ""))
    return "".join(parts).strip()


def to_display(tokens):
    """UI-readable rendering of the corrupted lyric.

    Uppercase the glyph for any stressed (digit=1) phoneme, lowercase otherwise. Insert a
    hyphen between consecutive phoneme glyphs within a word. Word breaks (spaces and
    punctuation from g2p) pass through unchanged.

    Example: 'i KWIK-lee kuh-LEK-tuhd' for tokens with stress on KWIK and LEK.

    ASCII-only, Merriam-Webster diacritics break F5-TTS's character tokenizer, so we keep
    this format compatible with the TTS input pipeline (the `pseudo` string remains the
    actual TTS input; `display` is for the Gradio readout only).
    """
    parts = []
    prev_was_phoneme = False
    for tok in tokens:
        base = tok.rstrip("012")
        stress = tok[len(base):]
        glyph = ARPABET_TO_SPELLING.get(base)
        if glyph is None:
            # Word break / punctuation
            parts.append(tok if not base.isalpha() else "")
            prev_was_phoneme = False
            continue
        if stress.startswith("1"):
            glyph = glyph.upper()
        if prev_was_phoneme:
            parts.append("-")
        parts.append(glyph)
        prev_was_phoneme = True
    return "".join(parts).strip()


def corrupt_sentence(sentence: str, level: int, lm, seed: int = 0):
    """Returns (arpabet_tokens, ipa, pseudo_spelling, display).

    pseudo_spelling is the lowercase TTS input. display is the UI readout.
    """
    rng = np.random.default_rng(seed)
    tokens = g2p_tokens(sentence)
    corrupted = corrupt(tokens, level, lm, rng)
    return corrupted, to_ipa(corrupted), to_spelling(corrupted), to_display(corrupted)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentence", required=True)
    p.add_argument("--level", type=int, required=True, choices=[0, 1, 2, 3, 4])
    p.add_argument("--lm", default="data/phoneme_lm.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    lm = load_lm(Path(args.lm))
    arpa_orig = g2p_tokens(args.sentence)
    corrupted, ipa, pseudo, display = corrupt_sentence(args.sentence, args.level, lm, args.seed)

    print(f"original ARPABET : {' '.join(t for t in arpa_orig if t.strip())}")
    print(f"level {args.level} (p={LEVEL_P[args.level]:.2f}, T={temperature(args.level):.3f})")
    print(f"  ARPABET : {' '.join(t for t in corrupted if t.strip())}")
    print(f"  IPA     : {ipa}")
    print(f"  PSEUDO  : {pseudo}")
    print(f"  DISPLAY : {display}")


if __name__ == "__main__":
    main()
