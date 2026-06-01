"""Phoneme-level corruption for the Coherence Dial.

Given a sentence + a dial level (0..4), returns the same sentence's phoneme sequence with
p(level) of phonemes substituted, drawn from the CMUdict bigram LM conditioned on the previous
phoneme, constrained to the same class (vowel<->vowel, consonant<->consonant). Stress markers
and syllable count are preserved (substitutions only, no insertions/deletions). Word
boundaries (spaces/punctuation from g2p_en) pass through unchanged.

Outputs three views of the corrupted phonemes:
  - ARPAbet  (with stress digits)         — for inspection / training labels
  - IPA      (no stress)                  — for F5-TTS phoneme input
  - pseudo   (rough English orthography)  — fallback for TTS that needs grapheme input

p_level: { 0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0 }
"""

import argparse
import sys
from pathlib import Path

import numpy as np

LEVEL_P = [0.0, 0.25, 0.50, 0.75, 1.0]

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
    return {
        "phonemes": list(d["phonemes"]),
        "vowel_mask": d["vowel_mask"],
        "unigram": d["unigram"],
        "bigram": d["bigram"],
    }


_G2P = None


def g2p_tokens(sentence: str):
    """Returns the raw g2p_en token stream (interleaved phonemes + spaces/punctuation)."""
    global _G2P
    if _G2P is None:
        from g2p_en import G2p
        _G2P = G2p()
    return [t for t in _G2P(sentence) if t != ""]


def corrupt(tokens, level: int, lm, rng):
    """Substitute each phoneme with prob p(level), same class, bigram-conditional."""
    p = LEVEL_P[level]
    phonemes = lm["phonemes"]
    idx = {ph: i for i, ph in enumerate(phonemes)}
    vmask = lm["vowel_mask"]
    uni = lm["unigram"]
    bi = lm["bigram"]
    out = []
    prev_i = None
    for tok in tokens:
        base = tok.rstrip("012")
        stress = tok[len(base):]
        if base not in idx:
            out.append(tok)
            continue
        if rng.random() < p:
            class_mask = vmask if base in VOWELS else (~vmask)
            dist = bi[prev_i] if prev_i is not None else uni
            d = dist * class_mask
            if d.sum() == 0:
                d = uni * class_mask
            d = d / d.sum()
            new_i = int(rng.choice(len(phonemes), p=d))
            new_base = phonemes[new_i]
        else:
            new_i = idx[base]
            new_base = base
        out.append(new_base + stress)
        prev_i = new_i
    return out


def to_ipa(tokens):
    parts = []
    for tok in tokens:
        base = tok.rstrip("012")
        parts.append(ARPABET_TO_IPA.get(base, tok))
    return "".join(parts)


def to_spelling(tokens):
    """Rough English orthography for TTS systems that don't accept IPA."""
    parts = []
    for tok in tokens:
        base = tok.rstrip("012")
        parts.append(ARPABET_TO_SPELLING.get(base, tok if not base.isalpha() else ""))
    return "".join(parts).strip()


def corrupt_sentence(sentence: str, level: int, lm, seed: int = 0):
    """Convenience: returns (arpabet_tokens, ipa, pseudo_spelling)."""
    rng = np.random.default_rng(seed)
    tokens = g2p_tokens(sentence)
    corrupted = corrupt(tokens, level, lm, rng)
    return corrupted, to_ipa(corrupted), to_spelling(corrupted)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentence", required=True)
    p.add_argument("--level", type=int, required=True, choices=[0, 1, 2, 3, 4])
    p.add_argument("--lm", default="data/phoneme_lm.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    lm = load_lm(Path(args.lm))
    arpa_orig = g2p_tokens(args.sentence)
    corrupted, ipa, pseudo = corrupt_sentence(args.sentence, args.level, lm, args.seed)

    print(f"original ARPABET : {' '.join(t for t in arpa_orig if t.strip())}")
    print(f"level {args.level} (p={LEVEL_P[args.level]:.2f})")
    print(f"  ARPABET : {' '.join(t for t in corrupted if t.strip())}")
    print(f"  IPA     : {ipa}")
    print(f"  PSEUDO  : {pseudo}")


if __name__ == "__main__":
    main()
