"""Build a phoneme unigram + bigram model from CMUdict.

Outputs data/phoneme_lm.npz with: phonemes (39-list), vowel_mask (39-bool), unigram (39-float),
bigram (39x39 row-normalized float). Used by scripts/corrupt_phonemes.py to draw class-constrained
phoneme substitutions for graded corruption.

One-shot: downloads CMUdict on first run, caches to data/cmudict.dict.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

PHONEMES = ["AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY","F","G","HH",
            "IH","IY","JH","K","L","M","N","NG","OW","OY","P","R","S","SH","T","TH","UH",
            "UW","V","W","Y","Z","ZH"]
VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"


def load_cmudict(cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    print(f"downloading CMUdict to {cache_path}...", file=sys.stderr)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    txt = urllib.request.urlopen(CMUDICT_URL, timeout=60).read().decode("utf-8", errors="ignore")
    cache_path.write_text(txt)
    return txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/phoneme_lm.npz")
    p.add_argument("--cmudict-cache", default="data/cmudict.dict")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt = load_cmudict(Path(args.cmudict_cache))

    idx = {ph: i for i, ph in enumerate(PHONEMES)}
    n = len(PHONEMES)
    uni = np.zeros(n, dtype=np.float64)
    bi = np.zeros((n, n), dtype=np.float64)

    nlines = 0
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        phones = [tok.rstrip("012") for tok in parts[1:]]
        prev = None
        for ph in phones:
            if ph not in idx:
                continue
            i = idx[ph]
            uni[i] += 1
            if prev is not None:
                bi[prev, i] += 1
            prev = i
        nlines += 1

    # normalize
    uni /= max(uni.sum(), 1.0)
    row_sums = bi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    bi = bi / row_sums

    vowel_mask = np.array([ph in VOWELS for ph in PHONEMES], dtype=bool)
    np.savez(out_path, phonemes=np.array(PHONEMES), vowel_mask=vowel_mask,
             unigram=uni, bigram=bi)
    print(f"fit {nlines} CMUdict pronunciations -> {out_path}")
    print(f"  {n} phonemes ({int(vowel_mask.sum())} vowels / {int((~vowel_mask).sum())} consonants)")
    print(f"  top-5 unigram: " + ", ".join(
        f"{PHONEMES[i]}={uni[i]:.3f}" for i in np.argsort(uni)[::-1][:5]))


if __name__ == "__main__":
    main()
