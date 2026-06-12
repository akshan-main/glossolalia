"""Build a phoneme unigram + bigram model from CMUdict.

Outputs data/phoneme_lm.npz with: phonemes (39-list), vowel_mask (39-bool), unigram (39-float),
bigram (39x39 row-normalized float). Used by scripts/corrupt_phonemes.py to draw class-constrained
phoneme substitutions for graded corruption.

A `--bias` preset reshapes the substitution distribution toward the empirical phoneme inventory of
wordless vocal music. Default is `none` (raw English CMUdict distribution). Other presets are
grounded in cited phonetic literature — see DECISIONS.md for the rationale + citations.

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

# ARPAbet -> IPA for panphon distance computation. Diphthongs are passed as two-segment
# strings to panphon; its hamming_feature_edit_distance handles the alignment.
_ARPABET_TO_IPA = {
    "AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ","AW":"aʊ","AY":"aɪ","EH":"ɛ","ER":"ɝ","EY":"eɪ",
    "IH":"ɪ","IY":"i","OW":"oʊ","OY":"ɔɪ","UH":"ʊ","UW":"u",
    "B":"b","CH":"tʃ","D":"d","DH":"ð","F":"f","G":"ɡ","HH":"h","JH":"dʒ","K":"k","L":"l",
    "M":"m","N":"n","NG":"ŋ","P":"p","R":"ɹ","S":"s","SH":"ʃ","T":"t","TH":"θ","V":"v",
    "W":"w","Y":"j","Z":"z","ZH":"ʒ",
}


def build_distance_matrix() -> np.ndarray:
    """39x39 PanPhon feature-edit-distance matrix over ARPAbet, in raw feature-count units.

    Verified probes vs direct panphon measurement: P/B=1, S/SH=2, P/M=3, P/ZH=7, K/N=8, AA/P=11.
    Range observed on the full 39x39: 0 - 48 (high end includes diphthong/affricate pairs).
    """
    import panphon.distance
    d = panphon.distance.Distance()
    n = len(PHONEMES)
    D = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(PHONEMES):
        for j, b in enumerate(PHONEMES):
            if i == j:
                continue
            # hamming_feature_edit_distance returns a [0,1] normalized score; * 24 -> raw count.
            D[i, j] = float(d.hamming_feature_edit_distance(
                _ARPABET_TO_IPA[a], _ARPABET_TO_IPA[b]) * 24.0)
    return D


# Phoneme-bias presets. Each maps an ARPAbet symbol (no stress) to a positive multiplier on its
# sampling probability; unlisted phonemes default to 1.0 (no change). The bias is applied to both
# unigram marginals AND to the *target* column of the bigram transitions, then both are renormalized.
# It is also saved as a per-phoneme weight vector (`bias_weights`) for use as the importance term
# in the Boltzmann substitution kernel in scripts/corrupt_phonemes.py.
#
# IMPORTANT — provenance reality check:
#   The specific multiplier values below are a hand-tuned design heuristic, not derived from
#   published per-phoneme frequency tables. Earlier provenance comments overclaimed (Link &
#   Tomaschek 2024 does NOT publish per-phoneme multipliers; Samarin's own data shows obstruent
#   dominance in onset position, contradicting a "sonorant-heavy" claim; Crystal 1995 favors
#   closed vowels for phonaesthetic pleasantness, not open vowels). The reasoning IS the
#   reasoning: sonorants, voiced fricatives, and open back vowels are perceptually softer to
#   the author's ear; the dial=4 attractor sets the demo aesthetic. See DECISIONS.md
#   2026-06-11 "Citation audit: dreamy preset multipliers are hand-tuned, not corpus-derived."
BIAS_PRESETS = {
    "none": {},  # raw CMUdict English distribution
    # Design heuristic: dial=4 should land on a soft, sustained, sonorant-leaning palette.
    # Sonorants (M N L R W Y) + voiced fricatives (V Z ZH DH) get boosted; voiceless stops
    # and affricates (P T K CH) get suppressed; open back vowels (AA AO OW) get boosted.
    # Authored by feel; values are not from a corpus.
    "dreamy": {
        "AA": 1.8, "AO": 1.5, "OW": 1.4, "UW": 1.3, "AH": 1.3,
        "M": 1.6, "N": 1.5, "NG": 1.3, "L": 1.5, "R": 1.2, "W": 1.3, "Y": 1.3,
        "V": 1.2, "Z": 1.1, "ZH": 1.2, "DH": 1.1,
        "IY": 0.6, "IH": 0.6, "EH": 0.7, "EY": 0.8,
        "P": 0.4, "T": 0.4, "K": 0.4, "CH": 0.3, "S": 0.5, "SH": 0.6, "F": 0.5, "TH": 0.4,
        "HH": 0.7, "B": 0.7, "D": 0.7, "G": 0.7, "JH": 0.5,
    },
    # Hopelandic / Jonsi-leaning. Hand-tuned, NOT derived from a published Hopelandic phoneme
    # study (none exists). Glides + high-front vowels emphasized by feel; no source cited.
    "sigur-ros": {
        "IY": 1.8, "IH": 1.4, "EY": 1.5, "AY": 1.4, "OY": 1.3, "AA": 1.4, "OW": 1.3, "UW": 1.2,
        "Y": 1.8, "W": 1.5, "L": 1.4, "R": 1.3, "M": 1.3, "N": 1.3,
        "HH": 1.3, "S": 1.2, "SH": 1.1,
        "P": 0.5, "T": 0.5, "K": 0.6, "CH": 0.4, "JH": 0.5,
        "B": 0.7, "D": 0.7, "G": 0.7, "TH": 0.6, "DH": 0.7, "F": 0.7,
        "V": 0.8, "Z": 0.8, "ZH": 0.8, "NG": 0.9, "ER": 0.7,
    },
    # Cocteau Twins / Fraser-leaning. Hand-tuned. No published Fraser-lyric phoneme corpus
    # exists; this is a design heuristic only.
    "fraser": {
        "AA": 1.6, "AO": 1.4, "OW": 1.4, "UW": 1.3, "AH": 1.3, "EY": 1.3, "AY": 1.2,
        "M": 1.8, "N": 1.6, "L": 1.7, "NG": 1.2, "W": 1.4, "Y": 1.3, "R": 1.1,
        "V": 1.2, "DH": 1.2, "Z": 1.1, "ZH": 1.2, "HH": 1.2,
        "P": 0.4, "T": 0.4, "K": 0.4, "CH": 0.3, "JH": 0.4, "S": 0.5, "SH": 0.7, "F": 0.5, "TH": 0.5,
        "B": 0.7, "D": 0.6, "G": 0.7, "IH": 0.7, "IY": 0.8, "ER": 0.6,
    },
}


def apply_bias(uni: np.ndarray, bi: np.ndarray, preset: str):
    """Reshape unigram + bigram by per-phoneme multipliers, then renormalize."""
    weights = BIAS_PRESETS[preset]
    if not weights:
        return uni, bi
    w = np.array([weights.get(ph, 1.0) for ph in PHONEMES], dtype=np.float64)
    uni = uni * w
    uni /= max(uni.sum(), 1e-12)
    # bias the *destination* column (i.e., what we sample given a prev phoneme)
    bi = bi * w[None, :]
    row_sums = bi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    bi = bi / row_sums
    return uni, bi


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
    p.add_argument("--bias", default="none", choices=sorted(BIAS_PRESETS.keys()),
                   help="reshape sampling distribution toward a target wordless-vocal palette")
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

    # apply bias preset (no-op if "none")
    uni, bi = apply_bias(uni, bi, args.bias)

    vowel_mask = np.array([ph in VOWELS for ph in PHONEMES], dtype=bool)

    # 39x39 panphon feature-edit-distance matrix (verified empirically).
    print("building panphon distance matrix...", file=sys.stderr)
    dist_matrix = build_distance_matrix()

    # Per-phoneme bias weights (the dreamy/sigur-ros/fraser multipliers as a 39-vector)
    # for use as the importance term in corrupt_phonemes.py Boltzmann substitution.
    bias_w = BIAS_PRESETS[args.bias]
    bias_weights = np.array([bias_w.get(ph, 1.0) for ph in PHONEMES], dtype=np.float32)

    np.savez(out_path, phonemes=np.array(PHONEMES), vowel_mask=vowel_mask,
             unigram=uni, bigram=bi, bias=np.array(args.bias),
             dist_matrix=dist_matrix, bias_weights=bias_weights)
    print(f"fit {nlines} CMUdict pronunciations -> {out_path} (bias={args.bias})")
    print(f"  {n} phonemes ({int(vowel_mask.sum())} vowels / {int((~vowel_mask).sum())} consonants)")
    print(f"  top-5 unigram: " + ", ".join(
        f"{PHONEMES[i]}={uni[i]:.3f}" for i in np.argsort(uni)[::-1][:5]))
    print(f"  dist_matrix: shape {dist_matrix.shape}, range [{dist_matrix.min():.1f}, {dist_matrix.max():.1f}]")


if __name__ == "__main__":
    main()
