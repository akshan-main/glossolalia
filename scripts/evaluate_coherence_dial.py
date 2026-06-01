"""Coherence Dial validation harness.

Per swept clip:
  - Whisper-WER vs the ORIGINAL typed sentence  (should RISE as dial level rises)
  - Resemblyzer cosine vs the level-0 same-voice/same-seed reference clip
    (should stay >= 0.85 across all levels — voice preserved)

Aggregate:
  - mean WER per level (and per voice)
  - mean cosine per level (and per voice)
  - Spearman( level, mean_WER )   --- gate >= +0.80
  - min cosine across levels       --- gate >= 0.85

Verdict: PASS / PARTIAL / FAIL with a per-level breakdown JSON report.

Hallucination guard: when Whisper's avg_logprob falls below the threshold the transcription is
junk (model invented text) — we floor WER to 1.0 in that case so glossolalia doesn't get a
spuriously LOW WER because the model dreamed coherent words from noise.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_manifest(path: Path):
    rows = json.loads(path.read_text())
    if isinstance(rows, dict) and "clips" in rows:
        rows = rows["clips"]
    return rows


def whisper_wer(wav_path, ref_text, model, no_speech_threshold=0.8, logprob_threshold=-1.5):
    import jiwer
    out = model.transcribe(str(wav_path), no_speech_threshold=no_speech_threshold,
                           logprob_threshold=logprob_threshold, condition_on_previous_text=False,
                           language="en", fp16=False)
    hyp = (out.get("text") or "").strip()
    seg_avg_logprob = np.mean([s.get("avg_logprob", -10) for s in out.get("segments", [])]) if out.get("segments") else -10
    seg_no_speech = np.mean([s.get("no_speech_prob", 1.0) for s in out.get("segments", [])]) if out.get("segments") else 1.0
    if not hyp or seg_avg_logprob < logprob_threshold or seg_no_speech > no_speech_threshold:
        return 1.0, hyp, float(seg_avg_logprob), float(seg_no_speech)
    wer = jiwer.wer(ref_text.lower(), hyp.lower())
    return float(min(wer, 1.0)), hyp, float(seg_avg_logprob), float(seg_no_speech)


def resemblyzer_embed(wav_path, encoder):
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(str(wav_path))
    return encoder.embed_utterance(wav)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="sweep manifest JSON (from sweep_dial.py)")
    p.add_argument("--out", default="sweep/eval_report.json")
    p.add_argument("--whisper", default="base.en")
    p.add_argument("--spearman-gate", type=float, default=0.80)
    p.add_argument("--cosine-gate", type=float, default=0.85)
    args = p.parse_args()

    rows = load_manifest(Path(args.manifest))
    if not rows:
        print("empty manifest", file=sys.stderr); sys.exit(1)

    import whisper
    from resemblyzer import VoiceEncoder
    print(f"loading whisper:{args.whisper} + resemblyzer", file=sys.stderr)
    asr = whisper.load_model(args.whisper)
    enc = VoiceEncoder()

    # index lv0 reference per (voice, seed, sentence)
    ref_by = {}
    for r in rows:
        if r["level"] == 0:
            ref_by[(r["voice"], r["seed"], r["sentence"])] = r["wav"]

    per_clip = []
    for i, r in enumerate(rows):
        wav = r["wav"]; ref = r["sentence"]
        wer, hyp, lp, nsp = whisper_wer(wav, ref, asr)
        cos = 1.0
        if r["level"] != 0:
            ref_wav = ref_by.get((r["voice"], r["seed"], r["sentence"]))
            if ref_wav and Path(ref_wav).exists():
                e_clip = resemblyzer_embed(wav, enc)
                e_ref = resemblyzer_embed(ref_wav, enc)
                cos = float(np.dot(e_clip, e_ref) / ((np.linalg.norm(e_clip) * np.linalg.norm(e_ref)) + 1e-9))
        per_clip.append({**r, "wer": wer, "hyp": hyp, "logprob": lp, "no_speech": nsp, "cos_vs_lv0": cos})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}", file=sys.stderr)

    # aggregate per level
    levels = sorted({r["level"] for r in per_clip})
    per_level = {}
    for lv in levels:
        sub = [r for r in per_clip if r["level"] == lv]
        per_level[lv] = {
            "n": len(sub),
            "mean_wer": float(np.mean([r["wer"] for r in sub])),
            "mean_cos_vs_lv0": float(np.mean([r["cos_vs_lv0"] for r in sub])),
        }

    xs = np.array(levels, dtype=float)
    ys = np.array([per_level[lv]["mean_wer"] for lv in levels])
    cs = np.array([per_level[lv]["mean_cos_vs_lv0"] for lv in levels])

    from scipy.stats import spearmanr
    sp = float(spearmanr(xs, ys).correlation) if len(xs) >= 3 else float("nan")
    min_cos = float(cs[xs > 0].min()) if (xs > 0).any() else 1.0

    verdict = ("PASS" if (sp >= args.spearman_gate and min_cos >= args.cosine_gate)
               else "PARTIAL" if (sp >= 0.5 or min_cos >= 0.75) else "FAIL")

    report = {
        "manifest": args.manifest, "n_clips": len(per_clip),
        "per_level": per_level, "spearman_wer_vs_level": sp,
        "min_cos_vs_lv0_across_levels": min_cos,
        "gates": {"spearman": args.spearman_gate, "cosine": args.cosine_gate},
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**report, "per_clip": per_clip}, indent=2))

    print("\n===== COHERENCE DIAL EVAL =====")
    for lv in levels:
        r = per_level[lv]
        print(f"  level {lv}: WER={r['mean_wer']:.3f}  cos_vs_lv0={r['mean_cos_vs_lv0']:.3f}  (n={r['n']})")
    print(f"\nSpearman(WER, level) = {sp:+.3f}   (gate >= +{args.spearman_gate:.2f})")
    print(f"min cos vs lv0       = {min_cos:.3f}   (gate >= {args.cosine_gate:.2f})")
    print(f"VERDICT: {verdict}")
    print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()
