"""Generate the Coherence Dial training corpus.

For each (sentence, voice, level): corrupt the sentence's phonemes at p(level), synthesize the
result with the BASE F5-TTS in that voice, and write a wav + a row in manifest.jsonl. The
manifest pairs each synthesized clip with its ORIGINAL sentence + level + voice — that's the
(input, level) -> audio mapping the LoRA fine-tune later learns.

Honest scaling note (read before running on Colab):
  F5-TTS ~5-7s per generated clip on A100. So:
    500 sentences x 1 voice x 5 levels =  2,500 clips  ~  3-4 h  (single-voice spike)
    500 sentences x 3 voices x 5 levels = 7,500 clips  ~  10-12 h
   1500 sentences x 3 voices x 5 levels = 22,500 clips ~  30-40 h  (full scale; needs multi-session)
  Default below is the single-voice spike that fits a 10h wall-clock budget alongside everything
  else. Override --max-sentences and pass multiple --voice for larger runs.

Layout under --out:
  data/coherence/
    clip_00000_v1_lv0.wav
    clip_00000_v1_lv0.json        # per-clip metadata
    manifest.jsonl                # one line per clip
    SUMMARY.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_sentences(path: Path, n: int):
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if n and n < len(lines):
        lines = lines[:n]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", required=True, help="text file, one sentence per line")
    p.add_argument("--voice", action="append", required=True,
                   help="voice id + wav path + optional ref text, e.g. v1:data/voices/v1.wav:data/voices/v1.txt; pass multiple times")
    p.add_argument("--lm", default="data/phoneme_lm.npz")
    p.add_argument("--out", default="data/coherence")
    p.add_argument("--max-sentences", type=int, default=500,
                   help="cap sentence count (default 500 for single-voice spike)")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--input-mode", choices=["pseudo", "ipa", "text", "mondegreen"], default="pseudo",
                   help="pseudo/ipa = phoneme corruption (Tongues mode targets); mondegreen = "
                        "real-English-words phonetic ghost (Ghost mode targets); text = no corruption.")
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--model", default="F5TTS_v1_Base", help="F5-TTS variant identifier")
    p.add_argument("--remove-silence", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip clips whose wav already exists")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="When parallelizing across N workers, this worker's shard index (0..N-1). Used "
                        "to take sentences[shard_idx::shard_count] and to offset the clip index so "
                        "filenames don't collide across workers.")
    p.add_argument("--shard-count", type=int, default=1,
                   help="Total number of parallel workers (default 1 = no sharding).")
    args = p.parse_args()

    # ---- inputs ----
    sentences = load_sentences(Path(args.sentences), args.max_sentences)
    # Sharding: each worker takes a contiguous slice. We use CONTIGUOUS not strided
    # so the global clip index for a given (worker, local_idx) is just
    # `shard_idx * len(local_sentences) + local_idx`, which keeps filenames stable
    # across reruns and lets --resume work after a cancel.
    if args.shard_count > 1:
        n = len(sentences)
        per = (n + args.shard_count - 1) // args.shard_count  # ceil-div
        start = args.shard_idx * per
        end = min(start + per, n)
        sentences = sentences[start:end]
        clip_idx_offset = start * len(args.voice) * args.levels
        print(f"  shard {args.shard_idx}/{args.shard_count}: sentences[{start}:{end}] ({len(sentences)} sents), "
              f"clip_idx_offset={clip_idx_offset}", file=sys.stderr)
    else:
        clip_idx_offset = 0
    voices = []
    for v in args.voice:
        parts = v.split(":")
        vid = parts[0]
        wav = parts[1]
        ref_text_path = parts[2] if len(parts) > 2 else None
        ref_text = ""
        if ref_text_path and Path(ref_text_path).exists():
            ref_text = Path(ref_text_path).read_text(encoding="utf-8").strip()
        voices.append({"id": vid, "wav": wav, "ref_text": ref_text})

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corrupt_phonemes import load_lm, corrupt_sentence, LEVEL_P
    lm = load_lm(Path(args.lm))

    # Mondegreen index + DistilGPT-2 reranker (Ghost mode only). Heavy: ~5s + ~10s LM load.
    # Without the reranker the independent path picks weighted by inverse phoneme distance
    # alone, which surfaces rare CMUdict entries (surnames, archaic words). With the LM,
    # common English wins by ~30 nats per token, so the output reads as ordinary speech.
    mondegreen_idx = None
    mondegreen_reranker = None
    if args.input_mode == "mondegreen":
        from mondegreen import MondegreenIndex, LMReranker
        cmu_path = Path(args.lm).parent / "cmudict.dict"
        mondegreen_idx = MondegreenIndex(cmu_path)
        print(f"Mondegreen index: {mondegreen_idx.size} words", file=sys.stderr)
        print("Loading DistilGPT-2 reranker (CPU)...", file=sys.stderr)
        mondegreen_reranker = LMReranker()
        print("  reranker ready", file=sys.stderr)

    total = len(sentences) * len(voices) * args.levels
    print(f"Generating {total} clips: {len(sentences)} sentences x {len(voices)} voices x {args.levels} levels",
          file=sys.stderr)

    # ---- F5-TTS init ----
    try:
        from f5_tts.api import F5TTS
    except ImportError:
        print("ERROR: f5-tts is not installed. `pip install f5-tts` (typically on Colab GPU).", file=sys.stderr)
        sys.exit(1)
    tts = F5TTS(model=args.model)

    # Per-shard manifests so parallel workers don't clobber each other. The
    # build_coherence_dataset.py step concatenates `manifest*.jsonl` at merge time.
    manifest_name = f"manifest_shard{args.shard_idx}.jsonl" if args.shard_count > 1 else "manifest.jsonl"
    manifest_path = out / manifest_name
    manifest_f = manifest_path.open("a" if args.resume else "w")

    idx = 0
    written = 0
    skipped = 0
    for si, sentence in enumerate(sentences):
        # cache corrupted texts per (sentence, level) — same across voices
        corrupted_by_level = {}
        for lv in range(args.levels):
            seed = args.seed_base + si * 31 + lv
            arpa, ipa, pseudo, display = corrupt_sentence(sentence, lv, lm, seed=seed)
            mond = mondegreen_idx.substitute(sentence, lv, seed=seed,
                                              reranker=mondegreen_reranker) if mondegreen_idx else ""
            corrupted_by_level[lv] = {"arpabet": " ".join(t for t in arpa if t.strip()),
                                      "ipa": ipa, "pseudo": pseudo, "display": display,
                                      "mondegreen": mond}

        for voice in voices:
            for lv in range(args.levels):
                # Global clip index = local idx + shard offset, so each shard's filenames
                # land in a non-overlapping range and the merged corpus is contiguous.
                global_idx = clip_idx_offset + idx
                name = f"clip_{global_idx:05d}_{voice['id']}_lv{lv}"
                wav_path = out / f"{name}.wav"
                meta_path = out / f"{name}.json"
                idx += 1
                if args.resume and wav_path.exists():
                    skipped += 1
                    continue

                gen = corrupted_by_level[lv]
                gen_text = {"pseudo": gen["pseudo"], "ipa": gen["ipa"],
                            "text": sentence, "mondegreen": gen["mondegreen"]}[args.input_mode]
                if not gen_text.strip():
                    continue

                try:
                    tts.infer(
                        ref_file=voice["wav"], ref_text=voice["ref_text"],
                        gen_text=gen_text, file_wave=str(wav_path),
                        seed=args.seed_base + si * 7 + lv, remove_silence=args.remove_silence,
                    )
                except Exception as e:
                    print(f"  TTS fail on {name}: {e}", file=sys.stderr); continue

                meta = {
                    "name": name, "sentence_id": si, "sentence": sentence,
                    "voice": voice["id"], "voice_wav": voice["wav"],
                    "level": lv, "level_p": LEVEL_P[lv],
                    "input_mode": args.input_mode, "gen_text": gen_text,
                    "arpabet": gen["arpabet"], "ipa": gen["ipa"],
                    "pseudo": gen["pseudo"], "display": gen["display"],
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                manifest_f.write(json.dumps({
                    "audio_path": str(wav_path.relative_to(out.parent)),
                    "sentence": sentence, "level": lv,
                    "voice": voice["id"], "voice_wav": voice["wav"],
                    "voice_ref_text": voice["ref_text"], "input_text": gen_text,
                }) + "\n")
                manifest_f.flush()
                written += 1
                if written % 25 == 0:
                    print(f"  written {written}/{total - skipped} (skipped {skipped})", file=sys.stderr)

    manifest_f.close()

    summary = {
        "sentences": len(sentences), "voices": [v["id"] for v in voices],
        "levels": args.levels, "input_mode": args.input_mode,
        "written": written, "skipped": skipped, "total_planned": total,
        "manifest": str(manifest_path),
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE: wrote {written}, skipped {skipped} -> {out}")


if __name__ == "__main__":
    main()
