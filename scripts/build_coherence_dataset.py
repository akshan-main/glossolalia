"""Convert generate_coherence_data's manifest.jsonl into the format F5-TTS finetune expects.

F5-TTS finetune (per upstream README) consumes either:
  (a) a directory with per-clip `<name>.wav` + `<name>.txt`  (transcript-per-clip), or
  (b) a CSV/JSONL `audio_path|text` manifest.

For our run we emit BOTH:
  - <out>/wavs/<name>.wav  (copies/links the audio)
  - <out>/wavs/<name>.txt  (the LoRA's input text: original sentence + control token)
  - <out>/metadata.csv     (audio_path|text, pipe-separated, F5-TTS default)
  - <out>/metadata.jsonl   (richer record incl. voice + level for sweep filtering)

The IP is the `text` column: each clip's *input* during training is the ORIGINAL sentence +
the control token `tongues <level_word>`, while the *target audio* is the base-TTS render of
the corrupted-text we computed earlier. The LoRA learns the mapping from
  (original_sentence, level_token) -> corrupted_audio.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

LEVEL_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/coherence", help="dir produced by generate_coherence_data.py")
    p.add_argument("--out", default="data/coherence_ds", help="F5-TTS finetune dataset dir")
    p.add_argument("--stem", default="tongues", help="control-token stem")
    p.add_argument("--copy-audio", action="store_true", help="copy wavs into <out>/wavs (default: symlink)")
    p.add_argument("--max-rows", type=int, help="cap row count (for quick sanity)")
    args = p.parse_args()

    data = Path(args.data); out = Path(args.out)
    # Accept both single-manifest layout and multi-shard (manifest_shard*.jsonl) layout.
    # When shards are present we read them all so the parallel-fanout corpus is merged here.
    manifest_files = sorted(data.glob("manifest_shard*.jsonl"))
    if not manifest_files:
        single = data / "manifest.jsonl"
        if single.exists():
            manifest_files = [single]
    if not manifest_files:
        print(f"missing manifest in {data}", file=sys.stderr); sys.exit(1)
    print(f"reading {len(manifest_files)} manifest file(s)", file=sys.stderr)

    wavs_out = out / "wavs"
    wavs_out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "metadata.csv"
    jsonl_path = out / "metadata.jsonl"

    rows = 0
    skipped = 0
    with csv_path.open("w") as f_csv, jsonl_path.open("w") as f_json:
        f_csv.write("audio_path|text\n")
        manifest_lines = []
        for mf in manifest_files:
            with mf.open() as f_in:
                manifest_lines.extend(f_in.readlines())
        for line in manifest_lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            wav_src = Path(r["audio_path"])
            if not wav_src.is_absolute():
                wav_src = (data.parent / wav_src).resolve()
            if not wav_src.exists():
                skipped += 1; continue

            name = wav_src.stem
            wav_dst = wavs_out / wav_src.name
            if not wav_dst.exists():
                if args.copy_audio:
                    shutil.copy(wav_src, wav_dst)
                else:
                    try:
                        wav_dst.symlink_to(wav_src)
                    except OSError:
                        shutil.copy(wav_src, wav_dst)
            text = f"{r['sentence']} | {args.stem} {LEVEL_WORDS[r['level']]}"
            (wavs_out / f"{name}.txt").write_text(text, encoding="utf-8")
            f_csv.write(f"wavs/{wav_src.name}|{text}\n")
            f_json.write(json.dumps({
                "audio_path": f"wavs/{wav_src.name}", "text": text,
                "sentence": r["sentence"], "level": r["level"], "voice": r["voice"],
                "voice_wav": r["voice_wav"], "voice_ref_text": r.get("voice_ref_text", ""),
            }) + "\n")
            rows += 1
            if args.max_rows and rows >= args.max_rows:
                break

    print(f"wrote {rows} rows ({skipped} missing) -> {out}")
    print(f"  - {csv_path}")
    print(f"  - {jsonl_path}")
    print(f"  - {wavs_out} ({'copies' if args.copy_audio else 'symlinks'})")


if __name__ == "__main__":
    main()
