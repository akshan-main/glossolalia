"""Run the trained Coherence Dial LoRA across (sentence, voice, level, seed) tuples.

For each (sentence, voice, level, seed):
  - prompt = "{sentence} | {stem} {level_word}"  (the LoRA's expected input format)
  - call F5-TTS + LoRA inference
  - write sweep/{voice}_lv{level}_s{seed}_{idx}.wav

Default sweep: 10 hold-out sentences x N voices x 5 levels x 3 seeds.
"""

import argparse
import json
import sys
from pathlib import Path

# Make `patches/` (at the repo root) importable when this script is run via `python scripts/sweep_dial.py`
# (Python adds the script's dir to sys.path[0], not the cwd, so we add the repo root explicitly).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LEVEL_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven"]
DEFAULT_HOLDOUT_SENTENCES = [
    "the river was wide and calm in the morning light",
    "she opened the old book and began to read aloud",
    "a quiet wind moved through the empty stone courtyard",
    "the children laughed and ran across the wet grass",
    "he wrote her a letter that he never sent",
    "the train arrived late and emptied into the rain",
    "small lights flickered in windows along the harbor",
    "no one knew what the dog had seen in the trees",
    "she remembered a song her grandmother used to hum",
    "the city slept under a thin veil of new snow",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True, help="path to LoRA adapter dir or .safetensors")
    p.add_argument("--voices", required=True,
                   help="comma-separated voice specs: v1:data/voices/v1.wav:data/voices/v1.txt,...")
    p.add_argument("--out", default="sweep")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--seeds", default="42,123,7")
    p.add_argument("--sentences", help="optional path; defaults to a built-in 10-sentence holdout")
    p.add_argument("--max-sentences", type=int, default=10)
    p.add_argument("--stem", default="tongues")
    p.add_argument("--model", default="F5TTS_v1_Base")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sentences = (Path(args.sentences).read_text().splitlines()
                 if args.sentences else DEFAULT_HOLDOUT_SENTENCES)
    sentences = [s.strip() for s in sentences if s.strip()][: args.max_sentences]

    voices = []
    for v in args.voices.split(","):
        parts = v.split(":")
        vid, wav = parts[0], parts[1]
        ref_text_path = parts[2] if len(parts) > 2 else None
        ref_text = Path(ref_text_path).read_text().strip() if ref_text_path and Path(ref_text_path).exists() else ""
        voices.append({"id": vid, "wav": wav, "ref_text": ref_text})

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    try:
        import patches  # noqa: F401 — installs F5TTS.load_lora (DECISIONS.md "F5-TTS LoRA path = DIY PEFT")
    except ImportError as e:
        print(f"could not import patches/ from {Path(__file__).resolve().parent.parent}: {e}", file=sys.stderr); sys.exit(1)
    try:
        from f5_tts.api import F5TTS
    except ImportError as e:
        print(f"could not import f5_tts: {e}", file=sys.stderr); sys.exit(1)
    tts = F5TTS(model=args.model)
    tts.load_lora(args.lora)

    rows = []
    total = len(sentences) * len(voices) * args.levels * len(seeds)
    print(f"sweeping {total} clips", file=sys.stderr)
    i = 0
    for si, sentence in enumerate(sentences):
        for voice in voices:
            for lv in range(args.levels):
                for sd in seeds:
                    name = f"{voice['id']}_lv{lv}_s{sd}_s{si}"
                    wav_p = out / f"{name}.wav"
                    prompt = f"{sentence} | {args.stem} {LEVEL_WORDS[lv]}"
                    try:
                        tts.infer(ref_file=voice["wav"], ref_text=voice["ref_text"],
                                  gen_text=prompt, file_wave=str(wav_p), seed=sd)
                    except Exception as e:
                        print(f"  fail {name}: {e}", file=sys.stderr); continue
                    rows.append({"name": name, "wav": str(wav_p), "sentence": sentence,
                                 "voice": voice["id"], "level": lv, "seed": sd})
                    i += 1
                    if i % 10 == 0:
                        print(f"  {i}/{total}", file=sys.stderr)
    (out / "sweep_manifest.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} clips + {out}/sweep_manifest.json")


if __name__ == "__main__":
    main()
