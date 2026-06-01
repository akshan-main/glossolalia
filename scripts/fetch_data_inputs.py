"""Fetch the two inputs the data-generation pipeline needs:
  1. data/sentences.txt  - N short English sentences (5..15 words), one per line
  2. data/voices/v{1,2,3}.wav + v{1,2,3}.txt  - 3 distinct-speaker reference clips (~6-12s each)
     + their transcripts (for F5-TTS voice-cloning conditioning).

Source: LibriTTS-R via HuggingFace datasets (single-speaker studio quality, openly licensed).
Falls back to LibriSpeech if LibriTTS-R isn't accessible. Streaming mode, so no GB-scale download.
"""

import argparse
import sys
from pathlib import Path


def _scan_dataset(hf_id, split, sentence_limit, voice_count, voice_min_sec, voice_max_sec):
    """Stream a TTS dataset and harvest sentences + 1 clip per first N distinct speakers.

    hf_id may be "owner/name" or "owner/name:config" (some datasets require a config name).
    """
    from datasets import load_dataset
    config = None
    if ":" in hf_id:
        hf_id, config = hf_id.split(":", 1)
    label = f"{hf_id}::{config or '*'}::{split}"
    print(f"streaming {label}", file=sys.stderr)
    kw = {"split": split, "streaming": True, "trust_remote_code": True}
    if config:
        ds = load_dataset(hf_id, config, **kw)
    else:
        ds = load_dataset(hf_id, **kw)

    sentences, seen_sentence = [], set()
    voices = []          # list of dicts: {sid, audio (np), sr, text}
    seen_sid = set()

    for row in ds:
        # Robustly pick fields across libritts / librispeech variants
        text = (row.get("text_normalized") or row.get("text") or row.get("transcription")
                or row.get("normalized_text") or row.get("sentence") or "").strip()
        sid = (row.get("speaker_id") or row.get("speaker") or row.get("client_id")
               or row.get("id", "")).__str__()
        audio = row.get("audio")
        if not text or not audio:
            continue
        nw = len(text.split())
        # collect sentences (5-15 words, unique-ish)
        if 5 <= nw <= 15 and len(sentences) < sentence_limit and text not in seen_sentence:
            sentences.append(text)
            seen_sentence.add(text)
        # collect 1 clip per first N speakers (clip must be in length window)
        if sid and sid not in seen_sid and len(voices) < voice_count:
            arr, sr = audio["array"], audio["sampling_rate"]
            dur = len(arr) / sr
            if voice_min_sec <= dur <= voice_max_sec:
                voices.append({"sid": sid, "audio": arr, "sr": sr, "text": text})
                seen_sid.add(sid)
        if len(sentences) >= sentence_limit and len(voices) >= voice_count:
            break

    return sentences, voices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--n-sentences", type=int, default=500)
    p.add_argument("--voices", type=int, default=3)
    p.add_argument("--voice-min-sec", type=float, default=6.0)
    p.add_argument("--voice-max-sec", type=float, default=12.0)
    p.add_argument("--datasets",
                   default="mythicinfinity/libritts_r:clean,openslr/librispeech_asr:clean",
                   help="comma-separated HF dataset ids (each may be 'owner/name' or 'owner/name:config')")
    p.add_argument("--split", default="train.clean.100")
    args = p.parse_args()

    out = Path(args.out)
    (out / "voices").mkdir(parents=True, exist_ok=True)

    sentences, voices = [], []
    last_err = None
    for ds_id in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        try:
            sentences, voices = _scan_dataset(
                ds_id, args.split, args.n_sentences, args.voices,
                args.voice_min_sec, args.voice_max_sec,
            )
            if sentences and len(voices) >= args.voices:
                print(f"OK from {ds_id}", file=sys.stderr); break
        except Exception as e:
            print(f"  {ds_id} failed: {e}", file=sys.stderr); last_err = e; continue
    if not sentences or len(voices) < args.voices:
        print("ERROR: no usable source dataset. Last error:", last_err, file=sys.stderr)
        print("Manual fallback: drop ≥3 single-speaker 6-12s WAVs in data/voices/v1.wav v2.wav v3.wav",
              file=sys.stderr)
        print("and write data/sentences.txt yourself.", file=sys.stderr)
        sys.exit(2)

    (out / "sentences.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")
    import soundfile as sf
    for i, v in enumerate(voices, start=1):
        wav_p = out / "voices" / f"v{i}.wav"
        txt_p = out / "voices" / f"v{i}.txt"
        sf.write(str(wav_p), v["audio"], v["sr"])
        txt_p.write_text(v["text"], encoding="utf-8")
        print(f"  v{i} ({v['sid']}): {len(v['audio'])/v['sr']:.1f}s -> {wav_p}", file=sys.stderr)
    print(f"\nwrote {len(sentences)} sentences -> {out/'sentences.txt'}")
    print(f"wrote {len(voices)} voice refs -> {out/'voices'}")


if __name__ == "__main__":
    main()
