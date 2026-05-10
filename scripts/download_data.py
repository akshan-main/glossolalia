"""Download clips per subgenre from MTG-Jamendo (or FMA in v2)."""

import argparse
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SUBGENRES, DATA_ROOT, MTG_JAMENDO_REPO, FMA_REPO, CLIPS_PER_SUBGENRE


def download_from_mtg(subgenre, n_clips, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(MTG_JAMENDO_REPO, split="train", streaming=True)
    saved = 0
    for example in ds:
        genres = example.get("genres") or []
        if subgenre.tag not in genres:
            continue
        audio = example["audio"]
        path = out_dir / f"{saved:03d}.wav"
        sf.write(path, audio["array"], audio["sampling_rate"])
        saved += 1
        if saved >= n_clips:
            break
    return saved


def download_from_fma(subgenre, n_clips, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(FMA_REPO, split="train", streaming=True)
    saved = 0
    target = subgenre.tag.lower()
    for example in ds:
        genre_top = (example.get("genre_top") or "").lower()
        genres_all = [g.lower() for g in (example.get("genres_all") or [])]
        if target != genre_top and target not in genres_all:
            continue
        audio = example["audio"]
        path = out_dir / f"{saved:03d}.wav"
        sf.write(path, audio["array"], audio["sampling_rate"])
        saved += 1
        if saved >= n_clips:
            break
    return saved


SOURCE_DISPATCH = {
    "mtg": download_from_mtg,
    "fma": download_from_fma,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgenre", help="single subgenre name (default: all)")
    parser.add_argument("--clips", type=int, default=CLIPS_PER_SUBGENRE)
    args = parser.parse_args()

    targets = SUBGENRES
    if args.subgenre:
        targets = [s for s in SUBGENRES if s.name == args.subgenre]
        if not targets:
            print(f"Unknown subgenre: {args.subgenre}")
            sys.exit(1)

    for sg in targets:
        out = DATA_ROOT / sg.repo_slug / "audio"
        print(f"→ {sg.name} (source: {sg.source}, tag: {sg.tag})")
        downloader = SOURCE_DISPATCH[sg.source]
        n = downloader(sg, args.clips, out)
        print(f"  {n} clips → {out}")


if __name__ == "__main__":
    main()
