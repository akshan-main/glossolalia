"""Push the locally-prepared data inputs (sentences + voices + phoneme LM + CMUdict cache)
to a HuggingFace dataset repo so the Colab notebook can `snapshot_download` them in Cell 2.

Default target: akshan-main/glossolalia-inputs (override via --repo).
"""

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data", help="local dir with sentences.txt, voices/, phoneme_lm.npz, cmudict.dict")
    p.add_argument("--repo", default="akshan-main/glossolalia-inputs")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    required = ["sentences.txt", "phoneme_lm.npz", "voices"]
    missing = [r for r in required if not (src / r).exists()]
    if missing:
        print(f"ERROR: missing in {src}: {missing}", file=sys.stderr); sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"uploading {src} -> dataset {args.repo} (private={args.private})", file=sys.stderr)
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["sentences.txt", "phoneme_lm.npz", "cmudict.dict",
                        "voices/*.wav", "voices/*.txt"],
    )
    print(f"DONE: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
