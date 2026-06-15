"""One-command pipeline that takes a fresh LoRA from the Modal volume and ships it
to the HF Space at akshan-main/glossolalia.

Steps:
  1. Run push_lora_to_hf.py (Modal volume -> akshan-main/glossolalia-dial-lora).
  2. Sync the Gradio app files to the Space repo.
     Pushed: app.py, requirements.txt, README.md, config.py, patches.py,
             scripts/mondegreen.py, scripts/corrupt_phonemes.py, static/*,
             data/voices/ (the 9 inference voices), data/cmudict.dict,
             data/phoneme_lm.npz.
  3. Trigger Space rebuild (implicit on file push).

The Space pulls the LoRA at runtime via app.py's HF_LORA_REPO env (set in Space settings,
defaults to akshan-main/glossolalia-dial-lora in config.py).

Usage:
    python scripts/deploy_space.py
    python scripts/deploy_space.py --skip-lora    # only sync code, don't re-push adapter
    python scripts/deploy_space.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HF_SPACE = "akshan-main/glossolalia"

# Files relative to repo root that must be present on the Space for app.py to run.
SPACE_FILES = [
    "app.py",
    "config.py",
    "requirements.txt",
    "README.md",
    "patches/__init__.py",
    "patches/f5tts_lora.py",
    "scripts/__init__.py",  # so `from scripts.X import ...` resolves
    "scripts/mondegreen.py",
    "scripts/corrupt_phonemes.py",
    "scripts/post_fx.py",  # imported by app.py for the post-FX bus
]

# Directories that get pushed wholesale.
SPACE_DIRS = [
    "static",
    "data/voices",
]

# Single extra data files referenced at runtime (cmudict + phoneme LM).
SPACE_DATA_FILES = [
    "data/cmudict.dict",
    "data/phoneme_lm.npz",
]


def _hf_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path.home() / ".cache/huggingface/token"
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError("No HF token: set HF_TOKEN or ~/.cache/huggingface/token")


def run_lora_push() -> None:
    """Invoke push_lora_to_hf.py to ship the adapter."""
    script = Path(__file__).parent / "push_lora_to_hf.py"
    subprocess.check_call([sys.executable, str(script)])


def _resolve(repo_root: Path, p: str) -> Path:
    return repo_root / p


def sync_to_space(repo_root: Path, dry_run: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token())
    api.create_repo(HF_SPACE, repo_type="space", exist_ok=True,
                    space_sdk="gradio", private=False)

    uploaded = 0
    skipped = 0

    def push(local: Path, remote: str) -> None:
        nonlocal uploaded, skipped
        if not local.exists():
            print(f"  SKIP (missing): {local}")
            skipped += 1
            return
        print(f"  {local}  ->  {HF_SPACE}:{remote}")
        if dry_run:
            return
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=HF_SPACE,
            repo_type="space",
            commit_message=f"sync {remote}",
        )
        uploaded += 1

    # Plain files.
    for f in SPACE_FILES:
        push(_resolve(repo_root, f), f)
    for f in SPACE_DATA_FILES:
        push(_resolve(repo_root, f), f)

    # Whole dirs.
    for d in SPACE_DIRS:
        ld = _resolve(repo_root, d)
        if not ld.exists():
            print(f"  SKIP dir (missing): {ld}")
            continue
        for f in ld.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(repo_root))
                push(f, rel)

    print(f"Done. Uploaded {uploaded} files, skipped {skipped}. Space: https://huggingface.co/spaces/{HF_SPACE}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-lora", action="store_true",
                   help="Don't re-push the LoRA, only sync app code.")
    p.add_argument("--dry-run", action="store_true",
                   help="List files that would be pushed, don't actually push.")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if not args.skip_lora:
        print("=== Step 1: pushing LoRA adapter to model repo ===")
        run_lora_push()

    print()
    print("=== Step 2: syncing Gradio app to Space ===")
    sync_to_space(repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
