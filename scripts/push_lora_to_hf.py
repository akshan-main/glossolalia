"""Download the v5/v8 Tongues LoRA adapter from the Modal volume and push it to
`akshan-main/glossolalia-dial-lora` on HuggingFace.

Usage:
    python scripts/push_lora_to_hf.py
    python scripts/push_lora_to_hf.py --volume-path /v5_adapter   # default
    python scripts/push_lora_to_hf.py --dry-run

The Modal volume name and adapter path are pulled from `modal/app.py`. The HF token
must be in ~/.cache/huggingface/token or HF_TOKEN env.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MODAL_VOLUME = "glossolalia-v5"
DEFAULT_VOLUME_PATH = "/v5_adapter"
HF_REPO = "akshan-main/glossolalia-dial-lora"


def _hf_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path.home() / ".cache/huggingface/token"
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError("No HF token: set HF_TOKEN or ~/.cache/huggingface/token")


def download_from_volume(volume_path: str, out_dir: Path) -> None:
    """Pull the adapter dir off Modal volume into a local staging dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODAL_VOLUME}{volume_path} -> {out_dir}")
    subprocess.check_call([
        "modal", "volume", "get", "--force",
        MODAL_VOLUME, volume_path, str(out_dir),
    ])


def _list_local(d: Path) -> list[Path]:
    out: list[Path] = []
    for p in d.rglob("*"):
        if p.is_file():
            out.append(p)
    return out


def push_to_hf(local_dir: Path, dry_run: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token())
    # Make sure repo exists; harmless if it already does.
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=False)

    files = _list_local(local_dir)
    if not files:
        print(f"FAIL: no files under {local_dir} to upload")
        sys.exit(1)
    print(f"Uploading {len(files)} files to {HF_REPO}")
    for f in files:
        rel = f.relative_to(local_dir)
        # Strip the leading dir name (volume get nests the adapter under its own dir)
        # so the repo lands files at root (adapter_config.json, adapter_model.safetensors,...)
        # rather than v5_adapter/adapter_config.json.
        parts = list(rel.parts)
        if len(parts) > 1 and parts[0].startswith("v5_adapter"):
            parts = parts[1:]
        target = "/".join(parts) if parts else f.name
        print(f"  {f}  ->  {HF_REPO}:{target}")
        if dry_run:
            continue
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=target,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"upload {target}",
        )
    if dry_run:
        print("(dry-run; nothing pushed)")
    else:
        print(f"Done. Adapter live at https://huggingface.co/{HF_REPO}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--volume-path", default=DEFAULT_VOLUME_PATH,
                   help="Modal volume path to the adapter dir (default /v5_adapter)")
    p.add_argument("--dry-run", action="store_true",
                   help="List the files that would be pushed, don't actually push.")
    args = p.parse_args()

    with tempfile.TemporaryDirectory(prefix="lora_push_") as td:
        staging = Path(td)
        download_from_volume(args.volume_path, staging)
        push_to_hf(staging, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
