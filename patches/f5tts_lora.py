"""Monkey-patch F5TTS to add load_lora(path) and set_dial(level).

Loads both the PEFT LoRA adapter AND the learnable per-level embedding from the same
directory (or HF repo). Backward compatible: dirs without level_embed.safetensors still
work, just without the level dial.

Architecture precedent: textual inversion (Gal et al. 2022, arXiv:2208.01618). A small
set of learnable embedding vectors in the text-encoder space, trained while the base
model is frozen. NOT Concept Sliders (scalar-modulated LoRA, different surface).

Decision provenance: DECISIONS.md "Per-level learnable conditioning tokens".
"""

import json
import os
from pathlib import Path
import types

import torch
import torch.nn as nn
from f5_tts.api import F5TTS


def _install_level_patch(base_dit, level_embed):
    """Attach level_embed + per-call level slot + patched get_input_embed to a live DiT."""
    base_dit.add_module("level_embed", level_embed)
    base_dit._current_levels = None

    def _patched_get_input_embed(self, x, cond, text,
                                 drop_audio_cond=False, drop_text=False,
                                 cache=True, audio_mask=None):
        if self.text_uncond is None or self.text_cond is None or not cache:
            seq_len = x.shape[1] if audio_mask is None else audio_mask.sum(dim=1)
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed
        if cache:
            text_embed = self.text_uncond if drop_text else self.text_cond
        if (not drop_text) and (getattr(self, "_current_levels", None) is not None):
            e = self.level_embed(self._current_levels.to(text_embed.device))
            text_embed = text_embed + e.unsqueeze(1).to(text_embed.dtype)
            if audio_mask is not None:
                text_embed = text_embed * audio_mask.unsqueeze(-1).to(text_embed.dtype)
        return self.input_embed(x, cond, text_embed,
                                drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)

    base_dit.get_input_embed = types.MethodType(_patched_get_input_embed, base_dit)


def _resolve_base_dit(peft_or_dit):
    """PEFT-wrap may or may not be present. Walk down to the live DiT instance."""
    bm = getattr(peft_or_dit, "base_model", None)
    if bm is None:
        return peft_or_dit
    return getattr(bm, "model", bm)


def _load_lora(self, path):
    """Wrap cfm.transformer with PEFT + optionally load level_embed."""
    from peft import PeftModel
    from safetensors.torch import load_file
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None

    local_dir = path
    if not os.path.isdir(path) and snapshot_download is not None:
        local_dir = snapshot_download(repo_id=path)

    target = self.ema_model.transformer
    self.ema_model.transformer = PeftModel.from_pretrained(target, local_dir).to(self.device)
    self.ema_model.transformer.eval()

    le_path = Path(local_dir) / "level_embed.safetensors"
    meta_path = Path(local_dir) / "level_embed.json"
    if le_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        base_dit = _resolve_base_dit(self.ema_model.transformer)
        le = nn.Embedding(meta["num_levels"], meta["text_dim"])
        sd = load_file(str(le_path))
        le.load_state_dict({"weight": sd["weight"]})
        le = le.to(device=self.device, dtype=next(base_dit.parameters()).dtype)
        le.eval()
        _install_level_patch(base_dit, le)

    self._lora_path = path
    return self


def _set_dial(self, level):
    """Set the per-call glossolalia level for the next infer(). No-op if no level_embed."""
    base_dit = _resolve_base_dit(self.ema_model.transformer)
    if not hasattr(base_dit, "level_embed"):
        return self
    base_dit._current_levels = torch.tensor([int(level)], device=self.device, dtype=torch.long)
    base_dit.text_cond = None
    base_dit.text_uncond = None
    return self


def install_load_lora():
    if not hasattr(F5TTS, "load_lora"):
        F5TTS.load_lora = _load_lora
    if not hasattr(F5TTS, "set_dial"):
        F5TTS.set_dial = _set_dial
