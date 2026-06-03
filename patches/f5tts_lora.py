"""Monkey-patch F5TTS to add load_lora(path) and set_dial(level).

v4 architecture: scalar-modulated single direction (Concept Sliders / Gandikota et al.,
ECCV 2024, arXiv:2311.12092 pattern). ONE direction vector d of shape (text_dim,) is
attached to the DiT. At inference, set_dial(level) sets scalar = level / (NUM_LEVELS-1),
and bias = scalar * d is added to the post-conv text embedding inside DiT.get_input_embed.

Why this replaces v3 (per-level nn.Embedding): per-level discrete embeddings have NO
monotonicity prior - the loss only enforces (level_i -> level_i output), not that
level_2 sits geometrically between level_1 and level_3. v3 Cell 7 verdict was Spearman
+0.10 with non-monotonic WER ordering (0.13/0.50/1.00/0.15/0.13 across levels 0-4).
With v4, level k = (k/4) * d, so level 2.5 is exactly halfway between level 2 and 3.
Monotonicity is structural, not learned.

Insertion site verified: src/f5_tts/model/backbones/dit.py L284-312 (DiT.get_input_embed).
Backward compatible: dirs without direction.safetensors fall back to v3 level_embed if
present, then to no-bias if neither.

Decision provenance: DECISIONS.md "Scalar-modulated single direction (v4)".
"""

import json
import os
from pathlib import Path
import types

import torch
import torch.nn as nn
from f5_tts.api import F5TTS

NUM_LEVELS = 5  # dial 0..4


def _install_direction_patch(base_dit, direction):
    """Attach a single learnable direction + scalar slot + patched get_input_embed."""
    base_dit.register_parameter("direction", direction)
    base_dit._current_scalars = None  # set per-call to torch.FloatTensor[b]

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
        if (not drop_text) and (getattr(self, "_current_scalars", None) is not None):
            scalars = self._current_scalars.to(text_embed.device).to(text_embed.dtype)
            # bias = scalar * direction, broadcast to [b, 1, text_dim] then over seq axis
            bias = scalars.view(-1, 1, 1) * self.direction.view(1, 1, -1).to(text_embed.dtype)
            text_embed = text_embed + bias
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
    """Wrap cfm.transformer with PEFT + load v4 direction OR v3 level_embed if present."""
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
    base_dit = _resolve_base_dit(self.ema_model.transformer)
    dtype = next(base_dit.parameters()).dtype

    # v4: single direction (preferred)
    dir_path = Path(local_dir) / "direction.safetensors"
    dir_meta = Path(local_dir) / "direction.json"
    if dir_path.exists():
        sd = load_file(str(dir_path))
        d = nn.Parameter(sd["direction"].to(device=self.device, dtype=dtype),
                         requires_grad=False)
        _install_direction_patch(base_dit, d)
        self._dial_mode = "v4"
        self._lora_path = path
        return self

    # v3 fallback: per-level nn.Embedding (deprecated, kept for backward compat)
    le_path = Path(local_dir) / "level_embed.safetensors"
    meta_path = Path(local_dir) / "level_embed.json"
    if le_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        le = nn.Embedding(meta["num_levels"], meta["text_dim"])
        sd = load_file(str(le_path))
        le.load_state_dict({"weight": sd["weight"]})
        le = le.to(device=self.device, dtype=dtype).eval()
        # install v3 patch (per-level)
        def _v3_install(_base_dit, _le):
            _base_dit.add_module("level_embed", _le)
            _base_dit._current_levels = None
            def _v3_get_input_embed(self, x, cond, text,
                                    drop_audio_cond=False, drop_text=False,
                                    cache=True, audio_mask=None):
                if self.text_uncond is None or self.text_cond is None or not cache:
                    seq_len = x.shape[1] if audio_mask is None else audio_mask.sum(dim=1)
                    text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
                    if cache:
                        if drop_text: self.text_uncond = text_embed
                        else:         self.text_cond   = text_embed
                if cache:
                    text_embed = self.text_uncond if drop_text else self.text_cond
                if (not drop_text) and (getattr(self, "_current_levels", None) is not None):
                    e = self.level_embed(self._current_levels.to(text_embed.device))
                    text_embed = text_embed + e.unsqueeze(1).to(text_embed.dtype)
                    if audio_mask is not None:
                        text_embed = text_embed * audio_mask.unsqueeze(-1).to(text_embed.dtype)
                return self.input_embed(x, cond, text_embed,
                                        drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)
            _base_dit.get_input_embed = types.MethodType(_v3_get_input_embed, _base_dit)
        _v3_install(base_dit, le)
        self._dial_mode = "v3"

    self._lora_path = path
    return self


def _set_dial(self, level):
    """Set the dial for the next infer(). v4: scalar = level/(NUM_LEVELS-1). v3: int level."""
    base_dit = _resolve_base_dit(self.ema_model.transformer)
    mode = getattr(self, "_dial_mode", None)
    if mode == "v4" and hasattr(base_dit, "direction"):
        scalar = float(level) / float(NUM_LEVELS - 1)
        base_dit._current_scalars = torch.tensor([scalar], device=self.device, dtype=torch.float32)
    elif mode == "v3" and hasattr(base_dit, "level_embed"):
        base_dit._current_levels = torch.tensor([int(level)], device=self.device, dtype=torch.long)
    else:
        return self
    # bust the CFG text cache so the next forward picks up the new bias
    base_dit.text_cond = None
    base_dit.text_uncond = None
    return self


def install_load_lora():
    if not hasattr(F5TTS, "load_lora"):
        F5TTS.load_lora = _load_lora
    if not hasattr(F5TTS, "set_dial"):
        F5TTS.set_dial = _set_dial
