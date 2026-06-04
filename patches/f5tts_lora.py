"""Monkey-patch F5TTS to add load_lora(path) and set_dial(level).

v5 architecture: AdaLN-side learnable level conditioning. A LevelEmbed MLP
(Linear(1, dim) -> SiLU -> Linear(dim, dim)) maps a continuous scalar in [0,1]
to a vector ADDED to t after DiT.time_embed(time), before any transformer
block. t drives AdaLayerNorm in all 22 DiTBlocks + AdaLayerNorm_Final. Same
surface F5-TTS uses for time conditioning -- already trusted to globally
modulate hidden states at every layer.

Precedent: DiT adaLN-Zero (Peebles & Xie 2022, arXiv:2212.09748 sec 3.2)
sums timestep + class-label before SiLU+Linear. Concept Sliders
(Gandikota et al., ECCV 2024, arXiv:2311.12092) for scalar-modulated graded
control. FiLM (Perez 2017, arXiv:1709.07871).

Why v4 (text-embed bias) failed: text is consumed ONCE at the input
projection (concat with mel+cond, bare Linear). Blocks see only (x, t).
A text_embed bias gets averaged spatially and linearly absorbed; downstream
blocks ignore it. Whisper picked up sub-perceptual artifacts
(Carlini-Wagner 2018, arXiv:1801.01944), giving false Spearman +0.975
while spectral corr stayed at 0.97 between dial=0 and dial=4.

Backward compatible: dirs with the old level_embed.safetensors (v3) or
direction.safetensors (v4) still load with their original behavior.

Decision provenance: DECISIONS.md "v5 AdaLN-side level conditioning".
"""

import json
import os
from pathlib import Path
import types

import torch
import torch.nn as nn
from f5_tts.api import F5TTS

NUM_LEVELS = 5  # dial 0..4
DIM = 1024      # F5-TTS v1 Base DiT hidden dim


class _V5LevelEmbed(nn.Module):
    """Re-built at load time. Same architecture as Cell 5 LevelEmbed."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, s):
        return self.net(s)


def _install_v5_patch(base_dit, level_embed):
    """Patch DiT.forward to shift t by level_embed(scalar) before blocks."""
    base_dit.add_module("level_embed", level_embed)
    base_dit._current_scalars = None

    def _patched_forward(self, x, cond, text, time, mask=None,
                         drop_audio_cond=False, drop_text=False,
                         cfg_infer=False, cache=False):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        sc = getattr(self, "_current_scalars", None)
        if sc is not None:
            s = sc.to(device=t.device, dtype=t.dtype).view(-1, 1)
            if s.shape[0] == 1 and t.shape[0] > 1:
                s = s.expand(t.shape[0], 1)
            t = t + self.level_embed(s)
        seq_len = x.shape[1]
        if cfg_infer:
            x_c = self.get_input_embed(x, cond, text, drop_audio_cond=False,
                                       drop_text=False, cache=cache, audio_mask=mask)
            x_u = self.get_input_embed(x, cond, text, drop_audio_cond=True,
                                       drop_text=True, cache=cache, audio_mask=mask)
            x = torch.cat((x_c, x_u), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(x, cond, text,
                                     drop_audio_cond=drop_audio_cond,
                                     drop_text=drop_text, cache=cache, audio_mask=mask)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)
        if residual is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = self.norm_out(x, t)
        return self.proj_out(x)

    base_dit.forward = types.MethodType(_patched_forward, base_dit)


def _resolve_base_dit(peft_or_dit):
    bm = getattr(peft_or_dit, "base_model", None)
    if bm is None:
        return peft_or_dit
    return getattr(bm, "model", bm)


def _load_lora(self, path):
    """Wrap cfm.transformer with PEFT + optionally load level conditioning."""
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

    # v5 (preferred): level_to_time.safetensors holds LevelEmbed MLP state_dict
    v5_path = Path(local_dir) / "level_to_time.safetensors"
    if v5_path.exists():
        sd = load_file(str(v5_path))
        # Keys are 'net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias'
        le = _V5LevelEmbed(DIM)
        le.load_state_dict({k.replace("net.", "net."): v for k, v in sd.items()})
        le = le.to(device=self.device, dtype=dtype).eval()
        _install_v5_patch(base_dit, le)
        self._dial_mode = "v5"
        self._lora_path = path
        return self

    # v4 fallback: scalar-modulated single direction
    dir_path = Path(local_dir) / "direction.safetensors"
    if dir_path.exists():
        sd = load_file(str(dir_path))
        d = nn.Parameter(sd["direction"].to(device=self.device, dtype=dtype),
                         requires_grad=False)
        base_dit.register_parameter("direction", d)
        base_dit._current_scalars = None

        def _v4_get_input_embed(self, x, cond, text,
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
            if (not drop_text) and (getattr(self, "_current_scalars", None) is not None):
                scalars = self._current_scalars.to(text_embed.device).to(text_embed.dtype)
                bias = scalars.view(-1,1,1) * self.direction.view(1,1,-1).to(text_embed.dtype)
                text_embed = text_embed + bias
                if audio_mask is not None:
                    text_embed = text_embed * audio_mask.unsqueeze(-1).to(text_embed.dtype)
            return self.input_embed(x, cond, text_embed,
                                    drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)
        base_dit.get_input_embed = types.MethodType(_v4_get_input_embed, base_dit)
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
        base_dit.add_module("level_embed", le)
        base_dit._current_levels = None
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
        base_dit.get_input_embed = types.MethodType(_v3_get_input_embed, base_dit)
        self._dial_mode = "v3"

    self._lora_path = path
    return self


def _set_dial(self, level):
    """Set the dial. v5/v4: scalar = level/(NUM_LEVELS-1). v3: int level."""
    base_dit = _resolve_base_dit(self.ema_model.transformer)
    mode = getattr(self, "_dial_mode", None)
    if mode == "v5" and hasattr(base_dit, "level_embed"):
        scalar = float(level) / float(NUM_LEVELS - 1)
        base_dit._current_scalars = torch.tensor([scalar], device=self.device, dtype=torch.float32)
    elif mode == "v4" and hasattr(base_dit, "direction"):
        scalar = float(level) / float(NUM_LEVELS - 1)
        base_dit._current_scalars = torch.tensor([scalar], device=self.device, dtype=torch.float32)
        base_dit.text_cond = None
        base_dit.text_uncond = None
    elif mode == "v3" and hasattr(base_dit, "level_embed"):
        base_dit._current_levels = torch.tensor([int(level)], device=self.device, dtype=torch.long)
        base_dit.text_cond = None
        base_dit.text_uncond = None
    return self


def install_load_lora():
    if not hasattr(F5TTS, "load_lora"):
        F5TTS.load_lora = _load_lora
    if not hasattr(F5TTS, "set_dial"):
        F5TTS.set_dial = _set_dial
