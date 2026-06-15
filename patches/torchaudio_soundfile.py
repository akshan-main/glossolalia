"""Read audio via soundfile (libsndfile) instead of torchaudio's torchcodec path.

torchaudio 2.8+ removed its soundfile/ffmpeg load backends and routes `torchaudio.load`
through torchcodec. torchcodec's native libs do not load inside ZeroGPU's transient-GPU
sandbox (the GPU build needs CUDA runtime libs that don't resolve at dlopen time, and the
CPU build isn't ABI-compatible with the torch ZeroGPU pins). Our reference voices are all
WAV, so we read them with soundfile directly. This is not a "fallback" wrapper: it is the
exact operation torchaudio's deleted soundfile backend performed (libsndfile -> tensor).

Installed before f5_tts is imported so its `torchaudio.load(ref_audio)` call uses this.
"""

from __future__ import annotations


def install_soundfile_load() -> None:
    import torch
    import torchaudio
    import soundfile as sf

    if getattr(torchaudio, "_glossolalia_sf_load", False):
        return

    def _sf_load(uri, *args, **kwargs):
        # soundfile -> data [frames, channels] float32, sr int
        data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T).contiguous()  # -> [channels, frames]
        return wav, sr

    torchaudio.load = _sf_load
    torchaudio._glossolalia_sf_load = True
    print("[patches] torchaudio.load -> soundfile (libsndfile) installed")
