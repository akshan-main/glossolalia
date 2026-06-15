"""Glossolalia Dial patches package.

Importing this package installs runtime monkey-patches against the upstream f5-tts API that
we depend on but that aren't yet merged upstream (see DECISIONS.md "F5-TTS LoRA path = DIY
PEFT" entry for the reasoning + the F5-TTS LoRA path workflow that picked this approach).

Currently:
  - torchaudio.load -> soundfile: read WAV via libsndfile instead of torchaudio's
    torchcodec path (torchcodec native libs don't load on ZeroGPU). Installed FIRST,
    before f5_tts is imported anywhere, so its ref-audio load uses it.
  - F5TTS.load_lora(path) — accepts a PEFT adapter directory OR an HF model repo id; wraps
    the underlying CFM.transformer with PeftModel.from_pretrained. Required by app.py and
    scripts/sweep_dial.py which both call self._tts.load_lora(path).
"""

from .torchaudio_soundfile import install_soundfile_load

install_soundfile_load()

from .f5tts_lora import install_load_lora

install_load_lora()
