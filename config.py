"""glossolalia — Coherence Dial configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
ADAPTERS_ROOT = PROJECT_ROOT / "adapters"
STATIC_ROOT = PROJECT_ROOT / "static"

# TTS base + adapter
TTS_MODEL_HF = "SWivid/F5-TTS"       # F5-TTS, ~336M, voice-cloning + IPA input
TTS_MODEL_VARIANT = "F5TTS_v1_Base"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
}

# Dial config — control token = "<CONTROL_STEM> <LEVEL_WORDS[level]>"
CONTROL_STEM = "tongues"
LEVEL_WORDS = ["zero", "one", "two", "three", "four"]
LEVEL_P = [0.0, 0.25, 0.50, 0.75, 1.0]   # phoneme substitution probability per level

# Voice presets (reference clips for cloning; populated by data prep)
VOICE_PRESETS = {
    "v1": {"name": "warm alto",   "wav": "data/voices/v1.wav", "ref_text": "data/voices/v1.txt"},
    "v2": {"name": "bright tenor","wav": "data/voices/v2.wav", "ref_text": "data/voices/v2.txt"},
    "v3": {"name": "soft mezzo",  "wav": "data/voices/v3.wav", "ref_text": "data/voices/v3.txt"},
}

# Validation gates
WHISPER_MODEL = "base.en"
RESEMBLYZER_MIN_COSINE = 0.85
SPEARMAN_GATE = 0.80                 # WER vs level Spearman; positive (WER rises as dial rises)

# Audio
SAMPLE_RATE = 24000                  # F5-TTS native
GEN_MAX_SEC = 30

# HF repos
HF_DATA_REPO = "akshan-main/glossolalia-inputs"
HF_LORA_REPO = "akshan-main/glossolalia-dial-lora"
HF_SPACE_REPO = "build-small-hackathon/glossolalia"
