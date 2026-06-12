"""Glossolalia Dial — a single dial that grades a typed lyric into dreamy territory in two
distinct phonotactic paths:

  Ghost mode: lyric is rewritten as a sequence of real English words (mondegreen substitution).
              Constrained by syllable count, primary-stress position, PanPhon feature-edit
              distance; reranked by DistilGPT-2 for semantic coherence. F5-TTS base reads it.
  Tongues mode: clean lyric goes into F5-TTS + a fine-tuned LoRA + a learned scalar conditioner
                (LevelEmbed at AdaLN side). The LoRA produces graded glossolalic audio in the
                user's chosen voice — invented pseudowords, sonorant-leaning palette.

Both modes ride F5-TTS for voice cloning + audio synthesis. Off-the-Grid: no cloud APIs.

v1 Gradio app (gr.Blocks). v2 (Off-Brand badge) is in app_server.py.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from config import (
    CONTROL_STEM, HF_LORA_REPO, LEVEL_WORDS, RESEMBLYZER_MIN_COSINE,
    SAMPLE_RATE, VOICE_PRESETS, WHISPER_MODEL,
)
from scripts.post_fx import PRESETS as POSTFX_PRESETS, apply_post_fx

VOICE_IDS = list(VOICE_PRESETS.keys())
DEFAULT_VOICE = VOICE_IDS[0] if VOICE_IDS else "v1"
DEFAULT_TEXT = "the river was wide and calm in the morning light"
# default to the published LoRA repo so the Space loads it without needing env vars;
# COHERENCE_DIAL_LORA env var overrides for local dev against a checkpoint dir.
LORA_PATH = os.environ.get("COHERENCE_DIAL_LORA", HF_LORA_REPO)

MODE_GHOST = "Ghost"
MODE_TONGUES = "Tongues"
MODES = (MODE_GHOST, MODE_TONGUES)


# ----- inference engine (lazy-loaded; falls back to a silent stub if F5-TTS isn't installed) -----

class TTSEngine:
    """Dual-mode engine. One F5-TTS instance with the v8 LoRA loaded; mode is selected per
    inference call. For Ghost mode we set_dial(0) (LevelEmbed contributes ~zero) and feed
    mondegreen-substituted text. For Tongues mode we set_dial(level) and feed the clean
    lyric. The base LoRA attention adaptation is always on, but at dial=0 it produces audio
    indistinguishable from F5-TTS base (verified empirically by v5 sweep — lv0 sounded
    identical to base output)."""

    def __init__(self):
        self._tts = None
        self._asr = None
        self._enc = None
        self._mondegreen = None
        self._lm = None
        self._lora_loaded = False
        self.live = False

    def _ensure_mondegreen(self):
        """Lazy load the deterministic phonetic-ghost generator + DistilGPT-2 reranker."""
        if self._mondegreen is not None:
            return
        try:
            from scripts.mondegreen import MondegreenIndex, LMReranker
            self._mondegreen = MondegreenIndex("data/cmudict.dict")
            print(f"[engine] Mondegreen index loaded ({self._mondegreen.size} words)")
            self._lm = LMReranker()
            print("[engine] DistilGPT-2 reranker loaded")
        except Exception as e:
            print(f"[engine] mondegreen load FAILED ({e}); Ghost mode falls back to clean text")
            self._mondegreen = False
            self._lm = False

    def ghost_text(self, sentence: str, level: int, seed: int = 42) -> str:
        """Deterministic Ghost mode substitution. Returns the source if mondegreen unavailable."""
        self._ensure_mondegreen()
        if not self._mondegreen:
            return sentence
        return self._mondegreen.substitute(sentence, level, seed=seed,
                                            reranker=(self._lm or None))

    def _ensure(self):
        if self._tts is not None:
            return
        try:
            import patches  # noqa: F401 — installs F5TTS.load_lora before instantiation
            from f5_tts.api import F5TTS
            self._tts = F5TTS(model="F5TTS_v1_Base")
            self.live = True
            print(f"[engine] F5-TTS base loaded (model=F5TTS_v1_Base)")
            if LORA_PATH:
                try:
                    self._tts.load_lora(LORA_PATH)
                    self._lora_loaded = True
                    print(f"[engine] LoRA loaded from {LORA_PATH}")
                except Exception as e:
                    print(f"[engine] LoRA load FAILED ({e}); falling back to base model — Well-Tuned badge forfeit")
            else:
                print("[engine] no LoRA path configured; running base model only")
        except ImportError:
            print("[engine] f5-tts not installed; running with silent stub for layout testing")
            self.live = False

    def _ensure_asr(self):
        if self._asr is None:
            try:
                import whisper
                self._asr = whisper.load_model(WHISPER_MODEL)
            except Exception:
                self._asr = False
        return self._asr

    def _ensure_encoder(self):
        if self._enc is None:
            try:
                from resemblyzer import VoiceEncoder
                self._enc = VoiceEncoder()
            except Exception:
                self._enc = False
        return self._enc

    def generate(self, sentence: str, voice_id: str, level: int, seed: int = 42,
                  mode: str = MODE_TONGUES):
        """Returns (audio float32 mono, sample_rate, gen_text_used).

        mode=MODE_GHOST: substitute lyric via mondegreen at given level, set_dial(0), TTS reads it.
        mode=MODE_TONGUES: leave lyric clean, set_dial(level), LoRA conditions glossolalic audio.
        """
        self._ensure()
        voice = VOICE_PRESETS[voice_id]
        if mode == MODE_GHOST:
            gen_text = self.ghost_text(sentence, level, seed=seed)
            tts_dial = 0
        else:
            gen_text = sentence
            tts_dial = level
        if not self.live:
            return np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE, gen_text
        if hasattr(self._tts, "set_dial"):
            try:
                self._tts.set_dial(tts_dial)
            except Exception as e:
                print(f"[engine] set_dial({tts_dial}) FAILED ({e}); proceeding without conditioning")
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        ref_text = ""
        ref_txt = Path(voice["ref_text"])
        if ref_txt.exists():
            ref_text = ref_txt.read_text(encoding="utf-8").strip()
        self._tts.infer(ref_file=voice["wav"], ref_text=ref_text,
                        gen_text=gen_text, file_wave=out, seed=seed)
        import soundfile as sf
        y, sr = sf.read(out, always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y.astype(np.float32), sr, gen_text

    def transcribe_wer(self, y: np.ndarray, sr: int, ref_text: str) -> float | None:
        asr = self._ensure_asr()
        if not asr:
            return None
        import soundfile as sf, jiwer
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(tmp, y, sr)
        out = asr.transcribe(tmp, fp16=False, language="en", condition_on_previous_text=False,
                             no_speech_threshold=0.8, logprob_threshold=-1.5)
        hyp = (out.get("text") or "").strip()
        if not hyp:
            return 1.0
        return float(min(jiwer.wer(ref_text.lower(), hyp.lower()), 1.0))

    def voice_cosine(self, y: np.ndarray, sr: int, ref_y: np.ndarray, ref_sr: int) -> float | None:
        enc = self._ensure_encoder()
        if not enc:
            return None
        from resemblyzer import preprocess_wav
        import soundfile as sf
        a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        b = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(a, y, sr); sf.write(b, ref_y, ref_sr)
        ea = enc.embed_utterance(preprocess_wav(a))
        eb = enc.embed_utterance(preprocess_wav(b))
        return float(np.dot(ea, eb) / ((np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-9))


ENGINE = TTSEngine()


# ----- crossfading for Morph mode -----

def equal_power_concat(clips, sr, fade_ms=200):
    if not clips:
        return np.zeros(1, dtype=np.float32)
    fade_n = max(1, int(sr * fade_ms / 1000))
    t = np.linspace(0, 1, fade_n, dtype=np.float32)
    fi = np.sin(t * np.pi / 2.0)
    fo = np.cos(t * np.pi / 2.0)
    out = clips[0].astype(np.float32).copy()
    for c in clips[1:]:
        c = c.astype(np.float32)
        if len(out) < fade_n or len(c) < fade_n:
            out = np.concatenate([out, c]); continue
        head = out[-fade_n:] * fo
        tail = c[:fade_n] * fi
        out = np.concatenate([out[:-fade_n], head + tail, c[fade_n:]])
    return out


def _wav_to_filepath(y: np.ndarray, sr: int) -> str:
    import soundfile as sf
    if y.ndim == 2:
        y = y.T  # (channels, samples) -> (samples, channels)
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, y, sr)
    return path


# ----- readout (live metrics strip) -----

def readout(level: int | None = None, wer: float | None = None,
            cosine: float | None = None, status: str = "") -> str:
    cells = [
        ("DIAL", f"{level}" if level is not None else "—"),
        ("WER", f"{wer:.2f}" if wer is not None else "—"),
        ("VOICE-SIM", f"{cosine:.2f}" if cosine is not None else "—"),
        ("STATUS", status or ("live" if ENGINE.live else "stub")),
    ]
    return "<div class='readout'>" + "".join(
        f"<div class='readout-cell'><div class='readout-label'>{k}</div><div class='readout-val'>{v}</div></div>"
        for k, v in cells
    ) + "</div>"


# ----- speak + morph handlers -----

def _safe_int(v, lo: int = 0, hi: int = 4) -> int:
    try:
        return max(lo, min(hi, int(float(v))))
    except (TypeError, ValueError):
        return lo


def speak(sentence: str, voice_id: str, level, postfx_preset: str, mode: str, seed: int = 42):
    sentence = (sentence or "").strip()
    level = _safe_int(level)
    if not sentence:
        return None, readout(level, None, None, "type a sentence first"), ""
    y, sr, gen_text = ENGINE.generate(sentence, voice_id, level, seed=int(seed), mode=mode)
    if postfx_preset != "dry":
        out_audio, _ = apply_post_fx(y, sr, preset=postfx_preset)
    else:
        out_audio = y
    path = _wav_to_filepath(out_audio, sr)
    # In Ghost mode the substituted lyric IS the artifact; show it. In Tongues mode the
    # gen_text equals the source lyric — nothing useful to display.
    readout_text = gen_text if (mode == MODE_GHOST and gen_text and gen_text != sentence) else ""
    return path, readout(level, None, None, f"{mode.lower()} · lv{level}"), readout_text


def morph(sentence: str, voice_id: str, postfx_preset: str, mode: str,
          seed: int = 42, gap_ms: int = 250):
    sentence = (sentence or "").strip()
    if not sentence:
        return None, readout(None, None, None, "type a sentence first"), ""
    clips = []
    for lv in range(5):
        y, sr, _ = ENGINE.generate(sentence, voice_id, lv, seed=int(seed), mode=mode)
        clips.append(y)
    morphed = equal_power_concat(clips, sr, fade_ms=gap_ms)
    if postfx_preset != "dry":
        morphed, _ = apply_post_fx(morphed, sr, preset=postfx_preset)
    path = _wav_to_filepath(morphed, sr)
    return path, readout(None, None, None, f"{mode.lower()} · morphed 0->4"), ""


# ----- CSS (dreamy pastel theme: half-remembered photograph of dusk) -----

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,700;1,9..144,300;1,9..144,400&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;1,6..72,300;1,6..72,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* DUSK — Vaughan-Oliver-leaning aesthetic. Warm dust palette, asymmetric editorial composition,
   brass knob with vermillion arc as the focal ritual object. */

:root {
    --paper: #F2E6D2;
    --paper-warm: #ECDABE;
    --paper-deep: #D9B89A;
    --ink: #2A1F2D;
    --ink-soft: #5C4D52;
    --ink-quiet: #8B7C7F;
    --vermillion: #B7472A;
    --vermillion-glow: rgba(183, 71, 42, 0.32);
    --brass: #A8884D;
    --brass-light: #D6B888;
    --brass-deep: #6B5328;
    --hairline: rgba(42, 31, 45, 0.16);
    --hairline-strong: rgba(42, 31, 45, 0.32);
}

html, body, .gradio-container, .dark, .light, gradio-app {
    background:
      radial-gradient(1200px 700px at 78% -8%, rgba(217,184,154,0.55), transparent 60%),
      radial-gradient(900px 600px at 12% 110%, rgba(183, 71, 42, 0.10), transparent 65%),
      linear-gradient(180deg, var(--paper) 0%, var(--paper-warm) 60%, var(--paper-deep) 120%) !important;
    background-attachment: fixed !important;
    color: var(--ink) !important;
    font-family: 'Newsreader', Georgia, serif !important;
}

/* film grain veil */
body::before, gradio-app::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.92' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.16 0 0 0 0 0.12 0 0 0 0 0.18 0 0 0 0.6 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
    opacity: 0.18; mix-blend-mode: multiply;
}

.gradio-container { max-width: 760px !important; margin: 0 auto !important; padding: 88px 28px 120px !important; position: relative; z-index: 1; }

/* -------------------- HERO -------------------- */

#hero { text-align: left; margin-bottom: 64px; padding-bottom: 32px; border-bottom: 1px solid var(--hairline); position: relative; }
#hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-weight: 400; font-size: 11px;
    color: var(--vermillion); letter-spacing: 0.34em; text-transform: uppercase;
    margin-bottom: 18px;
}
#hero h1 {
    font-family: 'Fraunces', serif; font-variation-settings: 'opsz' 144, 'SOFT' 100;
    font-style: italic; font-weight: 300; font-size: 110px;
    letter-spacing: -0.025em; line-height: 0.86; margin: 0;
    color: var(--ink); text-transform: lowercase;
    text-shadow: 0 1px 0 rgba(255,255,255,0.18);
}
#hero h1 .dot { color: var(--vermillion); font-size: 0.45em; vertical-align: super; margin-left: 4px; }
#hero .tagline {
    font-family: 'Newsreader', serif; font-style: italic; font-weight: 300; font-size: 19px;
    color: var(--ink-soft); margin-top: 22px; max-width: 460px;
    line-height: 1.5;
}
#hero .meta {
    position: absolute; right: 0; top: 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: var(--ink-quiet); letter-spacing: 0.18em; text-transform: uppercase;
    text-align: right; line-height: 1.8;
}

/* -------------------- COMPONENTS -------------------- */

label, .gr-form > label, span[data-testid="block-info"], .label-wrap, label > span {
    color: var(--ink-soft) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-style: normal !important;
    font-weight: 400 !important;
    font-size: 10px !important;
    letter-spacing: 0.26em !important;
    text-transform: uppercase !important;
}

/* textarea — magazine pull-quote */
textarea.svelte-633qhp, textarea {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--hairline-strong) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'Fraunces', serif !important;
    font-style: italic !important;
    font-weight: 300 !important;
    font-size: 26px !important;
    line-height: 1.35 !important;
    padding: 14px 0 18px 0 !important;
    box-shadow: none !important;
    resize: none !important;
}
textarea::placeholder { color: var(--ink-quiet) !important; opacity: 0.5 !important; }
textarea:focus { border-bottom-color: var(--vermillion) !important; outline: none !important; box-shadow: none !important; }

/* other inputs - quieter, paper card */
input, select, .gr-input, .gr-dropdown, [role="listbox"], .gr-box {
    background: rgba(255, 248, 235, 0.55) !important;
    backdrop-filter: blur(2px);
    border: 1px solid var(--hairline) !important;
    color: var(--ink) !important;
    border-radius: 4px !important;
    font-family: 'Newsreader', serif !important;
    font-size: 15px !important;
}
input[type="text"], input[type="number"], select { padding: 10px 14px !important; }
input:focus, select:focus { border-color: var(--vermillion) !important; outline: none !important; box-shadow: 0 0 0 3px var(--vermillion-glow) !important; }

/* mode radio — hard-printed binary switch */
.gr-form fieldset, .wrap.svelte-1mwvhlq, [role="radiogroup"] {
    display: flex !important; gap: 0 !important;
    border: 1px solid var(--ink) !important;
    background: transparent !important;
    border-radius: 4px !important;
    padding: 0 !important;
    overflow: hidden;
}
[role="radio"] label, label.gr-input-label, [data-testid="block-label"] + div > label {
    background: transparent !important;
    color: var(--ink) !important;
    font-family: 'Fraunces', serif !important;
    font-style: italic !important;
    font-size: 15px !important;
    padding: 8px 22px !important;
    cursor: pointer !important;
    border-right: 1px solid var(--hairline) !important;
}
[role="radio"][aria-checked="true"] label, label.gr-input-label.selected,
input[type="radio"]:checked + label {
    background: var(--ink) !important;
    color: var(--paper) !important;
}

/* buttons — italic pill outlines */
button.primary, button[variant="primary"], .primary > button, button.lg, .gr-button {
    background: transparent !important;
    color: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    padding: 12px 30px !important;
    font-family: 'Fraunces', serif !important;
    font-style: italic !important;
    font-weight: 300 !important;
    font-size: 17px !important;
    letter-spacing: 0.01em !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    transition: background 0.22s, color 0.22s, transform 0.22s !important;
}
button.primary:hover, button[variant="primary"]:hover, .gr-button:hover {
    background: var(--ink) !important; color: var(--paper) !important; transform: translateY(-1px);
}
button.primary:nth-of-type(2), .gr-row button:nth-of-type(2) { border-color: var(--vermillion) !important; color: var(--vermillion) !important; }
button.primary:nth-of-type(2):hover, .gr-row button:nth-of-type(2):hover { background: var(--vermillion) !important; color: var(--paper) !important; }

/* audio output */
.gr-audio, audio {
    background: rgba(255, 248, 235, 0.55) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 4px !important;
}

/* readout — newspaper masthead row */
.readout { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; margin-top: 24px;
           border-top: 1px solid var(--hairline); border-bottom: 1px solid var(--hairline); padding: 14px 0; }
.readout-cell { background: transparent; padding: 4px 12px; text-align: left; border-right: 1px solid var(--hairline); }
.readout-cell:last-child { border-right: none; }
.readout-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.28em; color: var(--ink-quiet); margin-bottom: 6px; text-transform: uppercase; }
.readout-val { font-family: 'Fraunces', serif; font-style: italic; font-size: 22px; font-weight: 400; color: var(--ink); letter-spacing: -0.01em; }

/* ghost lyric readout */
.ghost-lyric, .ghost-lyric textarea, [data-testid="textbox"]:not(:first-of-type) textarea {
    font-family: 'Fraunces', serif !important;
    font-style: italic !important;
    font-size: 22px !important;
    color: var(--vermillion) !important;
    border: none !important;
    border-left: 2px solid var(--vermillion) !important;
    background: transparent !important;
    padding: 10px 0 10px 20px !important;
    border-radius: 0 !important;
}

#footer {
    margin-top: 96px; padding-top: 28px;
    border-top: 1px solid var(--hairline);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: var(--ink-quiet); letter-spacing: 0.22em;
    text-transform: uppercase;
    line-height: 1.8;
    display: grid; grid-template-columns: 1fr auto; gap: 24px;
}

.dial-label { display: none; }
.gr-slider, [data-testid="slider"] { display: none !important; }

/* hide the standard gradio header / footer chrome */
.show-api, footer.svelte-mpyp5e, gradio-app > footer { display: none !important; }

@media (max-width: 720px) {
    #hero h1 { font-size: 64px !important; }
    #hero .meta { display: none; }
    .gradio-container { padding: 40px 18px 64px !important; }
    textarea { font-size: 22px !important; }
}
"""


_THEME = gr.themes.Base(primary_hue=gr.themes.colors.pink, neutral_hue=gr.themes.colors.slate).set(
    body_background_fill="*neutral_950", block_background_fill="*neutral_900",
)


# ----- circular knob widget (gr.HTML + html_template + js_on_load + trigger('change'))
# Requires Gradio >= 6.9 for arbitrary event-name triggers (we pin >= 6.10 in requirements.txt).
# Pattern is from gradio.app/guides/custom-HTML-components.

KNOB_HTML = """
<style>
  #dial-knob { padding: 28px 0 12px 0; display: flex; flex-direction: column; align-items: center; }
  .knob-stage {
    position: relative; width: 280px; height: 280px;
    display: flex; align-items: center; justify-content: center;
  }
  /* outer brass ring */
  .knob-stage::before {
    content: ''; position: absolute; inset: 0; border-radius: 50%;
    background: conic-gradient(from 0deg,
      #B89968, #6B5328, #D6B888, #A8884D, #6B5328, #C6A468, #B89968);
    filter: blur(0.5px);
  }
  .knob-stage::after {
    content: ''; position: absolute; inset: 10px; border-radius: 50%;
    background: var(--paper);
    box-shadow:
      inset 0 0 0 1px rgba(42,31,45,0.18),
      0 18px 32px -16px rgba(42,31,45,0.42),
      0 2px 0 rgba(255,255,255,0.6) inset;
  }
  .knob-host {
    position: relative; z-index: 2; width: 192px; height: 192px;
    border-radius: 50%; cursor: grab; outline: none;
    background:
      radial-gradient(circle at 32% 28%, #FFF1D8 0%, #E8C7A0 30%, #B89968 64%, #6B5328 96%);
    box-shadow:
      inset 0 -8px 18px rgba(42,31,45,0.32),
      inset 0 2px 6px rgba(255,255,255,0.45),
      0 6px 16px rgba(42,31,45,0.22);
    touch-action: none; user-select: none;
    transition: box-shadow 0.18s, filter 0.22s;
  }
  .knob-host:focus-visible, .knob-host.dragging {
    box-shadow:
      inset 0 -8px 18px rgba(42,31,45,0.32),
      inset 0 2px 6px rgba(255,255,255,0.55),
      0 0 0 2px var(--vermillion),
      0 8px 24px var(--vermillion-glow);
    filter: saturate(1.08);
  }
  .knob-host.dragging { cursor: grabbing; }
  /* vermillion arc */
  .knob-arc {
    position: absolute; z-index: 3; inset: -16px; border-radius: 50%; pointer-events: none;
    background: conic-gradient(
      from 225deg,
      var(--vermillion) 0deg,
      var(--vermillion) var(--arc-deg, 0deg),
      transparent var(--arc-deg, 0deg) 270deg,
      transparent 360deg
    );
    -webkit-mask: radial-gradient(circle, transparent 116px, #000 117px, #000 124px, transparent 125px);
            mask: radial-gradient(circle, transparent 116px, #000 117px, #000 124px, transparent 125px);
    filter: drop-shadow(0 0 8px var(--vermillion-glow));
  }
  /* indicator — a tiny vermillion bar with subtle glow */
  .knob-indicator {
    position: absolute; z-index: 4; left: 50%; top: 22px;
    width: 4px; height: 28px; margin-left: -2px;
    background: var(--vermillion); border-radius: 2px;
    transform-origin: 50% 74px;
    transform: rotate(var(--knob-angle, -135deg));
    box-shadow: 0 0 12px var(--vermillion-glow);
    pointer-events: none;
    transition: transform 0.16s cubic-bezier(.34,1.36,.4,1);
  }
  /* the center numeral — italic Fraunces in ink */
  .knob-label {
    position: absolute; z-index: 5; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-style: italic; font-weight: 300;
    font-size: 92px; color: var(--ink);
    letter-spacing: -0.04em; pointer-events: none;
    text-shadow: 0 1px 0 rgba(255, 240, 220, 0.5), 0 -1px 0 rgba(42, 31, 45, 0.12);
  }
  /* ticks — italic editorial numerals around the knob */
  .knob-ticks {
    display: flex; justify-content: space-between; width: 320px; margin: 28px auto 0;
    font-family: 'Fraunces', serif; font-style: italic; font-weight: 300;
    font-size: 14px; color: var(--ink-quiet);
    padding: 0 18px;
  }
  .knob-ticks span {
    display: inline-flex; flex-direction: column; align-items: center; gap: 4px;
    cursor: pointer; transition: color 0.18s;
  }
  .knob-ticks span::before {
    content: ''; width: 1px; height: 8px; background: var(--ink-quiet);
    transition: background 0.18s;
  }
  .knob-ticks span.active { color: var(--vermillion); font-weight: 400; font-style: normal; }
  .knob-ticks span.active::before { background: var(--vermillion); height: 12px; }
  .knob-caption {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.34em; color: var(--ink-quiet);
    text-transform: uppercase; margin-top: 12px;
  }
</style>
<div id="dial-knob">
  <div class="knob-stage">
    <div class="knob-host" id="dial-knob-host" tabindex="0" role="slider"
         aria-valuemin="0" aria-valuemax="4" aria-valuenow="${value}" aria-label="dial">
      <div class="knob-arc"></div>
      <div class="knob-indicator"></div>
      <div class="knob-label">${value}</div>
    </div>
  </div>
  <div class="knob-ticks">
    <span data-lv="0">0</span><span data-lv="1">1</span><span data-lv="2">2</span><span data-lv="3">3</span><span data-lv="4">4</span>
  </div>
  <div class="knob-caption">turn · drag · scroll · arrow keys</div>
</div>
"""

KNOB_JS = """
(element, props) => {
  const host = element.querySelector('.knob-host');
  if (!host) return;
  const lab = element.querySelector('.knob-label');
  const ticks = Array.from(element.querySelectorAll('.knob-ticks span'));
  const LEVELS = 5;
  const ANGLE_MIN = -135, ANGLE_MAX = 135;
  let level = Math.max(0, Math.min(LEVELS - 1, Math.round(Number(props.value) || 0)));

  const render = (lv) => {
    const frac = lv / (LEVELS - 1);
    const ang  = ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * frac;
    const arc  = (ANGLE_MAX - ANGLE_MIN) * frac;
    host.style.setProperty('--knob-angle', ang + 'deg');
    host.style.setProperty('--arc-deg',  arc + 'deg');
    lab.textContent = lv;
    host.setAttribute('aria-valuenow', lv);
    ticks.forEach(t => t.classList.toggle('active', Number(t.dataset.lv) === lv));
  };

  const setLevel = (lv, fire) => {
    lv = Math.max(0, Math.min(LEVELS - 1, Math.round(lv)));
    if (lv === level && !fire) return;
    level = lv;
    render(level);
    if (fire) { props.value = level; trigger('change'); }
  };
  render(level);

  let dragging = false, startY = 0, startLv = 0;
  host.addEventListener('pointerdown', (e) => {
    dragging = true; startY = e.clientY; startLv = level;
    host.setPointerCapture(e.pointerId); host.classList.add('dragging');
  });
  host.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const dy = startY - e.clientY;   // drag up -> louder
    setLevel(startLv + dy / 40, true);
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    try { host.releasePointerCapture(e.pointerId); } catch (_) {}
    host.classList.remove('dragging');
  };
  host.addEventListener('pointerup', endDrag);
  host.addEventListener('pointercancel', endDrag);
  host.addEventListener('wheel', (e) => {
    e.preventDefault();
    setLevel(level + (e.deltaY < 0 ? 1 : -1), true);
  }, { passive: false });
  host.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') { e.preventDefault(); setLevel(level + 1, true); }
    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') { e.preventDefault(); setLevel(level - 1, true); }
    else if (e.key === 'Home') { e.preventDefault(); setLevel(0, true); }
    else if (e.key === 'End')  { e.preventDefault(); setLevel(LEVELS - 1, true); }
  });
  ticks.forEach(t => t.addEventListener('click', () => setLevel(Number(t.dataset.lv), true)));
}
"""

STYLE_PRESETS = {
    "dreamy":     "Goodman/Samarin glossolalia palette",
    "hopelandic": "glide-led, high-front vowel chains (Sigur Rós register)",
    "fraser":     "m/n/l onsets, open back vowels (Cocteau Twins register)",
}

with gr.Blocks(title="Glossolalia Dial") as demo:
    gr.HTML(
        """
        <div id="hero">
            <div class="eyebrow">Volume I · Thousand Token Wood</div>
            <h1>glossolalia<span class="dot">.</span></h1>
            <p class="tagline">a single dial that grades your lyric from speech into wordless
            tongues, in your own voice. two phonotactic paths, one melody.</p>
            <div class="meta">A field study<br>in dreamy dissolution</div>
        </div>
        """
    )

    # lyric — magazine pull-quote spanning full width
    sentence = gr.Textbox(label="the lyric", value=DEFAULT_TEXT, lines=2,
                          placeholder="anything; the dial will dissolve it",
                          elem_id="lyric-input")

    # voice + mode in a quiet two-up
    with gr.Row():
        voice = gr.Dropdown([(v["name"], k) for k, v in VOICE_PRESETS.items()],
                            value=DEFAULT_VOICE, label="the voice", scale=1)
        mode = gr.Radio(choices=list(MODES), value=MODE_TONGUES,
                        label="the path", scale=1)

    # the knob — focal ritual object, centered, large
    with gr.Row(elem_id="dial-row"):
        with gr.Column():
            gr.HTML(KNOB_HTML)
            level = gr.Slider(0, 4, value=0, step=1, label="", elem_id="dial-slider",
                              visible=True)
            gr.HTML(f"<script>{KNOB_JS}</script>")

    # post-fx + seed as a quiet adjuster row
    with gr.Row():
        postfx = gr.Dropdown(list(POSTFX_PRESETS.keys()), value="subtle",
                             label="post · reverb / chorus / octave", scale=2)
        seed = gr.Number(value=42, precision=0, label="seed", scale=1)

    # two actions, side by side, the morph one in vermillion
    with gr.Row(elem_classes="action-row"):
        speak_btn = gr.Button("speak once", variant="primary",
                              elem_classes="primary", scale=1)
        morph_btn = gr.Button("morph 0 → 4 in one breath", variant="primary",
                              elem_classes="primary", scale=1)

    audio_out = gr.Audio(label="the take", type="filepath", autoplay=False)
    ghost_lyric = gr.Textbox(label="the ghost — the deterministic substitution at this dial",
                              interactive=False, lines=2,
                              elem_classes="ghost-lyric")
    metrics = gr.HTML(readout())

    level.change(lambda lv: readout(level=_safe_int(lv)),
                 inputs=level, outputs=metrics)
    speak_btn.click(speak, inputs=[sentence, voice, level, postfx, mode, seed],
                    outputs=[audio_out, metrics, ghost_lyric])
    morph_btn.click(morph, inputs=[sentence, voice, postfx, mode, seed],
                    outputs=[audio_out, metrics, ghost_lyric])

    gr.HTML(
        """
        <div id="footer">
            <div>Open weights · all inference local · the dial is a fine-tuned
                  control surface, not a DSP effect.</div>
            <div>Build-Small · Thousand Token Wood</div>
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(theme=_THEME, css=CUSTOM_CSS)
