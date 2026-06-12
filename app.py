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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #EFE6D8;
    --bg-top: #B6BFD8;
    --bg-mid: #D9A6A0;
    --bg-bottom: #F2D6BD;
    --surface: #F5ECDF;
    --surface-soft: #EAD9C7;
    --ink: #3C4266;
    --ink-soft: #6B6E8A;
    --ink-quiet: #8C8FA8;
    --accent-rose: #D9A6A0;
    --accent-sage: #A8B89A;
    --accent-peach: #F2D6BD;
    --accent-coral: #E78558;
    --accent-wine: #503F52;
    --halo-warm: #F4C7B0;
    --halo-cool: #C8C4DE;
    --hairline: rgba(60, 66, 102, 0.18);
}

body, .gradio-container, .dark, .light {
    background:
      radial-gradient(1200px 800px at 20% 8%, rgba(244,199,176,0.55), transparent 60%),
      linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 45%, var(--bg-bottom) 100%) !important;
    background-attachment: fixed !important;
    color: var(--ink) !important;
    font-family: 'EB Garamond', Georgia, serif !important;
}

body::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.235 0 0 0 0 0.258 0 0 0 0 0.4 0 0 0 0.55 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
    opacity: 0.14; mix-blend-mode: overlay;
}

.gradio-container { max-width: 880px !important; margin: 0 auto !important; padding: 72px 28px 96px !important; position: relative; z-index: 1; }

#hero { text-align: center; margin-bottom: 44px; }
#hero h1 {
    font-family: 'Fraunces', serif; font-variation-settings: 'opsz' 144, 'SOFT' 100;
    font-style: italic; font-weight: 300; font-size: 76px;
    letter-spacing: 0.04em; line-height: 1; margin: 0;
    color: var(--ink); text-transform: lowercase;
}
#hero .tagline {
    font-family: 'EB Garamond', serif; font-style: italic; font-size: 18px;
    color: var(--ink-soft); margin-top: 12px; max-width: 540px; margin-left: auto; margin-right: auto;
    line-height: 1.5;
}
#hero .accent-line { display: inline-block; width: 64px; height: 1px; background: var(--ink-soft); opacity: 0.5; margin: 20px 0 0; }

label, .gr-form > label, span[data-testid="block-info"] {
    color: var(--ink-soft) !important;
    font-family: 'EB Garamond', serif !important;
    font-style: italic !important;
    font-weight: 400 !important;
    font-size: 13px !important;
    letter-spacing: 0.04em !important;
    text-transform: lowercase !important;
}

input, textarea, select,
.gr-input, .gr-text-input, .gr-dropdown, [role="listbox"],
.gr-box, .block {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    color: var(--ink) !important;
    border-radius: 14px !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 15px !important;
}
textarea, input[type="text"], input[type="number"] {
    padding: 12px 14px !important;
}
input:focus, textarea:focus, select:focus { border-color: var(--accent-coral) !important; outline: none !important; box-shadow: 0 0 0 3px rgba(231,133,88,0.18) !important; }

button.primary, button[variant="primary"], .primary > button, button.lg {
    background: transparent !important;
    color: var(--ink) !important;
    border: 1px solid var(--ink-soft) !important;
    padding: 12px 28px !important;
    font-family: 'Fraunces', serif !important;
    font-style: italic !important;
    font-weight: 400 !important;
    font-size: 16px !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    transition: background 0.18s, border-color 0.18s !important;
}
button.primary:hover, button[variant="primary"]:hover { background: rgba(60,66,102,0.06) !important; border-color: var(--ink) !important; }

.gr-audio, audio { background: var(--surface) !important; border: 1px solid var(--hairline) !important; border-radius: 14px !important; }

.readout { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; margin-top: 18px; }
.readout-cell { background: transparent; padding: 12px 8px; text-align: center; border-right: 1px solid var(--hairline); }
.readout-cell:last-child { border-right: none; }
.readout-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.18em; color: var(--ink-quiet); margin-bottom: 4px; text-transform: lowercase; }
.readout-val { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 500; color: var(--ink); letter-spacing: -0.01em; }

#footer {
    margin-top: 72px; padding-top: 24px;
    border-top: 1px solid var(--hairline);
    text-align: center;
    font-family: 'EB Garamond', serif; font-style: italic;
    font-size: 12px; color: var(--ink-quiet);
    line-height: 1.6;
}

.dial-label { display: none; }
.gr-slider, [data-testid="slider"] { display: none !important; }

@media (max-width: 720px) {
    #hero h1 { font-size: 48px !important; }
    .gradio-container { padding: 32px 16px !important; }
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
  #dial-knob { padding: 4px 0 8px 0; }
  .dial-label { color: var(--text-secondary); font-size: 12px; font-weight: 500;
                text-transform: uppercase; letter-spacing: 0.08em;
                font-family: 'JetBrains Mono', monospace; margin-bottom: 8px; }
  .knob-host {
    position: relative; width: 168px; height: 168px; margin: 4px auto 6px;
    border-radius: 50%; cursor: grab;
    background: radial-gradient(circle at 30% 30%, #1a1a26 0%, #0e0e16 70%, #08080d 100%);
    box-shadow:
      inset 0 0 0 1px var(--border),
      inset 0 8px 24px rgba(0,0,0,0.6),
      0 0 0 1px var(--border-bright),
      0 0 32px rgba(255, 61, 146, 0.08);
    touch-action: none; user-select: none; outline: none;
    transition: box-shadow 0.18s;
  }
  .knob-host:focus-visible, .knob-host.dragging {
    box-shadow:
      inset 0 0 0 1px var(--accent),
      inset 0 8px 24px rgba(0,0,0,0.6),
      0 0 0 1px var(--accent),
      0 0 48px var(--accent-glow);
  }
  .knob-host.dragging { cursor: grabbing; }
  .knob-arc {
    position: absolute; inset: -6px; border-radius: 50%; pointer-events: none;
    background: conic-gradient(
      from 225deg,
      var(--accent) 0deg,
      var(--accent) var(--arc-deg, 0deg),
      transparent var(--arc-deg, 0deg) 270deg,
      transparent 360deg
    );
    -webkit-mask: radial-gradient(circle, transparent 78px, #000 79px, #000 86px, transparent 87px);
            mask: radial-gradient(circle, transparent 78px, #000 79px, #000 86px, transparent 87px);
    filter: drop-shadow(0 0 6px var(--accent-glow));
  }
  .knob-indicator {
    position: absolute; left: 50%; top: 14px;
    width: 3px; height: 24px; margin-left: -1.5px;
    background: var(--accent); border-radius: 2px;
    transform-origin: 50% 70px;
    transform: rotate(var(--knob-angle, -135deg));
    box-shadow: 0 0 12px var(--accent-glow);
    pointer-events: none;
    transition: transform 0.12s cubic-bezier(.4,1.2,.4,1);
  }
  .knob-label {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-size: 56px; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.03em; pointer-events: none;
  }
  .knob-ticks {
    display: flex; justify-content: space-between; width: 220px; margin: 8px auto 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted);
    letter-spacing: 0.1em;
  }
  .knob-ticks span.active { color: var(--accent); }
</style>
<div class="knob-host" id="dial-knob-host" tabindex="0" role="slider"
     aria-valuemin="0" aria-valuemax="4" aria-valuenow="${value}" aria-label="Tongues dial">
  <div class="knob-arc"></div>
  <div class="knob-indicator"></div>
  <div class="knob-label">${value}</div>
</div>
<div class="knob-ticks">
  <span data-lv="0">0</span><span data-lv="1">1</span><span data-lv="2">2</span><span data-lv="3">3</span><span data-lv="4">4</span>
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
            <h1>glossolalia</h1>
            <p class="tagline">type a sentence, pick a voice, turn the dial.<br>
            hear it dissolve from speech to wordless tongues, in the same voice.</p>
            <div class="accent-line"></div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            sentence = gr.Textbox(label="lyric", value=DEFAULT_TEXT, lines=2,
                                  placeholder="anything")
        with gr.Column(scale=2):
            voice = gr.Dropdown([(v["name"], k) for k, v in VOICE_PRESETS.items()],
                                value=DEFAULT_VOICE, label="voice")

    with gr.Row():
        mode = gr.Radio(
            choices=list(MODES),
            value=MODE_TONGUES,
            label="mode",
            info=(
                "Ghost: real English words that sound like the source · "
                "Tongues: invented pseudowords from the fine-tuned LoRA dial"
            ),
        )

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(KNOB_HTML)
            level = gr.Slider(0, 4, value=0, step=1, label="", elem_id="dial-slider", visible=True)
            gr.HTML(f"<script>{KNOB_JS}</script>")
        with gr.Column(scale=2):
            postfx = gr.Dropdown(list(POSTFX_PRESETS.keys()), value="subtle",
                                 label="Post-FX (reverb · chorus · octave)")

    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed", scale=1)
        speak_btn = gr.Button("Speak", variant="primary", elem_classes="primary", scale=2)
        morph_btn = gr.Button("Morph 0 → 4 (one continuous take)", variant="primary",
                              elem_classes="primary", scale=2)

    audio_out = gr.Audio(label="output", type="filepath", autoplay=False)
    ghost_lyric = gr.Textbox(label="ghost lyric (deterministic substitution at current level)",
                              interactive=False, lines=2)
    metrics = gr.HTML(readout())

    # twisting the knob just updates the readout; Speak button drives generation
    level.change(lambda lv: readout(level=_safe_int(lv)),
                 inputs=level, outputs=metrics)
    speak_btn.click(speak, inputs=[sentence, voice, level, postfx, mode, seed],
                    outputs=[audio_out, metrics, ghost_lyric])
    morph_btn.click(morph, inputs=[sentence, voice, postfx, mode, seed],
                    outputs=[audio_out, metrics, ghost_lyric])

    gr.HTML(
        """
        <div id="footer">
            Open weights · runs locally on your machine · the dial is a fine-tuned control token,
            not a DSP effect. In the lineage of Sigur Rós' Hopelandic, Lisa Gerrard, and the long
            tradition of wordless vocal music.<br>
            BUILD-SMALL-HACKATHON · THOUSAND TOKEN WOOD
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(theme=_THEME, css=CUSTOM_CSS)
