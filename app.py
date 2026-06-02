"""The Un-Language Slider — a single dial that grades a typed sentence from intelligible speech
to phonotactically-valid English-native glossolalia in the same voice. Powered by F5-TTS + a
fine-tuned LoRA where one control token (`tongues zero..four`) maps to the dissolution level.

This is the v1 Gradio app (gr.Blocks). v2 is in app_server.py (gradio.Server custom HTML knob).
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
DEFAULT_TEXT = "I had a dream last night about the ocean."
# default to the published LoRA repo so the Space loads it without needing env vars;
# COHERENCE_DIAL_LORA env var overrides for local dev against a checkpoint dir.
LORA_PATH = os.environ.get("COHERENCE_DIAL_LORA", HF_LORA_REPO)


# ----- inference engine (lazy-loaded; falls back to a silent stub if F5-TTS isn't installed) -----

class TTSEngine:
    def __init__(self):
        self._tts = None
        self._asr = None
        self._enc = None
        self._lora_loaded = False
        self.live = False

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

    def generate(self, sentence: str, voice_id: str, level: int, seed: int = 42):
        """Returns (mono numpy float32, sample_rate)."""
        self._ensure()
        voice = VOICE_PRESETS[voice_id]
        prompt = f"{sentence} | {CONTROL_STEM} {LEVEL_WORDS[level]}"
        if not self.live:
            # silent stub at SAMPLE_RATE, ~3s, for UI testing
            return np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        ref_text = ""
        ref_txt = Path(voice["ref_text"])
        if ref_txt.exists():
            ref_text = ref_txt.read_text(encoding="utf-8").strip()
        self._tts.infer(ref_file=voice["wav"], ref_text=ref_text,
                        gen_text=prompt, file_wave=out, seed=seed)
        import soundfile as sf
        y, sr = sf.read(out, always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y.astype(np.float32), sr

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


def speak(sentence: str, voice_id: str, level, postfx_preset: str, seed: int = 42):
    sentence = (sentence or "").strip()
    level = _safe_int(level)
    if not sentence:
        return None, readout(level, None, None, "type a sentence first")
    y, sr = ENGINE.generate(sentence, voice_id, level, seed=int(seed))
    # post-fx
    if postfx_preset != "dry":
        y_wet, _ = apply_post_fx(y, sr, preset=postfx_preset)
        out_audio = y_wet
    else:
        out_audio = y
    # metrics
    wer = ENGINE.transcribe_wer(y, sr, sentence)
    cos = None
    if level != 0:
        try:
            ref_y, ref_sr = ENGINE.generate(sentence, voice_id, 0, seed=int(seed))
            cos = ENGINE.voice_cosine(y, sr, ref_y, ref_sr)
        except Exception:
            cos = None
    path = _wav_to_filepath(out_audio, sr)
    return path, readout(level, wer, cos, "ok")


def morph(sentence: str, voice_id: str, postfx_preset: str, seed: int = 42, gap_ms: int = 250):
    sentence = (sentence or "").strip()
    if not sentence:
        return None, readout(None, None, None, "type a sentence first")
    clips = []
    for lv in range(5):
        y, sr = ENGINE.generate(sentence, voice_id, lv, seed=int(seed))
        clips.append(y)
    morphed = equal_power_concat(clips, sr, fade_ms=gap_ms)
    if postfx_preset != "dry":
        morphed, _ = apply_post_fx(morphed, sr, preset=postfx_preset)
    path = _wav_to_filepath(morphed, sr)
    return path, readout(None, None, None, "morphed 0->4")


# ----- CSS (reused verbatim from prior app: dark theme, magenta accent, monospace readouts) -----

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-card: #14141c;
    --bg-card-hover: #1a1a26;
    --border: #2a2a3a;
    --border-bright: #3a3a52;
    --text-primary: #f5f5fa;
    --text-secondary: #8a8a9a;
    --text-muted: #5a5a6a;
    --accent: #ff3d92;
    --accent-glow: rgba(255, 61, 146, 0.35);
    --accent-soft: rgba(255, 61, 146, 0.12);
}

body, .gradio-container, .dark {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 920px !important;
    margin: 0 auto !important;
    padding: 56px 24px 80px 24px !important;
}

#hero { text-align: center; margin-bottom: 40px; position: relative; }
#hero::before {
    content: ''; position: absolute; top: -40px; left: 50%; transform: translateX(-50%);
    width: 240px; height: 240px;
    background: radial-gradient(circle, var(--accent-soft) 0%, transparent 70%);
    z-index: -1; pointer-events: none;
}
#hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-size: 72px; font-weight: 700;
    letter-spacing: -0.045em; line-height: 0.95; margin: 0;
    background: linear-gradient(180deg, #ffffff 0%, #888899 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
#hero .tagline { font-size: 16px; color: var(--text-secondary); margin-top: 14px; max-width: 560px; margin-left: auto; margin-right: auto; }
#hero .accent-line { display: inline-block; width: 48px; height: 3px; background: var(--accent); margin: 22px 0 0 0; border-radius: 2px; box-shadow: 0 0 20px var(--accent-glow); }

label { color: var(--text-secondary) !important; font-size: 12px !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; font-family: 'JetBrains Mono', monospace !important; }

input, textarea, select, .gr-input, .gr-text-input {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 12px !important;
    padding: 12px 16px !important; font-size: 14px !important; font-family: 'Inter', sans-serif !important;
}
input:focus, textarea:focus, select:focus { border-color: var(--accent) !important; outline: none !important; }

.gr-dropdown, [role="listbox"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }

button.primary, button[variant="primary"], .primary > button {
    background: var(--accent) !important; color: white !important; border: none !important;
    padding: 14px 36px !important; font-weight: 600 !important; font-size: 15px !important;
    border-radius: 12px !important; box-shadow: 0 0 32px var(--accent-glow) !important;
    transition: transform 0.15s, box-shadow 0.15s !important; font-family: 'Inter', sans-serif !important;
}
button.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 0 48px var(--accent-glow) !important; }

.gr-audio, audio { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }

.readout { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-top: 14px; }
.readout-cell { background: var(--bg-card); padding: 14px 12px; text-align: center; }
.readout-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.16em; color: var(--text-muted); margin-bottom: 6px; }
.readout-val { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 500; color: var(--accent); letter-spacing: -0.02em; }

#footer { margin-top: 64px; padding-top: 28px; border-top: 1px solid var(--border); text-align: center; font-size: 12px; color: var(--text-muted); line-height: 1.7; }
#footer a { color: var(--text-secondary); text-decoration: none; border-bottom: 1px solid var(--border); }
#footer a:hover { color: var(--accent); }

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

with gr.Blocks(title="Glossolalia Dial") as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1>GLOSSOLALIA</h1>
            <p class="tagline">Type a sentence. Pick a voice. Turn the dial.<br>
            Hear it dissolve from speech to wordless tongues — in the same voice.</p>
            <div class="accent-line"></div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            sentence = gr.Textbox(label="Sentence", value=DEFAULT_TEXT, lines=2,
                                  placeholder="type anything")
        with gr.Column(scale=2):
            voice = gr.Dropdown([(v["name"], k) for k, v in VOICE_PRESETS.items()],
                                value=DEFAULT_VOICE, label="Voice")

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<div class='dial-label'>PLAIN &nbsp;↔&nbsp; TONGUES &nbsp;(0 → 4)</div>")
            level = gr.HTML(
                value=0,
                html_template=KNOB_HTML,
                js_on_load=KNOB_JS,
                elem_id="dial-knob",
            )
        with gr.Column(scale=2):
            postfx = gr.Dropdown(list(POSTFX_PRESETS.keys()), value="subtle",
                                 label="Post-FX (reverb · chorus · octave)")

    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed", scale=1)
        speak_btn = gr.Button("Speak", variant="primary", elem_classes="primary", scale=2)
        morph_btn = gr.Button("Morph 0 → 4 (one continuous take)", variant="primary",
                              elem_classes="primary", scale=2)

    audio_out = gr.Audio(label="Output", type="filepath", autoplay=False)
    metrics = gr.HTML(readout())

    # twisting the knob just updates the readout; Speak button drives generation
    level.change(lambda lv: readout(level=_safe_int(lv)),
                 inputs=level, outputs=metrics)
    speak_btn.click(speak, inputs=[sentence, voice, level, postfx, seed],
                    outputs=[audio_out, metrics])
    morph_btn.click(morph, inputs=[sentence, voice, postfx, seed],
                    outputs=[audio_out, metrics])

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
    demo.launch(theme=_THEME, css=CUSTOM_CSS, show_api=False)
