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


def _blend_with_music(vocal: np.ndarray, vocal_sr: int, music_path: str,
                       vocal_gain_db: float = 0.0, music_gain_db: float = -8.0,
                       tempo_lock: bool = True) -> tuple[np.ndarray, int]:
    """Mix the TTS vocal over an uploaded music track.

    - Detect music tempo via librosa.beat.beat_track
    - Detect music key (rough) via chroma + Krumhansl-Schmuckler profile
    - Optionally time-stretch the vocal so it lands in a tempo grid compatible with the music
    - Sum the two streams; trim to the longer of (music length, vocal length)
    - All local: librosa + numpy. Off-the-Grid stays clean.
    """
    import librosa
    music, music_sr = librosa.load(music_path, sr=vocal_sr, mono=True)
    if tempo_lock and len(music) > vocal_sr * 2:
        try:
            mtempo, _ = librosa.beat.beat_track(y=music, sr=music_sr)
            # estimate vocal speech "tempo" by syllable-rate proxy (rms onset rate)
            vtempo, _ = librosa.beat.beat_track(y=vocal, sr=vocal_sr)
            if mtempo and vtempo and abs(np.log2(mtempo / vtempo)) < 1.5:
                # cap stretch at 25% in either direction to avoid robotic artifacts
                ratio = float(np.clip(vtempo / mtempo, 0.78, 1.28))
                if abs(ratio - 1.0) > 0.04:
                    vocal = librosa.effects.time_stretch(vocal, rate=ratio)
        except Exception as e:
            print(f"[blend] tempo lock failed: {e}; mixing without stretch")
    # Pad / truncate to the longer length
    n = max(len(music), len(vocal))
    if len(vocal) < n: vocal = np.pad(vocal, (0, n - len(vocal)))
    if len(music) < n: music = np.pad(music, (0, n - len(music)))
    vocal_gain = 10.0 ** (vocal_gain_db / 20.0)
    music_gain = 10.0 ** (music_gain_db / 20.0)
    out = vocal_gain * vocal + music_gain * music
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out.astype(np.float32), vocal_sr

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
                  mode: str = MODE_TONGUES, custom_voice_path: str | None = None,
                  custom_voice_text: str = ""):
        """Returns (audio float32 mono, sample_rate, gen_text_used).

        mode=MODE_GHOST: substitute lyric via mondegreen at given level, set_dial(0), TTS reads it.
        mode=MODE_TONGUES: leave lyric clean, set_dial(level), LoRA conditions glossolalic audio.

        custom_voice_path: if a path is supplied, F5-TTS clones this clip instead of the
        preset voice. custom_voice_text is the transcript of that clip (improves clone quality;
        an empty string lets F5-TTS auto-transcribe via Whisper).
        """
        self._ensure()
        if custom_voice_path:
            voice_wav = custom_voice_path
            voice_ref_text = (custom_voice_text or "").strip()
        else:
            voice = VOICE_PRESETS[voice_id]
            voice_wav = voice["wav"]
            ref_txt = Path(voice["ref_text"])
            voice_ref_text = ref_txt.read_text(encoding="utf-8").strip() if ref_txt.exists() else ""
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
        self._tts.infer(ref_file=voice_wav, ref_text=voice_ref_text,
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


def speak(sentence, voice_id, level, postfx_preset, mode, seed,
          custom_voice, custom_voice_text, music_path, music_gain_db):
    sentence = (sentence or "").strip()
    level = _safe_int(level)
    if not sentence:
        return None, readout(level, None, None, "type a sentence first"), ""
    y, sr, gen_text = ENGINE.generate(
        sentence, voice_id, level, seed=int(seed), mode=mode,
        custom_voice_path=custom_voice or None,
        custom_voice_text=custom_voice_text or "",
    )
    if postfx_preset != "dry":
        y, _ = apply_post_fx(y, sr, preset=postfx_preset)
    if music_path:
        try:
            y, sr = _blend_with_music(y, sr, music_path,
                                       vocal_gain_db=0.0,
                                       music_gain_db=float(music_gain_db),
                                       tempo_lock=True)
        except Exception as e:
            print(f"[blend] failed: {e}; returning dry vocal")
    path = _wav_to_filepath(y, sr)
    readout_text = gen_text if (mode == MODE_GHOST and gen_text and gen_text != sentence) else ""
    return path, readout(level, None, None, f"{mode.lower()} · lv{level}"), readout_text


def morph(sentence, voice_id, postfx_preset, mode, seed,
          custom_voice, custom_voice_text, music_path, music_gain_db, gap_ms: int = 250):
    sentence = (sentence or "").strip()
    if not sentence:
        return None, readout(None, None, None, "type a sentence first"), ""
    clips = []
    for lv in range(5):
        y, sr, _ = ENGINE.generate(
            sentence, voice_id, lv, seed=int(seed), mode=mode,
            custom_voice_path=custom_voice or None,
            custom_voice_text=custom_voice_text or "",
        )
        clips.append(y)
    morphed = equal_power_concat(clips, sr, fade_ms=gap_ms)
    if postfx_preset != "dry":
        morphed, _ = apply_post_fx(morphed, sr, preset=postfx_preset)
    if music_path:
        try:
            morphed, sr = _blend_with_music(morphed, sr, music_path,
                                             vocal_gain_db=0.0,
                                             music_gain_db=float(music_gain_db),
                                             tempo_lock=True)
        except Exception as e:
            print(f"[blend] failed: {e}; returning dry vocal")
    path = _wav_to_filepath(morphed, sr)
    return path, readout(None, None, None, f"{mode.lower()} · morphed 0->4"), ""


# ----- CSS (dreamy pastel theme: half-remembered photograph of dusk) -----

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Pinyon+Script&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

/* HEAVEN OR LAS VEGAS — direct reference. Long-exposure Christmas lights, deep midnight
   violet background, hot red-orange sun in the lower-right hemisphere, gold light trails
   swooping diagonally. Title in hand-drawn flowing italic script (Pinyon Script ≈ the
   Vaughan Oliver "Heaven or Las Vegas" lettering). Photographic, luminous, analog. */

:root {
    --night: #1A0F2D;
    --night-deep: #0E0820;
    --violet: #3D1E54;
    --violet-glow: #5A2F75;
    --sun-core: #FF4D2C;
    --sun-mid: #FF7A3D;
    --sun-halo: rgba(255, 120, 60, 0.55);
    --gold: #F5C56B;
    --gold-bright: #FFDB8A;
    --gold-glow: rgba(245, 197, 107, 0.42);
    --cream: #F3E7C3;
    --cream-mute: #D6C998;
    --ink-light: #E8DBB3;
    --ink-mute: #A89A6A;
    --hairline: rgba(245, 197, 107, 0.22);
}

html, body, .gradio-container, .dark, .light, gradio-app {
    background:
      /* deep violet wash with night gradient */
      radial-gradient(1400px 900px at 30% 18%, var(--violet-glow) 0%, var(--violet) 32%,
                       var(--night) 72%, var(--night-deep) 100%) !important;
    background-attachment: fixed !important;
    color: var(--ink-light) !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    min-height: 100vh;
}

/* THE SUN — fixed, lower-right, the focal hot red-orange sphere from the cover */
.gradio-container::before {
    content: ''; position: fixed; z-index: 0; pointer-events: none;
    right: -120px; bottom: -120px;
    width: 720px; height: 720px;
    border-radius: 50%;
    background:
      radial-gradient(circle at 38% 32%,
        #FFE7B0 0%,
        #FFB070 12%,
        var(--sun-core) 28%,
        #C7311A 52%,
        #6B1808 78%,
        transparent 100%);
    filter: blur(8px);
    opacity: 0.92;
    box-shadow:
      0 0 220px 60px rgba(255, 120, 60, 0.45),
      0 0 460px 120px rgba(255, 80, 40, 0.22);
    animation: sunBreath 14s ease-in-out infinite alternate;
}
@keyframes sunBreath {
    0%   { transform: translate(0, 0) scale(1.00); opacity: 0.88; }
    100% { transform: translate(-18px, -8px) scale(1.06); opacity: 0.96; }
}

/* Gold long-exposure light trails — diagonal swoops, painted with SVG */
body::after, gradio-app::after {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 900' preserveAspectRatio='xMidYMid slice'><defs><filter id='glow'><feGaussianBlur stdDeviation='6'/></filter><linearGradient id='g1' x1='0' y1='0' x2='1' y2='0'><stop offset='0' stop-color='%23F5C56B' stop-opacity='0'/><stop offset='0.5' stop-color='%23FFDB8A' stop-opacity='0.85'/><stop offset='1' stop-color='%23F5C56B' stop-opacity='0'/></linearGradient></defs><g filter='url(%23glow)' fill='none' stroke='url(%23g1)' stroke-width='2.5' stroke-linecap='round'><path d='M -50 280 C 250 60, 700 200, 1250 80'/><path d='M -50 460 C 320 220, 760 380, 1250 240' stroke-width='1.8'/><path d='M -50 640 C 220 420, 780 560, 1250 420' stroke-width='1.4'/><path d='M -50 800 C 280 580, 740 760, 1250 600' stroke-width='1.2'/></g></svg>");
    background-size: cover;
    opacity: 0.55; mix-blend-mode: screen;
}

/* film grain — make it feel photographic, not digital */
body::before, gradio-app::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.94' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 0.78 0 0 0 0 0.42 0 0 0 0.5 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
    opacity: 0.12; mix-blend-mode: overlay;
}

.gradio-container { max-width: 800px !important; margin: 0 auto !important; padding: 64px 32px 100px !important; position: relative; z-index: 1; }

/* -------------------- HERO -------------------- */

#hero { text-align: center; margin-bottom: 72px; position: relative; padding-top: 24px; }
#hero .wordmark-rule {
    display: inline-block;
    border: 1px solid var(--gold);
    padding: 4px 12px 5px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 400; font-size: 10px;
    color: var(--gold); letter-spacing: 0.36em; text-transform: uppercase;
    margin-bottom: 38px;
    background: rgba(14, 8, 32, 0.55);
}
#hero h1 {
    font-family: 'Pinyon Script', cursive;
    font-weight: 400; font-size: 220px;
    letter-spacing: -0.012em; line-height: 0.74; margin: 0;
    color: var(--gold-bright);
    text-shadow:
      0 0 22px rgba(245, 197, 107, 0.62),
      0 0 70px rgba(245, 197, 107, 0.32),
      0 0 130px rgba(245, 197, 107, 0.18),
      0 3px 0 rgba(0, 0, 0, 0.45);
    text-transform: lowercase;
    transform: rotate(-3deg) translateX(-12px);
    display: inline-block;
}
#hero .tagline {
    font-family: 'Cormorant Garamond', serif; font-style: italic; font-weight: 300; font-size: 19px;
    color: var(--cream); margin-top: 28px; max-width: 480px;
    margin-left: auto; margin-right: auto;
    line-height: 1.65;
    text-shadow: 0 1px 8px rgba(0, 0, 0, 0.6);
}

/* -------------------- COMPONENTS -------------------- */

label, .gr-form > label, span[data-testid="block-info"], .label-wrap, label > span,
[data-testid="block-label"] {
    color: var(--gold) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-style: normal !important;
    font-weight: 400 !important;
    font-size: 10px !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    opacity: 0.85 !important;
}

/* strip the default block chrome from EVERY gradio container so the page reads
   as light + type + air, not stacked UI cards. */
.block, .block-container, .form, .gradio-container > div > div,
[data-testid="block"], .gr-form, .gr-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
.svelte-vt1mxs, .svelte-1ipelgc, .gr-padded { background: transparent !important; }

/* lyric textarea — handwriting on light, no card, just a hot gold underline */
textarea {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(245, 197, 107, 0.32) !important;
    border-radius: 0 !important;
    color: var(--gold-bright) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-weight: 300 !important;
    font-size: 32px !important;
    line-height: 1.32 !important;
    padding: 14px 4px 18px !important;
    box-shadow: none !important;
    resize: none !important;
    text-shadow: 0 0 14px rgba(245, 197, 107, 0.32);
}
textarea::placeholder { color: var(--cream-mute) !important; opacity: 0.45 !important; font-style: italic !important; }
textarea:focus { border-bottom-color: var(--gold-bright) !important; outline: none !important; box-shadow: none !important; }

/* dropdowns + number inputs — minimal, gold underline only.
   Need to nuke svelte's wrapper containers AND the inner input.
   DO NOT collapse min-height on the slider's .wrap — it would hide the track. */
input, select, .gr-input, .gr-dropdown, [role="listbox"],
.secondary-wrap, .container,
[data-testid="dropdown"], [data-testid="number"] > div {
    background: transparent !important;
    backdrop-filter: none !important;
    border: none !important;
    color: var(--cream) !important;
    border-radius: 0 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 19px !important;
    box-shadow: none !important;
}
/* Slider's own .wrap needs to keep its layout intact */
#dial-slider .wrap.svelte-8epfm4 { background: transparent !important; }
/* head row (label + number input + reset button) */
#dial-slider .head.svelte-8epfm4 { display: none !important; }
/* min/max value labels flanking the track */
#dial-slider .min_value, #dial-slider .max_value {
    font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 18px;
    color: var(--cream-mute); padding: 0 14px; align-self: center;
}
#dial-slider .slider_input_container.svelte-8epfm4 {
    display: flex !important; align-items: center !important; gap: 0 !important;
    padding: 12px 0 !important;
}
input, select {
    border-bottom: 1px solid rgba(245, 197, 107, 0.32) !important;
    font-style: italic !important;
    color: var(--gold-bright) !important;
}
input[type="text"], input[type="number"], select { padding: 8px 4px !important; }
input:focus, select:focus {
    border-bottom-color: var(--gold-bright) !important; outline: none !important;
    box-shadow: none !important;
}
/* dropdown trigger button (the visible selection) */
[data-testid="dropdown"] input, .gr-dropdown input {
    background: transparent !important;
    color: var(--gold-bright) !important;
}

/* dropdown popup options — dark velvet so they read on the violet bg */
[role="listbox"] [role="option"], .options ul li {
    background: rgba(14, 8, 32, 0.96) !important;
    color: var(--cream) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
}
[role="listbox"] [role="option"]:hover { background: rgba(245, 197, 107, 0.18) !important; color: var(--gold-bright) !important; }

/* mode radio — luminous gold tabs.
   The radio group's outer wrap on its own variant of the svelte class */
.wrap.svelte-1mwvhlq, [role="radiogroup"] {
    display: inline-flex !important; gap: 0 !important;
    border: 1px solid var(--gold) !important;
    background: transparent !important;
    border-radius: 999px !important;
    padding: 3px !important;
    overflow: hidden;
}

/* kill the boxy gray container around the voice + post-fx dropdowns AND seed number */
.gradio-container [data-testid="dropdown"],
.gradio-container [data-testid="number"],
.gradio-container .form,
[class^="block"], [class*=" block"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* The dropdown internal button look — the gradio "selected value pill" */
.gradio-container .gr-dropdown,
.gradio-container [data-testid="dropdown"] > div > div {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(245, 197, 107, 0.32) !important;
    border-radius: 0 !important;
    padding: 6px 4px !important;
}
[role="radio"] label, .gr-input-label,
input[type="radio"] + label, label[for*="radio"] {
    background: transparent !important;
    color: var(--cream-mute) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-weight: 400 !important;
    font-size: 15px !important;
    padding: 8px 28px !important;
    cursor: pointer !important;
    border-radius: 999px !important;
    transition: all 0.22s !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}
[role="radio"][aria-checked="true"] label,
input[type="radio"]:checked + label {
    background: linear-gradient(180deg, var(--sun-core), var(--sun-mid)) !important;
    color: var(--night-deep) !important;
    box-shadow: 0 0 22px var(--sun-halo), inset 0 1px 0 rgba(255, 220, 180, 0.5) !important;
}

/* buttons — glowing pill on dark, gold borders */
button.primary, button[variant="primary"], .primary > button, button.lg, .gr-button {
    background: rgba(14, 8, 32, 0.62) !important;
    color: var(--gold-bright) !important;
    border: 1px solid var(--gold) !important;
    padding: 13px 32px !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-weight: 400 !important;
    font-size: 17px !important;
    letter-spacing: 0.02em !important;
    border-radius: 999px !important;
    box-shadow: 0 0 24px rgba(245, 197, 107, 0.18), inset 0 0 12px rgba(245, 197, 107, 0.08) !important;
    transition: all 0.24s !important;
    text-shadow: 0 0 8px rgba(245, 197, 107, 0.4);
}
button.primary:hover, button[variant="primary"]:hover, .gr-button:hover {
    background: var(--gold-bright) !important;
    color: var(--night-deep) !important;
    text-shadow: none;
    box-shadow: 0 0 36px var(--gold-glow), inset 0 0 18px rgba(255, 219, 138, 0.4) !important;
}
/* second action — Morph — in the hot vermillion / red-orange sun palette */
.action-row button:nth-of-type(2) {
    border-color: var(--sun-mid) !important;
    color: var(--sun-mid) !important;
    text-shadow: 0 0 10px rgba(255, 122, 61, 0.4);
    box-shadow: 0 0 24px rgba(255, 122, 61, 0.22) !important;
}
.action-row button:nth-of-type(2):hover {
    background: linear-gradient(180deg, var(--sun-core), var(--sun-mid)) !important;
    color: var(--night-deep) !important;
    text-shadow: none;
    box-shadow: 0 0 40px var(--sun-halo) !important;
}

/* audio output card */
.gr-audio, audio {
    background: rgba(14, 8, 32, 0.62) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 6px !important;
    color: var(--cream) !important;
}
audio::-webkit-media-controls-panel { background: rgba(14, 8, 32, 0.85) !important; }

/* THE DIAL — focal luminous filament with a sun-orb thumb. Slider IS the dial. */
#dial-frame {
    margin: 28px 0 8px;
    text-align: center;
}
.dial-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: var(--gold); letter-spacing: 0.34em; text-transform: uppercase;
    opacity: 0.8; margin-bottom: 12px;
}
.dial-ticks {
    display: flex; justify-content: space-between;
    font-family: 'Cormorant Garamond', serif; font-style: italic;
    color: var(--cream-mute); font-size: 22px;
    margin: 24px 6px 4px;
}
.dial-ticks span {
    display: flex; flex-direction: column; align-items: center;
    line-height: 1;
}
.dial-ticks span small {
    font-family: 'IBM Plex Mono', monospace; font-style: normal;
    font-size: 9px; color: var(--cream-mute); opacity: 0.6;
    letter-spacing: 0.22em; text-transform: uppercase;
    margin-top: 8px;
}

#dial-slider, [elem_id="dial-slider"] {
    padding: 0 !important; margin: 0 !important;
    background: transparent !important; border: none !important;
}
#dial-slider > div, #dial-slider .wrap-inner { background: transparent !important; }

/* The actual rail: a horizontal filament glowing in gold. */
input[type="range"] {
    -webkit-appearance: none; appearance: none;
    width: 100% !important; height: 6px !important;
    background:
      linear-gradient(90deg,
        rgba(245, 197, 107, 0.85),
        var(--gold-bright) 30%,
        var(--gold) 60%,
        rgba(245, 197, 107, 0.45)) !important;
    border-radius: 999px !important;
    outline: none !important;
    box-shadow:
      0 0 14px var(--gold-glow),
      0 0 36px rgba(245, 197, 107, 0.25);
    cursor: pointer;
    margin: 6px 0 !important;
}
/* The thumb is the red-orange sun — large, glowing, the focal object */
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 44px !important; height: 44px !important;
    border-radius: 50%;
    background:
      radial-gradient(circle at 32% 30%, #FFE0B0 0%, var(--sun-mid) 28%,
                       var(--sun-core) 60%, #8B2710 100%);
    box-shadow:
      0 0 18px rgba(255, 200, 120, 0.6),
      0 0 38px var(--sun-halo),
      0 0 80px rgba(255, 80, 40, 0.45),
      inset 0 -4px 8px rgba(120, 20, 0, 0.5);
    cursor: grab;
    border: 1px solid rgba(245, 197, 107, 0.7);
    margin-top: -19px;
    transition: box-shadow 0.18s, transform 0.18s;
}
input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.06);
    box-shadow:
      0 0 24px rgba(255, 220, 160, 0.8),
      0 0 56px var(--sun-halo),
      0 0 120px rgba(255, 80, 40, 0.55),
      inset 0 -4px 8px rgba(120, 20, 0, 0.5);
}
input[type="range"]:active::-webkit-slider-thumb { cursor: grabbing; }

input[type="range"]::-moz-range-thumb {
    width: 44px; height: 44px; border-radius: 50%;
    background:
      radial-gradient(circle at 32% 30%, #FFE0B0 0%, var(--sun-mid) 28%,
                       var(--sun-core) 60%, #8B2710 100%);
    box-shadow:
      0 0 18px rgba(255, 200, 120, 0.6),
      0 0 38px var(--sun-halo),
      0 0 80px rgba(255, 80, 40, 0.45),
      inset 0 -4px 8px rgba(120, 20, 0, 0.5);
    cursor: grab; border: 1px solid rgba(245, 197, 107, 0.7);
}
input[type="range"]::-moz-range-track { background: transparent; }

/* readout — newspaper-strip-like, gold rule */
.readout { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; margin-top: 24px;
           border-top: 1px solid var(--hairline); border-bottom: 1px solid var(--hairline); padding: 16px 0; }
.readout-cell { background: transparent; padding: 4px 14px; text-align: left; border-right: 1px solid var(--hairline); }
.readout-cell:last-child { border-right: none; }
.readout-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 0.32em; color: var(--gold); margin-bottom: 6px; text-transform: uppercase; opacity: 0.7; }
.readout-val { font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 22px; font-weight: 400; color: var(--cream); letter-spacing: -0.01em; }

/* ghost lyric — luminous pulled-quote in gold */
.ghost-lyric textarea, [data-testid="textbox"]:not(:first-of-type) textarea {
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-size: 22px !important;
    color: var(--gold-bright) !important;
    border: none !important;
    border-left: 2px solid var(--gold) !important;
    background: rgba(245, 197, 107, 0.04) !important;
    padding: 14px 0 14px 22px !important;
    border-radius: 0 !important;
    text-shadow: 0 0 10px rgba(245, 197, 107, 0.3);
}

#footer {
    margin-top: 88px; padding-top: 28px;
    border-top: 1px solid var(--hairline);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: var(--cream-mute); letter-spacing: 0.26em;
    text-transform: uppercase;
    line-height: 1.9;
    display: grid; grid-template-columns: 1fr auto; gap: 24px;
    opacity: 0.7;
}

.dial-label { display: none; }

/* hide the standard gradio header / footer chrome */
.show-api, footer.svelte-mpyp5e, gradio-app > footer { display: none !important; }

@media (max-width: 720px) {
    #hero h1 { font-size: 88px !important; }
    .gradio-container { padding: 36px 18px 64px !important; }
    textarea { font-size: 20px !important; }
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

    # custom voice — record or upload a 10-sec clip; F5-TTS clones it
    with gr.Accordion("Or clone your own voice (record / upload, 6-12 sec)",
                      open=False, elem_id="custom-voice-accord"):
        with gr.Row():
            custom_voice = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="reference clip — clean speech, no background music, 6-12 sec",
                scale=2,
            )
            custom_voice_text = gr.Textbox(
                label="(optional) transcript of the clip",
                placeholder="leave empty to auto-transcribe (Whisper)",
                lines=2,
                scale=1,
            )

    # background music — upload an instrumental; we tempo-lock + mix the vocal over it
    with gr.Accordion("Or drop in a backing track to mix the dial over",
                      open=False, elem_id="music-accord"):
        with gr.Row():
            music_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="backing track (instrumental, any length, any tempo)",
                scale=2,
            )
            music_gain = gr.Slider(
                -24, 6, value=-8, step=1,
                label="music level (dB)",
                scale=1,
            )

    # the dial — focal luminous element (Gradio slider styled as glowing filament + sun orb)
    gr.HTML(
        """
        <div id="dial-frame">
            <div class="dial-tag">the dial</div>
            <div class="dial-ticks">
                <span>0<small>clean</small></span>
                <span>1</span>
                <span>2<small>half-dissolved</small></span>
                <span>3</span>
                <span>4<small>tongues</small></span>
            </div>
        </div>
        """
    )
    level = gr.Slider(0, 4, value=0, step=1, label="", elem_id="dial-slider",
                      show_label=False, visible=True)

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
    speak_btn.click(
        speak,
        inputs=[sentence, voice, level, postfx, mode, seed,
                custom_voice, custom_voice_text, music_input, music_gain],
        outputs=[audio_out, metrics, ghost_lyric],
    )
    morph_btn.click(
        morph,
        inputs=[sentence, voice, postfx, mode, seed,
                custom_voice, custom_voice_text, music_input, music_gain],
        outputs=[audio_out, metrics, ghost_lyric],
    )

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
