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
import re
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

# ZeroGPU support (org Space). `spaces` is only installed on the ZeroGPU deployment;
# where it's absent (the T4 Space, local dev) gpu_task is a no-op pass-through, so the
# SAME app.py runs on both. On ZeroGPU it allocates a transient GPU for each call.
try:
    import spaces

    _ON_ZEROGPU = True

    def gpu_task(fn):
        return spaces.GPU(duration=120)(fn)
except Exception:
    _ON_ZEROGPU = False

    def gpu_task(fn):
        return fn

from config import (
    CONTROL_STEM, HF_LORA_REPO, LEVEL_WORDS, RESEMBLYZER_MIN_COSINE,
    SAMPLE_RATE, VOICE_PRESETS, WHISPER_MODEL,
)
from scripts.post_fx import PRESETS as POSTFX_PRESETS, apply_post_fx


def _parse_per_word_overrides(text: str) -> dict[str, tuple[str, float]]:
    """Parse a string like 'river=ree-vuh:1.3 calm=kawm light=:1.6' into a dict:
        { 'river': ('ree-vuh', 1.3), 'calm': ('kawm', 1.0), 'light': ('light', 1.6) }
    The 'word=replacement:stretch' format is intentionally simple:
      - word: the source word in the input lyric (case-insensitive)
      - replacement: the pronunciation guide we feed to F5-TTS (empty -> keep original)
      - stretch: a speed multiplier for THAT word's audio chunk (1.0 = normal,
        <1.0 = faster, >1.0 = slower / more sustained). Default 1.0 if omitted.
    """
    out: dict[str, tuple[str, float]] = {}
    for tok in (text or "").split():
        if "=" not in tok:
            continue
        word, rest = tok.split("=", 1)
        word = word.strip().lower()
        if ":" in rest:
            repl, sval = rest.split(":", 1)
            try:
                stretch = float(sval)
            except ValueError:
                stretch = 1.0
        else:
            repl, stretch = rest, 1.0
        stretch = max(0.5, min(2.5, stretch))
        out[word] = (repl.strip() if repl.strip() else word, stretch)
    return out


def _chunked_generate_with_overrides(
        engine, sentence: str, voice_id: str, level: int, seed: int, mode: str,
        custom_voice_path: str | None, custom_voice_text: str,
        overrides: dict[str, tuple[str, float]]) -> tuple[np.ndarray, int, str]:
    """Generate audio chunk-by-chunk so per-word stretch + pronunciation overrides apply.
    Each chunk goes through F5-TTS at its own speed, then we equal-power concat.
    Falls back to the single-shot path if no overrides are present."""
    if not overrides:
        return engine.generate(sentence, voice_id, level, seed=seed, mode=mode,
                                custom_voice_path=custom_voice_path,
                                custom_voice_text=custom_voice_text)
    words = re.findall(r"[A-Za-z']+|[^A-Za-z']+", sentence)
    chunks: list[tuple[str, float]] = []
    for w in words:
        key = w.lower().strip()
        if key in overrides:
            repl, stretch = overrides[key]
            chunks.append((repl, stretch))
        else:
            chunks.append((w, 1.0))
    # Merge adjacent chunks with stretch == 1.0 so we don't call F5-TTS 30 times for one sentence
    merged: list[tuple[str, float]] = []
    for text_part, st in chunks:
        if merged and abs(merged[-1][1] - 1.0) < 1e-6 and abs(st - 1.0) < 1e-6:
            merged[-1] = (merged[-1][0] + text_part, 1.0)
        else:
            merged.append((text_part, st))
    # Generate each chunk
    audios = []
    sr_out = SAMPLE_RATE
    last_gen_text_acc = ""
    for text_part, stretch in merged:
        if not text_part.strip():
            continue
        y, sr, gen_text = engine.generate(
            text_part, voice_id, level, seed=seed, mode=mode,
            custom_voice_path=custom_voice_path,
            custom_voice_text=custom_voice_text,
        )
        sr_out = sr
        if abs(stretch - 1.0) > 0.02:
            try:
                import librosa
                y = librosa.effects.time_stretch(y.astype(np.float32), rate=1.0 / stretch)
            except Exception as e:
                print(f"[chunked] time_stretch failed for '{text_part}': {e}")
        audios.append(y)
        last_gen_text_acc += gen_text + " "
    if not audios:
        return engine.generate(sentence, voice_id, level, seed=seed, mode=mode,
                                custom_voice_path=custom_voice_path,
                                custom_voice_text=custom_voice_text)
    final = equal_power_concat(audios, sr_out, fade_ms=80)
    return final.astype(np.float32), sr_out, last_gen_text_acc.strip()


def _blend_with_music(vocal: np.ndarray, vocal_sr: int, music_path: str,
                       vocal_gain_db: float = 0.0, music_gain_db: float = -8.0,
                       tempo_lock: bool = True, autotune: bool = True
                       ) -> tuple[np.ndarray, int]:
    """Mix the TTS vocal over an uploaded music track with TEMPO LOCK and AUTOTUNE.

    Tempo lock:
      - Detect music tempo via librosa.beat.beat_track
      - Detect vocal "tempo" proxy from the speech onset rate
      - Time-stretch the vocal to align (cap at +/-25% to keep formants).

    Autotune (rough whole-clip pitch shift):
      - Detect music's dominant pitch class via chroma_cqt average
      - Detect vocal median f0 via librosa.yin
      - Compute semitones from vocal_f0 to the nearest octave of the music's root note
      - Pitch-shift the vocal by that many semitones (capped at +/-7 to avoid chipmunk)

    All local: librosa + numpy. Off-the-Grid stays clean.
    """
    import librosa
    import math

    music, music_sr = librosa.load(music_path, sr=vocal_sr, mono=True)

    # --- tempo lock ---
    if tempo_lock and len(music) > vocal_sr * 2:
        try:
            mtempo, _ = librosa.beat.beat_track(y=music, sr=music_sr)
            vtempo, _ = librosa.beat.beat_track(y=vocal, sr=vocal_sr)
            if mtempo and vtempo and abs(np.log2(mtempo / vtempo)) < 1.5:
                ratio = float(np.clip(vtempo / mtempo, 0.78, 1.28))
                if abs(ratio - 1.0) > 0.04:
                    vocal = librosa.effects.time_stretch(vocal, rate=ratio)
        except Exception as e:
            print(f"[blend] tempo lock failed: {e}")

    # --- autotune: detect music root + pitch-shift vocal toward it ---
    if autotune:
        try:
            # 12-bin chroma -> dominant pitch class is the music's key center
            chroma = librosa.feature.chroma_cqt(y=music, sr=music_sr, hop_length=2048)
            key_idx = int(np.argmax(chroma.mean(axis=1)))  # 0=C, 1=C#, ..., 11=B
            key_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            # vocal median f0 via YIN
            f0 = librosa.yin(vocal.astype(np.float32),
                             fmin=70, fmax=500, sr=vocal_sr, frame_length=2048)
            f0 = f0[~np.isnan(f0) & (f0 > 0)]
            if len(f0) > 5:
                vocal_f0 = float(np.median(f0))
                # music root note across plausible vocal octaves: 65.4 Hz (C2) .. 523 Hz (C5)
                root_freqs = [
                    27.5 * (2 ** ((key_idx - 9 + 12 * o) / 12.0))  # A0=27.5; key_idx 9 == A
                    for o in range(2, 7)
                ]
                # pick the root octave closest in log-space to the vocal's median pitch
                root = min(root_freqs, key=lambda r: abs(math.log2(r / vocal_f0)))
                semitones = round(12 * math.log2(root / vocal_f0))
                semitones = max(-7, min(7, semitones))
                if semitones != 0:
                    vocal = librosa.effects.pitch_shift(
                        vocal.astype(np.float32), sr=vocal_sr, n_steps=semitones)
                print(f"[blend] autotune: music key={key_names[key_idx]}, "
                      f"vocal_f0={vocal_f0:.1f}Hz -> root={root:.1f}Hz, "
                      f"shift {semitones} semitones")
        except Exception as e:
            print(f"[blend] autotune failed: {e}")

    # --- mix ---
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
DEFAULT_TEXT = "she sells seashells by the seashore"
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
            # Lazy-loaded inside generate(), which on ZeroGPU runs under @spaces.GPU, so
            # F5-TTS's auto device detection picks the allocated GPU. (On the T4 Space it
            # loads to cuda directly; locally it falls back to CPU.)
            self._tts = F5TTS(model="F5TTS_v1_Base")
            self.live = True
            print(f"[engine] F5-TTS base loaded (device={self._tts.device})")
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
            # The uploaded/recorded clip may be mp3/m4a/etc. Our soundfile reader (and a
            # clean F5-TTS clone) want a 24kHz mono WAV, so transcode it first via librosa
            # (handles many formats through ffmpeg). Trim to <=15s to keep cloning fast.
            try:
                import librosa, soundfile as sf
                ref, _ = librosa.load(custom_voice_path, sr=24000, mono=True)
                ref = ref[: 24000 * 15]
                tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                sf.write(tmp_ref, ref, 24000, subtype="PCM_16")
                voice_wav = tmp_ref
            except Exception as e:
                print(f"[engine] custom voice transcode failed ({e}); using raw path")
                voice_wav = custom_voice_path
            voice_ref_text = (custom_voice_text or "").strip()
            # F5-TTS needs a transcript of the reference. If the user left it blank,
            # transcribe the clip ourselves so cloning is reliable instead of depending
            # on F5-TTS's internal auto-transcribe path.
            if not voice_ref_text:
                asr = self._ensure_asr()
                if asr:
                    try:
                        voice_ref_text = asr.transcribe(voice_wav)["text"].strip()
                        print(f"[engine] auto-transcribed clone ref: {voice_ref_text[:60]!r}")
                    except Exception as e:
                        print(f"[engine] clone ref transcribe failed: {e}")
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
        ("DIAL", f"{level}" if level is not None else "·"),
        ("WER", f"{wer:.2f}" if wer is not None else "·"),
        ("VOICE-SIM", f"{cosine:.2f}" if cosine is not None else "·"),
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


def _render_token_preview(lyric: str, overrides_text: str) -> str:
    """Render the lyric as clickable gold token chips. Edited tokens show their override.
    Python-driven: gets called on every lyric / overrides change so the markup stays in sync."""
    overrides = _parse_per_word_overrides(overrides_text or "")
    if not lyric or not lyric.strip():
        return '<div class="token-row token-empty">type a lyric above; click any word here to hand-tune it</div>'
    parts = re.findall(r"[A-Za-z']+|\s+|[^A-Za-z'\s]+", lyric)
    pieces = []
    for p in parts:
        if not p:
            continue
        if not re.search(r"[A-Za-z]", p):
            # whitespace / punctuation passes through, render \n as <br>
            pieces.append(p.replace("\n", "<br>"))
            continue
        key = p.lower().strip("'")
        if key in overrides:
            repl, stretch = overrides[key]
            label = f"{repl} ({stretch:.2f}x)"
            pieces.append(f'<span class="token edited" data-word="{key}" data-pron="{repl}" '
                          f'data-stretch="{stretch:.2f}" title="{label}">{p}</span>')
        else:
            pieces.append(f'<span class="token" data-word="{key}" data-pron="{p}" '
                          f'data-stretch="1.00" title="click to hand-tune">{p}</span>')
    return '<div class="token-row">' + "".join(pieces) + '</div>'


def _apply_output_fx(y_dry, sr, postfx_preset, music_path, music_gain_db):
    """Cheap post-generation DSP: post-fx bus + optional music blend (CPU only, NOT
    @gpu_task). Kept separate from F5-TTS generation so changing post-fx / music
    re-renders instantly from the cached dry voice instead of re-running the neural net."""
    y = y_dry
    if postfx_preset and postfx_preset != "dry":
        y, _ = apply_post_fx(y, sr, preset=postfx_preset)
    if music_path:
        try:
            y, sr = _blend_with_music(y, sr, music_path,
                                       vocal_gain_db=0.0,
                                       music_gain_db=float(music_gain_db),
                                       tempo_lock=True)
        except Exception as e:
            print(f"[blend] failed: {e}; returning dry vocal")
    return _wav_to_filepath(y, sr)


@gpu_task
def speak(sentence, voice_id, level, postfx_preset, mode, seed,
          custom_voice, custom_voice_text, music_path, music_gain_db, overrides_text):
    sentence = (sentence or "").strip()
    level = _safe_int(level)
    if not sentence:
        return None, readout(level, None, None, "type a sentence first"), "", None
    overrides = _parse_per_word_overrides(overrides_text or "")
    y, sr, gen_text = _chunked_generate_with_overrides(
        ENGINE, sentence, voice_id, level, int(seed), mode,
        custom_voice or None, custom_voice_text or "", overrides,
    )
    # Save the DRY voice to its own file and cache the PATH (a cheap string), so post-fx /
    # music changes re-render from it without re-inference. We cache a path rather than the
    # raw array because speak runs in ZeroGPU's forked GPU worker and its return values are
    # serialized back across the process boundary; a path is trivial to pass, an array isn't.
    dry_path = _wav_to_filepath(y, sr)
    path = _apply_output_fx(y, sr, postfx_preset, music_path, music_gain_db)
    readout_text = gen_text if (mode == MODE_GHOST and gen_text and gen_text != sentence) else ""
    return path, readout(level, None, None, f"{mode.lower()} · lv{level}"), readout_text, dry_path


@gpu_task
def morph(sentence, voice_id, postfx_preset, mode, seed,
          custom_voice, custom_voice_text, music_path, music_gain_db,
          overrides_text, gap_ms: int = 250):
    sentence = (sentence or "").strip()
    if not sentence:
        return None, readout(None, None, None, "type a sentence first"), "", None
    overrides = _parse_per_word_overrides(overrides_text or "")
    clips = []
    sr_out = SAMPLE_RATE
    for lv in range(5):
        y, sr, _ = _chunked_generate_with_overrides(
            ENGINE, sentence, voice_id, lv, int(seed), mode,
            custom_voice or None, custom_voice_text or "", overrides,
        )
        sr_out = sr
        clips.append(y)
    morphed = equal_power_concat(clips, sr_out, fade_ms=gap_ms)
    dry_path = _wav_to_filepath(morphed, sr_out)
    path = _apply_output_fx(morphed, sr_out, postfx_preset, music_path, music_gain_db)
    return path, readout(None, None, None, f"{mode.lower()} · morphed 0->4"), "", dry_path


def reapply_fx(dry_path, postfx_preset, music_path, music_gain_db):
    """Re-render the output from the cached DRY voice file when post-fx / music changes,
    so the user hears the effect immediately without re-running F5-TTS. Nothing generated
    yet (no cached path) -> leave the player as-is. Pure CPU DSP, NOT @gpu_task."""
    if not dry_path:
        return gr.update()
    try:
        import soundfile as sf
        y, sr = sf.read(str(dry_path), dtype="float32")
    except Exception:
        return gr.update()
    return _apply_output_fx(y, sr, postfx_preset, music_path, music_gain_db)


# ----- CSS (dreamy pastel theme: half-remembered photograph of dusk) -----

CUSTOM_CSS = """
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
    --gold: #FFD66B;            /* brighter, more saturated for higher contrast on violet AND on the sun side */
    --gold-bright: #FFE9A3;
    --gold-glow: rgba(255, 214, 107, 0.6);
    --cream: #FFEFC9;
    --cream-mute: #E8D9A0;
    --ink-light: #FFE9B8;
    --ink-mute: #BFAD78;
    --hairline: rgba(255, 214, 107, 0.32);
}

/* BASE: NO opaque ancestor backgrounds (gradio-app + body default to opaque dark and
   would paint over a background layer), and NO background-attachment:fixed (jank on
   scroll, ignored on iOS Safari). The dark base + sun live on the injected #bg-sun
   layer only. Verified: no ancestor sets transform/filter/will-change/contain, so a
   position:fixed child of <body> anchors to the viewport and is never clipped. */
html, body, gradio-app, .gradio-container, .dark, .light {
    background: transparent !important;
    color: var(--ink-light) !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    min-height: 100vh;
}
/* Solid dark on <body> so first paint (before the JS mounts #bg-sun) shows no flash. */
body { background: var(--night) !important; }

/* THE FIXED SUN LAYER: a real <div id="bg-sun"> prepended to <body> by demo.load(js).
   position:fixed + inset:0 => pinned to the viewport, never scrolls, never clipped.
   The sun is a DEFINED ~560px circular ball tucked into the bottom-right corner
   (a `circle 280px` radial-gradient = 560px diameter), centered ~150px past the corner
   so most of the ball is visible like a little sun, fading cleanly to the dark base.
   NO giant viewport-wide wash. translateZ(0) = own GPU layer, no repaint on scroll. */
#bg-sun {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
      radial-gradient(circle 420px at calc(100% - 160px) calc(100% - 160px),
        #FFE7B0 0%,
        #FFB070 14%,
        #FF7A3D 30%,
        var(--sun-core) 46%,
        #C7311A 64%,
        rgba(107, 24, 8, 0.45) 82%,
        transparent 100%),
      var(--night);
    transform: translateZ(0);
}

/* CONTENT sits above the fixed sun layer. */
.gradio-container {
    background: transparent !important;
    position: relative;
    z-index: 1;
}

/* SUN-OVERLAP READABILITY: any gold/cream text gets a dark halo so it pops against
   the bright orange/red of the sun, and a subtle stroke for hard edges. Reads as
   "lit from behind" on dark bg, as "outlined" on bright bg. Plus on the right-half
   accordion labels (which directly overlap the sun corner), apply mix-blend-mode so
   the text auto-inverts as it crosses the sun. */
.gradio-container button.lg,
.gradio-container button.primary,
.gradio-container [data-testid="block-info"],
.gradio-container input[role="listbox"],
.gradio-container label.svelte-19qdtil span.svelte-19qdtil,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container summary,
.gradio-container [class^="label"],
#token-preview .token,
.knob-ticks span {
    text-shadow:
      0 0 1px rgba(0, 0, 0, 0.95),
      0 0 4px rgba(0, 0, 0, 0.85),
      0 0 14px rgba(0, 0, 0, 0.65) !important;
}
/* Sun-overlap masking restored: dark semi-opaque pills on labels so the bright sun
   doesn't bleed through into text. Text stays gold; pill blocks the sun behind. */
.gradio-container .label-wrap,
.gradio-container summary,
.gradio-container [data-testid="block-info"] {
    background: rgba(14, 8, 32, 0.78) !important;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 214, 107, 0.22) !important;
    border-radius: 999px !important;
    padding: 10px 22px !important;
    margin: 4px 0 !important;
}
.gradio-container [data-testid="block-info"] {
    display: inline-block !important;
    padding: 3px 12px !important;
    font-size: 12px !important;
    margin-bottom: 6px !important;
}

/* (Light trails + film grain layers removed. Flat dark bg + viewport-fixed sun only.) */

.gradio-container { max-width: 1040px !important; margin: 0 auto !important; padding: 72px 48px 120px !important; position: relative; z-index: 1; }

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
   as light + type + air, not stacked UI cards.
   Aggressive: nuke any "block" class background + border + radius. */
.block, .block-container, .form, .gradio-container > div > div,
[data-testid="block"], .gr-form, .gr-box,
.svelte-vt1mxs, .svelte-1ipelgc, .gr-padded,
[class*=" block"], [class^="block"],
.svelte-633qhp, .svelte-1plpy97, .svelte-1mwvhlq,
.gradio-container [class*="container"],
.gradio-container [class*="form"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
/* leave the slider container with vertical padding so the slider track is visible */
#dial-slider { padding: 18px 0 24px !important; }
/* keep accordion content padded so the audio drop boxes don't collapse */
[data-testid="accordion"] > div:nth-child(2),
.gradio-container [class*="accordion"] > div:not(:first-child) {
    padding: 16px 0 8px !important;
}

/* Rows: consistent gaps, vertical alignment */
.gradio-container .gr-row, .gradio-container [class*="row"] {
    gap: 28px !important;
    align-items: end !important;
}

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

/* Dropdown popup panel: dark solid background on the entire popup container plus
   each option, so neither the wrapper gaps nor the options show the page through. */
ul[role="listbox"],
[role="listbox"]:not(input),
.options,
.options ul,
.choices,
ul.choices,
.svelte-1xfsv4t .options,
.svelte-1xfsv4t .options ul {
    background: rgba(14, 8, 32, 0.98) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6) !important;
    padding: 6px 0 !important;
    z-index: 100 !important;
}
ul[role="listbox"] [role="option"],
[role="option"],
.options ul li,
.choices li {
    background: transparent !important;        /* the parent's dark covers everything */
    color: var(--cream) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-size: 16px !important;
    padding: 8px 18px !important;
    border-radius: 0 !important;
}
ul[role="listbox"] [role="option"]:hover,
[role="option"]:hover,
.options ul li:hover,
.choices li:hover {
    background: rgba(245, 197, 107, 0.18) !important;
    color: var(--gold-bright) !important;
}

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
/* Gradio v6 dropdown internals: the .wrap holds the white default; override the whole
   stack down to the bare <input role="listbox">. */
.gradio-container .wrap.svelte-1xfsv4t,
.gradio-container .wrap-inner.svelte-1xfsv4t,
.gradio-container .secondary-wrap.svelte-1xfsv4t {
    background: rgba(14, 8, 32, 0.92) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 18px rgba(255, 214, 107, 0.12) !important;
}
.gradio-container input[role="listbox"],
.gradio-container input.border-none.svelte-1xfsv4t {
    background: transparent !important;
    color: var(--gold-bright) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-size: 17px !important;
    padding: 8px 18px !important;
    border: none !important;
}
.gradio-container input[role="listbox"]::placeholder { color: var(--cream-mute) !important; opacity: 1 !important; }
.gradio-container .dropdown-arrow.svelte-loyhyk { fill: var(--gold-bright) !important; }

/* Mode radio (Ghost / Tongues) — Gradio v6 svelte-19qdtil */
.gradio-container label.svelte-19qdtil {
    background: rgba(14, 8, 32, 0.7) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 999px !important;
    padding: 8px 22px !important;
    margin: 0 6px 0 0 !important;
    cursor: pointer !important;
    transition: all 0.22s !important;
}
.gradio-container label.svelte-19qdtil span.svelte-19qdtil {
    color: var(--gold-bright) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-weight: 500 !important;
    font-size: 16px !important;
}
.gradio-container label.svelte-19qdtil:has(input:checked) {
    background: linear-gradient(180deg, var(--sun-core), var(--sun-mid)) !important;
    border-color: var(--sun-core) !important;
    box-shadow: 0 0 22px var(--sun-halo) !important;
}
.gradio-container label.svelte-19qdtil:has(input:checked) span.svelte-19qdtil {
    color: var(--night-deep) !important;
}
/* The native radio dot — hide it; we use the chip-style instead */
.gradio-container input[type="radio"].svelte-19qdtil { display: none !important; }
[role="radio"][aria-checked="true"] label,
input[type="radio"]:checked + label {
    background: linear-gradient(180deg, var(--sun-core), var(--sun-mid)) !important;
    color: var(--night-deep) !important;
    box-shadow: 0 0 22px var(--sun-halo), inset 0 1px 0 rgba(255, 220, 180, 0.5) !important;
}

/* buttons — glowing pill on dark, gold borders */
button.primary, button[variant="primary"], .primary > button, button.lg, .gr-button {
    background: rgba(14, 8, 32, 0.92) !important;       /* opaque enough to read over the sun */
    color: var(--gold-bright) !important;
    border: 2px solid var(--gold) !important;            /* thicker for visibility */
    padding: 14px 34px !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-weight: 500 !important;                          /* heavier weight */
    font-size: 18px !important;
    letter-spacing: 0.02em !important;
    border-radius: 999px !important;
    box-shadow: 0 0 28px rgba(255, 214, 107, 0.32), inset 0 0 14px rgba(255, 214, 107, 0.12) !important;
    transition: all 0.24s !important;
    text-shadow: 0 0 6px rgba(255, 214, 107, 0.55);
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

/* ALL AUDIO WIDGETS (output "the take" + the voice-clone / backing-track upload
   inputs). Gradio renders these white by default, which clashes with the dark theme
   and reads as a "broken white player". Force the whole stack dark + gold. */
.gradio-container [data-testid="audio"],
#audio-out {
    background: rgba(14, 8, 32, 0.88) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    box-shadow: 0 0 20px rgba(255, 214, 107, 0.14) !important;
    margin: 10px 0 !important;
}
/* The white "Drop Audio Here / Click to Upload" drop zone inside audio inputs. */
.gradio-container [data-testid="audio"] .wrap,
.gradio-container [data-testid="audio"] .upload-container,
.gradio-container [data-testid="audio"] .file-upload,
.gradio-container [data-testid="audio"] [class*="upload"],
.gradio-container [data-testid="audio"] > div {
    background: transparent !important;
    color: var(--cream) !important;
    border-color: var(--hairline) !important;
}
/* The white block-label tab (e.g. "the take") that floats top-left. */
.gradio-container [data-testid="audio"] [data-testid="block-label"],
.gradio-container [data-testid="block-label"],
.gradio-container label.svelte-19djge9 {
    background: rgba(14, 8, 32, 0.92) !important;
    color: var(--gold) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 8px !important;
}
.gradio-container [data-testid="block-label"] svg,
.gradio-container [data-testid="block-label"] span { color: var(--gold) !important; fill: var(--gold) !important; }
.gradio-container [data-testid="audio"] audio {
    width: 100% !important; height: 46px !important; display: block !important;
}
/* The native HTML5 audio control bar: tint it to fit the dark theme. */
.gradio-container [data-testid="audio"] audio::-webkit-media-controls-panel {
    background: rgba(20, 12, 40, 0.9) !important;
}

/* LIVE PREVIEW TEXTBOX (#ghost-lyric): big, gold-on-dark, italic — it's the
   "see what the voice will say" surface and must be prominent. */
#ghost-lyric, #ghost-lyric > div, #ghost-lyric .wrap, #ghost-lyric > label > div {
    background: rgba(14, 8, 32, 0.92) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 14px !important;
}
#ghost-lyric textarea, #ghost-lyric input {
    background: transparent !important;
    color: var(--gold-bright) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-size: 22px !important;
    line-height: 1.4 !important;
    text-align: center !important;
    border: none !important;
    padding: 18px 22px !important;
    min-height: 80px !important;
}

/* THE DIAL — brass knob with vermillion arc and a pointer needle.
   Tick numbers sit in a semicircle above the knob: 0 on the left, 4 on the right. */
#dial-stack {
    display: flex; flex-direction: column; align-items: center;
    margin: 32px 0 8px;
}
.dial-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: var(--gold); letter-spacing: 0.34em; text-transform: uppercase;
    opacity: 0.8; margin-bottom: 12px;        /* gap below the tag; ticks ride above the knob on their own */
}
.knob-stage {
    position: relative;
    width: 420px; height: 400px;          /* big enough to hold ticks at radius 156 above and around the knob */
    display: flex; align-items: flex-end; justify-content: center;
}
/* Tick semicircle hugging the brass ring of the knob.
   Positioned with its origin AT the knob center, so the JS polar coords are direct. */
.knob-ticks {
    position: absolute;
    left: 50%;
    bottom: 156px;   /* knob-wrap margin-bottom (16) + knob radius (140) = knob center */
    width: 0; height: 0; pointer-events: none;
}
.knob-ticks span {
    position: absolute;
    left: 0; top: 0;
    transform: translate(calc(var(--tx) - 50%), calc(var(--ty) - 50%));
    display: flex; flex-direction: column; align-items: center;
    font-family: 'Cormorant Garamond', serif; font-style: italic;
    font-size: 24px; color: var(--cream-mute);
    cursor: pointer; pointer-events: auto;
    transition: color 0.22s, transform 0.22s;
    line-height: 1;
}
.knob-ticks span small {
    font-family: 'IBM Plex Mono', monospace; font-style: normal;
    font-size: 8px; color: var(--cream-mute); opacity: 0.55;
    letter-spacing: 0.22em; text-transform: uppercase;
    margin-top: 4px; max-width: 80px; text-align: center;
}
.knob-ticks span.active {
    color: var(--gold-bright); font-weight: 500; font-style: normal;
    text-shadow: 0 0 12px var(--gold-glow);
    transform: translate(calc(var(--tx) - 50%), calc(var(--ty) - 50%)) scale(1.22);
}
.knob-ticks span:hover { color: var(--gold-bright); }

/* Old-tech knob: knurled outer ring + flat dark face + thin cream indicator line.
   Inspired by 1970s hi-fi tuner knobs (Marantz, Moog, Bakelite). */
.knob-wrap {
    position: relative;
    width: 280px; height: 280px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 16px;
    filter: drop-shadow(0 14px 28px rgba(0,0,0,0.65));
}
.knob {
    position: relative; z-index: 2;
    width: 280px; height: 280px; border-radius: 50%;
    cursor: grab; outline: none;
    /* The flat face: warm walnut / dark Bakelite */
    background:
      radial-gradient(circle at 38% 32%,
        rgba(255, 220, 175, 0.18) 0%,
        rgba(255, 220, 175, 0.0) 28%),
      radial-gradient(circle at 50% 50%,
        #2C1B0E 0%, #1B0F08 70%, #100804 100%);
    box-shadow:
      inset 0 0 0 1px rgba(255, 220, 180, 0.1),
      inset 0 -6px 12px rgba(0,0,0,0.55),
      inset 0 4px 10px rgba(255, 220, 180, 0.08);
    touch-action: none; user-select: none;
    transition: filter 0.22s;
}
/* Knurled rim — fine repeating ridges in brass */
.knob::before {
    content: ''; position: absolute; inset: 0; border-radius: 50%; pointer-events: none;
    background: repeating-conic-gradient(
      from 0deg,
      #B89968 0deg 2deg,
      #4A3618 2deg 4deg,
      #8B6E3A 4deg 6deg
    );
    -webkit-mask: radial-gradient(circle, transparent 102px, #000 103px, #000 136px, transparent 137px);
            mask: radial-gradient(circle, transparent 102px, #000 103px, #000 136px, transparent 137px);
    filter: brightness(0.92);
}
/* Subtle inner highlight ring between knurled rim and flat face */
.knob::after {
    content: ''; position: absolute; inset: 12px; border-radius: 50%; pointer-events: none;
    box-shadow:
      inset 0 0 0 1px rgba(255, 220, 180, 0.18),
      inset 0 0 18px rgba(0,0,0,0.45);
}
.knob:focus-visible, .knob.dragging {
    filter: brightness(1.04);
}
.knob.dragging { cursor: grabbing; }
/* Faint gold arc OUTSIDE the knurled rim showing dial position.
   Knob radius 140, rim outer ~136, so the arc sits at radius 144..150. The element
   extends to inset:-16px so the mask circle fits. */
.knob-arc {
    position: absolute; z-index: 1; inset: -16px; border-radius: 50%; pointer-events: none;
    background: conic-gradient(from 270deg,
      var(--gold) 0deg,
      var(--gold-bright) var(--arc-deg, 0deg),
      transparent var(--arc-deg, 0deg) 180deg,
      transparent 360deg);
    -webkit-mask: radial-gradient(circle, transparent 144px, #000 145px, #000 152px, transparent 153px);
            mask: radial-gradient(circle, transparent 144px, #000 145px, #000 152px, transparent 153px);
    filter: drop-shadow(0 0 8px var(--gold-glow));
    opacity: 0.9;
}
/* THE INDICATOR LINE — a thin cream line painted on the knob face, from center
   to the edge of the knurled rim. Rotates with the knob value. */
.knob-pointer {
    position: absolute; z-index: 4;
    left: 50%; top: 18px;
    width: 5px; height: 96px;
    margin-left: -2.5px;
    background: linear-gradient(180deg, #FFF6D8 0%, #EFDBA8 80%, rgba(239, 219, 168, 0.0) 100%);
    border-radius: 2px;
    box-shadow:
      0 0 8px rgba(255, 240, 200, 0.7),
      0 0 18px rgba(255, 240, 200, 0.35),
      inset 0 1px 0 rgba(255, 255, 240, 0.85);
    transform-origin: 50% 122px;  /* pivot at the knob center (140 - 18) */
    transform: rotate(var(--knob-angle, -90deg));
    pointer-events: none;
    transition: transform 0.24s cubic-bezier(.34,1.36,.4,1);
}
.knob-pin { display: none; }

/* Hide the actual gradio slider — the knob drives it via JS */
#dial-slider { display: none !important; }
#dial-slider .head, #dial-slider .slider_input_container { display: none !important; }

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

/* per-word token chips + inline editor */
#token-preview { margin-top: 6px; min-height: 40px; }
.token-row {
    line-height: 2.2; font-family: 'Cormorant Garamond', serif; font-style: italic;
    font-size: 22px; color: var(--cream);
}
.token {
    cursor: pointer; padding: 2px 4px; border-radius: 4px;
    border-bottom: 1px dashed rgba(245, 197, 107, 0.32);
    transition: background 0.18s, color 0.18s;
}
.token:hover { background: rgba(245, 197, 107, 0.14); color: var(--gold-bright); }
.token.edited {
    color: var(--gold-bright);
    background: rgba(255, 80, 40, 0.18);
    border-bottom: 1px solid var(--vermillion);
    text-shadow: 0 0 12px var(--gold-glow);
}
.token-editor {
    display: inline-block; vertical-align: middle;
    background: rgba(14, 8, 32, 0.95); border: 1px solid var(--gold);
    border-radius: 8px; padding: 12px 14px; margin: 6px 8px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.5), 0 0 0 2px rgba(245, 197, 107, 0.12);
    color: var(--cream);
    font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 15px;
    z-index: 12; position: relative;
}
.token-editor .te-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.token-editor .te-head { font-style: italic; color: var(--gold); border-bottom: 1px solid var(--hairline); padding-bottom: 6px; margin-bottom: 8px; }
.token-editor .te-head b { color: var(--gold-bright); font-weight: 500; }
.token-editor input.te-pron {
    background: transparent !important; border: none !important;
    border-bottom: 1px solid rgba(245, 197, 107, 0.4) !important;
    color: var(--gold-bright) !important; font-style: italic !important;
    font-size: 16px !important; padding: 4px 6px !important; min-width: 130px;
}
.token-editor input.te-stretch {
    -webkit-appearance: none; appearance: none;
    width: 150px; height: 3px; background: rgba(245, 197, 107, 0.32);
    border-radius: 999px; outline: none; cursor: pointer;
    box-shadow: 0 0 6px var(--gold-glow);
}
.token-editor input.te-stretch::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, var(--sun-mid), var(--sun-core));
    box-shadow: 0 0 8px var(--sun-halo);
    cursor: grab;
}
.token-editor .te-val { color: var(--cream-mute); min-width: 50px; text-align: right; font-size: 13px; }
.token-editor .te-buttons { justify-content: flex-end; }
.token-editor button {
    background: transparent; border: 1px solid rgba(245, 197, 107, 0.42);
    color: var(--cream); padding: 4px 14px; border-radius: 999px;
    font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 14px;
    cursor: pointer; transition: all 0.18s;
}
.token-editor button:hover { background: rgba(245, 197, 107, 0.18); color: var(--gold-bright); }
.token-editor button.te-apply { border-color: var(--vermillion); color: var(--vermillion); }
.token-editor button.te-apply:hover { background: var(--vermillion); color: var(--paper); }

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
    # Inject fonts via <link> — Gradio's CSS-in-JS strips @import, so the stylesheet
    # rule has to land in the document head as a tag, not in the CSS string.
    gr.HTML(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Pinyon+Script&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
        <div id="hero">
            <h1>glossolalia</h1>
            <p class="tagline">type a lyric. pick a voice. turn the dial.<br>
            hear it dissolve into wordless tongues, in your own voice.</p>
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
    with gr.Accordion("🎤  Clone your own voice  (record or upload 6-12 sec)",
                      open=False, elem_id="custom-voice-accord"):
        with gr.Row():
            custom_voice = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="reference clip (clean speech, no background music, 6-12 sec)",
                scale=2,
            )
            custom_voice_text = gr.Textbox(
                label="(optional) transcript of the clip",
                placeholder="leave empty to auto-transcribe (Whisper)",
                lines=2,
                scale=1,
            )

    # background music — upload an instrumental; we tempo-lock + mix the vocal over it
    with gr.Accordion("🎵  Add background music  (mix a backing track under the voice)",
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

    # per-word overrides — stretch + pronunciation per individual word.
    # Token preview HTML is Python-rendered on every lyric / state change so the markup
    # is always in sync. JS only handles the click-to-edit popover and writes the new
    # state back into the textbox below; Python then re-renders on that change.
    with gr.Accordion("✎  Stretch or re-spell individual words  (click any word to edit)",
                      open=False, elem_id="overrides-accord"):
        token_preview = gr.HTML(_render_token_preview(DEFAULT_TEXT, ""),
                                 elem_id="token-preview")
        overrides_text = gr.Textbox(
            label="override state (the click-to-edit writes here; you can also type by hand)",
            placeholder='advanced: word=pronunciation:stretch  e.g.  river=ree-vuh:1.4',
            lines=1,
            elem_id="overrides-state",
            value="",
        )
        # Click-to-edit on .token chips. Gradio v6 strips inline <script> tags inside
        # gr.HTML, so we register the delegated listener via demo.load(js=...) below.

    # the dial — brass knob with vermillion arc and a Fraunces numeral center.
    # Tick numbers sit in a semicircle above the knob (0 on the left, 4 on the right).
    # The knob's indicator is a pointer needle that lines up with the active tick.
    gr.HTML(
        """
        <div id="dial-stack">
            <div class="dial-tag">the dial</div>
            <div class="knob-stage">
                <div class="knob-ticks" id="knob-ticks">
                    <span data-lv="0">0</span>
                    <span data-lv="1">1</span>
                    <span data-lv="2">2</span>
                    <span data-lv="3">3</span>
                    <span data-lv="4">4</span>
                </div>
                <div class="knob-wrap">
                    <div class="knob" id="knob" tabindex="0" role="slider"
                         aria-valuemin="0" aria-valuemax="4" aria-valuenow="0">
                        <div class="knob-arc"></div>
                        <div class="knob-pointer"></div>
                        <div class="knob-pin"></div>
                    </div>
                </div>
            </div>
        </div>
        """
    )
    level = gr.Slider(0, 4, value=0, step=1, label="", elem_id="dial-slider",
                      show_label=False, visible=True)

    # Live preview of what the audio will say, updates as the dial / mode / lyric change.
    ghost_lyric = gr.Textbox(label="what the voice will say at this dial position",
                              interactive=False, lines=2,
                              value="",
                              elem_classes="ghost-lyric",
                              elem_id="ghost-lyric")

    # post-fx as a quiet adjuster: how much room / reverb to wrap the voice in.
    postfx = gr.Dropdown(list(POSTFX_PRESETS.keys()), value="subtle",
                         label="space (dry = bare voice, cathedral = drenched in reverb)")
    # seed is wired but hidden from the UI (it powers determinism but most users don't care).
    seed = gr.Number(value=42, precision=0, label="seed", visible=False)

    # two actions: hear ONE dial position, or hear the WHOLE 0->4 dissolution in one take.
    # morph works in whichever mode is selected (Tongues or Ghost), so the label is neutral.
    with gr.Row(elem_classes="action-row"):
        speak_btn = gr.Button("play this dial", variant="primary",
                              elem_classes="primary", scale=1)
        morph_btn = gr.Button("dissolve · sweep 0 → 4", variant="primary",
                              elem_classes="primary", scale=1)

    audio_out = gr.Audio(label="the take", type="filepath", autoplay=True,
                          elem_id="audio-out")
    # Holds the last DRY (pre-effect) voice so changing post-fx / music re-renders
    # instantly from it instead of re-running F5-TTS.
    dry_cache = gr.State(None)
    # Hidden — stub metrics block, only shown after we wire real Whisper-WER + Resemblyzer.
    metrics = gr.HTML(readout(), visible=False)

    # Bind the brass knob to the hidden slider. Gradio strips inline <script> tags,
    # Inject the fixed background sun layer as a real <div> on <body>. A real node
    # (not a ::before pseudo) is immune to Gradio re-rendering its subtree and never
    # fights another element's pseudo slot. Idempotent so a reconnect can't duplicate it.
    demo.load(
        fn=None,
        js="""() => {
            if (!document.getElementById('bg-sun')) {
                const d = document.createElement('div');
                d.id = 'bg-sun';
                document.body.prepend(d);
            }
            return [];
        }""",
    )

    # so we inject this via demo.load(js=...) which runs once on page mount.
    demo.load(
        fn=None,
        js="""() => {
            const LEVELS = 5;
            // 0 on the LEFT, 4 on the RIGHT, sweep over the TOP.
            const ANGLE_MIN = -90, ANGLE_MAX = 90;
            function getInp() { return document.querySelector('#dial-slider input[type="range"]'); }
            function levelToAngle(level) {
                const frac = level / (LEVELS - 1);
                return ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * frac;
            }
            function positionTicks() {
                const ticks = document.querySelectorAll('.knob-ticks span');
                // .knob-ticks is positioned at the knob center, so polar coords directly:
                // tx = sin(angle)*R, ty = -cos(angle)*R. Knurled rim outer is at 88px, so
                // a radius of 102 sits the tick just outside the rim.
                const radius = 178;
                ticks.forEach(t => {
                    const lv = Number(t.dataset.lv);
                    const angle = levelToAngle(lv);
                    const rad = angle * Math.PI / 180;
                    const tx = Math.sin(rad) * radius;
                    const ty = -Math.cos(rad) * radius;
                    t.style.setProperty('--tx', tx + 'px');
                    t.style.setProperty('--ty', ty + 'px');
                });
            }
            function render(level) {
                const knob = document.getElementById('knob'); if (!knob) return;
                const ticks = document.querySelectorAll('.knob-ticks span');
                const angle = levelToAngle(level);
                knob.style.setProperty('--knob-angle', angle + 'deg');
                knob.style.setProperty('--arc-deg', (angle - ANGLE_MIN) + 'deg');
                knob.setAttribute('aria-valuenow', level);
                ticks.forEach(t => t.classList.toggle('active', Number(t.dataset.lv) === level));
            }
            function setLevel(level, fire) {
                level = Math.max(0, Math.min(LEVELS - 1, Math.round(level)));
                render(level);
                if (fire) {
                    const inp = getInp();
                    if (inp) {
                        inp.value = String(level);
                        inp.dispatchEvent(new Event('input',  {bubbles: true}));
                        inp.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                }
            }
            function bind() {
                const knob = document.getElementById('knob');
                const inp  = getInp();
                if (!knob || !inp || knob.dataset.bound === '1') return !!knob && !!inp;
                knob.dataset.bound = '1';
                positionTicks();
                render(parseInt(inp.value || '0', 10));
                inp.addEventListener('input', () => render(parseInt(inp.value || '0', 10)));
                // Rotational drag: convert pointer position to angle from knob center,
                // map that angle into 0..(LEVELS-1).
                let dragging = false;
                function angleFromEvent(e) {
                    const r = knob.getBoundingClientRect();
                    const cx = r.left + r.width / 2;
                    const cy = r.top + r.height / 2;
                    // 0 deg = pointing up; clockwise positive (standard CSS rotation).
                    let a = Math.atan2(e.clientX - cx, cy - e.clientY) * 180 / Math.PI;
                    return a; // range -180..+180
                }
                function angleToLevel(a) {
                    // ANGLE_MIN = -135 -> level 0, ANGLE_MAX = +135 -> level 4
                    let clamped = Math.max(ANGLE_MIN, Math.min(ANGLE_MAX, a));
                    const frac = (clamped - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN);
                    return Math.round(frac * (LEVELS - 1));
                }
                knob.addEventListener('pointerdown', e => {
                    dragging = true;
                    knob.setPointerCapture(e.pointerId);
                    knob.classList.add('dragging');
                    setLevel(angleToLevel(angleFromEvent(e)), true);
                });
                knob.addEventListener('pointermove', e => {
                    if (!dragging) return;
                    setLevel(angleToLevel(angleFromEvent(e)), true);
                });
                const end = e => {
                    if (!dragging) return;
                    dragging = false;
                    try { knob.releasePointerCapture(e.pointerId); } catch(_) {}
                    knob.classList.remove('dragging');
                };
                knob.addEventListener('pointerup', end);
                knob.addEventListener('pointercancel', end);
                knob.addEventListener('wheel', e => {
                    e.preventDefault();
                    setLevel(parseInt(inp.value || '0', 10) + (e.deltaY < 0 ? 1 : -1), true);
                }, {passive: false});
                knob.addEventListener('keydown', e => {
                    const cur = parseInt(inp.value || '0', 10);
                    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') { e.preventDefault(); setLevel(cur + 1, true); }
                    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') { e.preventDefault(); setLevel(cur - 1, true); }
                });
                document.querySelectorAll('.knob-ticks span').forEach(t => {
                    t.addEventListener('click', () => setLevel(Number(t.dataset.lv), true));
                });
                return true;
            }
            // Gradio mounts components asynchronously; retry until both are present.
            let n = 0;
            const iv = setInterval(() => { if (bind() || ++n > 60) clearInterval(iv); }, 150);
        }""",
    )

    # Click-to-edit on .token chips. Same demo.load(js=...) pattern as the knob —
    # inline <script> in gr.HTML is stripped by Gradio v6.
    demo.load(
        fn=None,
        js=r"""() => {
            function getStateTA() {
                const wrap = document.getElementById('overrides-state');
                return wrap ? wrap.querySelector('textarea, input') : null;
            }
            function parseState(s) {
                const out = {};
                (s || '').trim().split(/\s+/).forEach(t => {
                    if (!t.includes('=')) return;
                    const [w, rest] = t.split('=', 2);
                    let repl = rest, stretch = '1.0';
                    if (rest && rest.includes(':')) { [repl, stretch] = rest.split(':', 2); }
                    out[w.toLowerCase()] = { repl: repl || w, stretch: parseFloat(stretch) || 1.0 };
                });
                return out;
            }
            function serializeState(state) {
                return Object.entries(state)
                    .map(([w, v]) => `${w}=${v.repl}:${v.stretch.toFixed(2)}`)
                    .join(' ');
            }
            function openEditor(anchor) {
                const stateTA = getStateTA();
                if (!stateTA) return;
                const word = anchor.dataset.word;
                const state = parseState(stateTA.value || '');
                const cur = state[word] || {
                    repl: anchor.dataset.pron || word,
                    stretch: parseFloat(anchor.dataset.stretch || '1.0'),
                };
                const old = document.querySelector('.token-editor');
                if (old) old.remove();
                const ed = document.createElement('div');
                ed.className = 'token-editor';
                ed.innerHTML = `
                    <div class="te-row te-head">editing <b>${word}</b></div>
                    <label class="te-row">pronunciation <input class="te-pron" type="text" value="${cur.repl}"></label>
                    <label class="te-row">stretch <span class="te-val">${cur.stretch.toFixed(2)}x</span>
                        <input class="te-stretch" type="range" min="0.5" max="2.5" step="0.05" value="${cur.stretch}"></label>
                    <div class="te-row te-buttons">
                        <button type="button" class="te-clear">clear</button>
                        <button type="button" class="te-apply">apply</button>
                    </div>`;
                anchor.insertAdjacentElement('afterend', ed);
                const pron = ed.querySelector('.te-pron');
                const sl = ed.querySelector('.te-stretch');
                const val = ed.querySelector('.te-val');
                sl.oninput = () => { val.textContent = parseFloat(sl.value).toFixed(2) + 'x'; };
                ed.querySelector('.te-apply').onclick = () => {
                    const ns = parseState(stateTA.value || '');
                    ns[word] = { repl: pron.value || word, stretch: parseFloat(sl.value) };
                    stateTA.value = serializeState(ns);
                    stateTA.dispatchEvent(new Event('input', { bubbles: true }));
                    ed.remove();
                };
                ed.querySelector('.te-clear').onclick = () => {
                    const ns = parseState(stateTA.value || '');
                    delete ns[word];
                    stateTA.value = serializeState(ns);
                    stateTA.dispatchEvent(new Event('input', { bubbles: true }));
                    ed.remove();
                };
            }
            // Delegate clicks: works even after Python re-renders the token preview.
            document.addEventListener('click', e => {
                const t = e.target.closest('#token-preview .token');
                if (t) openEditor(t);
            }, true);
        }""",
    )

    level.change(lambda lv: readout(level=_safe_int(lv)),
                 inputs=level, outputs=metrics)
    sentence.change(_render_token_preview,
                    inputs=[sentence, overrides_text], outputs=token_preview)
    overrides_text.change(_render_token_preview,
                          inputs=[sentence, overrides_text], outputs=token_preview)

    # Live preview of "what the audio will say" as the dial / mode / lyric changes.
    # Ghost mode: mondegreen-substituted real English words.
    # Tongues mode: phoneme-corrupted pseudo-text.
    def _live_preview(text, lv, m, sd):
        try:
            lv = _safe_int(lv)
            sd = _safe_int(sd) or 42
            if not text or not text.strip():
                return ""
            # Dial=0 always returns the original lyric in either mode.
            if lv <= 0:
                return text.strip()
            if m == MODE_GHOST:
                # Ghost: real English words substituted via mondegreen + DistilGPT-2 rerank.
                return ENGINE.ghost_text(text, lv, seed=sd)
            else:
                # Tongues: pseudo-ASCII rendering of the phoneme corruption that the TTS
                # actually reads. Trimmed of inter-phoneme hyphens for readability.
                from scripts.corrupt_phonemes import load_lm, corrupt_sentence
                from pathlib import Path
                _lm = getattr(_live_preview, "_lm", None)
                if _lm is None:
                    _lm = load_lm(Path("data/phoneme_lm.npz"))
                    _live_preview._lm = _lm
                _, _, pseudo, _ = corrupt_sentence(text, lv, _lm, seed=sd)
                return pseudo
        except Exception as e:
            return f"(preview unavailable: {e})"

    for trig in (level, sentence, mode, seed):
        trig.change(_live_preview,
                    inputs=[sentence, level, mode, seed],
                    outputs=ghost_lyric)
    # Also compute the preview once on initial page load so the textbox isn't empty.
    demo.load(_live_preview,
              inputs=[sentence, level, mode, seed],
              outputs=ghost_lyric)
    speak_btn.click(
        speak,
        inputs=[sentence, voice, level, postfx, mode, seed,
                custom_voice, custom_voice_text,
                music_input, music_gain, overrides_text],
        outputs=[audio_out, metrics, ghost_lyric, dry_cache],
    )
    morph_btn.click(
        morph,
        inputs=[sentence, voice, postfx, mode, seed,
                custom_voice, custom_voice_text,
                music_input, music_gain, overrides_text],
        outputs=[audio_out, metrics, ghost_lyric, dry_cache],
    )
    # Live: changing post-fx / music re-renders from the cached dry voice (no re-inference).
    for trig in (postfx, music_input, music_gain):
        trig.change(reapply_fx,
                    inputs=[dry_cache, postfx, music_input, music_gain],
                    outputs=audio_out)

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
