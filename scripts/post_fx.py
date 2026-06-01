"""Audio post-FX bus for the Un-Language Slider.

Closes the dry-TTS-sounds-like-a-phone-call gap with pedalboard DSP — NOT a Fraser claim,
just a way to keep the toy from undermining itself. All effects are toggleable in the UI;
default preset is gentle ('lush'). All processing is done on a stereo signal.

Effects (in order):
  1. Gain stage
  2. Light chorus (modulation)
  3. Slap delay (short stereo delay, low feedback)
  4. Plate reverb (long tail, ~50% wet)
  5. Octave-up self-layer mixed under (subtle harmonics)

API:
  from scripts.post_fx import apply_post_fx
  y_wet, sr = apply_post_fx(y, sr, preset="lush", octave_mix=0.18)
"""

from __future__ import annotations

import numpy as np


PRESETS = {
    "dry": {"chorus": 0.0, "delay": 0.0, "reverb": 0.0, "octave_mix": 0.0},
    "subtle": {"chorus": 0.20, "delay": 0.10, "reverb": 0.25, "octave_mix": 0.08},
    "lush":   {"chorus": 0.35, "delay": 0.15, "reverb": 0.50, "octave_mix": 0.18},
    "cathedral": {"chorus": 0.30, "delay": 0.20, "reverb": 0.75, "octave_mix": 0.22},
}


def _to_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.stack([y, y], axis=0)
    return y


def _pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Phase-vocoder pitch shift via librosa (kept dependency-light)."""
    import librosa
    if y.ndim == 2:
        return np.stack([librosa.effects.pitch_shift(y=y[c], sr=sr, n_steps=semitones)
                         for c in range(y.shape[0])], axis=0)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)


def apply_post_fx(y: np.ndarray, sr: int, preset: str = "subtle",
                  chorus: float | None = None, delay: float | None = None,
                  reverb: float | None = None, octave_mix: float | None = None) -> tuple[np.ndarray, int]:
    """Apply the post-FX bus. Returns (wet_stereo, sr)."""
    p = PRESETS.get(preset, PRESETS["subtle"]).copy()
    if chorus is not None: p["chorus"] = chorus
    if delay is not None: p["delay"] = delay
    if reverb is not None: p["reverb"] = reverb
    if octave_mix is not None: p["octave_mix"] = octave_mix

    from pedalboard import Pedalboard, Chorus, Delay, Reverb, Gain
    y = _to_stereo(np.asarray(y, dtype=np.float32))

    # 1. octave-up self-layer mixed under (subtle harmonic)
    if p["octave_mix"] > 0:
        y_oct = _pitch_shift(y, sr, semitones=12.0).astype(np.float32)
        # length-match (pitch_shift preserves length for librosa)
        L = min(y.shape[-1], y_oct.shape[-1])
        y = y[..., :L] + p["octave_mix"] * y_oct[..., :L]

    # 2. effect chain via pedalboard (operates on float32 stereo of shape [2, N])
    board_fx = [Gain(gain_db=-1.0)]
    if p["chorus"] > 0:
        board_fx.append(Chorus(rate_hz=0.8, depth=0.25, centre_delay_ms=8.0,
                               feedback=0.0, mix=float(p["chorus"])))
    if p["delay"] > 0:
        board_fx.append(Delay(delay_seconds=0.11, feedback=0.15, mix=float(p["delay"])))
    if p["reverb"] > 0:
        board_fx.append(Reverb(room_size=0.8, damping=0.35, wet_level=float(p["reverb"]),
                               dry_level=1.0 - 0.5 * float(p["reverb"]), width=1.0))

    board = Pedalboard(board_fx)
    # pedalboard expects float32, shape (channels, samples)
    wet = board(y.astype(np.float32), sample_rate=sr)
    # soft clip to avoid peaks
    peak = float(np.max(np.abs(wet))) if wet.size else 1.0
    if peak > 0.98:
        wet = wet * (0.98 / peak)
    return wet, sr


def main():
    import argparse
    import soundfile as sf
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--preset", default="lush", choices=list(PRESETS.keys()))
    args = p.parse_args()
    y, sr = sf.read(args.inp, always_2d=False)
    if y.ndim == 2:
        y = y.T  # soundfile gives (samples, channels); we want (channels, samples)
    wet, sr = apply_post_fx(y, sr, preset=args.preset)
    sf.write(args.out, wet.T, sr)
    print(f"wrote {args.out} (preset={args.preset}, sr={sr}, dur={wet.shape[-1]/sr:.2f}s)")


if __name__ == "__main__":
    main()
