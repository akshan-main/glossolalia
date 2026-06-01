# The Un-Language Slider — Field Notes

*A dial that turns any sentence, in any voice, into wordless tongues. Built for HuggingFace + Gradio's Build Small Hackathon, Thousand Token Wood track.*

---

## The question

If you fine-tune a small TTS so a single token in the prompt grades the *intelligibility* of what comes out — clean speech at 0, phonotactically-valid English-native glossolalia at 4 — does the *middle* of the dial actually exist? Is there a coherent dial=2 where a sentence half-dissolves while staying in the same voice and the same prosodic shape, or does training collapse to two presets with a crossfade?

That question is the whole project. The toy ships only if the middle is real.

## Where the idea came from (and what almost killed it)

Eleven music-domain instantiations died on the way here — most of them killed by Google Lyria, Suno, Udio, and Magenta RealTime, in some order. The meta-lesson, eventually: **music generation is the most saturated, best-funded corner of generative AI right now**, and originality there is structurally capped against four billion-dollar incumbents.

Open/local *speech* has different incumbents (Hume in the cloud for emotion sliders, ElevenLabs for prosody) and one genuinely uncontested niche: **single-axis continuous attribute knobs on a fine-tuned open TTS, validated for monotonicity and independence**. Research workflows confirmed no shipped product hits the four-spec intersection of *typed-input + graded + voice-locked + dissolves-to-glossolalia*. The nearest misses — ProtoDisent-TTS / DARS for clinical dysarthria, F5-TTS dysarthric clone for pathology reconstruction, ACE-Step flow-edit for discrete lyric replacement — all target different problems.

## Why DSP can't do this

The honest version of "why does this need a fine-tune" — a question we kept asking ourselves because the wrong answer would mean shipping a plugin in a tuxedo. Signal-shaping (formant shift, reverb, time-stretch, vocoder, granular synth) acts *uniformly* on the audio that's already there. Our toy needs to:

- Read a typed sentence the user *just* invented,
- Generate audio that says it cleanly at one end and progressively *erodes specific words into different but plausible phonemes* at the other,
- Hold the syllable count, stress pattern, and voice identity locked across the trajectory.

You can't reach that with effects. The lexical resynthesis is the load-bearing thing, and only a model can do it.

A linguistically-curious counter-claim worth addressing: glossolalia, per Goodman 1972 and Samarin 1972, is well-known to reuse the speaker's native phonotactics. So could you just record someone speaking spontaneously, slap a reverb on it, and call it done? No — that gives you the *endpoint* as a separate take, never a controllable *trajectory* tied to a specific user-typed sentence. The Goodman finding actually supports the fine-tune: a structured native-phonotactic distribution is exactly what a graded conditioning token can learn cleanly and DSP cannot synthesize.

## The corrupt-text-as-training-signal trick

There is no pre-existing dataset of "sentences gradually dissolving into nonsense." Cocteau Twins albums are copyrighted, and even if they weren't, they're not labeled `level 0 / level 1 / level 2`. So we manufactured the dataset.

The trick has three steps and one twist:

1. **Take 500 ordinary English sentences** (LibriTTS-R transcripts, length-filtered to 5–15 words).
2. **For each sentence, generate 5 corrupted text variants** at substitution probabilities `p ∈ {0, 0.25, 0.5, 0.75, 1.0}`. The corruption is per-phoneme: g2p the original sentence into ARPAbet, then for each phoneme flip a Bernoulli at `p_level` and, if it flips, substitute a phoneme of the *same class* (vowel→vowel, consonant→consonant) drawn from a CMUdict bigram distribution conditioned on the previous phoneme. Stress markers and syllable count are preserved. The result is phonotactically valid English-flavored nonsense at the high end.
3. **Use the base F5-TTS (no LoRA yet) as a puppet**: feed it each corrupted text in each of 3 voice references, let it synthesize. Pair each synthesized clip with its *original* sentence + level + voice. That's 7,500 (audio, original-sentence, level, voice) tuples — the training set.
4. **The twist**: fine-tune the LoRA so that, *given the original sentence + a "tongues N" control token + the voice reference*, it produces audio matching the level-N synthesized version. The model never sees the corrupted text at training time; it sees the *original sentence* labeled with a level. At inference, the user types a sentence and turns the dial — the model has learned the mapping.

This is synthetic-data bootstrapping. It's why a single person with a Colab notebook can build this in 10 hours instead of needing a labeled corpus that doesn't exist.

## The spike: does the middle of the dial work?

*(Spike result + plots go here after the GPU run lands.)*

The gates were locked in advance, written into the validation script before any GPU spend:

- **Whisper-WER must rise monotonically across levels 0..4, Spearman ≥ +0.80.** WER goes up because the output is less intelligible — that's the dial doing its job. Hallucination-guarded: when Whisper's avg-logprob falls below threshold, WER floors at 1.0 so glossolalia doesn't get a spuriously *low* WER because Whisper invented coherent words from noise.
- **Resemblyzer cosine vs the level-0 reference clip stays ≥ 0.85 across all levels.** Voice preserved.
- **Hand-listen the dial=2 wavs**: they must be a *partial* dissolution — recognizably the same sentence, blurred, in the same voice. Not bimodal collapse to clean-or-mush.

If gate three fails, the originality drops from a defensible 5 to a 3 ("two presets with a crossfade"), because that collapse is something a DSP chain + two TTS calls *could* fake. The middle is the IP.

## The toy

`v1` is a standard Gradio UI: text input, voice picker, slider 0..4, Speak / Morph buttons, an audio player, and a live WER + voice-similarity readout. `v2` is a custom HTML/JS frontend served via `gradio.Server` with a real circular knob widget (CSS conic-gradient + JS pointer-drag) — same backend, same `@app.api()` endpoint, much more tactile.

A small post-FX bus (Pedalboard reverb + chorus + octave layer) sits between the model and the speaker, toggleable as `dry / subtle / lush / cathedral`. It's not a sonic claim — it just keeps the dry TTS from undermining itself. The dial does the meaning-making.

What it isn't: a Liz Fraser clone, a Cocteau Twins generator, an ElevenLabs replacement. Fraser's signature is sung soprano + multitracked harmonies + 4AD reverb-soaked production + non-English phoneme sourcing — none of which a dry monophonic open-TTS dial honors, and we didn't pretend otherwise. The lineage we *do* sit in is the older one: Sigur Rós' Hopelandic, Lisa Gerrard, Meredith Monk, scat, religious glossolalia. The continuous-graded-control axis is the contribution.

## What we learned

- The hardest part of an originality bet at a hackathon judged by a world-class field isn't the model work. It's killing your own ideas honestly enough, often enough, to get to one that isn't already shipped by someone with a thousand engineers.
- Workflow-based adversarial verification beats vibes. The Cocteau Twins frame *felt* right; the verification dossier killed it cleanly with citations, and the renamed toy is more honest *and* more interesting.
- A measured property + a clean label + a graded control token + a learning model + a validation harness with hard gates is a generic enough recipe that we used it twice (once in music, once in TTS). The transferable artifact is the recipe.

---

*Code: [GitHub](https://github.com/akshan-main/glossolalia) · Space: [HF](https://huggingface.co/spaces/build-small-hackathon/glossolalia) · LoRA: [HF model](https://huggingface.co/akshan-main/glossolalia-dial-lora)*
