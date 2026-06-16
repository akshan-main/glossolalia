---
title: Glossolalia Dial
emoji: 🍄
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.10.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: One dial, clean speech to wordless tongues, same voice
tags:
  - gradio
  - build-small-hackathon
  - track:wood
  - text-to-speech
  - voice-cloning
  - f5-tts
  - lora
  - glossolalia
  - achievement:welltuned
  - achievement:offgrid
  - achievement:fieldnotes
  - achievement:offbrand
---

# Glossolalia Dial

Type a sentence. Pick a voice. Turn the dial.

At **0** you hear it spoken cleanly. At **4** you hear it as wordless glossolalia: invented words that obey English sound-rules but mean nothing, in the same voice. **The middle of the dial is the point.** At 2 the sentence is half-dissolved, recognizable but slipping, not a clean cut between speech and noise.

The dial is a learned scalar conditioner. A small network maps the dial position to a vector added into F5-TTS's time embedding (the same AdaLN pathway the model uses for the diffusion timestep), co-trained with a LoRA. The naive version (appending a `tongues N` token to the prompt) failed: F5-TTS has no language-model front end, so it read the level word aloud and intelligibility moved the wrong way (Spearman -0.70). Making the conditioning a non-text scalar means the model cannot speak it, and the LoRA only has to learn one thing: the per-level audio transformation.

**Live:** turn the dial, hit *play this dial*. Or hit *dissolve* to hear the whole 0 to 4 sweep crossfaded into one take.

## Why it is worth a look

No shipped product, open or closed, gives you a *typed-input, graded, voice-locked* slide into glossolalia. Emotion and prosody sliders (Hume, ElevenLabs) move other axes and optimise *for* intelligibility. The closest research (dysarthric-speech clones, discrete lyric-swap edits) solves a different problem. The originality here is the interaction, not the model: a continuous, learned intelligibility axis on one token.

It is also not a DSP trick. Reverb, formant-shift, and vocoders act uniformly on audio that already exists. They cannot read a sentence you just invented and erode specific words into different but plausible ones while holding syllable count, stress, and voice. Only a model trained for it can, which is why this needed a fine-tune.

## Two modes

- **Tongues**: true glossolalia. The dial conditions the LoRA to slur the sentence into invented, pronounceable pseudo-words. `she sells seashells by the seashore` becomes something like `she'll sell sicials by the sohar` at the middle, wordless tongues at the top.
- **Ghost**: mondegreen. Real English words are swapped for similar-sounding real words (`seashells` to `seagulls`), the misheard-lyric effect. More words change as the dial rises. **This is pareidolia, not glossolalia**, and is labeled as such.

## How it was built

There is no dataset of "sentences gradually dissolving into nonsense", so we made one. This is the whole reason a single person can build this: instead of hunting for labeled data that does not exist, we manufacture the training target from plain text.

1. Take 3000 public-domain sentences (Project Gutenberg + LibriSpeech). Public-domain on purpose: the corpus is rights-clean and the build is reproducible end to end, which also keeps the project Off-the-Grid (no scraped or licensed text, no cloud calls).
2. For each, generate five corrupted phoneme variants at substitution rates 0, 0.25, 0.5, 0.75, 1.0. The corruption keeps the English phoneme inventory, preserves syllable count and stress, and leans toward open CV syllables. This is grounded in the phonetics of real glossolalia (Samarin 1972, Goodman 1972; Link & Tomaschek 2024 measured 95.7% CV structure across 7,486 glossolalic syllables).
3. Have base F5-TTS read each corrupted variant in two reference voices. That gives 30,000 (audio, original-sentence, level, voice) tuples.
4. Fine-tune a LoRA so that, given the *original* sentence plus a `tongues N` token, it reproduces the level-N audio. The model never sees the corrupted text. It learns the mapping from the dial alone.

Training ran on Modal (A100). Ghost mode runs the mondegreen search live (CMUdict + PanPhon phonetic distance, re-ranked by DistilGPT-2 for common-word coherence), no second model trained.

## Validation gates

The gates were written into the evaluation script before any GPU spend, so the dial had to earn its result:
- **Whisper-WER rises across levels.** The output is meant to get less intelligible, so word-error-rate should climb monotonically with the dial. Hallucination-guarded, so invented words at the top do not score a spuriously low WER when Whisper invents coherent text from noise.
- **Voice preserved.** Resemblyzer speaker-embedding cosine between dial-0 and the higher levels should stay close, so the words dissolve but the speaker does not.
- **The middle has to exist.** Hand-listen at dial 2 for partial dissolution, not a bimodal jump from clean to gibberish.

The honest evidence: turn the dial and the output slides from the sentence as typed at 0, to a half-dissolved version at 2 that is recognizably the same sentence but slipping, to wordless at 4, in the same voice. The exact words change run to run because it is sampling a generative model, but the graded slide is stable. The middle is real.

## Models (all under 32B, all local)

| Model | Size | Role |
|---|---|---|
| F5-TTS v1 Base | ~336M | flow-matching TTS, zero-shot voice clone |
| Glossolalia LoRA | rank-16 adapter | the dial (published, see below) |
| DistilGPT-2 | ~82M | Ghost-mode word re-ranking |
| Whisper base.en | ~74M | clone-reference transcription, validation |

Nothing calls a cloud API. Every model runs on the Space.

## Badges

- 🎯 **Well-Tuned**: fine-tuned LoRA published at [`akshan-main/glossolalia-dial-lora`](https://huggingface.co/akshan-main/glossolalia-dial-lora).
- 🔌 **Off the Grid**: no cloud APIs anywhere; `requirements.txt` has zero cloud SDKs.
- 🎨 **Off-Brand**: the dial is a hand-built circular knob (CSS conic-gradient rim, pointer-drag JS, arc indicator, injected HTML) driving the model, not a default Gradio slider.
- 📓 **Field Notes**: full write-up in [`BLOG.md`](https://github.com/akshan-main/glossolalia/blob/main/BLOG.md).

## Use it

- **Voice:** pick one of nine presets (warm and calm, high and arch, deep and slow, plus theatrical, haunted, and storyteller character voices from public-domain LibriVox), or open *clone your own voice* to upload or record a 6-12s clip.
- **Background music:** drop in an instrumental and it tempo-locks, tunes the vocal toward the track's key, then mixes it over.
- **Hand-tune words:** click any word to change its pronunciation or stretch it.
- **Space (reverb):** dry to cathedral, applied live without re-running the model.

## Links

- **Model (LoRA):** https://huggingface.co/akshan-main/glossolalia-dial-lora
- **Dataset (inputs):** https://huggingface.co/datasets/akshan-main/glossolalia-inputs
- **Code:** https://github.com/akshan-main/glossolalia
- **Field Notes:** https://github.com/akshan-main/glossolalia/blob/main/BLOG.md
- **Demo video:** https://youtu.be/dDOaBNfihyo
- **Social post:** https://x.com/frutigeraerosol/status/2066667649338417367

## Team

- [`akshan-main`](https://huggingface.co/akshan-main)
