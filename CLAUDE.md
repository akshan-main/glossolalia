# Build Small Hackathon — project context

This directory is the user's entry to HuggingFace + Gradio's **Build Small Hackathon**, track 🍄 **Thousand Token Wood**. Submissions close **2026-06-15**.

## Project: Glossolalia Dial

**One-line premise:** a single dial that grades a typed sentence from intelligible speech to phonotactically-valid English-native glossolalia — in the same voice — using one fine-tuned LoRA control token. (Earlier internal name during scoping: "The Un-Language Slider." Kept as a poetic subtitle only; the canonical product name is "Glossolalia Dial," which matches the HF Space, model, dataset, and GitHub repo.)

You type *"I had a dream last night about the ocean."* Pick a voice. The slider goes 0 → 4. At 0 you hear it spoken cleanly. At 2 you hear it half-dissolved ("I hade dremlas nigh abou the oshen…"). At 4 you hear it as wordless tongues that *sound* like a real language but aren't ("kah leh nah doh seh meh nah"), still recognizably the same speaker. A **Morph** button sweeps 0 → 4 across one continuous take. A small post-FX bus (reverb + chorus + octave layer) keeps the dry TTS from sounding like a phone call.

## Why this is original (verified, not asserted)

Three workflow audits during planning, each adversarially scored:

1. **Landscape**: no shipped product — open or closed — exposes a continuous *typed-input + graded + voice-locked + dissolves-to-glossolalia* control on a TTS. Nearest misses (ProtoDisent-TTS, ACE-Step flow-edit, F5-TTS dysarthric clone) target a different problem (clinical dysarthria / discrete lyric replace / single-point pathology). ElevenLabs / OpenAI / Hume / Sesame sliders all optimize *for* intelligibility or control orthogonal axes (emotion, prosody, speaker).
2. **DSP critique killed**: the Gemini-style "spontaneous speech + voice modification effect" replication is engineering-incoherent — signal-shaping cannot do lexical resynthesis from a typed input, cannot produce coherent middle states with locked syllable count/stress, and never gives a continuous *trajectory*. Goodman 1972 / Samarin 1972 phonotactic findings about glossolalia *support* the fine-tune (a structured native-phonotactic distribution is exactly what a graded conditioning token can learn).
3. **Cocteau Twins frame audit — failed and dropped**: an earlier framing as "Liz Fraser Dial" was honestly cargo-cult. Fraser's signature is sung soprano + multitracked harmonies + 4AD reverb + non-English phoneme sourcing; our dry monophonic TTS dial honors none of those. The toy *is* in the broader lineage of wordless vocal music (Sigur Rós Hopelandic, Lisa Gerrard, Meredith Monk, scat, religious glossolalia — Samarin/Goodman) but does not claim to *be* Cocteau Twins. The frame is the long tradition, not a single artist.

**Originality 5 is defensible, conditional on the spike validating the *middle* of the dial** — dial=2 must be a perceptual partial dissolution, not bimodal collapse to "clean at 0, gibberish at 3+." The middle is the IP.

## Architecture

- **Base model**: F5-TTS (~336M, flow-matching, voice-cloning, accepts IPA / phoneme input). Open weights. CPU/CUDA. The IPA path is load-bearing because we inject corrupted phonemes.
- **Adapter**: plain LoRA, rank 16, alpha 16, target the F5-TTS DiT attention modules. Plain LoRA chosen over LoKr because LoRA is composable, mergeable, and proven for graded control (Concept Sliders ECCV 2024). LoKr's "10× faster" claim is conflated with caching and irrelevant here.
- **Control token**: `{stem} {level_word}` appended to the prompt. Stem `tongues`. Level words `zero | one | two | three | four`. Five trained levels.
- **Training data — manufactured, not collected**: 500 sentences × 3 voices × 5 levels = 7,500 clips (full-scale variant: 1500 × 3 × 5 = 22,500 — uses multi-session Colab time). Each clip is synthesized by base F5-TTS reading a corrupted version of the source sentence (g2p → bigram-conditional phoneme substitution at p ∈ {0, 0.25, 0.50, 0.75, 1.0}, syllable count + stress preserved). Labels pair the *original* sentence + level with the *corrupted* base-TTS audio; the LoRA learns the mapping.
- **Validation gates** (all must pass):
  - **Whisper-WER** monotonic across levels, Spearman ≥ +0.80 (positive sign — WER *rises* as dial rises).
  - **Resemblyzer cosine** dial-0 vs every other level ≥ 0.85 (voice preserved).
  - **Perceptual gate**: hand-listen dial=2 wavs — partial dissolution, not bimodal.
  - **Hallucination guard** on Whisper: `--no_speech_threshold 0.8 --logprob_threshold -1.5`, floor WER at 1.0 when avg-logprob falls below threshold (don't let glossolalia get a spuriously LOW WER because Whisper invented coherent words from noise).
- **Post-FX bus** (Pedalboard, gentle defaults): plate reverb + light chorus + slap delay + optional octave-up layer mixed under. Toggleable in the UI as `dry / subtle / lush / cathedral`. This is *not* a Cocteau Twins claim — it just keeps the dry TTS from undermining itself.

## UI

- **v1** (`app.py`): standard `gr.Blocks` Gradio app. Text input + voice picker + slider 0..4 + post-FX preset + Speak + Morph buttons + audio output + live readout (WER + voice-similarity).
- **v2** (`app_server.py` + `static/`): `gradio.Server` (FastAPI + Gradio engine) serving a custom HTML/JS page with a real circular knob widget (CSS conic-gradient + JS pointer-drag, snaps to integer levels), connected to the same backend via the Gradio JS client. **Earns the Off-Brand badge.**

## Badges targeted

- 🎯 **Well-Tuned** — the fine-tuned LoRA published to `akshan-main/glossolalia-dial-lora` on HF.
- 🎨 **Off-Brand** — the v2 custom HTML knob UI via `gradio.Server`.
- 🔌 **Off the Grid** — no cloud APIs anywhere; `requirements.txt` audited (zero `openai` / `elevenlabs` / `anthropic` / `google-cloud` packages). All inference local.
- 📓 **Field Notes** — `BLOG.md` walks the audit → corruption pipeline → spike result → toy story.
- ❌ **Llama Champion** — skipped; F5-TTS isn't llama.cpp-native.
- ❌ **Sharing is Caring** — *doesn't apply* (that badge is for *agent traces*; this is a generative toy, not an LLM-tools-loop agent — don't fake it).

## Hard constraints

- ≤ 32B params (F5-TTS is ~336M, well under).
- Pure Gradio app (v1) + `gradio.Server` (v2) on a HF Space under the `build-small-hackathon` org.
- No cloud APIs.
- Submission: Space link + ≤90s demo video + social post.

## Timeline (absolute dates)

- 2026-06-03 — registration closes (already registered).
- 2026-06-05 — hack window begins.
- 2026-06-15 — submissions close (Space, demo video, social post).

## Repo layout

```
config.py                       # TTS model + LoRA config + voice presets + gates
requirements.txt                # f5-tts + g2p_en + jiwer + whisper + resemblyzer + pedalboard + gradio
app.py                          # v1 gr.Blocks UI
app_server.py                   # v2 gradio.Server UI
static/{index.html,style.css,knob.js}   # v2 custom frontend
scripts/
  build_phoneme_lm.py           # CMUdict -> phoneme unigram + bigram LM
  corrupt_phonemes.py           # g2p + Markov substitution at level p, syllable + stress preserved
  fetch_data_inputs.py          # pulls 500 sentences + 3 voice refs from LibriTTS-R/LibriSpeech
  generate_coherence_data.py    # base-TTS-synthesizes the 7500-clip training corpus
  build_coherence_dataset.py    # turns it into F5-TTS finetune CSV/JSONL + symlinked wavs
  sweep_dial.py                 # runs the trained LoRA across (sentence, voice, level, seed)
  evaluate_coherence_dial.py    # Whisper-WER + Resemblyzer cosine + Spearman + verdict JSON
  post_fx.py                    # pedalboard reverb/chorus/delay/octave bus
  push_data_inputs.py           # uploads data/ to HF dataset repo
data/                           # sentences.txt, voices/, phoneme_lm.npz, cmudict.dict (gitignored)
notebooks/coherence_dial_spike.ipynb   # Colab: install -> data pull -> train -> sweep -> validate
BLOG.md                         # Field Notes
IDEA-AUDIT.md                   # killed concepts + research verdicts (provenance)
```

## Working notes for Claude

- **Assume world-class competition. This is non-negotiable.** Run by OpenAI / NVIDIA / OpenBMB / Cohere with real cash + GPU prizes; every participant is highly skilled and fully motivated for 1st place. NEVER downplay other entries, never reassure by assuming "most won't do X," "the field is weak," or "we stand out because others won't bother." Our entry must win on **absolute merit** — be excellent assuming everyone else is too, and is doing the hard thing well. Justifying our position by imagining weak competitors is banned; it produces complacent strategy and bad calls.
- **The originality is in the interaction (graded intelligibility on a single fine-tuned control token), not the model.** Don't drift back into "we made Cocteau Twins" framing — that was honestly cargo-cult and we killed it.
- **Toy framing, not product framing.** Optimize for surprise + show-a-friend in 30s, not for breadth.
- Persist project context here in CLAUDE.md, not in `memory/project_*.md`.
- Minimal commit messages, never mention Claude/Anthropic/tests.

## Working discipline — pre-move research, post-move audit, decisions log

This project is being judged against world-class competition with a real cash + GPU prize pool. Slop loses. Every load-bearing move follows this loop:

1. **Pre-move research (workflow).** Before any substantive design or technical change — new feature, framing pivot, model choice, data pipeline alteration, scope shift — run a `Workflow` first. Independent agents gather sources, adversarial verifiers refute weak claims, the synthesis produces a citable verdict. No vibes-based design. If the verdict is "don't do this," don't do it.
2. **Make the move.** Implement against the verified plan, not against intuition.
3. **Post-move audit (workflow).** After committing a non-trivial change, run an audit `Workflow` covering: regressions in the existing pipeline, consistency with stated decisions, compliance with hackathon rules, internal contradiction between code / docs / decisions log. Fix what it finds before the next move.
4. **Decisions log.** Every load-bearing choice gets a `DECISIONS.md` entry with: the choice, what was rejected, the *why* (one paragraph), the cited sources, and the workflow that produced the verdict. A claim without a verified citation is not a claim — it's a guess we haven't done yet. The writeup (`BLOG.md`, demo voiceover, social post) cites the *actual* sources, not paraphrased authority.
5. **Mechanical edits skip the loop.** Typo fixes, file renames, dependency bumps, formatter passes — just do them. The loop is for moves a judge could plausibly ask "why this and not X" about.

This discipline is the difference between "nice toy" and "this person did proper work." Make every key decision in the demo video / BLOG.md / Q&A back to a verified source. Specifically: when explaining the LM rebias, cite the actual Goodman 1972 / Samarin 1972 / Hopelandic findings — not "we picked dreamy phonemes." The audience that knows the difference is exactly the audience the prizes come from.
