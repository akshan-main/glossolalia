# Idea Audit — Provenance for The Un-Language Slider

The point of this file is to remember **why** we landed on the current concept and **what** we killed along the way, so we don't relapse. All verdicts came out of adversarial multi-agent research workflows during planning.

---

## Locked: The Un-Language Slider

Graded intelligibility on a TTS, controlled by a single fine-tuned token. Originality 5 conditional on the spike validating the middle of the dial (Whisper-WER monotonic ≥ +0.80, Resemblyzer cosine ≥ 0.85, dial=2 perceptually partial). See `CLAUDE.md` for the concept details + architecture.

### What fine-tuning uniquely unlocks (the test every idea must pass)

A fine-tune is justified when the control it adds is:

1. **Not reachable by prompting** the base model. If you can type a sentence and get the same effect, the LoRA is decorative.
2. **Not reachable by DSP** on the output. If a Pedalboard chain reproduces it, you're shipping a plugin.
3. **Teaches the model a NEW conditioning signal** the base never saw — a learned mapping from a control token to a measurable acoustic property, validated for monotonicity and independence.

Graded intelligibility on TTS clears all three: prompting can't grade dissolution; DSP can't lexically resynthesize a different sentence into the same voice's prosody with locked syllable count; and the control token is a fresh learned mapping.

---

## Dead instantiations + reasons (do NOT recycle without genuinely new evidence)

| # | Idea | Why dead |
|---|---|---|
| 1 | Music-subgenre LoRA catalog (32-64 subgenre LoRAs) | Saturated by prompting in the base music model. Reads as breadth, not originality. Killed during early scoping. |
| 2 | Blender — runtime multi-LoRA blend across subgenres (sliders) | Collides head-on with Google Lyria RealTime (paid cloud, smooth live morph, already shipped). LoKr adapters don't weight-merge; "Diffusers `set_adapters`" path for ACE-Step is undocumented + unproven. |
| 3 | Timeline — schedule subgenres across a song via cue pins | Per-step LoRA scheduling ≠ per-time-region control (denoising step index ≠ audio time axis). Proper version needs MuseControlLite/LiLAC-grade architecture work — research-grade, not hackathon-grade. Cheap "sequential a2a chain" version relies on ACE-Step a2a, which is the model's known-buggiest path (issues #287, #302). |
| 4 | Field — record non-musical audio (coffee shop / yawn) → music in chosen subgenre | ACE-Step has a music-only prior at *both* the DCAE encoder and the diffusion denoiser. No empirical example of non-musical → coherent music. Strength tuning can't rescue it (low strength keeps OOD latent, high strength deletes input). Stable Audio handles this; ACE-Step does not. |
| 5 | "Impossible Dial" — dials for the residual of music after subtracting the language subspace, via CLAP | Squeezed: CLAP is contrastive audio-text, so its principal axes are *exactly* what the residual buries — residual axes either correlate strongly with nameable directions (PC1 ≈ 0.69 nameable) or are perceptually incoherent (axis 3 had post-rock at both extremes). Fundamental, not a CLAP bug. |
| 5b | Impossible Dial via MERT | Substrate validated (residual survival 86%, top-axis split-half r=0.70). But the GPU spike showed the axis was *audibly distinct* yet *uninterpretable to humans* — user verdict: "knob should mean something." Killed for being un-fun, even though originality was genuinely 5. |
| 6 | Music character knobs — grit, density, punch (as labels on measured audio features) | Two failures in one. (a) "Density" already shipped by Lyria RealTime (continuous). (b) The knob labels were measurements with arbitrary names ("grit" = spectral flatness; "punch" = crest factor) — user verdict: "the knobs and what they control are made up, can't have that at all." A knob is only fun if it means something universal, not "this measurement we named X." |
| 7 | Echo Chamber — region-local recomposition (click a bar → recompose just it via repaint + the knob LoRA) | The interaction is genuinely uncontested in audio, and feasibility was 4/5 (reuses repaint + retake + the knob LoRA). But: no bar grid exists for our instrumental corpus, regions become arbitrary time-slices, seams degrade, and the framing was anyway built on the "music knob" the user already killed. |
| 8 | Wrongifier — "make music deliberately wrong but still a song" by degree | User verdict: gimmicky one-laugh — fails the "keep coming back" replayability bar. Strong originality, weak product. |
| 9 | Hall of Mirrors — knob for how much a piece quotes itself (self-similarity tightness) | Most genuinely uncontested axis, but feasibility 2 — long-range compositional form, different class than the frame-level scalars we proved we can train. Unproven that a scalar token can teach long-range form at all. |
| 10 | Conversation — knob for instrument turn-taking (concentration vs spread of active voices, density-held-flat) | Pitch's central claim ("every piece already proven in-repo") was factually wrong — Demucs was only wired for 2-stem vocals, not the 4-stem split this needs. Also density-collapse cheat path. |
| 11 | "Liz Fraser Dial" framing of the current toy | Honest dossier: fails 4 of 5 of Fraser's irreducible properties (sung soprano vs spoken TTS; multitracking; 4AD reverb-soaked production; non-English phoneme sourcing). Calling a dry monophonic TTS dial "Liz Fraser" is cargo-cult and would mislead anyone who knows the records. **Frame dropped → renamed "The Un-Language Slider."** The product framing now points at the *tradition* (Sigur Rós, Lisa Gerrard, Monk, scat, religious glossolalia) rather than a specific artist. |

### Adapter choice (settled with research, do not re-relitigate)

**LoRA, not LoKr.** Concept Sliders (ECCV 2024) used low-rank LoRA *because* the low-rank constraint is the disentanglement mechanism for continuous graded composable control — that's our exact problem. LoKr's "10× faster" is a confounded vendor claim (different epoch/LR defaults) and irrelevant on a cached-latent training pipeline anyway. LoKr also doesn't weight-merge and breaks the clean composition path. We proved LoRA works (density knob Spearman 0.90, cross-drift 0.10); we have zero evidence for LoKr on graded control.

### Music corpus + ACE-Step pivot (rationale for the domain change)

After 11 dead instantiations in music, the meta-truth: **music generation is the most saturated, best-funded corner of generative AI right now** (Lyria 3, Suno, Udio, Magenta RT, MusicGen). Originality there is structurally capped — our best adversarially-screened idea topped out at "solid 4" and the user (rightly) felt none of them passed the "wait what" bar. Speech processing has Hume + ElevenLabs in the closed/cloud lane, but the **open/local single-axis attribute-knob niche** is genuinely empty (verified). The pivot to TTS Coherence Dial sits in that niche.

### What's still allowed to be true here (open questions, gated by the spike)

- The middle of the dial works (dial=2 is a perceptual partial dissolution, not bimodal). If this fails, originality collapses to ~3 — "two presets with a crossfade." We mitigate by flattening the p-schedule + curriculum-training level-2-weighted mini-batches.
- Voice stays the same person across the dial (Resemblyzer cosine ≥ 0.85). Mitigation if it collapses at level 4: lower LoRA rank, restrict target_modules to text-cross-attn only, optionally train two LoRAs (low-half + high-half) and blend.
- Whisper-WER monotonic. Mitigation if not: tighten the Markov palette toward a sonorant-friendly inventory; mind the hallucination-guard thresholds.

If two of these gates fail and mitigations don't recover, the honest move is to fall back to a renamed "Coherence Slider" with fewer claims (no Hopelandic/Monk lineage in the pitch), or to declare the spike a no-go and reconsider — not to ship a broken toy.
