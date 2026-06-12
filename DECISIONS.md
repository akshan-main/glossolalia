# Decisions Log — Glossolalia Dial

Every load-bearing choice in this project, with the reasoning + citations. Each entry lets a judge or contributor reconstruct *why* we did this — not just what we did. If a claim isn't here with a verified source, treat it as not made.

**Format:** date — title; choice; rejected; why; sources; the workflow that produced the verdict (if any).

---

## 2026-05-31 — TTS base = F5-TTS (not Spark / Chatterbox / Orpheus / Higgs / ElevenLabs)

- **Choice:** F5-TTS v1 Base (~336M, flow-matching, character-level input, zero-shot voice cloning).
- **Rejected:** Spark-TTS, Chatterbox, Orpheus, Higgs Audio v2, ElevenLabs, XTTS-v2.
- **Why this one:** F5-TTS has **no LLM front-end**. LLM-backed TTS auto-corrects out-of-distribution input — corrupted phoneme strings, the load-bearing input for the dial — back toward real English, collapsing the trajectory to two presets with a crossfade in the middle. F5-TTS pronounces corrupted spellings as written, which is what makes the *middle* of the dial possible. Smaller-model penalty is worth paying to keep the dial coherent.
- **Sources:** F5-TTS paper (Chen et al. 2024, arXiv:2410.06885); HuggingFace model card (`SWivid/F5-TTS`).
- **Workflow:** *TTS-choice research workflow*, 2026-05-31. Adversarially verified across 11 candidate models including F5R-TTS and 2026 newcomers.

## 2026-05-31 — Adapter = LoRA, not LoKr

- **Choice:** Plain LoRA, rank 16, alpha 16, target F5-TTS DiT attention modules (`to_q`, `to_k`, `to_v`, `to_out.0`).
- **Rejected:** LoKr, full fine-tune, prefix tuning, IA³.
- **Why this one:** Concept Sliders (Gandikota et al., ECCV 2024) demonstrated that LoRA's low-rank constraint *is* the disentanglement mechanism for graded continuous control — that's our exact problem. LoKr's "10× faster" claim is confounded with caching defaults and irrelevant on our cached-latent pipeline; LoKr also doesn't weight-merge, breaking the clean composition path. Prior internal proof: density-knob experiment shipped Spearman 0.90 monotonicity + cross-drift 0.10.
- **Sources:** Concept Sliders, Gandikota et al., ECCV 2024 (arXiv:2311.12092).
- **Workflow:** Internal density-knob pilot (2026-05); LoKr literature scan (2026-05).

## 2026-05-31 — Cocteau Twins / Liz Fraser framing dropped

- **Choice:** Naming = "The Un-Language Slider" → "Glossolalia Dial." Lineage = the wider tradition of wordless vocal music (Sigur Rós Hopelandic, Lisa Gerrard, Meredith Monk, scat, religious tongues).
- **Rejected:** "Liz Fraser Dial," "Cocteau Twins Generator," any single-artist framing.
- **Why this one:** Fraser's irreducible signature is sung soprano + multitracked harmonies + 4AD reverb-soaked production + non-English phoneme sourcing. Our dry monophonic spoken-TTS dial honors none of those. Naming her specifically is cargo-cult and would be instantly clocked as a bait-and-switch by anyone familiar with the records. The broader lineage holds; the single-artist claim doesn't.
- **Sources:** Cocteau Twins discography (4AD reissues 2003-2008); Reynolds 1991 *Blissed Out*; Hopelandic linguistic analyses on Sigur Rós official site.
- **Workflow:** *Cocteau Twins frame audit*, 2026-05-31. Five-property test; failed 4 of 5.

## 2026-05-31 — Off-the-Grid hard rule (no cloud APIs)

- **Choice:** All inference local. Zero cloud-API client packages in `requirements.txt`. F5-TTS, Whisper, Resemblyzer, Pedalboard, Demucs (if added) all run on Space hardware.
- **Rejected:** ElevenLabs as alternate backend, OpenAI Whisper API, hosted-LLM voice conditioning.
- **Why this one:** Earns the Off-the-Grid badge (verbatim rule: *"No cloud APIs. The whole thing runs on the model in front of you."*). Preserves the originality moat — the LoRA is the IP, not a cloud TTS. Allows reproducibility on any HF Space hardware. Removes a class of judging risk (cloud-call latency, key leakage, vendor outage at demo time).
- **Sources:** [huggingface.co/build-small-hackathon](https://huggingface.co/build-small-hackathon) badge text.
- **Workflow:** *Hackathon compliance audit* (2026-05-31) verified the rule; *ElevenLabs integration audit* (2026-06-01) confirmed no architecture survives that adds cloud TTS without forfeiting the badge.

## 2026-06-01 — ElevenLabs ruled out as backend

- **Choice:** No ElevenLabs integration anywhere in the inference path.
- **Rejected:** Eight integration architectures (replace F5-TTS, alternate backend, level-0-only, level-4-only, side-by-side comparator, voice-reference provider, user-supplied-key, pre-rendered).
- **Why this one:** Closed-weights, no LoRA path — the control token (the entire IP) cannot exist there. SSML `<phoneme>` tags only work on two legacy English-only models (`eleven_flash_v2`, `eleven_monolingual_v1`) and silently skip on every current model. Text front-end normalizes out-of-distribution input, which is exactly the input the dial depends on. The only "intelligibility-degrading" knob (Stability → Creative) is officially documented as "prone to hallucinations" — that's a stochastic failure mode, not a learned graded trajectory.
- **Sources:** [docs.elevenlabs.io / pronunciation-dictionaries](https://elevenlabs.io/docs/eleven-api/guides/how-to/text-to-speech/pronunciation-dictionaries); ElevenLabs Best Practices docs.
- **Workflow:** *ElevenLabs integration audit*, 2026-06-01.

## 2026-06-01 — Music-upload mode dropped

- **Choice:** No vocal-isolate-and-replace mode in v1. Shape (h) "pad grid" retained as a *conditional* post-launch bonus only.
- **Rejected:** Shape (a) full vocal replacement via Demucs + Whisper + F5-TTS + remix; Shape (d) sung glossolalia via DiffSinger / ACE-Step swap; Shapes (b)(c)(e)(f)(g) other music shapes.
- **Why this one:** F5-TTS speaks, doesn't sing — verified by training-corpus inspection (Emilia, speech-only). Spoken glossolalia over a tempo'd mix reads as voice-over, not vocal. The IP — dial=2 partial dissolution — is a close-listen artifact; an instrumental bed masks the exact evidence a judge needs to hear. Shape (a) specifically has three compounding failure modes (Demucs sibilant residue ~9dB SDR, WhisperX alignment drift 200–500ms+ on sung melisma, time-stretch formant damage that breaks the Resemblyzer ≥0.85 cosine gate) at 14h cost against ~10h budget. Shape (d) costs 60h, throws away the LoRA + 7,500-clip corpus + all validation gates, and lands us back in the saturated Lyria/Suno lane we paid 11 dead ideas to escape.
- **Sources:** F5-TTS paper §2 (training data); WhisperX paper (Bain et al. 2023, sung-vocal alignment limits); Demucs htdemucs benchmark (~9dB vocal SDR on MUSDB-HQ).
- **Workflow:** *Music-video-mode feasibility*, 2026-06-01.

## 2026-06-02 — Per-level learnable conditioning tokens (textual-inversion-style), not text-control

- **Choice:** A learnable `nn.Embedding(5, 512)` (one vector per level) is attached to F5-TTS's DiT and added to the post-conv text embedding inside `DiT.get_input_embed` (verified site: `src/f5_tts/model/backbones/dit.py` L284-312). The input text is just the sentence with no control marker. The LoRA adapter and the level embeddings co-train. At inference, `F5TTS.set_dial(level)` writes the per-batch level slot and busts the CFG text cache so the next infer picks up the new bias.
- **Rejected:** (a) Text-control token (`"sentence | tongues N"`) — the v1 attempt. Cell 7 verdict was PARTIAL, Spearman -0.70, voice cosine 0.97. Listened to Cell 8 wavs: F5-TTS read the entire input as text, so the model spoke the level word verbatim and the LoRA could not in 50 steps simultaneously silence the marker and corrupt the audio. (b) Separator-as-signal trick (special unicode that F5-TTS might ignore) — still leaves the LoRA with the two-jobs-in-one task. (c) Scaling text-control to 7,500 clips × 5 epochs — architectural problem, not under-training.
- **Why this one:** The conditioning signal is not text, so F5-TTS cannot read it aloud. The LoRA only has to learn one thing (the audio transformation per bias direction). The level embedding is six orders of magnitude smaller than the base model (5 × 512 = 2,560 params vs 336M); together with rank-16 LoRA on the attention projections (~2.9M params), the trainable surface stays at ~0.85% of the model.
- **Sources:** Textual Inversion, Gal et al. 2022, [arXiv:2208.01618](https://arxiv.org/abs/2208.01618) — learnable embedding vectors in the text-encoder space, trained while the base model is frozen. The pattern transfers cleanly from Stable Diffusion's text encoder to F5-TTS's text-embedding pipeline. F5-TTS architecture: source-verified at [`src/f5_tts/model/backbones/dit.py`](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py) L284-312 and `cfm.py` L294.
- **NOT cited:** Concept Sliders (Gandikota et al., ECCV 2024) — verified at [`sliders/trainscripts/textsliders/lora.py`](https://github.com/rohitgandikota/sliders) L249, it is a SCALAR-MODULATED LORA (`org_forward(x) + lora_up(lora_down(x)) * multiplier * scale`), not a per-level embedding table. The signed-direction LECO training assumes opposing prompt pairs we do not have. Using it as precedent for our setup would be inaccurate.
- **Workflow:** *F5-TTS per-level embedding architecture*, 2026-06-02. 14 of 18 claims survived adversarial verification.

## 2026-06-11 — Phoneme substitution kernel: PanPhon Boltzmann × dreamy_weight (replaces bigram-conditional draw)

- **Choice:** Replace the bigram-conditional class-masked draw in `scripts/corrupt_phonemes.py:corrupt()` with a Boltzmann distribution over the 39 ARPAbet phonemes:
  ```
  q(y | x, level) ∝ exp( -D[x,y] / T(level) ) * bias_weight(y)
  T(level) = 0.5 * exp(2.5 * p_level)   # T(0)=0.50, T(2)=1.75, T(4)=6.09
  ```
  `D` is the 39×39 PanPhon feature-edit-distance matrix, computed at LM-build time and stored in `data/phoneme_lm.npz`. `bias_weight` is the per-phoneme multiplier from the active preset (`dreamy` by default). At lv3-4 we also apply CV cluster simplification (collapse CC onset runs).
- **Rejected:** (a) Keep the bigram-conditional draw — produces phonetically arbitrary substitutions; v7 ear-test confirmed lv4 audio sounded like a different sentence, not a phonetic ghost. (b) Use Hirjee & Brown 2010 mondegreen confusion matrix — refuted by workflow `wtrmaydq7` (mondegreens are lexically motivated; would produce misheard English words at lv2, opposite of glossolalia). (c) Miller & Nicely 1955 consonant confusion matrix — only 16 consonants, noise-condition-specific, no vowel data.
- **What this gives us vs. v7:** at lv0 the lyric stays nearly identity (T=0.5 puts ~98% weight on self for most phonemes); at lv2 substitutions stay within Hamming distance 2-3 (P↔B, S↔SH, F↔V — near-minimal phonetic pairs); at lv4 the distribution opens and bias_weight steers the attractor toward sonorants/voiced fricatives/open back vowels. Local diverse-input test verified syllable count is preserved across cluster-heavy ("Strange tides crash"), vowel-heavy ("Oh how I love"), proper-name ("Eleanor walked through Aberdeen"), and long-form ("The endless rain falls softly...") inputs at all 5 levels.
- **What survives verification, what doesn't:**
  - **PanPhon distance values verified** by direct measurement against the installed library: P/B=1, S/SH=2, P/M=3, P/ZH=7, K/N=8, AA/P=11 — all match. Matrix range [0, 48].
  - **F5-TTS character-mode confirmed** by issue #362 (SWivid, owner): "current base models are using characters rather than phonemes" — pseudo-ASCII input is in-distribution.
  - **CV preference grounded** in [Link & Tomaschek 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10916350/) (95.7% CV in 7,486 German-L1 glossolalia syllables; top-6 syllables `[na, ra, la, ja, ba, da]`) AND in [Samarin 1973 *Language and Speech* 16:1](https://archive.org/details/tonguesofmenange0000sama/) ("preference for open syllables" as one of four formal features of glossolalia). This justifies CV cluster simplification at lv3-4.
  - **T(level) exponential schedule** is a design choice. No published precedent for this exact formula on TTS phoneme corruption; chosen by feel + smoke-tested on diverse inputs.
  - **`bias_weight` (dreamy preset) values are a design heuristic.** Earlier comments overclaimed Link & Tomaschek 2024 + Crystal 1995 as sources for the multipliers — neither paper publishes per-phoneme frequency tables, and Samarin's own onset data shows obstruent dominance (~79%) over sonorants (~19%), directly contradicting any "sonorant-heavy" provenance claim. The multipliers are hand-tuned by the author for a soft/sustained aesthetic. See correction below to the 2026-06-01 entry.
  - **PanPhon paper provenance is honest now:** the library exposes 24 features (3 added post the original COLING 2016 paper which had 21) and uses an external ARPAbet→IPA conversion (handled by us). Cited as a library, not as theoretical validation.
  - **Concept Sliders r=4 α=1 1000 iter recipe** (used to inform v6) targets SD-UNet, not DiT — informed prior, not validated transfer. v6 empirical refutation (0/9 audible) confirmed the architecture difference matters.
- **Sources (only what was verified against the primary):** PanPhon library (Mortensen et al., COLING 2016, [aclanthology.org/C16-1328](https://aclanthology.org/C16-1328/)) — used as software; Link & Tomaschek 2024 ([PMC10916350](https://pmc.ncbi.nlm.nih.gov/articles/PMC10916350/)) — CV figure and top-6 syllables, NOT per-phoneme multipliers; Samarin 1973 *Language and Speech* 16:1 — CV preference; F5-TTS issue [#362](https://github.com/SWivid/F5-TTS/issues/362) (SWivid) — character input mode. Anything previously cited that did not survive the audit has been removed from this entry.
- **Workflows:** Feature-distance recipe synthesis [`wtrmaydq7`](#) (5 probes, 2 refuted in adversarial verify); citation audit [`wtq6yi1zc`](#) (10 citations, 6 refuted/partial — drove the rewrite of the dreamy provenance, this entry, and the build_phoneme_lm.py docstring).

## 2026-06-11 — CORRECTION to 2026-06-01 "Phoneme LM rebias toward dreamy / wordless-vocal palette"

The 2026-06-01 entry implied that the dreamy preset's per-phoneme multipliers (M=1.6, L=1.5, AA=1.8, etc.) were derived from Link & Tomaschek 2024 + Pattison 1968 + Bryant & O'Connell 1971 + Crystal 1995. Citation audit `wtq6yi1zc` 2026-06-11 refuted that:

- Link & Tomaschek 2024 documents syllable *structure* (95.7% CV) and top-6 *syllable types* `[na, ra, la, ja, ba, da]` — but does NOT publish per-phoneme frequency tables. The multipliers are not derivable from this source.
- Samarin's onset-consonant frequency data (the paper *is* Samarin 1973, *Language and Speech* 16:1, not 1972 *Tongues of Men and Angels*) shows obstruents ~79% vs sonorants ~19% in onset position — *contradicting* the "sonorant-heavy" framing.
- Crystal 1995 "Phonaesthetically Speaking" favors *closed* mid-to-high vowels for aesthetic pleasantness, not open vowels — *contradicts* the "open vowel = dreamy" framing.

Honest provenance for the dreamy preset multipliers (and the `sigur-ros` and `fraser` presets):

> Hand-tuned design heuristic. The author chose to boost sonorants and open back vowels and suppress voiceless stops/affricates because that palette sounds soft and sustained to the author's ear. The values are not derived from a published frequency table. Samarin's CV preference grounds the syllable-shape preservation; the per-phoneme weighting beyond CV is aesthetic choice.

This honest provenance now lives in [scripts/build_phoneme_lm.py](scripts/build_phoneme_lm.py) lines 50-70 (the docstring for BIAS_PRESETS).

## 2026-06-11 — v7 LoRA recipe: keep v5's capacity, take v6's training intensity *(v6 rejected)*

- **v6 result (FAIL, 0/9 audible):** the Sliders-style capacity strip (r=16→4, α=16→1, drop `attn_norm.linear`) collapsed the dial WORSE than v5's 4/9. v5 lv0-vs-lv4 corr was 0.79 / 0.90 / 0.79 across three sentences; v6 was 0.94 / 0.91 / 0.86. The capacity reduction broke the LoRA's ability to learn the corruption mapping at all.
- **Why v6 was wrong:** Concept Sliders' low-rank constraint is calibrated for **contrastive paired-prompt** training where the dial axis is *given* by the c+/c- prompt difference. Our setup is **supervised regression with a learned scalar pathway** — the LoRA must (a) receive the scalar signal through frozen AdaLN, (b) figure out what acoustic transformation each level should apply, (c) execute it while preserving voice. That requires capacity. Deep-research finding #1 explicitly flagged the mechanism mismatch ("Mechanism mismatch means extrapolation should err toward MORE, not equivalent") but I applied the numeric recipe too cleanly.
- **v7 choice:** Restore v5's `r=16 α=16` and `attn_norm.linear` in `target_modules`, keep v6's 40 epochs + 2e-4 LR_LORA + 2e-3 LEVEL_LR. Keep per-level loss + grad-ratio diagnostics.
- **Rejected (for v7):** (a) bigger corpus alongside v5 arch — two-variable change, can't isolate which fixed it. Save for v8 if v7 fails. (b) further increase capacity (r=32) — no signal that v5's r=16 was the binding constraint.
- **Why v5's attn_norm.linear-in-LoRA worked empirically (no published precedent):** Without an adapter on `attn_norm.linear`, the path from `time_embed + level_embed → block-level scale/shift` runs through frozen weights trained for time-only conditioning. The LoRA on attention alone can only react to whatever the frozen AdaLN already produces from the (slightly-shifted) `t`. With `attn_norm.linear` in LoRA targets, the model can adapt how it consumes the level component of `t` per block. v5→v6 ablation is the only empirical evidence either way; F5-TTS PEFT (Kwon Interspeech 2025) does NOT cover this surface, so this is our novel finding.
- **Sources:** [Concept Sliders ECCV 2024](https://arxiv.org/abs/2311.12092) §3.2 — interference 0.10 low-rank vs 0.19 unconstrained (only applies under contrastive c+/c- training). [DiT paper, Peebles & Xie 2022](https://arxiv.org/abs/2212.09748). [ICLR 2025 DiT ablation](https://openreview.net/pdf?id=E4roJSM9RM). [F5-TTS PEFT, Kwon Interspeech 2025](https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf). v5 vs v6 spectral corr comparison is our own ablation, logged here.
- **Workflow:** Deep-research *wf_ccca9848*, 2026-06-11 (drove v6); v6 empirical refutation is the post-move audit that motivated v7.
- **Diagnostic signature for v7:** at step 100, `per_lv_loss` should differ across levels by >2%; `grad_ratio(lora/level)` should drop below 8. If still flat by step 200 → escalate to v8 (bigger corpus + same arch).

## 2026-06-11 — v6 LoRA recipe: Sliders-validated baseline *(REJECTED — see v7 above)*

- **Choice (now rejected):** EPOCHS 6→40 (→ ~2000 steps), LR_LORA 1e-4→2e-4, LEVEL_LR 5e-4→2e-3, LoRA rank 16→4, alpha 16→1 (α/r=0.25), drop `attn_norm.linear` from `target_modules` (keep `to_q/to_k/to_v/to_out.0`). Add per-level loss + grad-ratio diagnostics.
- **Rejected:** (a) Keep r=16 α=16 with just more epochs — the high-rank attention LoRA out-competes the zero-init LevelEmbed in the gradient race, eating signal that should recruit the dial. (b) Add AdaLN linear to LoRA targets — no published validation; risks letting attention bypass the level scalar entirely. (c) Regenerate corpus larger before retraining — Whisper transcripts confirm corpus is graded (lv0 "The moon I gaze..." vs lv4 "Drdingar dik zona shizaman"); the LoRA, not the corpus, is the failure point. (d) Pivot to XTTS-v2 or other base — premature.
- **Why this one:** Empirical signature from the v5 micro-spike — 300 steps with r=16 α=16 produced a LoRA that maps every dial position to near-clean English at hold-out time (Whisper transcripts: lv4 "caught in more L.I." vs lv0 "calm in the morning light", ~1 word differs). This is the canonical zero-init gate starvation failure mode: ICLR 2025 ablation establishes zero-init is the dominant DiT performance factor *because* the gate starts at zero and only learns if the surrounding pathways don't already minimize loss alone. Our attention LoRA at r=16 has 4× the capacity Concept Sliders validates and was solving the regression target by attention alone, leaving LevelEmbed at near-zero norm. The fix is structural: starve the attention LoRA's capacity (low-rank constraint per Sliders §3.2 — disentanglement is the constraint, not an efficiency knob) AND boost the level pathway's escape velocity (LEVEL_LR 10× LR_LORA — empirical, since the DiT paper's "we did not tune LRs" leaves no published norm to match).
- **Sources:** Concept Sliders (Gandikota et al. ECCV 2024, [arXiv:2311.12092](https://arxiv.org/abs/2311.12092)) — Table 3 ablation interference 0.10 low-rank vs 0.19 unconstrained; default config r=4 α=1, LR 2e-4 AdamW 1000 iter ([github.com/rohitgandikota/sliders](https://github.com/rohitgandikota/sliders) `train_lora.py` config.yaml). DiT paper (Peebles & Xie 2022, [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)) — flat 1e-4 LR with no tuning. ICLR 2025 (Improving DiT Pretraining: [openreview.net/pdf?id=E4roJSM9RM](https://openreview.net/pdf?id=E4roJSM9RM)) — zero-init is the most influential adaLN factor. F5-TTS PEFT precedent (Kwon et al. Interspeech 2025, [isca-archive.org/interspeech_2025/kwon25_interspeech.pdf](https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf)) — adapts conditioning surfaces but on text-embed + post-concat linear, NOT per-block AdaLN.
- **Workflow:** *Deep-research wf_ccca9848*, 2026-06-11 (104 agents, 3.3M tokens, 737 tool uses, 9 high/medium confidence findings adversarially verified). *F5-TTS input-mode workflow wf_8050a0dd*, 2026-06-10 — initial synthesis rejected: its premise that "corpus targets are acoustically near-identical" is refuted by Whisper transcripts.
- **Diagnostic signature for the next run:** at step 100, `lvl_norms` at s=1.0 should be in 0.5-3.0; `per_lv_loss` should differ across levels; `grad_ratio(lora/level)` should drop below 5 after warmup. If any of these fail by step 200, the dial pathway is still being starved and we escalate (lower rank further OR boost LEVEL_LR another 2×).

## 2026-06-01 — Phoneme LM rebias toward dreamy / wordless-vocal palette  *(in flight)*

- **Choice (proposed):** Bias the substitution distribution toward sonorants (L, M, N, R, W, J), open vowels (AY, OY, EE, OW, AH), voiced fricatives (V, Z, ZH); away from voiceless stops (K, P, T) and harsh affricates (CH).
- **Rejected:** Generic English bigram distribution from CMUdict over all 135K words.
- **Why this one:** The generic LM corrupts toward English-flavored nonsense — slight respellings of the input ("luhv ehz ah baltahfreel" for "love is a battlefield"). That's not teaching anyone anything; a singer reaches that with a typo. Empirical phonetic analyses of glossolalia (Goodman 1972 *Speaking in Tongues*, Samarin 1972 *Tongues of Men and Angels*) document that wordless-tongues distributions are *structured subsets* of the speaker's L1 phoneme inventory, biased toward sonorants and open vowels — the same phonotactic technique artists in the dreamy lineage (Fraser, Gerrard, Jónsi) use. Without this rebias, dial=2 risks bimodal collapse (English-or-mush), and the originality claim of "graded structured glossolalia" technically holds for English-native but doesn't deliver the *dreamy* English-native the framing implies.
- **Sources:** *(workflow-pending — will be filled in with verified citations from Goodman 1972, Samarin 1972, and any quantitative Hopelandic / Cocteau phonotactic studies)*.
- **Workflow:** *LM-rebias research*, 2026-06-01 (in progress).

---

## What goes in here

A decision belongs in this log if any of these are true:

- A judge could plausibly ask "why did you pick X over Y" at the demo.
- The choice is non-obvious from reading the code alone.
- The choice closes off a path that someone else (or future-us) might be tempted to reopen.
- The choice cites an external authority (paper, dataset spec, hackathon rule, vendor doc).

A decision does **not** belong here if it's mechanical (file rename, dependency bump, typo fix) or if the reasoning is already self-evident from one read of the code.
