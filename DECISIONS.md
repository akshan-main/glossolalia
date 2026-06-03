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
