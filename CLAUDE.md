# Build Small Hackathon — project context

This directory is for the user's entry to HuggingFace + Gradio's **Build Small Hackathon**.

## Project: Genre Engine

**One-line premise:** turn any sound or any text into a banger in any of 20+ subgenres spanning multiple genre families, all from one fine-tuned model.

**Three interaction modes, one model:**

1. **Audio → Subgenre** — upload or hum audio, pick subgenre, get the same melodic ideas reshaped. Uses ACE-Step audio2audio + timbre conditioning.
2. **Lyrics → Banger** — type lyrics, pick subgenre, get a full short song with vocals + accompaniment.
3. **Live mode** — continuous mic capture, auto-segment on silence, transformations stream past as the user hums or sings.

**Single model, many LoRAs:** all three modes run on the same ACE-Step 1.5 XL base. Subgenre identity is loaded as a LoRA at inference time. The LoRAs are the project's IP.

## Product framing (not demo framing)

The optimization function is **"would the internet actually use this,"** not *"would the judges chuckle at a 30-second clip."* Implications:

- Output quality must be reliable across diverse user inputs — not just cherry-picked demo cases
- Subgenre breadth matters: 20+ LoRAs across genre families so any visitor finds what they want first try
- The Space itself is the product — pure Gradio on HF Spaces, no backend, no DB, no auth
- Persistence lives in the user's browser (bookmarked URL) and on their disk (downloads), not on our side
- Discoverability comes from HF's trending / community surfaces and the Collection page — that's the only growth loop

## Subgenre coverage (target ≥20)

Across genre families:
- **Electronic:** drum and bass, house, techno, vaporwave, hyperpop, ambient, footwork, phonk
- **Hip-hop:** boom bap, trap, lofi
- **Rock / pop:** indie, synthpop, post-rock
- **World / other:** afrobeats, reggaeton, bossa nova
- **R&B / soul:** neo-soul, R&B
- **Jazz / classical:** lounge jazz, cinematic / orchestral

Source data: MTG-Jamendo (HF: `rkstgr/mtg-jamendo`) — 55k+ CC-licensed tracks across 195 tags.

## Architecture (verified end-to-end 2026-05-10)

- **Base model:** **ACE-Step 1.5 XL** (April 2026 release, 4B params). Current open-source SOTA — SongEval 8.12 / AudioBox 7.76, highest scores ever recorded. Surpasses Suno v5 across all 11 evaluation dimensions.
- **Architecture:** hybrid — autoregressive LM (Composer Agent planner) + Diffusion Transformer (acoustic renderer). On HF Hub at `ACE-Step/Ace-Step1.5` with Diffusers integration.
- **Audio2audio:** native, reference audio + strength + separate timbre conditioning ([B, 64, T] latent).
- **Fine-tuning:** LoKr adapters (10× faster than vanilla LoRA, ~5 min per subgenre on a decent GPU). Hyperparameters: rank 16, alpha 32, dropout 0.1, learning rate 1e-4, target modules `[q_proj, v_proj, k_proj, out_proj]`, batch size 1 with gradient accumulation 8.
- **Captioning:** `ACE-Step/acestep-captioner` for auto-labeling MTG-Jamendo clips, augmented with explicit subgenre tags.
- **Audio length:** 10–600s per training clip, 48kHz stereo internally compressed via ACE-Step's 1D VAE to 64-dim latent at 25Hz.
- **VRAM:** 16GB minimum, 20GB+ recommended; `woct0rdho/ACE-Step` fork available for <10GB if needed.
- **Inference time:** ~2s per song on A100 (ZeroGPU class), ~10s on RTX 3090. Cold start ~30–60s for 4B model load.

## LoRA hosting structure

**20 separate HF repos, grouped by an HF Collection.**

- Each LoRA gets its own repo: `<user>/genre-engine-dnb`, `<user>/genre-engine-house`, `<user>/genre-engine-vaporwave`, etc.
- One HF Collection (*"Genre Engine LoRAs"*) groups them all with a unified narrative
- Each repo's README links back to the Collection and the Space
- App reads the Collection's API at startup to discover available LoRAs — adding the 21st LoRA needs no code change
- Stronger HF discoverability than a single repo (20 trending chances, 20 model card pages, 20 download stat counters)

## Track + approach (locked)

- **Track:** 🍄 Thousand Token Wood
- **Approach:** fine-tune ACE-Step 1.5 XL with LoKr adapters per subgenre across genre families; three interaction modes share the same fine-tuned base
- **Idea:** Genre Engine (committed 2026-05-10)

## Hard constraints

- ≤ 32B params (ACE-Step 1.5 XL is 4B, well under)
- Pure Gradio app on HF Space under build-small-hackathon org — no external services, no DB, no auth
- `gradio.Server` (v6.10.0+) for the UI; ZeroGPU for inference compute (confirmed via hackathon)
- Submission: Space link + demo video + social post

## Timeline (absolute dates)

- 2026-05-07 — Registration opened
- 2026-05-10 — Project locked, prep starts
- 2026-05-27 — Registration closes
- 2026-05-29 — Hack window begins
- 2026-06-08 — Submissions close

Pre-window (~3 weeks): MTG-Jamendo curation across all target subgenres, captioning runs, LoKr training across 20+ subgenres, baseline Space scaffold with all three modes wired.

Hack window (~11 days): UI polish, additional subgenres if time, quality validation across user inputs, Space optimization, Collection assembly, demo video, social post.

## UX within pure-Gradio-on-Spaces constraints

What we ship inside the Gradio app:
- Three tabs (Audio / Lyrics / Live)
- Subgenre dropdown shared across tabs (populated from the HF Collection)
- `gr.Audio` output with built-in HTML5 player + download button
- `gr.State` for in-tab session history during one browsing session (transient, no claim of persistence)
- Stable Space URL users bookmark in their browser

What we don't try to build:
- User accounts, server-side history, favorites that persist across sessions, community boards — all need backends we don't have

## Bonus quests targeted (5 of 6)

- 🎯 **Well-Tuned** — 20+ subgenre LoKr adapters published as separate HF repos + grouped in a Collection
- 🎨 **Off-Brand** — custom UI via `gr.Server`
- 📡 **Sharing is Caring** — published LoRAs (open weights), curation scripts, sample outputs, Collection page
- 📓 **Field Notes** — blog post on multi-subgenre ACE-Step LoKr fine-tuning
- 🔌 **Off the Grid** — fully local inference (model + audio never leave the device)

**Skipped:** 🦙 Llama Champion. ACE-Step is hybrid LM + diffusion, not llama.cpp-native.

## Demo video angle (side effect, not optimization target)

Three beats: hum a melody → drum and bass; type lyrics → hyperpop song with vocals; switch to live mode → continuous transformations stream while user hums freely. Demonstrates breadth of subgenres, breadth of input modes, and that the same model handles all of it.

## Honest residual risks

- LoKr training quality varies per subgenre. Popular tags have plenty of MTG-Jamendo data; niche tags may need extra curation.
- Cold-start latency on Space load is real (4B model). Persistent loading via ZeroGPU mitigates after first request per session.
- Audio2audio quality on ACE-Step has known issues per the repo's own GitHub issues — distortion in some cases. Validate per subgenre before publishing.
- Live mode latency: not true real-time (2–10s per phrase) — communicates as "stream of transformations," not as zero-latency live performance.
- 20+ LoRAs is real curation work. Half a day per subgenre for dataset cleanup is realistic; the training itself is fast (~5 min per LoKr).

## Working notes for Claude

- All three modes use the same model + LoRAs. Don't reintroduce a separate model for any mode.
- LoRAs live in 20 separate repos grouped by an HF Collection. App reads the Collection at startup.
- This is a product, not a demo. Optimize for quality and breadth, not for one cherry-picked demo moment.
- Persist project context here, not in `memory/project_*.md` — see the feedback memory on persistence location.
