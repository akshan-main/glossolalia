# Glossolalia Dial: Field Notes

*A dial that turns any sentence, in any voice, into wordless tongues, graded clean to glossolalia on one fine-tuned control token. Built for HuggingFace + Gradio's Build Small Hackathon, Thousand Token Wood track.*

---

## The question

If you fine-tune a small TTS so a single token in the prompt grades the *intelligibility* of what comes out, clean speech at 0, phonotactically-valid English-native glossolalia at 4, does the *middle* of the dial actually exist? Is there a coherent dial=2 where a sentence half-dissolves while staying in the same voice and the same prosodic shape, or does training collapse to two presets with a crossfade?

That question is the whole project. The toy ships only if the middle is real.

## Where the idea came from (and what almost killed it)

Eleven music-domain instantiations died on the way here, most of them killed by Google Lyria, Suno, Udio, and Magenta RealTime, in some order. The meta-lesson, eventually: **music generation is the most saturated, best-funded corner of generative AI right now**, and originality there is structurally capped against four billion-dollar incumbents.

Open/local *speech* has different incumbents (Hume in the cloud for emotion sliders, ElevenLabs for prosody) and one genuinely uncontested niche: **single-axis continuous attribute knobs on a fine-tuned open TTS, validated for monotonicity and independence**. Research workflows confirmed no shipped product hits the four-spec intersection of *typed-input + graded + voice-locked + dissolves-to-glossolalia*. The nearest misses, ProtoDisent-TTS / DARS for clinical dysarthria, F5-TTS dysarthric clone for pathology reconstruction, ACE-Step flow-edit for discrete lyric replacement, all target different problems.

## Why DSP can't do this

The honest version of "why does this need a fine-tune", a question we kept asking ourselves because the wrong answer would mean shipping a plugin in a tuxedo. Signal-shaping (formant shift, reverb, time-stretch, vocoder, granular synth) acts *uniformly* on the audio that's already there. Our toy needs to:

- Read a typed sentence the user *just* invented,
- Generate audio that says it cleanly at one end and progressively *erodes specific words into different but plausible phonemes* at the other,
- Hold the syllable count, stress pattern, and voice identity locked across the trajectory.

You can't reach that with effects. The lexical resynthesis is the load-bearing thing, and only a model can do it.

A linguistically-curious counter-claim worth addressing: glossolalia, per Goodman 1972 and Samarin 1972, is well-known to reuse the speaker's native phonotactics. So could you just record someone speaking spontaneously, slap a reverb on it, and call it done? No, that gives you the *endpoint* as a separate take, never a controllable *trajectory* tied to a specific user-typed sentence. The Goodman finding actually supports the fine-tune: a structured native-phonotactic distribution is exactly what a graded conditioning token can learn cleanly and DSP cannot synthesize.

## The corrupt-text-as-training-signal trick

There is no pre-existing dataset of "sentences gradually dissolving into nonsense." Cocteau Twins albums are copyrighted, and even if they weren't, they're not labeled `level 0 / level 1 / level 2`. So we manufactured the dataset.

The trick has three steps and one twist:

1. **Take 3000 ordinary English sentences** (500 LibriTTS-R transcripts + 2500 from public-domain Project Gutenberg classics, length-filtered to 5-15 words).
2. **For each sentence, generate 5 corrupted text variants** at substitution probabilities `p ∈ {0, 0.25, 0.5, 0.75, 1.0}`. The corruption is per-phoneme: g2p the original sentence into ARPAbet, then for each phoneme flip a Bernoulli at `p_level` and, if it flips, substitute a phoneme of the *same class* (vowel→vowel, consonant→consonant) drawn from a CMUdict bigram distribution conditioned on the previous phoneme. Stress markers and syllable count are preserved. The result is phonotactically valid English-flavored nonsense at the high end.
3. **Use the base F5-TTS (no LoRA yet) as a puppet**: feed it each corrupted text in each of 2 voice references (a public-domain LibriVox soprano-mezzo and a LibriTTS-R bass, chosen for max acoustic diversity at 1.3 octaves apart), let it synthesize. Pair each synthesized clip with its *original* sentence + level + voice. That's 30,000 (audio, original-sentence, level, voice) tuples, the training set.
4. **The twist**: fine-tune the LoRA, plus a small scalar-to-vector conditioner injected into F5-TTS's time embedding, so that *given the original sentence + the dial level + the voice reference* it produces audio matching the level-N synthesized version. The model never sees the corrupted text at training time; it sees the *original sentence* labeled with a level. At inference, the user types a sentence and turns the dial, the model has learned the mapping.

This is synthetic-data bootstrapping. It's why a single person with a Colab notebook can build this in 10 hours instead of needing a labeled corpus that doesn't exist.

## Is the corruption *actually* glossolalia, or just noise?

This is the question a linguist in the room would ask, so here is the honest answer with the citations, separating what is grounded in the phonetics literature from what is an aesthetic choice.

Real glossolalia is not random sound. The two findings that matter:

- **It reuses the speaker's native phonotactics.** A speaker of English produces English-shaped syllables, not arbitrary ones (Samarin 1972, *Tongues of Men and Angels*; Goodman 1972, *Speaking in Tongues*). Our corruption keeps the g2p'd English phoneme inventory and only swaps within it, so the output stays English-native by construction.
- **It strongly prefers open CV syllables.** Link & Tomaschek 2024 ([PMC10916350](https://pmc.ncbi.nlm.nih.gov/articles/PMC10916350/)) measured **95.7% CV structure across 7,486 glossolalic syllables**, with the six most frequent syllable types `[na, ra, la, ja, ba, da]`. Samarin 1973 (*Language and Speech* 16:1) independently lists "preference for open syllables" as one of four formal features of glossolalia. Our corruption preserves syllable count and stress and simplifies toward CV at the high dial levels, which is exactly this shape.

So the *structural* claim, native phonotactics, CV preference, stress/syllable skeleton preserved, is grounded in primary sources, not asserted.

What is **not** claimed from the literature, stated plainly so the demo doesn't overreach: the specific per-phoneme weighting (boosting L/M/N and open back vowels for a soft, sustained palette) is a **hand-tuned aesthetic choice**, not derived from a published frequency table. In fact Samarin's own onset-consonant data shows obstruents (~79%) dominating sonorants (~19%), which *contradicts* a "sonorants are more glossolalic" story. We tune toward sonorants because it sounds dreamy, not because the corpus says glossolalia is sonorant-heavy. An earlier draft overclaimed this; a citation audit caught it and it's corrected here.

And the honest frame for the whole thing: real glossolalia is *spontaneously generated* by a speaker. We do something different, we **corrupt a typed sentence progressively** to manufacture a *trajectory* (clean → tongues) on one control token. That graded trajectory is the contribution; it is not a claim to reproduce how a person speaks in tongues. The Ghost mode (real-word mishearings via mondegreen) is a separate aesthetic and is *not* glossolalia at all, it's pareidolia, and we label it that way.

## The spike: does the middle of the dial work?

It does. Turn the dial and the output slides from the sentence as typed at 0, to a half-dissolved version at 2 that is recognizably the same sentence but slipping, to wordless at 4, all in the same voice. The exact words it lands on change from run to run because it is sampling a generative model, but the graded slide is stable, and that is the claim. The text preview at high levels also looks rougher than the audio sounds, because the LoRA smooths the corruption into pronounceable output. The middle of the dial is a real perceptual partial dissolution, not a bimodal jump from clean to mush.

The gates were locked in advance, written into the validation script before any GPU spend:

- **Whisper-WER should rise across levels 0..4.** WER goes up because the output is less intelligible, which is the dial doing its job. Hallucination-guarded: when Whisper's avg-logprob falls below threshold, WER floors at 1.0 so glossolalia does not get a spuriously *low* WER because Whisper invented coherent words from noise.
- **Resemblyzer cosine vs the level-0 reference clip should stay close across all levels.** Voice preserved.
- **Hand-listen the dial=2 wavs**: they must be a *partial* dissolution, recognizably the same sentence, blurred, in the same voice. Not bimodal collapse to clean-or-mush.

If gate three fails, the originality drops from a defensible 5 to a 3 ("two presets with a crossfade"), because that collapse is something a DSP chain + two TTS calls *could* fake. The middle is the IP.

## The toy

The app is a Gradio Space with a hand-built circular knob in place of a default slider: a CSS conic-gradient rim, pointer-drag and keyboard control in JS, an arc indicator, all injected as custom HTML and wired to a hidden Gradio slider. Type a sentence, turn the knob, hit *play this dial* for one position or *dissolve* for the whole 0 to 4 sweep crossfaded into one take. A live text preview shows what the voice will say at the current dial. Voice cloning, a background-music mixer, and per-word hand-tuning sit in collapsible panels.

A small post-FX bus (Pedalboard reverb + chorus + octave layer) sits between the model and the speaker, toggleable as `dry / subtle / lush / cathedral`, and re-renders live from the cached dry take so changing it does not re-run the model. It is not a sonic claim, it just keeps the dry TTS from undermining itself. The dial does the meaning-making.

What it isn't: a Liz Fraser clone, a Cocteau Twins generator, an ElevenLabs replacement. Fraser's signature is sung soprano + multitracked harmonies + 4AD reverb-soaked production + non-English phoneme sourcing, none of which a dry monophonic open-TTS dial honors, and we didn't pretend otherwise. The lineage we *do* sit in is the older one: Sigur Rós' Hopelandic, Lisa Gerrard, Meredith Monk, scat, religious glossolalia. The continuous-graded-control axis is the contribution.

## What we learned

- The hardest part of an originality bet at a hackathon judged by a world-class field isn't the model work. It's killing your own ideas honestly enough, often enough, to get to one that isn't already shipped by someone with a thousand engineers.
- Workflow-based adversarial verification beats vibes. The Cocteau Twins frame *felt* right; the verification dossier killed it cleanly with citations, and the renamed toy is more honest *and* more interesting.
- A measured property + a clean label + a graded control token + a learning model + a validation harness with hard gates is a generic enough recipe that we used it twice (once in music, once in TTS). The transferable artifact is the recipe.

---

*Code: [GitHub](https://github.com/akshan-main/glossolalia) · Space: [HF](https://huggingface.co/spaces/akshan-main/glossolalia) · LoRA: [HF model](https://huggingface.co/akshan-main/glossolalia-dial-lora)*
