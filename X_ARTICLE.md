A dial that makes any sentence fall apart, in your own voice

This is my submission to the HuggingFace and Gradio Build Small hackathon (Thousand Token Wood track). It is also my first time fine-tuning an audio model, and most of what follows is what that taught me.

I trained a single dial that dissolves any sentence you type, in any voice you pick. At 0 it speaks the sentence cleanly. Turn it up and the words come apart, and you choose how: into invented words that sound like a language but aren't, or into other real words you mishear.

Try it: [Space link]

The hard part nobody asks about is the middle. Anyone can make "clean speech" and "noise." The question is whether dial 2 is a real half-dissolved state, recognizably the same sentence but slipping, or whether training just collapses to two presets with a crossfade. The whole thing only ships if the middle is real.

Two modes, one dial

Same sentence, "she sells seashells by the seashore," same voice:

- Tongues, top of the dial: "schausel says luzor." Invented, pronounceable, meaningless.
- Ghost, top of the dial: "she sills seagulls by the thistle." Every word swapped for a different real word that sounds close.

Tongues is speaking-in-tongues. Ghost is what your brain does when you mishear a lyric.

Why this isn't a reverb plugin

The obvious objection is that this is a DSP effect. It isn't, and the reason is the interesting part. Reverb, formant-shift, vocoders, granular synthesis all act uniformly on audio that already exists. This has to read a sentence you invented two seconds ago and erode specific words into different but plausible ones, while holding the syllable count, the stress, and the voice fixed across the whole slide. You cannot reach that with effects. It needs lexical resynthesis from typed input, and only a model does that.

Why F5-TTS, and why most TTS models would break this

This is the choice the whole project rests on. Most modern TTS has a language-model front end. That front end is trained to be helpful: hand it something out of distribution and it quietly corrects it back toward real, well-formed English. For a normal product that is exactly what you want. For this it is fatal, because the input the dial depends on is deliberately malformed. An LLM-fronted TTS would un-corrupt my training targets and collapse the five levels into two.

F5-TTS has no LLM front end. It is a flow-matching model that takes characters and pronounces what you give it, faithfully, including spellings that are not words. That faithfulness is the entire reason the middle of the dial can exist. It is a smaller, less "smart" model than the alternatives, and that is the point. I traded raw quality for a model that does not argue with me.

The trick: manufacture the dataset

There is no dataset of "sentences gradually dissolving into nonsense," so I made one.

Take 3000 plain English sentences. For each, generate five corrupted versions at rising substitution rates. The corruption works on phonemes: convert the sentence to its sounds, then swap a growing fraction of them for same-class sounds, keeping syllable count and stress intact. Have the base TTS read each corrupted version out loud. Pair each clip with the original clean sentence and its level. That gives 30,000 (audio, clean sentence, level) tuples.

Then fine-tune so that, given the clean sentence and the dial, the model reproduces the level-N audio. The model never sees the corrupted text. It only ever sees the clean sentence and a number, and learns the mapping from the dial alone. That inversion is the training signal: the corruption builds the target, the model learns to reach it from clean input.

Fine-tuning it, and the bug that almost killed it

The dial is two pieces trained together: a rank-16 LoRA on the model's attention, and a small network that turns the dial number into a vector added into the model's timestep embedding, the same pathway it already uses for the diffusion step. The scalar network starts at zero so it does no harm before it learns anything.

My first runs failed in a way I did not understand for a while. Every dial position produced near-clean speech. Dial 4 and dial 0 differed by about one word. The corruption was in the data, the loss was going down, and the dial did nothing.

The cause, once I found it, is a clean lesson. The LoRA and the zero-initialized dial network are in a race for the same gradient. The LoRA has plenty of capacity, so it learns to minimize the training loss using attention alone, and it gets there before the dial network ever leaves zero. The loss is satisfied, so the dial is never recruited. The fix was not more data or more steps. It was to give the dial pathway a much higher learning rate than the LoRA, ten times, so it escapes zero and starts contributing before the LoRA closes off the loss. After that the levels separated. That gradient-race failure is the single most useful thing I learned doing this.

Ghost, and why the reranking matters

Ghost does not use the LoRA. It runs live. For each word it searches a pronunciation dictionary for real words that sound close, using a phonetic feature distance, not just spelling. The problem is that "close-sounding" alone surfaces garbage: rare surnames, abbreviations, words no one says. Early versions turned "seashells" into things like "selz."

Two things fix it. A frequency filter throws out anything that is not common English. Then a small language model (DistilGPT-2) does a beam search over the candidate words and scores whole sequences by how likely they are as a sentence, so the output reads like a plausible mishearing instead of word salad. The dial controls how many words get swapped and how far the swap is allowed to go. Without the reranking, Ghost is noise. With it, "the river was wide and calm" becomes "the rougher was wide and come." That step is the whole mode.

Naming the two things honestly

Only one of these is glossolalia, and saying so is the point. Real speaking-in-tongues is not random. It reuses the speaker's native phonemes (Samarin 1972, Goodman 1972) and prefers open consonant-vowel syllables. Link and Tomaschek (2024) measured 95.7% CV structure across 7,486 glossolalic syllables. The Tongues corruption keeps the English phoneme inventory, preserves syllable shape, and leans toward open syllables, so it sits inside that structure. What I do not claim: the exact per-phoneme weighting is a taste choice, not something a paper supports, and a citation pass caught me overstating it.

Ghost is not glossolalia at all. It is mondegreen, pareidolia, the misheard-lyric effect, and I label it that way in the app. I did not put one word on two different things.

Did the middle work

Turn the dial and listen. At 0 the model speaks the sentence as typed. At 2 it is half-dissolved, recognizably the same sentence but slipping, in the same voice. At 4 it is wordless. The exact words it lands on change from run to run, because it is sampling a generative model, not replaying a recording, but the graded slide from clean to gone is stable. That slide, the middle actually existing instead of a hard cut between clean and noise, was the only thing I cared about.

What it can't do

It is English only, because the phonetics and the dictionary are English. The dial is trained on five points, so it is graded, not infinitely continuous. Ghost depends on the sentence: a three-word input has fewer distinct states than a long one. Tongues at the very top can still produce the occasional hard cluster. None of these break the toy, but they are real edges.

Everything runs locally on the Space. F5-TTS for the voice (about 336M parameters), a rank-16 LoRA plus the scalar network for the dial, a small language model for Ghost, Whisper for transcription. No cloud APIs, nothing over 32B. You can clone your own voice from a short clip, mix a backing track, and the dial works on whatever you type.

The lineage is the older one: wordless vocal music, Sigur Rós, Lisa Gerrard, Meredith Monk, scat, and on the Ghost side, every lyric anyone has ever misheard. The contribution is putting one continuous, learned control on that axis.

Code, model, and the full write-up in the replies.
