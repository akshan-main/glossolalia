# Glossolalia Dial

Glossolalia Dial is a text-to-speech toy with one knob. Type a sentence and it speaks it. Turn the knob up and the words come apart into wordless babble that still sounds like a language, in the same voice the whole way. The trick is the middle, where the sentence is half-dissolved instead of cutting straight from speech to noise.

Two modes:
- **Tongues** is the words slurring into made-up pseudo-words. A LoRA and a learned dial trained into F5-TTS.
- **Ghost** swaps every word for a real one that sounds close (seashells becomes seagulls), the misheard-lyric thing. No model, runs live.

Live: https://huggingface.co/spaces/build-small-hackathon/glossolalia

look into this (write-up)[https://x.com/frutigeraerosol/status/2066667649338417367] for a more detailed description for better understanding

Built for the HuggingFace and Gradio Build Small hackathon, Thousand Token Wood track, which wants small things that are genuinely original and a bit weird. The sound it chases is the wordless-vocals lineage, Cocteau Twins and Liz Fraser especially, where the voice is an instrument and the words stop mattering.

## Glossary

- **Glossolalia** is speaking in tongues: fluent-sounding vocalizing that follows a language's sound rules but means nothing. That's Tongues mode.
- **Mondegreen** is a misheard lyric, where your brain hears real words that were never there ("'scuse me while I kiss this guy"). That's Ghost mode.

## Run it

```
pip install -r requirements.txt
python app.py
```

It grabs F5-TTS and the dial LoRA on first run. CPU works, a GPU is much faster. 
Base model as in https://huggingface.co/SWivid/F5-TTS

## Train the dial

The pipeline lives in `modal/app.py` and runs on Modal:

```
pip install modal
modal token new
modal run modal/app.py
```

There's no dataset of sentences falling apart into nonsense, so it builds one: corrupt each sentence's phonemes at five rising rates, have base F5-TTS read each, then train the dial to reproduce that slide from the clean sentence alone. The model never sees the corrupted text. Ghost mode trains nothing, it just searches CMUdict for close-sounding words live and reranks them with DistilGPT-2.

## Writeup

[How it was built, and why it needs a finetune and not a reverb plugin.](https://x.com/frutigeraerosol/status/2066667649338417367)

- Checkpoint: https://huggingface.co/akshan-main/glossolalia-dial-lora
- Dataset: https://huggingface.co/datasets/akshan-main/glossolalia-inputs
- Space: https://huggingface.co/spaces/build-small-hackathon/glossolalia
