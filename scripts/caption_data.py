"""Generate per-clip captions for downloaded audio using Qwen2-Audio-Instruct."""

import sys
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SUBGENRES, DATA_ROOT


CAPTIONER_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
SR = 16000
MAX_NEW_TOKENS = 96


def load_captioner():
    processor = AutoProcessor.from_pretrained(CAPTIONER_MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        CAPTIONER_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return processor, model


def caption_one(audio_path, subgenre, processor, model):
    audio, _ = librosa.load(audio_path, sr=SR)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {
                    "type": "text",
                    "text": (
                        f"This is a {subgenre.name} track. In one sentence under 30 words, "
                        f"describe its instruments, rhythm, and mood. Do not mention BPM, key, "
                        f"or duration. Do not use the words 'audio' or 'track'."
                    ),
                },
            ],
        },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    out_ids = ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return response


def main():
    processor, model = load_captioner()
    for sg in SUBGENRES:
        audio_dir = DATA_ROOT / sg.repo_slug / "audio"
        if not audio_dir.exists():
            print(f"skip {sg.name}: no audio dir at {audio_dir}")
            continue
        wavs = sorted(audio_dir.glob("*.wav"))
        print(f"→ {sg.name}: {len(wavs)} clips")
        for i, wav in enumerate(wavs, 1):
            cap_path = wav.with_suffix(".caption.txt")
            if cap_path.exists():
                continue
            try:
                raw = caption_one(wav, sg, processor, model)
            except Exception as e:
                print(f"  [{i}/{len(wavs)}] failed: {wav.name} ({e})")
                continue
            cap_path.write_text(f"{sg.name}, {raw}")
            if i % 10 == 0:
                print(f"  [{i}/{len(wavs)}] captioned")


if __name__ == "__main__":
    main()
