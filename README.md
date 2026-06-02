---
title: Glossolalia Dial
emoji: 🍄
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.15.2
app_file: app_server.py
pinned: false
license: apache-2.0
short_description: A dial that grades any sentence from clean speech to wordless tongues, in the same voice
---

# Glossolalia Dial

Type a sentence. Pick a voice. Turn the dial.

At 0 you hear it spoken cleanly. At 4 you hear it as wordless glossolalia — phonotactically valid English-sounding tongues, in the same voice. The middle of the dial is where words half-dissolve.

Built on F5-TTS + a fine-tuned LoRA where one control token (`tongues zero..four`) maps to the dissolution level.
