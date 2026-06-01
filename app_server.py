"""The Un-Language Slider — v2 UI.

A custom HTML/JS frontend served via gradio.Server (FastAPI + Gradio's queuing/SSE/ZeroGPU
engine). The page renders a real circular knob widget the user drags; on change it calls the
@app.api() inference endpoint through the Gradio JS client, which keeps the queue + concurrency
controls Gradio provides while letting us ship a UI that's not the default gr.Blocks look.

This earns the Off-Brand badge.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server

from app import morph as gr_morph
from app import speak as gr_speak

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Server()


@app.api()
def speak(sentence: str, voice_id: str = "v1", level: int = 0,
          postfx_preset: str = "subtle", seed: int = 42) -> str:
    """Generate one take. Returns the wav file path the queue gives a URL for."""
    path, _readout = gr_speak(sentence, voice_id, int(level), postfx_preset, int(seed))
    return path or ""


@app.api()
def morph(sentence: str, voice_id: str = "v1",
          postfx_preset: str = "subtle", seed: int = 42) -> str:
    """One continuous take sweeping levels 0→4 with equal-power crossfade."""
    path, _readout = gr_morph(sentence, voice_id, postfx_preset, int(seed))
    return path or ""


@app.get("/")
async def homepage():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# Serve static assets (CSS, JS, fonts)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    app.launch(show_error=True)
