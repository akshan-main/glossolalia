"""tune-fine — turn any sound or any text into a banger in 12 subgenres (expanding to 20).

Three modes share one fine-tuned ACE-Step 1.5 XL backbone:
  - Audio: melody-conditioned audio2audio
  - Lyrics: text-to-song with vocals
  - Live: continuous mic capture with auto-segmentation
"""

import gradio as gr

from config import SUBGENRES, SUBGENRE_BY_NAME

SUBGENRE_NAMES = [s.name for s in SUBGENRES]


def describe_subgenre(name):
    s = SUBGENRE_BY_NAME.get(name)
    if not s:
        return ""
    return (
        f"<div class='subgenre-card'>"
        f"<span class='family-tag'>{s.family}</span>"
        f"<div class='subgenre-desc'>{s.desc}</div>"
        f"</div>"
    )


# ----- Inference stubs (replace when adapters are trained) -----

def transform_audio(audio_path, subgenre, history):
    if audio_path is None:
        return None, history
    new_history = history + [("audio", subgenre)]
    return None, new_history


def generate_from_lyrics(lyrics, subgenre, history):
    if not lyrics or not lyrics.strip():
        return None, history
    new_history = history + [("lyrics", subgenre)]
    return None, new_history


def live_capture(audio_chunk, subgenre, history):
    if audio_chunk is None:
        return None, history
    new_history = history + [("live", subgenre)]
    return None, new_history


def render_history(history):
    if not history:
        return "<div class='history-empty'>No generations yet this session.</div>"
    items = []
    for i, (mode, subgenre) in enumerate(reversed(history[-10:]), 1):
        mode_label = {"audio": "Audio", "lyrics": "Lyrics", "live": "Live"}[mode]
        items.append(
            f"<div class='history-item'>"
            f"<span class='history-num'>{i:02d}</span>"
            f"<span class='history-mode'>{mode_label}</span>"
            f"<span class='history-arrow'>→</span>"
            f"<span class='history-subgenre'>{subgenre}</span>"
            f"</div>"
        )
    return "<div class='history-list'>" + "".join(items) + "</div>"


# ----- Visual identity -----

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-card: #14141c;
    --bg-card-hover: #1a1a26;
    --border: #2a2a3a;
    --border-bright: #3a3a52;
    --text-primary: #f5f5fa;
    --text-secondary: #8a8a9a;
    --text-muted: #5a5a6a;
    --accent: #ff3d92;
    --accent-glow: rgba(255, 61, 146, 0.35);
    --accent-soft: rgba(255, 61, 146, 0.12);
}

body, .gradio-container, .dark {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 56px 24px 80px 24px !important;
}

/* HERO */
#hero {
    text-align: center;
    margin-bottom: 56px;
    position: relative;
}
#hero::before {
    content: '';
    position: absolute;
    top: -40px;
    left: 50%;
    transform: translateX(-50%);
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, var(--accent-soft) 0%, transparent 70%);
    z-index: -1;
    pointer-events: none;
}
#hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 88px;
    font-weight: 700;
    letter-spacing: -0.045em;
    line-height: 0.95;
    margin: 0;
    background: linear-gradient(180deg, #ffffff 0%, #888899 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
#hero .tagline {
    font-size: 18px;
    color: var(--text-secondary);
    margin-top: 16px;
    font-weight: 400;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
}
#hero .accent-line {
    display: inline-block;
    width: 48px;
    height: 3px;
    background: var(--accent);
    margin: 28px 0 0 0;
    border-radius: 2px;
    box-shadow: 0 0 20px var(--accent-glow);
}

/* TABS */
.tabs > .tab-nav, .tab-nav {
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
    margin-bottom: 32px !important;
}
.tabs > .tab-nav button, .tab-nav button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 14px 22px !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.tabs > .tab-nav button.selected, .tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* SUBGENRE CARD */
.subgenre-card {
    padding: 18px 22px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    transition: border-color 0.2s;
}
.subgenre-card:hover {
    border-color: var(--border-bright);
}
.family-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    background: var(--accent-soft);
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 8px;
}
.subgenre-desc {
    font-size: 14px;
    color: var(--text-primary);
    line-height: 1.5;
    margin-top: 8px;
}

/* TAB HELP */
.tab-help {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 24px;
    line-height: 1.6;
}

/* BUTTONS */
button.primary, button[variant="primary"], .primary > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    padding: 14px 36px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    box-shadow: 0 0 32px var(--accent-glow) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.01em !important;
}
button.primary:hover, button[variant="primary"]:hover, .primary > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 48px var(--accent-glow) !important;
}

/* INPUT FIELDS */
input, textarea, select, .gr-input, .gr-text-input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.15s !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--accent) !important;
    outline: none !important;
}

/* DROPDOWN */
.gr-dropdown, [role="listbox"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* LABELS */
label {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* AUDIO COMPONENT */
.gr-audio, audio {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

/* HISTORY */
.history-empty {
    color: var(--text-muted);
    font-style: italic;
    text-align: center;
    padding: 40px 0;
}
.history-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.history-item {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 18px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 14px;
    transition: border-color 0.15s;
}
.history-item:hover {
    border-color: var(--border-bright);
}
.history-num {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    font-size: 12px;
}
.history-mode {
    color: var(--accent);
    font-weight: 500;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
}
.history-arrow {
    color: var(--text-muted);
}
.history-subgenre {
    color: var(--text-primary);
}

/* FOOTER */
#footer {
    margin-top: 80px;
    padding-top: 32px;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.7;
}
#footer a {
    color: var(--text-secondary);
    text-decoration: none;
    border-bottom: 1px solid var(--border);
    transition: color 0.15s;
}
#footer a:hover {
    color: var(--accent);
}

/* MOBILE */
@media (max-width: 720px) {
    #hero h1 { font-size: 56px !important; }
    .gradio-container { padding: 32px 16px !important; }
}
"""


with gr.Blocks(
    title="tune-fine",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_900",
    ),
    css=CUSTOM_CSS,
) as demo:
    history_state = gr.State([])

    gr.HTML(
        """
        <div id="hero">
            <h1>TUNE-FINE</h1>
            <p class="tagline">Turn any sound or any text into a banger.<br>
            One fine-tuned model, three ways in.</p>
            <div class="accent-line"></div>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------- AUDIO MODE ----------------
        with gr.Tab("Audio"):
            gr.HTML(
                "<div class='tab-help'>Hum, whistle, or upload a clip. Pick a subgenre. "
                "The model reshapes your melody into that sonic identity.</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    audio_subgenre = gr.Dropdown(
                        SUBGENRE_NAMES,
                        label="Subgenre",
                        value=SUBGENRE_NAMES[0],
                    )
                with gr.Column(scale=3):
                    audio_subgenre_desc = gr.HTML(describe_subgenre(SUBGENRE_NAMES[0]))
            audio_subgenre.change(describe_subgenre, audio_subgenre, audio_subgenre_desc)

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your audio",
            )
            audio_btn = gr.Button("Transform", variant="primary", elem_classes="primary")
            audio_output = gr.Audio(label="Result", type="filepath")

        # ---------------- LYRICS MODE ----------------
        with gr.Tab("Lyrics"):
            gr.HTML(
                "<div class='tab-help'>Type any words. Pick a subgenre. "
                "Get a full short song with vocals and accompaniment.</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    lyrics_subgenre = gr.Dropdown(
                        SUBGENRE_NAMES,
                        label="Subgenre",
                        value=SUBGENRE_NAMES[0],
                    )
                with gr.Column(scale=3):
                    lyrics_subgenre_desc = gr.HTML(describe_subgenre(SUBGENRE_NAMES[0]))
            lyrics_subgenre.change(describe_subgenre, lyrics_subgenre, lyrics_subgenre_desc)

            lyrics_input = gr.Textbox(
                lines=6,
                placeholder="Type lyrics here. Short lines work best.\n\nsun is rising / over the city / ready to begin / a brand new day",
                label="Lyrics",
            )
            lyrics_btn = gr.Button("Generate", variant="primary", elem_classes="primary")
            lyrics_output = gr.Audio(label="Result", type="filepath")

        # ---------------- LIVE MODE ----------------
        with gr.Tab("Live"):
            gr.HTML(
                "<div class='tab-help'>Continuous mic capture with auto-segmentation. "
                "Hum or sing freely; transformations stream past as you go.</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    live_subgenre = gr.Dropdown(
                        SUBGENRE_NAMES,
                        label="Subgenre",
                        value=SUBGENRE_NAMES[0],
                    )
                with gr.Column(scale=3):
                    live_subgenre_desc = gr.HTML(describe_subgenre(SUBGENRE_NAMES[0]))
            live_subgenre.change(describe_subgenre, live_subgenre, live_subgenre_desc)

            live_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Mic — start humming",
            )
            live_output = gr.Audio(label="Live transformations", type="filepath", streaming=True)

        # ---------------- SESSION HISTORY ----------------
        with gr.Tab("Session"):
            gr.HTML(
                "<div class='tab-help'>Generations from this browser tab. "
                "Cleared on refresh — download anything you want to keep.</div>"
            )
            history_display = gr.HTML(render_history([]))

    # Wire up handlers
    audio_btn.click(
        transform_audio,
        inputs=[audio_input, audio_subgenre, history_state],
        outputs=[audio_output, history_state],
    ).then(render_history, history_state, history_display)

    lyrics_btn.click(
        generate_from_lyrics,
        inputs=[lyrics_input, lyrics_subgenre, history_state],
        outputs=[lyrics_output, history_state],
    ).then(render_history, history_state, history_display)

    live_input.stream(
        live_capture,
        inputs=[live_input, live_subgenre, history_state],
        outputs=[live_output, history_state],
    ).then(render_history, history_state, history_display)

    gr.HTML(
        """
        <div id="footer">
            Built on a fine-tuned <a href="https://huggingface.co/ACE-Step/Ace-Step1.5">ACE-Step 1.5 XL</a>.
            Subgenre LoRAs in the <a href="#">tune-fine Collection</a>.<br>
            Open weights · runs in your browser · your audio stays local.
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(show_api=False)
