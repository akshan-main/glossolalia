// The Un-Language Slider — vanilla JS circular knob widget + Gradio JS client wiring.
// Drag the knob to rotate; snaps to integer levels 0..4. Triggers Speak on release.

import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

// ---------- knob widget ----------
class Knob {
  constructor(el, opts = {}) {
    this.el = el;
    this.min = opts.min ?? 0;
    this.max = opts.max ?? 4;
    this.value = opts.value ?? 0;
    this.onCommit = opts.onCommit || (() => {});
    this.valueEl = el.querySelector(".knob-value");
    this._dragging = false;
    this._setupEvents();
    this._render();
  }
  setValue(v, commit = false) {
    v = Math.max(this.min, Math.min(this.max, Math.round(v)));
    if (v !== this.value) {
      this.value = v;
      this._render();
    }
    if (commit) this.onCommit(this.value);
  }
  _render() {
    const t = (this.value - this.min) / (this.max - this.min); // 0..1
    const angle = -135 + t * 270;                                // -135..+135 deg
    const fill = t * 270;                                        // 0..270 deg
    this.el.style.setProperty("--angle", angle + "deg");
    this.el.style.setProperty("--fill", fill + "deg");
    this.el.setAttribute("aria-valuenow", String(this.value));
    if (this.valueEl) this.valueEl.textContent = String(this.value);
  }
  _angleFromPoint(x, y) {
    const r = this.el.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    let deg = (Math.atan2(y - cy, x - cx) * 180) / Math.PI + 90;
    if (deg > 180) deg -= 360;
    return Math.max(-135, Math.min(135, deg));
  }
  _setValueFromPoint(x, y) {
    const deg = this._angleFromPoint(x, y);
    const t = (deg + 135) / 270;
    const raw = this.min + t * (this.max - this.min);
    this.setValue(raw);
  }
  _setupEvents() {
    const onDown = (e) => {
      this._dragging = true;
      e.preventDefault();
      this.el.setPointerCapture?.(e.pointerId);
      this._setValueFromPoint(e.clientX, e.clientY);
    };
    const onMove = (e) => {
      if (!this._dragging) return;
      this._setValueFromPoint(e.clientX, e.clientY);
    };
    const onUp = (e) => {
      if (!this._dragging) return;
      this._dragging = false;
      this.el.releasePointerCapture?.(e.pointerId);
      this.onCommit(this.value);
    };
    this.el.addEventListener("pointerdown", onDown);
    this.el.addEventListener("pointermove", onMove);
    this.el.addEventListener("pointerup", onUp);
    this.el.addEventListener("pointercancel", onUp);
    // keyboard a11y
    this.el.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowDown") this.setValue(this.value - 1, true);
      if (e.key === "ArrowRight" || e.key === "ArrowUp")  this.setValue(this.value + 1, true);
    });
    // scroll to nudge
    this.el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.setValue(this.value + (e.deltaY < 0 ? 1 : -1), true);
    }, { passive: false });
  }
}

// ---------- voice pill selection ----------
const voicePills = document.querySelectorAll("#voicePills .pill");
let currentVoice = "v1";
voicePills.forEach((p) => {
  p.addEventListener("click", () => {
    voicePills.forEach((q) => q.classList.remove("active"));
    p.classList.add("active");
    currentVoice = p.dataset.voice;
  });
});

// ---------- Gradio JS client connection ----------
const statusEl = document.getElementById("status");
const audioEl = document.getElementById("audio");
const setStatus = (s) => { if (statusEl) statusEl.textContent = s; };

let client = null;
async function ensureClient() {
  if (!client) {
    setStatus("connecting…");
    client = await Client.connect(window.location.origin);
    setStatus("connected");
  }
  return client;
}

function gradioFileToUrl(item) {
  // Gradio JS client returns either a string filepath or { path, url } etc.
  if (!item) return null;
  if (typeof item === "string") {
    if (item.startsWith("http")) return item;
    return `/file=${item}`;
  }
  if (item.url) return item.url;
  if (item.path) return `/file=${item.path}`;
  return null;
}

async function callApi(name, args) {
  const c = await ensureClient();
  setStatus(`${name}…`);
  const t0 = performance.now();
  const r = await c.predict(`/${name}`, args);
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  setStatus(`${name} ok · ${dt}s`);
  return Array.isArray(r.data) ? r.data[0] : r.data;
}

async function speakOnce(level) {
  const sentence = document.getElementById("sentence").value;
  const postfx = document.getElementById("postfx").value;
  const seed = parseInt(document.getElementById("seed").value, 10) || 42;
  if (!sentence.trim()) { setStatus("type a sentence first"); return; }
  const out = await callApi("speak", {
    sentence, voice_id: currentVoice, level: Number(level),
    postfx_preset: postfx, seed,
  });
  const url = gradioFileToUrl(out);
  if (url) { audioEl.src = url; audioEl.play().catch(() => {}); }
}

async function morphSweep() {
  const sentence = document.getElementById("sentence").value;
  const postfx = document.getElementById("postfx").value;
  const seed = parseInt(document.getElementById("seed").value, 10) || 42;
  if (!sentence.trim()) { setStatus("type a sentence first"); return; }
  const out = await callApi("morph", {
    sentence, voice_id: currentVoice, postfx_preset: postfx, seed,
  });
  const url = gradioFileToUrl(out);
  if (url) { audioEl.src = url; audioEl.play().catch(() => {}); }
}

// ---------- bootstrap ----------
const knob = new Knob(document.getElementById("knob"), {
  min: 0, max: 4, value: 0,
  onCommit: (v) => { speakOnce(v).catch((e) => setStatus("error: " + e.message)); },
});

document.getElementById("speakBtn").addEventListener("click",
  () => speakOnce(knob.value).catch((e) => setStatus("error: " + e.message)));
document.getElementById("morphBtn").addEventListener("click",
  () => morphSweep().catch((e) => setStatus("error: " + e.message)));
