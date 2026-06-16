"""Modal app: train v5 glossolalia dial (AdaLN-side level conditioning + LoRA) on A100-40GB.

Replaces Colab kernel-reset thrash with a clean reproducible runner. Same architecture
the workflow synthesized + verified, with the dtype handling fixed (no PEFT
modules_to_save quirks, level_embed attached directly to DiT, optimized as a separate
param group).

Run:
    modal run modal/app.py                  # full pipeline (~50 min, ~$2.50)
    modal run modal/app.py::sweep_and_verify  # re-sweep with existing adapter

Architecture: see DECISIONS.md "v5 AdaLN-side level conditioning".
"""
import modal

APP_NAME = "glossolalia-dial-v5"
REPO_URL = "https://github.com/akshan-main/glossolalia.git"
REPO_DIR = "/root/repo"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.5.0",
        "torchaudio==2.5.0",
        # transformers >= 4.55 uses torch.float8_e8m0fnu (needs torch >= 2.6).
        # Pin to a version that works with torch 2.5.
        "transformers==4.46.3",
        "f5-tts",
        "peft>=0.14.0,<0.16.0",  # peft 0.16+ may also require newer torch
        # NO torchao: torchao>=0.16 needs torch>=2.11 (uses torch.int1).
        # peft works without it as long as we don't use quantized models.
        "soundfile",
        "librosa",
        "scipy",
        "numpy",
        "huggingface_hub",
        "safetensors",
        "g2p_en",
        "panphon",
        "jiwer",
        "openai-whisper",
        "resemblyzer",
        "cached_path",
    )
    .run_commands(
        # Pre-download NLTK assets, caches into image
        "python -c \"import nltk; "
        "nltk.download('averaged_perceptron_tagger_eng', quiet=True); "
        "nltk.download('averaged_perceptron_tagger', quiet=True); "
        "nltk.download('cmudict', quiet=True)\""
    )
    # NOTE: no git clone here. Docker would cache it and lose later commits.
    # Each function clones fresh at runtime.
)

vol = modal.Volume.from_name("glossolalia-v5", create_if_missing=True)
app = modal.App(APP_NAME, image=image)

# Lightweight image for FLUX-schnell plate generation (no f5_tts / heavy audio stack)
# Lets diffusers and transformers resolve compatible versions on their own (avoid pinning
# transformers to 4.46.3 which lacks Dinov2WithRegistersConfig that recent diffusers needs).
flux_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",       # FLUX-schnell prefers torch>=2.6
        "diffusers==0.31.0",
        "transformers>=4.45,<4.55",
        "accelerate>=0.30",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "Pillow",
    )
)


@app.function(gpu="A10G", image=flux_image, volumes={"/vol": vol}, timeout=60 * 30)
def gen_plates():
    """Generate dreamy photographic background plates with SDXL-Turbo (non-gated, free).
    Saves PNGs to /vol/plates/. Plates are bundled into the Space.
    SDXL-Turbo: ~1-4 steps, 512x512 native, fast on A10G."""
    import os
    import torch
    from diffusers import AutoPipelineForText2Image

    os.makedirs("/vol/plates", exist_ok=True)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe.enable_attention_slicing()

    prompts = {
        "plate_sun":
            "extreme close-up of a hot red-orange spherical sun-orb glowing in deep night, "
            "soft photographic out-of-focus, dreamy 4AD record sleeve aesthetic, "
            "warm rim light, deep violet background, hand-tinted analog photograph",
        "plate_lights":
            "long-exposure photograph of warm golden Christmas lights swooping in motion "
            "across deep purple velvet background, ribbon of light, "
            "dreamy ethereal, photographic, soft focus, no people, no text",
        "plate_streaks":
            "abstract photograph of golden ribbon light trails painted in the night, "
            "deep violet to magenta gradient sky, swooping curves of warm yellow light, "
            "analog film grain, 4AD record sleeve, dreamy",
        "plate_velvet":
            "photograph of deep velvet purple night with subtle painterly texture, "
            "tiny floating warm light specks, ethereal, dreamy ambient backdrop, no figures",
    }
    saved = []
    for name, prompt in prompts.items():
        out_p = f"/vol/plates/{name}.png"
        if os.path.exists(out_p):
            print(f"  skip {name} (exists)")
            saved.append(out_p)
            continue
        gen = torch.Generator("cuda").manual_seed(hash(name) & 0xFFFF)
        img = pipe(
            prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=512,
            width=1024,
            generator=gen,
        ).images[0]
        img.save(out_p, format="PNG", optimize=True)
        print(f"  saved {name}")
        saved.append(out_p)
    vol.commit()
    print(f"plates: {saved}")


@app.local_entrypoint()
def make_plates():
    gen_plates.remote()
    print(">> plates on volume at /vol/plates")


def _setup_repo():
    """Clone (or pull) the repo at runtime. Called at the top of every function."""
    import os, subprocess
    if not os.path.isdir(REPO_DIR):
        subprocess.check_call(["git", "clone", "-q", REPO_URL, REPO_DIR])
    else:
        subprocess.check_call(["git", "-C", REPO_DIR, "pull", "-q", "origin", "main"])
    import sys
    sys.path.insert(0, REPO_DIR)


def _stage_inputs():
    """Pull sentences + voices + phoneme LM from HF to local repo dir + /vol/voices."""
    import os, shutil
    _setup_repo()
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id="akshan-main/glossolalia-inputs",
        repo_type="dataset",
        local_dir=f"{REPO_DIR}/data_pull",
    )
    os.makedirs(f"{REPO_DIR}/data/voices", exist_ok=True)
    for f in ("sentences.txt", "phoneme_lm.npz", "cmudict.dict"):
        s = os.path.join(p, f)
        if os.path.exists(s):
            shutil.copy(s, f"{REPO_DIR}/data/{f}")
    vd = os.path.join(p, "voices")
    if os.path.isdir(vd):
        for f in os.listdir(vd):
            shutil.copy(os.path.join(vd, f), f"{REPO_DIR}/data/voices/{f}")
    os.makedirs("/vol/voices", exist_ok=True)
    for f in os.listdir(f"{REPO_DIR}/data/voices"):
        shutil.copy(f"{REPO_DIR}/data/voices/{f}", f"/vol/voices/{f}")


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60 * 12,
)
def pull_and_generate():
    """Single-worker corpus gen (sequential). Pulls data, generates all 30k clips
    on one A100, then builds the F5-TTS dataset. ~8h. See generate_chunk +
    main_parallel for the 4-way fanout version."""
    import os, shutil, subprocess
    _stage_inputs()
    print("  sentences:", sum(1 for _ in open(f"{REPO_DIR}/data/sentences.txt")))

    if os.path.isdir("/vol/coherence_ds") and os.path.exists("/vol/coherence_ds/metadata.csv"):
        print("  corpus already on volume, skipping generation")
        vol.commit()
        return

    # Write clips DIRECTLY to /vol/coherence so partial progress persists across
    # container restarts and the --resume flag actually works after a cancel.
    os.makedirs("/vol/coherence", exist_ok=True)
    os.chdir(REPO_DIR)

    import threading, time
    stop_commit = threading.Event()
    def periodic_commit():
        while not stop_commit.wait(300):  # commit every 5 minutes
            try:
                vol.commit()
                print(f"  [commit] volume snapshot at {time.strftime('%H:%M:%S')}", flush=True)
            except Exception as e:
                print(f"  [commit] failed: {e}", flush=True)
    commit_thread = threading.Thread(target=periodic_commit, daemon=True)
    commit_thread.start()

    try:
        subprocess.check_call([
            "python", "scripts/generate_coherence_data.py",
            "--sentences", "data/sentences.txt",
            "--voice", "v2:data/voices/v2.wav:data/voices/v2.txt",
            "--voice", "v3:data/voices/v3.wav:data/voices/v3.txt",
            "--lm", "data/phoneme_lm.npz",
            "--out", "/vol/coherence",
            "--max-sentences", "3000",
            "--levels", "5",
            "--input-mode", "pseudo",
            "--resume",
        ])
    finally:
        stop_commit.set()
        commit_thread.join(timeout=10)
        vol.commit()

    subprocess.check_call([
        "python", "scripts/build_coherence_dataset.py",
        "--data", "/vol/coherence",
        "--out", "/vol/coherence_ds",
    ])
    vol.commit()
    print("  corpus saved to volume")


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60 * 6,
)
def generate_chunk(shard_idx: int, shard_count: int = 4):
    """One worker in the parallel corpus-gen fanout. Each worker takes a contiguous
    slice of sentences (shard_idx of shard_count), writes clips with a non-overlapping
    global index range to /vol/coherence, and emits its own manifest_shard{N}.jsonl.
    The build step (run separately) globs all shard manifests and merges them."""
    import os, shutil, subprocess, threading, time
    _stage_inputs()
    print(f"  sentences:", sum(1 for _ in open(f"{REPO_DIR}/data/sentences.txt")),
          f"shard={shard_idx}/{shard_count}", flush=True)

    os.makedirs("/vol/coherence", exist_ok=True)
    os.chdir(REPO_DIR)

    stop_commit = threading.Event()
    def periodic_commit():
        while not stop_commit.wait(300):
            try:
                vol.commit()
                print(f"  [shard{shard_idx} commit] {time.strftime('%H:%M:%S')}", flush=True)
            except Exception as e:
                print(f"  [shard{shard_idx} commit] failed: {e}", flush=True)
    commit_thread = threading.Thread(target=periodic_commit, daemon=True)
    commit_thread.start()

    try:
        subprocess.check_call([
            "python", "scripts/generate_coherence_data.py",
            "--sentences", "data/sentences.txt",
            "--voice", "v2:data/voices/v2.wav:data/voices/v2.txt",
            "--voice", "v3:data/voices/v3.wav:data/voices/v3.txt",
            "--lm", "data/phoneme_lm.npz",
            "--out", "/vol/coherence",
            "--max-sentences", "3000",
            "--levels", "5",
            "--input-mode", "pseudo",
            "--resume",
            "--shard-idx", str(shard_idx),
            "--shard-count", str(shard_count),
        ])
    finally:
        stop_commit.set()
        commit_thread.join(timeout=10)
        vol.commit()
    print(f"  shard {shard_idx} done")


@app.function(
    gpu="A10G",
    volumes={"/vol": vol},
    timeout=60 * 30,
)
def build_dataset_after_fanout():
    """Run AFTER all 4 generate_chunk workers complete. Globs all manifest_shard*.jsonl
    files, merges into the F5-TTS finetune dataset format at /vol/coherence_ds."""
    import os, subprocess
    _setup_repo()
    os.chdir(REPO_DIR)
    subprocess.check_call([
        "python", "scripts/build_coherence_dataset.py",
        "--data", "/vol/coherence",
        "--out", "/vol/coherence_ds",
    ])
    vol.commit()
    print("  merged dataset saved to /vol/coherence_ds")


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60 * 10,
)
def train_v5():
    """Train v5: LoRA on attention + attn_norm.linear, plus LevelEmbed MLP added to t."""
    import os, re, json, time, types
    from pathlib import Path
    _setup_repo()

    import torch
    import torch.nn as nn
    import torchaudio
    import soundfile as sf
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from f5_tts.api import F5TTS
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    NUM_LEVELS = 5
    DIM = 1024
    BATCH = 4
    # Sliders' ~1000-2000-step regime. Full-scale 30000 clips / batch 4 = 7500 steps per epoch.
    # 2 epochs lands at 15000 steps, well above Sliders' 1000 floor (2-voice corpus, see DECISIONS.md).
    EPOCHS = 2
    LR_LORA = 2e-4     # Sliders config.yaml: AdamW 2e-4 bf16
    LEVEL_LR = 2e-3    # 10x LR_LORA: empirical, give zero-init MLP a chance to escape origin
    SEED = 42
    # Allow override via env vars so train_v9_ghost can reuse this function for the Ghost corpus.
    OUT_DIR = os.environ.get("V5_OUT_DIR", "/vol/v5_adapter")
    CORPUS_DIR = os.environ.get("V5_CORPUS_DIR", "/vol/coherence_ds")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda"

    f5 = F5TTS(model="F5TTS_v1_Base")
    cfm = f5.ema_model
    dit = cfm.transformer
    assert dit.dim == DIM
    model_dtype = next(dit.parameters()).dtype
    print(f"  base DiT dtype: {model_dtype}")

    for p in cfm.parameters():
        p.requires_grad_(False)

    class LevelEmbed(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, s):
            return self.net(s)

    # CRITICAL: level_embed stays in fp32 to escape bf16 mantissa-epsilon kill.
    # bf16 epsilon ~7.8e-3 vs |t| ~ O(1-10): zero-init outputs (steps 0-30) round away.
    # See audit C12 in workflow w49ptme7h.
    level_embed = LevelEmbed(DIM).to(device=device)  # fp32
    for p in level_embed.parameters():
        p.requires_grad_(True)
    dit.add_module("level_embed", level_embed)
    dit._current_scalars = None

    def _patched_forward(self, x, cond, text, time, mask=None,
                         drop_audio_cond=False, drop_text=False,
                         cfg_infer=False, cache=False):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        sc = getattr(self, "_current_scalars", None)
        if sc is not None:
            orig_dtype = t.dtype
            s = sc.to(device=t.device, dtype=torch.float32).view(-1, 1)
            if s.shape[0] == 1 and t.shape[0] > 1:
                s = s.expand(t.shape[0], 1)
            # Force fp32 math so the small zero-init level shift doesn't get rounded away in bf16.
            with torch.amp.autocast("cuda", enabled=False):
                t = (t.float() + self.level_embed(s)).to(orig_dtype)
        seq_len = x.shape[1]
        if cfg_infer:
            x_c = self.get_input_embed(x, cond, text, drop_audio_cond=False,
                                       drop_text=False, cache=cache, audio_mask=mask)
            x_u = self.get_input_embed(x, cond, text, drop_audio_cond=True,
                                       drop_text=True, cache=cache, audio_mask=mask)
            x = torch.cat((x_c, x_u), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(x, cond, text,
                                     drop_audio_cond=drop_audio_cond,
                                     drop_text=drop_text, cache=cache, audio_mask=mask)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)
        if residual is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = self.norm_out(x, t)
        return self.proj_out(x)

    dit.forward = types.MethodType(_patched_forward, dit)

    # v7 LoRA config: revert v6's capacity strip (0/9 audible vs v5's 4/9), empirically
    # the attn_norm.linear surface was load-bearing on supervised-regression scalar
    # control, contrary to Sliders' contrastive-pair recipe. Keep v5's r=16 alpha=16 +
    # attn_norm.linear targets, with v6's 40-epoch / 2e-4 LR_LORA / 2e-3 LEVEL_LR.
    lora_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "attn_norm.linear"],
    )
    cfm.transformer = get_peft_model(dit, lora_cfg).to(device)
    base_dit = cfm.transformer.base_model.model
    assert hasattr(base_dit, "level_embed"), "level_embed lost on PEFT wrap"
    assert base_dit is dit, "PEFT deep-copied the base model; patched forward lost"
    assert base_dit.forward.__func__ is _patched_forward, "patched forward overridden"

    LV_RE = re.compile(r"_lv(\d)\.wav$")
    TONGUES_RE = re.compile(r"\s*\|\s*tongues\s+\w+\s*$", re.IGNORECASE)

    class CoherenceDS(Dataset):
        def __init__(self, csv_path, max_dur=15.0):
            self.rows = []
            base = Path(csv_path).parent
            with open(csv_path) as f:
                f.readline()
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    rel_path, text = line.split("|", 1)
                    wav_p = base / rel_path.strip()
                    try:
                        info = sf.info(str(wav_p))
                        if info.frames / info.samplerate > max_dur:
                            continue
                    except Exception:
                        continue
                    m = LV_RE.search(rel_path)
                    lvl = int(m.group(1)) if m else 0
                    text = TONGUES_RE.sub("", text).strip()
                    self.rows.append((str(wav_p), text, lvl))
            from torchaudio.transforms import MelSpectrogram
            self.mel = MelSpectrogram(
                sample_rate=24000, n_fft=1024, hop_length=256,
                win_length=1024, n_mels=100, power=1, center=True, normalized=False,
            )

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            wav_p, text, lvl = self.rows[i]
            y, sr = torchaudio.load(wav_p)
            if sr != 24000:
                y = torchaudio.functional.resample(y, sr, 24000)
            if y.shape[0] > 1:
                y = y.mean(0, keepdim=True)
            m = self.mel(y).squeeze(0).clamp(min=1e-5).log()
            return {
                "mel": m.transpose(0, 1),
                "text": text,
                "mel_len": m.shape[-1],
                "scalar": float(lvl) / float(NUM_LEVELS - 1),
            }

    def collate(batch):
        mels = pad_sequence([b["mel"] for b in batch], batch_first=True, padding_value=0.0)
        lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
        return {
            "mel": mels,
            "mel_lens": lens,
            "text": [b["text"] for b in batch],
            "scalars": torch.tensor([b["scalar"] for b in batch], dtype=torch.float32),
        }

    ds = CoherenceDS(f"{CORPUS_DIR}/metadata.csv")
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate,
                    num_workers=2, drop_last=True)
    print(f"  dataset rows: {len(ds)}, batches/epoch: {len(dl)}")
    assert len(ds) > 0

    # Identity-based exclusion (substring filter is brittle if PEFT renames anything)
    level_param_ids = {id(p) for p in level_embed.parameters()}
    lora_params = [p for _, p in cfm.named_parameters()
                   if p.requires_grad and id(p) not in level_param_ids]
    level_params = list(level_embed.parameters())
    for p in level_params:
        p.requires_grad_(True)
    print(f"  LoRA params : {sum(p.numel() for p in lora_params):,}")
    print(f"  Level params: {sum(p.numel() for p in level_params):,}")

    # eps=1e-8 since level params now fp32 (was 1e-6 for bf16 safety)
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LR_LORA},
            {"params": level_params, "lr": LEVEL_LR},
        ],
        betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8,
    )

    def clear_text_cache(d):
        d.text_cond = None
        d.text_uncond = None

    # Per-level loss accumulator (deep-research diag: if flat across levels => dial dead)
    per_level_loss = {i: [] for i in range(NUM_LEVELS)}

    cfm.train()
    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        for batch in dl:
            mel = batch["mel"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            scalars = batch["scalars"].to(device)
            base_dit._current_scalars = scalars
            clear_text_cache(base_dit)
            lr_scale = min(1.0, (step + 1) / 50.0)
            opt.param_groups[0]["lr"] = LR_LORA * lr_scale
            opt.param_groups[1]["lr"] = LEVEL_LR * lr_scale
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = cfm(mel, text=batch["text"], lens=mel_lens)
            loss.backward()
            # Diag: grad-norm RATIO before clipping. If LoRA grad >> level grad after
            # warmup, the attention pathway is starving the zero-init gate (deep-research
            # finding #3: zero-init gates lose gradient races to competing pathways).
            with torch.no_grad():
                gn_lora = torch.norm(torch.stack([p.grad.norm() for p in lora_params if p.grad is not None]))
                gn_level = torch.norm(torch.stack([p.grad.norm() for p in level_params if p.grad is not None]))
            torch.nn.utils.clip_grad_norm_(lora_params + level_params, 1.0)
            opt.step()
            # Track per-level loss bucket
            for s_val, _ in zip(scalars.tolist(), [None] * len(scalars)):
                lvl = int(round(s_val * (NUM_LEVELS - 1)))
                per_level_loss[lvl].append(float(loss.item()))
            base_dit._current_scalars = None
            clear_text_cache(base_dit)
            # Print at steps 1, 5, then every 25, log per-level loss + grad ratio
            if step in (1, 5) or step % 25 == 0:
                with torch.no_grad():
                    probe = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]],
                                         device=device, dtype=torch.float32)
                    norms = level_embed(probe).norm(dim=-1).tolist()
                ratio = float(gn_lora / max(gn_level, torch.tensor(1e-9))) if gn_level > 0 else float('inf')
                # Per-level recent mean (last 20 examples of each level)
                pl_mean = {}
                for lv in range(NUM_LEVELS):
                    recent = per_level_loss[lv][-20:]
                    pl_mean[lv] = sum(recent) / max(len(recent), 1)
                pl_str = "|".join(f"{pl_mean[lv]:.3f}" for lv in range(NUM_LEVELS))
                print(f"  ep{ep} step{step:4d} loss={loss.item():.4f} "
                      f"lvl_norms={[f'{x:.3f}' for x in norms]} "
                      f"per_lv_loss=[{pl_str}] "
                      f"grad_ratio(lora/lvl)={ratio:.2f} "
                      f"elapsed={time.time() - t0:.0f}s")
            step += 1

    cfm.transformer.save_pretrained(OUT_DIR, safe_serialization=True)
    le_state = {f"net.{k}": v.detach().cpu().contiguous()
                for k, v in level_embed.net.state_dict().items()}
    save_file(le_state, str(Path(OUT_DIR) / "level_to_time.safetensors"))
    (Path(OUT_DIR) / "level_to_time.json").write_text(json.dumps({
        "format": "v5_adaln",
        "dim": int(DIM),
        "num_levels": int(NUM_LEVELS),
    }, indent=2))
    vol.commit()
    print(f"\nsaved adapter + level_to_time -> {OUT_DIR}")
    print("  files:", sorted(os.listdir(OUT_DIR)))


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60 * 6,
)
def sweep_and_verify():
    """Sweep dial 0..4 on 3 holdout sentences, spectral diff between dial=0 and dial=4."""
    import os, numpy as np, soundfile as sf
    from scipy.signal import stft
    _setup_repo()

    import patches  # installs F5TTS.load_lora + set_dial
    from f5_tts.api import F5TTS

    OUT = os.environ.get("SWEEP_ADAPTER_DIR", "/vol/v5_adapter")
    SWEEP = os.environ.get("SWEEP_OUT_DIR", "/vol/sweep")
    os.makedirs(SWEEP, exist_ok=True)

    tts = F5TTS(model="F5TTS_v1_Base")
    tts.load_lora(OUT)
    print(f"  dial_mode: {getattr(tts, '_dial_mode', None)}")

    sentences = [
        "the river was wide and calm in the morning light",
        "she opened the old book and began to read aloud",
        "a quiet wind moved through the empty stone courtyard",
    ]
    voice_wav = "/vol/voices/v1.wav"
    voice_txt = "/vol/voices/v1.txt"
    voice_ref_text = open(voice_txt).read().strip()

    for si, sentence in enumerate(sentences):
        for lv in range(5):
            tts.set_dial(lv)
            out_p = os.path.join(SWEEP, f"v1_lv{lv}_s{si}.wav")
            tts.infer(
                ref_file=voice_wav, ref_text=voice_ref_text,
                gen_text=sentence, file_wave=out_p, seed=42,
            )
    vol.commit()

    def spec_diff(wa, wb, n_fft=1024, hop=256):
        y1, sr = sf.read(wa)
        y2, _ = sf.read(wb)
        n = min(len(y1), len(y2))
        y1, y2 = y1[:n], y2[:n]
        if y1.ndim > 1: y1 = y1.mean(axis=1)
        if y2.ndim > 1: y2 = y2.mean(axis=1)
        _, _, S1 = stft(y1, sr, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, S2 = stft(y2, sr, nperseg=n_fft, noverlap=n_fft - hop)
        M1, M2 = np.log1p(np.abs(S1)), np.log1p(np.abs(S2))
        return {
            "logmag_corr": float(np.corrcoef(M1.flatten(), M2.flatten())[0, 1]),
            "logmag_mae": float(np.mean(np.abs(M1 - M2))),
        }

    print("\n=== v5 spectral diff (EARS gate, ignore Whisper) ===")
    n_audible = 0
    for si in range(3):
        for pair in [(0, 2), (0, 4), (2, 4)]:
            wa = os.path.join(SWEEP, f"v1_lv{pair[0]}_s{si}.wav")
            wb = os.path.join(SWEEP, f"v1_lv{pair[1]}_s{si}.wav")
            d = spec_diff(wa, wb)
            verdict = "AUDIBLE" if d["logmag_corr"] < 0.85 else "near-identical"
            if d["logmag_corr"] < 0.85:
                n_audible += 1
            print(f"  s{si} lv{pair[0]} vs lv{pair[1]}: corr={d['logmag_corr']:.4f} "
                  f"mae={d['logmag_mae']:.4f} -> {verdict}")
    print(f"\n  audible pairs: {n_audible}/9")
    print(f"  gate (>=4 audible to pass): {'PASS' if n_audible >= 4 else 'FAIL'}")


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60 * 12,
)
def pull_and_generate_ghost():
    """v9: regenerate corpus with --input-mode=mondegreen for Ghost mode LoRA training.
    Saves to /vol/coherence_ghost and /vol/coherence_ghost_ds, does not overwrite v8 corpus."""
    import os, shutil, subprocess
    _setup_repo()

    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id="akshan-main/glossolalia-inputs",
        repo_type="dataset",
        local_dir=f"{REPO_DIR}/data_pull",
    )
    os.makedirs(f"{REPO_DIR}/data/voices", exist_ok=True)
    for f in ("sentences.txt", "phoneme_lm.npz", "cmudict.dict"):
        s = os.path.join(p, f)
        if os.path.exists(s):
            shutil.copy(s, f"{REPO_DIR}/data/{f}")
    vd = os.path.join(p, "voices")
    if os.path.isdir(vd):
        for f in os.listdir(vd):
            shutil.copy(os.path.join(vd, f), f"{REPO_DIR}/data/voices/{f}")
    print("  sentences:", sum(1 for _ in open(f"{REPO_DIR}/data/sentences.txt")))

    os.makedirs("/vol/voices", exist_ok=True)
    for f in os.listdir(f"{REPO_DIR}/data/voices"):
        shutil.copy(f"{REPO_DIR}/data/voices/{f}", f"/vol/voices/{f}")

    if os.path.isdir("/vol/coherence_ghost_ds") and \
       os.path.exists("/vol/coherence_ghost_ds/metadata.csv"):
        print("  ghost corpus already on volume, skipping")
        vol.commit()
        return

    os.chdir(REPO_DIR)
    subprocess.check_call([
        "python", "scripts/generate_coherence_data.py",
        "--sentences", "data/sentences.txt",
        "--voice", "v2:data/voices/v2.wav:data/voices/v2.txt",
        "--voice", "v3:data/voices/v3.wav:data/voices/v3.txt",
        "--lm", "data/phoneme_lm.npz",
        "--out", "data/coherence_ghost",
        "--max-sentences", "3000",
        "--levels", "5",
        "--input-mode", "mondegreen",
        "--resume",
    ])
    subprocess.check_call([
        "python", "scripts/build_coherence_dataset.py",
        "--data", "data/coherence_ghost",
        "--out", "data/coherence_ghost_ds",
    ])
    shutil.copytree(f"{REPO_DIR}/data/coherence_ghost", "/vol/coherence_ghost", dirs_exist_ok=True)
    shutil.copytree(f"{REPO_DIR}/data/coherence_ghost_ds", "/vol/coherence_ghost_ds", dirs_exist_ok=True)
    vol.commit()


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60 * 10,
)
def train_v9_ghost():
    """v9 LoRA: DEPRECATED (2026-06-12). See DECISIONS.md "Ghost LoRA dropped" entry.

    Ghost mode in app.py sets dial=0 at inference, so a Ghost LoRA never functionally
    applies. Kept here only so the entrypoint stays callable if someone later switches
    Ghost inference to set_dial(level). For now, the v9 corpus + adapter are NOT built
    or shipped. Tongues mode (v8 LoRA) is the only trained adapter.
    """
    import os
    os.environ["V5_CORPUS_DIR"] = "/vol/coherence_ghost_ds"
    os.environ["V5_OUT_DIR"] = "/vol/v9_adapter"
    os.environ.setdefault("V5_EPOCHS", "2")
    train_v5.local()


@app.local_entrypoint()
def main():
    """Full-scale Tongues run: 3000 sentences x 2 voices x 5 levels = 30,000 clips, 2 epochs.
    Sequential (single A100). Use main_parallel for 4-way fanout (~4x faster)."""
    import os
    os.environ.setdefault("V5_EPOCHS", "2")
    print(">> step 1: pull data + generate corpus (Tongues = pseudo, 30000 clips)")
    pull_and_generate.remote()
    print("\n>> step 2: train v8 (Tongues mode)")
    train_v5.remote()
    print("\n>> step 3: sweep + spectral verify")
    sweep_and_verify.remote()
    print("\n>> done. Tongues adapter on volume at /vol/v5_adapter")


@app.local_entrypoint()
def main_parallel(num_shards: int = 4):
    """Parallel Tongues run: fans out corpus generation across `num_shards` concurrent
    A100 containers via Function.map(), each handling a contiguous slice of sentences.

    Same 30,000-clip target as main. Wall-clock ~8h / num_shards for corpus, then
    sequential build (~5 min) + training (~4h). Total cost stays roughly the same
    (more containers, less time each). Default 4 shards: ~2h corpus + ~4h train.

    Filenames carry a shard-aware global index so the merged corpus is contiguous;
    each worker writes its own manifest_shard{N}.jsonl, which build_dataset_after_fanout
    globs and merges before training."""
    import os
    os.environ.setdefault("V5_EPOCHS", "2")
    print(f">> step 1: parallel corpus gen (Tongues, {num_shards} workers, ~30k clips)")
    # starmap unpacks each tuple as positional args; blocks until all shards finish.
    shard_args = [(i, num_shards) for i in range(num_shards)]
    list(generate_chunk.starmap(shard_args))
    print("\n>> step 2: merge shard manifests into F5-TTS finetune dataset")
    build_dataset_after_fanout.remote()
    print("\n>> step 3: train v8 (Tongues mode)")
    train_v5.remote()
    print("\n>> step 4: sweep + spectral verify")
    sweep_and_verify.remote()
    print("\n>> done. Tongues adapter on volume at /vol/v5_adapter")


@app.function(gpu="A100-40GB", volumes={"/vol": vol}, timeout=60 * 30)
def sweep_v9_ghost():
    import os
    os.environ["SWEEP_ADAPTER_DIR"] = "/vol/v9_adapter"
    os.environ["SWEEP_OUT_DIR"] = "/vol/sweep_ghost"
    sweep_and_verify.local()


@app.local_entrypoint()
def sweep_ghost():
    sweep_v9_ghost.remote()


@app.local_entrypoint()
def main_ghost():
    """Full-scale Ghost run: 3000 sentences x 2 voices x 5 levels = 30,000 clips, 2 epochs.
    Same 2-voice plan as v8 (v2 Karen Savage + v3 bass) for diversity. See DECISIONS.md."""
    print(">> step 1: pull data + generate corpus (Ghost = mondegreen, 30000 clips)")
    pull_and_generate_ghost.remote()
    print("\n>> step 2: train v9 (Ghost mode LoRA)")
    train_v9_ghost.remote()
    print("\n>> done. Ghost adapter on volume at /vol/v9_adapter")
