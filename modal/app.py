"""Modal app: train v5 glossolalia dial (AdaLN-side level conditioning + LoRA) on A100-40GB.

Replaces Colab kernel-reset thrash with a clean reproducible runner. Same architecture
the workflow synthesized + verified, with the dtype handling fixed (no PEFT
modules_to_save quirks — level_embed attached directly to DiT, optimized as a separate
param group).

Run:
    modal run modal/app.py                  # full pipeline (~50 min, ~$2.50)
    modal run modal/app.py::dry_run         # 30s sanity check (~$0.05) — DO THIS FIRST
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
        "f5-tts",
        "peft>=0.14.0",
        # NOTE: NO torchao. torchao>=0.16 needs torch>=2.11 (uses torch.int1),
        # torchao<0.16 has its own issues with peft 0.14. Skip it on this image —
        # peft works without it as long as we don't use quantized models. The Colab
        # need for torchao was a different version-conflict trap that doesn't apply
        # to a clean Modal Debian build.
        "soundfile",
        "librosa",
        "scipy",
        "numpy",
        "huggingface_hub",
        "safetensors",
        "g2p_en",
        "jiwer",
        "openai-whisper",
        "resemblyzer",
        "cached_path",
    )
    .run_commands(
        # Pre-download NLTK assets — caches into image
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


def _setup_repo():
    """Clone (or pull) the repo at runtime. Called at the top of every function."""
    import os, subprocess
    if not os.path.isdir(REPO_DIR):
        subprocess.check_call(["git", "clone", "-q", REPO_URL, REPO_DIR])
    else:
        subprocess.check_call(["git", "-C", REPO_DIR, "pull", "-q", "origin", "main"])
    import sys
    sys.path.insert(0, REPO_DIR)


@app.function(gpu="A100-40GB", timeout=60 * 2)
def dry_run():
    """30-second sanity check: GPU works, F5-TTS imports, repo clones. Costs ~$0.05."""
    import torch
    _setup_repo()
    print(f"  torch: {torch.__version__} | cuda: {torch.cuda.is_available()}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    from f5_tts.api import F5TTS
    from peft import LoraConfig
    import patches  # repo's patches/__init__.py
    print("  f5_tts + peft + patches all import OK")
    print("  dry_run PASS")


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60,
)
def pull_and_generate():
    """Pull data from HF + generate 200-clip training corpus. Idempotent (caches to volume)."""
    import os, shutil, subprocess
    _setup_repo()

    # Stage data inputs (small files, fine to re-pull each run)
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

    # Mirror voices to volume so sweep_and_verify can read them
    os.makedirs("/vol/voices", exist_ok=True)
    for f in os.listdir(f"{REPO_DIR}/data/voices"):
        shutil.copy(f"{REPO_DIR}/data/voices/{f}", f"/vol/voices/{f}")

    # Generate corpus (cached on volume — skip if already done)
    if os.path.isdir("/vol/coherence_ds") and os.path.exists("/vol/coherence_ds/metadata.csv"):
        print("  corpus already on volume, skipping generation")
        vol.commit()
        return

    os.chdir(REPO_DIR)
    subprocess.check_call([
        "python", "scripts/generate_coherence_data.py",
        "--sentences", "data/sentences.txt",
        "--voice", "v1:data/voices/v1.wav:data/voices/v1.txt",
        "--lm", "data/phoneme_lm.npz",
        "--out", "data/coherence",
        "--max-sentences", "40",
        "--levels", "5",
        "--input-mode", "pseudo",
        "--resume",
    ])
    subprocess.check_call([
        "python", "scripts/build_coherence_dataset.py",
        "--data", "data/coherence",
        "--out", "data/coherence_ds",
    ])
    shutil.copytree(f"{REPO_DIR}/data/coherence", "/vol/coherence", dirs_exist_ok=True)
    shutil.copytree(f"{REPO_DIR}/data/coherence_ds", "/vol/coherence_ds", dirs_exist_ok=True)
    vol.commit()
    print("  corpus saved to volume")


@app.function(
    gpu="A100-40GB",
    volumes={"/vol": vol},
    timeout=60 * 60,
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
    EPOCHS = 6
    LR_LORA = 1e-4
    LEVEL_LR = 5e-4
    SEED = 42
    OUT_DIR = "/vol/v5_adapter"
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

    ds = CoherenceDS("/vol/coherence_ds/metadata.csv")
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
            torch.nn.utils.clip_grad_norm_(lora_params + level_params, 1.0)
            opt.step()
            base_dit._current_scalars = None
            clear_text_cache(base_dit)
            # Print at steps 1, 5, then every 10 — early-detect a dead dial in the first minute
            if step in (1, 5) or step % 10 == 0:
                with torch.no_grad():
                    # fp32 probe — level_embed is fp32 now
                    probe = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]],
                                         device=device, dtype=torch.float32)
                    norms = level_embed(probe).norm(dim=-1).tolist()
                print(f"  ep{ep} step{step:4d} loss={loss.item():.4f} "
                      f"lvl_norms={[f'{x:.3f}' for x in norms]} "
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
    timeout=60 * 60,
)
def sweep_and_verify():
    """Sweep dial 0..4 on 3 holdout sentences, spectral diff between dial=0 and dial=4."""
    import os, numpy as np, soundfile as sf
    from scipy.signal import stft
    _setup_repo()

    import patches  # installs F5TTS.load_lora + set_dial
    from f5_tts.api import F5TTS

    OUT = "/vol/v5_adapter"
    SWEEP = "/vol/sweep"
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


@app.local_entrypoint()
def main():
    print(">> step 1: pull data + generate corpus")
    pull_and_generate.remote()
    print("\n>> step 2: train v5")
    train_v5.remote()
    print("\n>> step 3: sweep + spectral verify")
    sweep_and_verify.remote()
    print("\n>> done. Adapter on volume at /vol/v5_adapter")
