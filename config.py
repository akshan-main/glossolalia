"""tune-fine configuration. Single source of truth for subgenres, paths, hyperparameters."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
LATENTS_ROOT = PROJECT_ROOT / "latents"
ADAPTERS_ROOT = PROJECT_ROOT / "adapters"


@dataclass
class Subgenre:
    name: str
    family: str
    source: str
    tag: str
    desc: str
    repo_slug: str


SUBGENRES = [
    Subgenre("deep house", "Electronic", "mtg", "deephouse", "Warm four-on-the-floor, ~120 BPM, soulful and steady.", "tunefine-deep-house"),
    Subgenre("techno", "Electronic", "mtg", "techno", "Driving, hypnotic, dark atmosphere, relentless kick.", "tunefine-techno"),
    Subgenre("drum and bass", "Electronic", "mtg", "drumnbass", "174 BPM breakbeats, deep sub-bass, rolling rhythm.", "tunefine-dnb"),
    Subgenre("ambient", "Electronic", "mtg", "ambient", "Slow-moving textures, drones, atmosphere over rhythm.", "tunefine-ambient"),
    Subgenre("trip-hop", "Electronic", "mtg", "triphop", "Atmospheric, downtempo, dusty samples and breakbeats.", "tunefine-triphop"),
    Subgenre("hip-hop", "Hip-hop", "mtg", "hiphop", "Boom-bap drums, sampled grooves, urban energy.", "tunefine-hiphop"),
    Subgenre("soul", "R&B", "mtg", "soul", "Warm vocal-driven, organic instruments, emotive.", "tunefine-soul"),
    Subgenre("funk", "Funk", "mtg", "funk", "Groove-driven, syncopated bass, horn stabs.", "tunefine-funk"),
    Subgenre("indie rock", "Rock", "mtg", "indie", "Jangly guitars, mid-tempo, earnest energy.", "tunefine-indie"),
    Subgenre("post-rock", "Rock", "mtg", "postrock", "Atmospheric, building dynamics, instrumental.", "tunefine-postrock"),
    Subgenre("synthpop", "Pop", "mtg", "synthpop", "Bright synths, drum machine, danceable nostalgia.", "tunefine-synthpop"),
    Subgenre("bossa nova", "World", "mtg", "bossanova", "Soft Brazilian, gentle nylon guitar, brushed drums.", "tunefine-bossa-nova"),
]

SUBGENRE_BY_NAME = {s.name: s for s in SUBGENRES}


BASE_MODEL = "ACE-Step/Ace-Step1.5"
MTG_JAMENDO_REPO = "rkstgr/mtg-jamendo"
FMA_REPO = "benjamin-paine/free-music-archive-full"
CAPTIONER_MODEL = "ACE-Step/acestep-captioner"


TRAINING = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "num_epochs": 100,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1.0e-4,
    "warmup_steps": 50,
    "use_lokr": True,
}


CLIPS_PER_SUBGENRE = 200
TARGET_CLIP_DURATION_SEC = 30
SAMPLE_RATE = 48000
