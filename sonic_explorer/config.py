"""Shared constants and paths. Same file whether imported locally or inside Colab."""

from pathlib import Path

# --- audio / windowing ---
CLAP_SR = 48000
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_DIM = 512

WINDOW_SEC = 5.0
HOP_SEC = 2.5  # library-scale hop; wider than the single-song notebook's 1s hop to bound segment count

STRUCTURE_DIM = 32  # fixed-length resampled self-similarity profile per segment

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_DIR = DATA_DIR / "fma_metadata"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
STRUCTURE_DIR = ARTIFACTS_DIR / "structure"

DB_PATH = ARTIFACTS_DIR / "sonic_explorer.db"
SOUND_INDEX_PATH = ARTIFACTS_DIR / "sound.index"
STRUCTURE_INDEX_PATH = ARTIFACTS_DIR / "structure.index"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False
