"""Shared constants and paths. Same file whether imported locally or inside Colab."""

from pathlib import Path

# --- audio / windowing ---
CLAP_SR = 48000
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_DIM = 512

STRUCTURE_SR = 22050  # standard librosa analysis rate for chroma/beat tracking; independent of CLAP_SR

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

# Marker file written by scripts/seed_dev_data.py -- its presence means the UI is
# currently pointed at synthetic placeholder data, not the real synced library.
# Delete it (or just overwrite data/artifacts/ with the real sync) once real data lands.
DEV_DATA_MARKER = ARTIFACTS_DIR / ".dev_data"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False
