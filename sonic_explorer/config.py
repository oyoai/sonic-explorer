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

# data/ (gitignored, the full local/real dataset) wins if present; deploy_data/
# (committed to git, a small stratified subset -- see scripts/build_deploy_subset.py)
# is the fallback for environments like Streamlit Community Cloud that only ever
# see what's actually in the repo. Resolved once at import time.
_CANDIDATE_DATA_DIRS = [PROJECT_ROOT / "data", PROJECT_ROOT / "deploy_data"]
DATA_DIR = next(
    (d for d in _CANDIDATE_DATA_DIRS if (d / "artifacts" / "sonic_explorer.db").exists()),
    _CANDIDATE_DATA_DIRS[0],
)
AUDIO_DIR = DATA_DIR / "audio"
METADATA_DIR = DATA_DIR / "fma_metadata"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
STRUCTURE_DIR = ARTIFACTS_DIR / "structure"

DB_PATH = ARTIFACTS_DIR / "sonic_explorer.db"
SOUND_INDEX_PATH = ARTIFACTS_DIR / "sound.index"
STRUCTURE_INDEX_PATH = ARTIFACTS_DIR / "structure.index"

# Persisted mean/std for harmony's post-hoc embedding whitening (see
# analysis/embedding_whitening.py's module docstring for the real bug this
# fixes) -- HarmonyFacet.embed() loads this if present and applies it so
# every fresh embedding lands in the same space as what's already indexed.
# Absent on a fresh checkout before scripts/whiten_harmony_index.py has
# ever been run against that environment's data dir -- HarmonyFacet treats
# that as "no whitening yet," not an error.
HARMONY_WHITENER_PATH = ARTIFACTS_DIR / "harmony_whitener.npz"

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


def audio_path_for(song) -> Path:
    """Resolves a Song's actual audio file in *this* environment, ignoring
    song.filepath -- an absolute path written in one environment (Colab, local,
    deployed) is never valid in another (hit once already: repath_audio_paths.py
    fixes the Colab->local case; this avoids needing an equivalent script for
    local->deployed). Falls back to the stored filepath for anything not named
    {fma_track_id}.mp3 -- e.g. scripts/seed_dev_data.py's dev_{track_id}.wav
    synthetic files, which are always consumed in the same environment they were
    written in, so the stored path is already correct for them."""
    candidate = AUDIO_DIR / f"{song.fma_track_id}.mp3"
    return candidate if candidate.exists() else Path(song.filepath)


ALBUM_ART_DIR = DATA_DIR / "album_art"


def album_art_path_for(song) -> Path | None:
    """Resolves a song's AI-generated album art image (scripts/export_
    album_art_prompts.py -> notebooks/12_album_art_generation.ipynb), if one
    exists for THIS environment's resolved DATA_DIR. Returns None, not a
    guessed fallback path, when it doesn't -- unlike audio_path_for's
    song.filepath fallback, there's no "wrong but valid" path to fall back
    to for an image that was never generated.

    Deliberately keyed by song.id under DATA_DIR specifically, not e.g.
    song.fma_track_id: album art was generated against deploy_data's own
    song_id numbering, and data/'s song_ids are a DIFFERENT numbering for
    the same underlying songs (build_deploy_subset.py reassigns ids when it
    builds the subset). Resolving via DATA_DIR (not a hardcoded deploy_data
    path) means this only ever looks for art under whichever environment is
    actually active -- on Streamlit Cloud (DATA_DIR resolves to deploy_data)
    that's exactly right; running locally against the full data/ library
    (DATA_DIR resolves to data/, which has no album_art/ directory at all)
    this correctly returns None rather than risking a same-numbered-but-
    different-song mismatch by reaching into deploy_data's art from there."""
    candidate = ALBUM_ART_DIR / f"{song.id}.png"
    return candidate if candidate.exists() else None
