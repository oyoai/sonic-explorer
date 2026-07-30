"""Standalone measurement: how much does CLAP's embedding move under pure
loudness perturbation alone, with everything else about the audio held
fixed? Run this BEFORE building a full perturbation/robustness suite on top
of this pipeline -- see sonic_explorer/evaluation/clap_gain_sensitivity.py's
own module docstring for why this needs answering first (no stage in this
pipeline normalizes raw gain before feature extraction, so a robustness
test's measured similarity changes could otherwise be conflating a real
content-sensitivity signal with an unmeasured loudness artifact).

Real audio, real segments, sampled from the actual library -- not synthetic
tones, since whether CLAP is gain-sensitive plausibly depends on real
spectral content, not a single sine wave. Each song contributes its middle
segment specifically, to avoid a track's intro/outro silence or fade
disproportionately influencing the result.

Doesn't touch the FAISS index or EmbeddingRepository at all -- every vector
here is computed fresh via SoundFacet.embed_batch(), the same call any
future perturbation test would make, so this measures exactly the
sensitivity that test would actually be exposed to."""

import librosa
import numpy as np

from sonic_explorer.config import CLAP_SR, DB_PATH, audio_path_for
from sonic_explorer.evaluation.clap_gain_sensitivity import measure_gain_sensitivity
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository

SAMPLE_SIZE = 30
GAIN_LEVELS_DB = [-12.0, -6.0, -3.0, 3.0, 6.0, 12.0]
SEED = 42


def _load_sample_windows(song_repo: SongRepository, sample_size: int, seed: int) -> list[np.ndarray]:
    songs = song_repo.list_songs()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(songs), size=min(sample_size, len(songs)), replace=False)

    windows = []
    for i in indices:
        song = songs[int(i)]
        segments = song_repo.get_segments(song.id)
        if not segments:
            continue
        seg = segments[len(segments) // 2]
        audio, sr = librosa.load(str(audio_path_for(song)), sr=CLAP_SR, mono=True)
        windows.append(audio[int(seg.start_sec * sr):int(seg.end_sec * sr)])
    return windows


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)

    windows = _load_sample_windows(song_repo, SAMPLE_SIZE, SEED)
    print(f"Measuring CLAP gain sensitivity on {len(windows)} real segments, {len(GAIN_LEVELS_DB)} gain levels...")

    results = measure_gain_sensitivity(windows, GAIN_LEVELS_DB, sr=CLAP_SR)

    print(f"\n{'gain (dB)':>10} | {'mean cos sim':>13} | {'mean drift':>11} | {'min sim':>8} | {'max sim':>8}")
    print("-" * 62)
    for r in results:
        sims = r.cosine_similarities
        print(
            f"{r.gain_db:>10.1f} | {r.mean_similarity:>13.4f} | {r.mean_drift:>11.4f} | "
            f"{min(sims):>8.4f} | {max(sims):>8.4f}"
        )
    print(
        "\nA drift near 0 (similarity near 1.0) at every gain level means CLAP is effectively "
        "loudness-invariant for this library; a drift that grows with |gain_db| means loudness "
        "alone measurably moves the embedding -- factor that in before attributing a perturbation "
        "test's similarity changes to content sensitivity."
    )


if __name__ == "__main__":
    main()
