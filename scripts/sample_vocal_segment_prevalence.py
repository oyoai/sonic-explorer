"""Prevalence sample: before committing to a ~20-24 hour full-library run of
the per-segment AST vocal-presence check, measure how common the problem
actually is on a random sample of segments drawn from the whole library (not
restricted to any genre -- the failure mode, e.g. an instrumental bridge in
an otherwise-vocal song, can happen anywhere). Also reports the score
distribution so the ~0.018 threshold (validated on only 9 songs) can be
checked against a more diverse sample before being trusted at scale.

Read-only -- does not modify the vocal facet index or DB. Prints per-segment
detail plus a summary, and writes the raw results to a CSV for later
inspection.
"""

import csv
from pathlib import Path

import librosa
import numpy as np

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH, audio_path_for
from sonic_explorer.pipeline.vocal_presence import AST_SAMPLE_RATE, MIN_VOCAL_CONFIDENCE, best_vocal_label_score
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

FACET_NAME = "vocal"
SAMPLE_SIZE = 400
SEED = 42
THRESHOLD = MIN_VOCAL_CONFIDENCE
OUT_CSV = Path(__file__).resolve().parent / "vocal_segment_prevalence_sample.csv"


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)

    rng = np.random.default_rng(SEED)

    # (segment_id, song_id) for every currently-'done' vocal-facet segment
    all_pairs = []
    songs_by_id = {}
    for song in song_repo.list_songs():
        songs_by_id[song.id] = song
        for seg in song_repo.get_segments(song.id):
            if embedding_repo.status(seg.id, FACET_NAME) == "done":
                all_pairs.append((seg.id, song.id))

    print(f"{len(all_pairs)} total 'done' vocal-facet segments in the library")

    sample_idx = rng.choice(len(all_pairs), size=min(SAMPLE_SIZE, len(all_pairs)), replace=False)
    sample_pairs = [all_pairs[i] for i in sample_idx]

    # group by song so each song's audio is loaded exactly once
    segs_by_song: dict[int, list[int]] = {}
    for seg_id, song_id in sample_pairs:
        segs_by_song.setdefault(song_id, []).append(seg_id)

    print(f"Sampling {len(sample_pairs)} segments across {len(segs_by_song)} distinct songs...")

    from transformers import pipeline
    clf = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", top_k=15)

    seg_lookup = {seg.id: seg for song_id in segs_by_song for seg in song_repo.get_segments(song_id)}

    rows = []
    failed_songs = []
    for i, (song_id, seg_ids) in enumerate(segs_by_song.items()):
        song = songs_by_id[song_id]
        try:
            audio, sr = librosa.load(str(audio_path_for(song)), sr=AST_SAMPLE_RATE, mono=True)
        except Exception as exc:
            failed_songs.append((song.fma_track_id, str(exc)))
            continue

        for seg_id in seg_ids:
            seg = seg_lookup[seg_id]
            start, end = int(seg.start_sec * sr), int(seg.end_sec * sr)
            window = audio[start:end]
            if len(window) < sr * 0.5:
                continue
            preds = clf(window, sampling_rate=sr)
            score, label = best_vocal_label_score(preds)
            rows.append({
                "segment_id": seg_id, "song_id": song_id, "title": song.title, "genre": song.genre_top,
                "start_sec": seg.start_sec, "end_sec": seg.end_sec, "score": score, "label": label,
            })

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(segs_by_song)} songs processed, {len(rows)} segments scored so far")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scores = np.array([r["score"] for r in rows])
    below_threshold = scores < THRESHOLD

    print(f"\n=== Summary ({len(rows)} segments scored, {len(failed_songs)} songs failed to load) ===")
    print(f"score distribution: mean={scores.mean():.4f} median={np.median(scores):.4f} "
          f"p10={np.percentile(scores, 10):.4f} p25={np.percentile(scores, 25):.4f} "
          f"p75={np.percentile(scores, 75):.4f} p90={np.percentile(scores, 90):.4f}")
    print(f"segments below threshold ({THRESHOLD}): {below_threshold.sum()}/{len(rows)} "
          f"({below_threshold.mean():.1%})")

    # how many distinct SONGS have at least one flagged segment, for context
    flagged_songs = {r["song_id"] for r, below in zip(rows, below_threshold) if below}
    print(f"distinct songs with >=1 flagged segment: {len(flagged_songs)}/{len(segs_by_song)} "
          f"({len(flagged_songs) / len(segs_by_song):.1%})")

    print(f"\nRaw results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
