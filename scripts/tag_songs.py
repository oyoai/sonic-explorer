"""Batch AST-tags every song missing sound_tags -- local, free, CPU-only.
Makes ZERO network/API calls of any kind: no anthropic import anywhere in
this file, no secrets.toml read, nothing that could incur cost even by
accident. Split out from generate_song_descriptions.py, which used to do
this AND the separate, paid-API description-synthesis step in one
combined pass -- that coupling meant there was no way to "just get the
free tags" without also either paying for an LLM call per song or already
having a description to skip it, which in practice no untagged song ever
did. Run this first; generate_song_descriptions.py now only describes
songs this script has already tagged.

AST tags a representative 10s clip per song (not the whole track -- see
CLIP_DURATION_SEC's own comment for why that's not optional), same
sonic_explorer.pipeline.sound_tagging.get_descriptive_tags() every other
AST call site in this codebase already uses. Checkpointed, safe to re-run:
only songs with no sound_tags yet are processed."""

import librosa

from sonic_explorer.config import DB_PATH, audio_path_for
from sonic_explorer.pipeline.sound_tagging import AST_SAMPLE_RATE, get_descriptive_tags, serialize_tags
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository

CHECKPOINT_EVERY = 25

# AST ("...-10-10-...") is trained on 10s clips -- feeding it a whole song
# (several minutes) rather than a short window is what actually stalled an
# earlier version of this pass for over an hour with zero songs completed.
# Every other AST call site in this codebase (vocal_presence.py's
# per-segment check, sample_vocal_segment_prevalence.py) slices a short
# window first; this must too. The middle of the song is used rather than
# the start, to avoid a cold-open silence/fade-in skewing the tags.
CLIP_DURATION_SEC = 10.0


def _representative_clip(audio, sr):
    clip_len = int(CLIP_DURATION_SEC * sr)
    if len(audio) <= clip_len:
        return audio
    start = (len(audio) - clip_len) // 2
    return audio[start : start + clip_len]


def tag_song(song_repo, song) -> None:
    audio, sr = librosa.load(str(audio_path_for(song)), sr=AST_SAMPLE_RATE, mono=True)
    clip = _representative_clip(audio, sr)
    tags = get_descriptive_tags(clip, sr)
    song_repo.update_sound_tags(song.id, serialize_tags(tags))


def run_batch(song_repo, todo: list) -> tuple[int, int]:
    n_done, n_failed = 0, 0
    for i, song in enumerate(todo):
        try:
            tag_song(song_repo, song)
            n_done += 1
        except Exception as exc:
            print(f"  failed on {song.title!r}: {exc}")
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  ...{i + 1}/{len(todo)} processed ({n_done} done, {n_failed} failed)")

    print(f"Done. {n_done} succeeded, {n_failed} failed.")
    return n_done, n_failed


def main(db_path=DB_PATH):
    conn = init_db(db_path)
    song_repo = SongRepository(conn)

    todo = [s for s in song_repo.list_songs() if not s.sound_tags]
    print(f"{len(todo)} songs need sound tags (of {len(song_repo.list_songs())} total).")
    run_batch(song_repo, todo)


if __name__ == "__main__":
    main()
