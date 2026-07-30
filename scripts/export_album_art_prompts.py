"""Batch-exports one deterministic AI-album-art prompt per song in
deploy_data/ (the "deployed set" -- same DB build_deploy_subset.py
produces) -- step 3 of the album-art feature. Steps 1-2 (descriptor
bundling + phrase templating) live in analysis/album_art_prompt.py and are
tested there directly (no audio, no DB); this script is just the batch
orchestration + real per-song data gathering that feeds it. Step 4 (Colab
image generation) is a separate notebook that reads this script's output.
Step 5 (wiring generated images into the app) is deliberately not built
yet -- explicit scope, not an oversight.

tempo_bpm/energy/brightness/sound_tags are already-persisted DB columns
(SongRepository.list_songs) -- no recomputation. key_tonic/key_mode is NOT
persisted anywhere in this schema (see analysis/key_chord.py's own
docstring), so this script computes it live per song the same way
Explore's Selected Song panel already does (chroma_for_display +
estimate_key) -- a real per-song audio decode, the one genuinely new cost
this script adds over what's already stored. Pure CPU, no GPU needed --
same as every other descriptor this pulls.

Sound tag coverage depends on scripts/tag_songs.py (free, local AST
tagging, split out from generate_song_descriptions.py so it never requires
an API key or incurs API cost) having been run against this DB first --
this script reports real, live coverage numbers up front rather than
assuming; a song with no tags yet just gets a prompt with no sound-tag
phrase, an honest gap build_album_art_prompt() already degrades
gracefully for, not a bug."""

import csv
import json

from sonic_explorer.analysis.album_art_prompt import SongDescriptors, build_album_art_prompt
from sonic_explorer.analysis.key_chord import estimate_key
from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.analysis.waveform_preview import chroma_for_display
from sonic_explorer.config import PROJECT_ROOT, audio_path_for
from sonic_explorer.pipeline.sound_tagging import deserialize_tags
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository

DEPLOY_DB_PATH = PROJECT_ROOT / "deploy_data" / "artifacts" / "sonic_explorer.db"
OUTPUT_DIR = PROJECT_ROOT / "album_art"
CHECKPOINT_EVERY = 25


def _descriptors_for_song(song) -> SongDescriptors:
    """Key is the one descriptor with no stored column -- computed live,
    same chroma_for_display()+estimate_key() pair Song X-Ray/Explore
    already use, wrapped defensively since a real audio-decode failure on
    one song (corrupt file, near-silent clip) shouldn't abort the whole
    batch -- that song's prompt just skips the mood phrase, same graceful
    handling build_album_art_prompt() already does for None."""
    try:
        chroma, _times = chroma_for_display(audio_path_for(song))
        key_estimate = estimate_key(chroma)
        key_tonic, key_mode = key_estimate.tonic, key_estimate.mode
    except Exception as exc:
        print(f"  key estimation failed for {song.title!r}: {exc}")
        key_tonic, key_mode = None, None

    tags = [label for label, _score in deserialize_tags(song.sound_tags)]
    return SongDescriptors(
        song_id=song.id, tempo_bpm=song.tempo_bpm, energy=song.energy, brightness=song.brightness,
        key_tonic=key_tonic, key_mode=key_mode, sound_tags=tags,
    )


def main():
    conn = init_db(DEPLOY_DB_PATH)
    song_repo = SongRepository(conn)
    songs = song_repo.list_songs()
    print(f"{len(songs)} songs in the deployed set.")

    normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in songs])

    n_tagged = sum(1 for s in songs if s.sound_tags)
    print(
        f"{n_tagged}/{len(songs)} songs have sound tags -- untagged songs' prompts below will lack a "
        "sound-tag phrase until scripts/tag_songs.py is run against this DB."
    )

    rows = []
    for i, song in enumerate(songs):
        descriptors = _descriptors_for_song(song)
        prompt = build_album_art_prompt(descriptors, normalizer)
        rows.append({
            "song_id": song.id, "title": song.title, "artist": song.artist, "genre": song.genre_top,
            "prompt": prompt.prompt_text, "trace": prompt.trace,
        })
        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  ...{i + 1}/{len(songs)} processed")

    OUTPUT_DIR.mkdir(exist_ok=True)

    json_path = OUTPUT_DIR / "prompts.json"
    json_path.write_text(json.dumps(rows, indent=2))

    # CSV drops `trace` -- a nested dict doesn't flatten sensibly into a
    # spreadsheet cell, and the JSON export above is the source of truth for
    # inspecting groundedness; the CSV exists for a quick human skim and as
    # the simplest possible input for the Colab notebook's pandas read.
    csv_path = OUTPUT_DIR / "prompts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["song_id", "title", "artist", "genre", "prompt"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "trace"})

    print(f"Wrote {len(rows)} prompts to {json_path} and {csv_path}.")


if __name__ == "__main__":
    main()
