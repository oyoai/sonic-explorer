"""Batch-precomputes a short natural-language description per song: AST tags
(pipeline/sound_tagging.py) + song DNA synthesized into a phrase by the same
LLM layer used throughout the app (llm/explain.py's generate_description) --
same pattern as Ask the DJ's explanations, applied to a new use case.

Must be a precompute/batch step, not a live Streamlit feature: AST needs
torch/transformers, which the deployed app deliberately never installs (see
requirements.txt's docstring -- the deployed app only ever reads precomputed
artifacts). Runs entirely on CPU. Checkpointed (commits to the DB every
CHECKPOINT_EVERY songs) since a full-library run makes real LLM API calls
and takes real wall-clock time either way. Safe to re-run: songs that
already have a description are skipped.
"""

import tomllib
from pathlib import Path

import anthropic
import librosa

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.config import ARTIFACTS_DIR, CLAP_SR, DB_PATH, audio_path_for
from sonic_explorer.llm.explain import ExplanationClient
from sonic_explorer.pipeline.sound_tagging import AST_SAMPLE_RATE, get_descriptive_tags
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository

CHECKPOINT_EVERY = 25

# AST ("...-10-10-...") is trained on 10s clips -- feeding it a whole song
# (several minutes) rather than a short window is what actually stalled the
# first run of this script for over an hour with zero songs completed.
# Every other AST call site in this codebase (vocal_presence.py's per-segment
# check, scripts/sample_vocal_segment_prevalence.py) slices a short window
# before calling the classifier; this script must too. The middle of the
# song is used rather than the start, to avoid a cold-open silence/fade-in.
CLIP_DURATION_SEC = 10.0


def _representative_clip(audio, sr):
    clip_len = int(CLIP_DURATION_SEC * sr)
    if len(audio) <= clip_len:
        return audio
    start = (len(audio) - clip_len) // 2
    return audio[start : start + clip_len]


def main():
    secrets = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml").read_text())
    api_key = secrets["ANTHROPIC_API_KEY"]

    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    llm_client = ExplanationClient(anthropic.Anthropic(api_key=api_key))

    all_songs = song_repo.list_songs()
    dna_normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in all_songs])

    todo = [s for s in all_songs if not s.description]
    print(f"{len(all_songs)} total songs, {len(todo)} need a description")

    n_done, n_failed = 0, 0
    for i, song in enumerate(todo):
        raw_dna = {axis: getattr(song, axis) for axis in AXES}
        if any(v is None for v in raw_dna.values()):
            n_failed += 1
            continue

        try:
            audio, sr = librosa.load(str(audio_path_for(song)), sr=AST_SAMPLE_RATE, mono=True)
            clip = _representative_clip(audio, sr)
            tags = get_descriptive_tags(clip, sr)

            norm_dna = dna_normalizer.normalize(raw_dna)
            description = llm_client.generate_description(
                title=song.title, artist=song.artist, genre=song.genre_top, tags=tags,
                tempo_bpm=norm_dna["tempo_bpm"], energy=norm_dna["energy"], brightness=norm_dna["brightness"],
                harmonic_complexity=norm_dna["harmonic_complexity"], rhythmic_density=norm_dna["rhythmic_density"],
            )
            song_repo.update_description(song.id, description)
            n_done += 1
        except Exception as exc:
            print(f"  failed on {song.title!r}: {exc}")
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  ...{i + 1}/{len(todo)} processed ({n_done} done, {n_failed} failed)")

    print(f"\nDone. {n_done} descriptions generated, {n_failed} failed/skipped (no DNA or an error).")


if __name__ == "__main__":
    main()
