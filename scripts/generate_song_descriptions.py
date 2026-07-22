"""Batch-precomputes, per song: (1) a short natural-language description --
AST tags (pipeline/sound_tagging.py) + song DNA synthesized into a phrase by
the same LLM layer used throughout the app (llm/explain.py's
generate_description) -- same pattern as Ask the DJ's explanations, applied
to a new use case; and (2) the raw AST tags themselves, stored separately
(sound_tags column) because a short synthesized phrase necessarily drops
most detected tags, which makes it unreliable for exact sound-content search
(e.g. "crow sounds") -- the agent's search_by_sound_content tool needs the
raw tags, not just the phrase.

Must be a precompute/batch step, not a live Streamlit feature: AST needs
torch/transformers, which the deployed app deliberately never installs (see
requirements.txt's docstring -- the deployed app only ever reads precomputed
artifacts). Runs entirely on CPU. Checkpointed (commits to the DB every
CHECKPOINT_EVERY songs) since a full-library run makes real LLM API calls
and takes real wall-clock time either way. Safe to re-run: songs that
already have both a description and sound_tags are skipped. Songs that have
a description but no sound_tags (the 1399 already processed before this
column existed) get a tags-only pass -- re-tagged, but no new LLM call,
since the description they already have doesn't need regenerating.
"""

import tomllib
from pathlib import Path

import anthropic
import librosa

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.config import ARTIFACTS_DIR, CLAP_SR, DB_PATH, audio_path_for
from sonic_explorer.llm.explain import ExplanationClient
from sonic_explorer.pipeline.sound_tagging import AST_SAMPLE_RATE, get_descriptive_tags, serialize_tags
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


def _process(song_repo, llm_client, dna_normalizer, song, *, need_description: bool) -> bool:
    """Returns True on success. Always (re)tags; only calls the LLM (and
    updates description) when need_description is True -- the tags-only
    backfill path skips that call entirely, since a description that already
    exists doesn't need regenerating."""
    raw_dna = {axis: getattr(song, axis) for axis in AXES}
    if need_description and any(v is None for v in raw_dna.values()):
        return False

    audio, sr = librosa.load(str(audio_path_for(song)), sr=AST_SAMPLE_RATE, mono=True)
    clip = _representative_clip(audio, sr)
    tags = get_descriptive_tags(clip, sr)
    song_repo.update_sound_tags(song.id, serialize_tags(tags))

    if need_description:
        norm_dna = dna_normalizer.normalize(raw_dna)
        description = llm_client.generate_description(
            title=song.title, artist=song.artist, genre=song.genre_top, tags=tags,
            tempo_bpm=norm_dna["tempo_bpm"], energy=norm_dna["energy"], brightness=norm_dna["brightness"],
            harmonic_complexity=norm_dna["harmonic_complexity"], rhythmic_density=norm_dna["rhythmic_density"],
        )
        song_repo.update_description(song.id, description)
    return True


def _run_batch(song_repo, llm_client, dna_normalizer, todo, *, need_description: bool, label: str):
    n_done, n_failed = 0, 0
    for i, song in enumerate(todo):
        try:
            if _process(song_repo, llm_client, dna_normalizer, song, need_description=need_description):
                n_done += 1
            else:
                n_failed += 1
        except Exception as exc:
            print(f"  failed on {song.title!r}: {exc}")
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  ...{i + 1}/{len(todo)} {label} processed ({n_done} done, {n_failed} failed)")

    print(f"Done with {label}. {n_done} succeeded, {n_failed} failed/skipped.")
    return n_done, n_failed


def main():
    secrets = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml").read_text())
    api_key = secrets["ANTHROPIC_API_KEY"]

    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    llm_client = ExplanationClient(anthropic.Anthropic(api_key=api_key))

    all_songs = song_repo.list_songs()
    dna_normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in all_songs])

    needs_description = [s for s in all_songs if not s.description]
    needs_tags_only = [s for s in all_songs if s.description and not s.sound_tags]
    print(
        f"{len(all_songs)} total songs -- {len(needs_description)} need a description (+ tags), "
        f"{len(needs_tags_only)} have a description but need tags backfilled"
    )

    if needs_description:
        _run_batch(song_repo, llm_client, dna_normalizer, needs_description, need_description=True, label="description")
    if needs_tags_only:
        _run_batch(song_repo, llm_client, dna_normalizer, needs_tags_only, need_description=False, label="tags-only backfill")


if __name__ == "__main__":
    main()
