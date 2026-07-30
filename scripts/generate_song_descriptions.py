"""Batch-generates a short natural-language description per song via an
LLM -- llm/explain.py's generate_description(), the same layer Ask the
DJ's own explanations use. Two interchangeable backends, picked by
use_local:

- Real Anthropic API (default, use_local=False) -- the only place in this
  script that imports anthropic or reads secrets.toml, and only when this
  path is actually taken (a local-only run touches neither).
- LocalTransformersClient (use_local=True, see llm/local_client.py) --
  free, CPU-only, no network call of any kind once the model weights are
  cached locally. Same generate_description() call either way: explain.py
  needed zero changes to support this, since ExplanationClient was already
  written against a duck-typed `.messages.create(...) -> .content[0].text`
  interface, not the real Anthropic SDK specifically.

Split from tag_songs.py (free, local, CPU-only AST tagging) -- they used
to be one combined script/pass, which meant there was no way to get just
the free tags without either paying for an LLM call per song or already
having a description to skip it (in practice, no untagged song ever did).
Requires sound_tags already present: a song with no tags yet is skipped
here, not silently tagged as a side effect -- run tag_songs.py first.
That keeps THIS script's own cost (one LLM call per song processed,
whichever backend) never coupled to tagging's cost (CPU time only) again.

Checkpointed (commits to the DB every CHECKPOINT_EVERY songs) since a
full-library run takes real wall-clock time either way. Safe to re-run:
songs that already have a description are skipped."""

import tomllib
from pathlib import Path

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.config import DB_PATH
from sonic_explorer.llm.explain import ExplanationClient
from sonic_explorer.pipeline.sound_tagging import deserialize_tags
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository

CHECKPOINT_EVERY = 25


def _build_llm_client(use_local: bool):
    if use_local:
        from sonic_explorer.llm.local_client import LocalTransformersClient

        return ExplanationClient(LocalTransformersClient())

    import anthropic

    secrets = tomllib.loads((Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml").read_text())
    return ExplanationClient(anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"]))


def describe_song(song_repo, llm_client, dna_normalizer, song) -> bool:
    """Returns False (skips, no API call made) when this song's DNA scalars
    aren't fully computed yet -- generate_description's prompt needs all
    five, and a partial/guessed value would misrepresent the song."""
    raw_dna = {axis: getattr(song, axis) for axis in AXES}
    if any(v is None for v in raw_dna.values()):
        return False

    tags = deserialize_tags(song.sound_tags)
    norm_dna = dna_normalizer.normalize(raw_dna)
    description = llm_client.generate_description(
        title=song.title, artist=song.artist, genre=song.genre_top, tags=tags,
        tempo_bpm=norm_dna["tempo_bpm"], energy=norm_dna["energy"], brightness=norm_dna["brightness"],
        harmonic_complexity=norm_dna["harmonic_complexity"], rhythmic_density=norm_dna["rhythmic_density"],
    )
    song_repo.update_description(song.id, description)
    return True


def run_batch(song_repo, llm_client, dna_normalizer, todo: list) -> tuple[int, int]:
    n_done, n_failed = 0, 0
    for i, song in enumerate(todo):
        try:
            if describe_song(song_repo, llm_client, dna_normalizer, song):
                n_done += 1
            else:
                n_failed += 1
        except Exception as exc:
            print(f"  failed on {song.title!r}: {exc}")
            n_failed += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  ...{i + 1}/{len(todo)} processed ({n_done} done, {n_failed} failed)")

    print(f"Done. {n_done} succeeded, {n_failed} failed/skipped.")
    return n_done, n_failed


def main(db_path=DB_PATH, use_local: bool = False):
    conn = init_db(db_path)
    song_repo = SongRepository(conn)
    llm_client = _build_llm_client(use_local)

    all_songs = song_repo.list_songs()
    dna_normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in all_songs])

    needs_description = [s for s in all_songs if not s.description]
    not_yet_tagged = [s for s in needs_description if not s.sound_tags]
    todo = [s for s in needs_description if s.sound_tags]
    print(
        f"{len(all_songs)} total songs -- {len(needs_description)} need a description, of which "
        f"{len(not_yet_tagged)} aren't tagged yet (run tag_songs.py first, skipped here) and "
        f"{len(todo)} are ready to describe now."
    )

    if todo:
        run_batch(song_repo, llm_client, dna_normalizer, todo)


if __name__ == "__main__":
    main()
