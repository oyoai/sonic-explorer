"""Library-scale stem-separation + embedding: for each song, separate into
stems (pipeline/separation.py, Demucs -- GPU/Colab, see that module's
caveat) once, then segment and embed each stem independently through its own
facet (facets/stems.py), reusing the exact same compute-once/checkpoint
discipline as pipeline/embed_library.py.

A separate pipeline from run_batch_embedding rather than a generalization of
it -- separation is a fundamentally different (and far heavier) kind of
per-song work than sharing one already-loaded mix across facets, so keeping
it its own module leaves run_batch_embedding's simpler, already-proven logic
untouched.

separate_fn is injectable specifically so this module's segmentation/
embedding/checkpoint/error-isolation logic is fully testable with a fake
separator -- no real Demucs/GPU/torch needed for the test suite, the same
duck-typing pattern already used for FakeSoundFacet in test_embed_library.py.

Energy gate: a stem with essentially nothing in it (an instrumental track's
vocal stem, a rare a cappella track's drum stem) isn't skipped at
separation time -- Demucs always produces all four stems in one pass, so
there's no separation-time cost to save -- but IS skipped at embedding time,
via mark_skipped() rather than left 'pending'. Indexing a CLAP embedding of
near-silence would be worse than just wasted compute: near-silent segments
from *different* songs would all embed close together, making unrelated
instrumental tracks look "similar" on the vocal facet purely because they're
both quiet. mark_skipped() (distinct status) also keeps resumability honest
-- 'pending' would make the per-song skip-check re-run the whole Demucs
separation on every future invocation forever, for a result that was never
going to change.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.base import Facet
from sonic_explorer.pipeline.separation import separate_stems
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

# A stem window whose RMS is below this fraction of the full mix's RMS is
# treated as carrying no meaningful content for that facet. A heuristic, not
# a tuned constant -- reasonable enough to filter obvious silence/noise
# without being so strict it discards quiet-but-real vocals/instrumentation.
MIN_STEM_ENERGY_RATIO = 0.05

_FINISHED_STATUSES = ("done", "skipped")


def _has_meaningful_energy(stem_window: np.ndarray, mix_window: np.ndarray, ratio: float = MIN_STEM_ENERGY_RATIO) -> bool:
    stem_rms = float(np.sqrt(np.mean(np.square(stem_window)))) if len(stem_window) else 0.0
    mix_rms = float(np.sqrt(np.mean(np.square(mix_window)))) if len(mix_window) else 0.0
    return stem_rms >= ratio * (mix_rms + 1e-9)


def run_batch_stem_embedding(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    stem_facets: dict[str, Facet],
    checkpoint_every: int = 50,
    on_checkpoint: Callable[[int, int], None] | None = None,
    on_error: Callable[[int, Exception], None] | None = None,
    separate_fn=separate_stems,
) -> list[int]:
    """manifest_df needs columns: track_id, relative_path. Song rows (and
    their segments) are expected to already exist (via the sound-facet batch
    job) -- tracks not yet in the DB, or with no segments, are skipped rather
    than creating them here. stem_facets keys are the facet names to compute,
    e.g. {"vocal": VocalFacet(), "drums": DrumsFacet()}.

    A single separation/embedding failure must not take down the whole batch
    -- same isolate-and-retry-next-run discipline as
    build_structure_library.py's per-song failure handling. Returns the list
    of track_ids that failed."""
    import librosa

    total = len(manifest_df)
    failed_track_ids: list[int] = []
    pending_confirmation: dict[str, list[tuple[int, int | None]]] = {name: [] for name in stem_facets}

    def checkpoint():
        for facet_name, pending in pending_confirmation.items():
            if not pending:
                continue
            embedding_repo.save_index(facet_name)
            for seg_id, dim in pending:
                embedding_repo.mark_done(seg_id, facet_name, vector_store_id=seg_id, dim=dim)
            pending.clear()

    for i, row in enumerate(manifest_df.itertuples()):
        song = song_repo.get_song_by_fma_track_id(int(row.track_id))
        if song is None or not song.segments:
            continue

        if all(
            embedding_repo.status(seg.id, facet_name) in _FINISHED_STATUSES
            for seg in song.segments
            for facet_name in stem_facets
        ):
            continue  # every requested stem facet already resolved (embedded or skipped) for every segment

        try:
            filepath = str(audio_dir / row.relative_path)
            audio, sr = librosa.load(filepath, sr=CLAP_SR, mono=True)
            stems = separate_fn(audio, sr)

            for facet_name, facet in stem_facets.items():
                stem_audio = stems[facet_name]
                pending_segments = [
                    seg for seg in song.segments
                    if embedding_repo.status(seg.id, facet_name) not in _FINISHED_STATUSES
                ]
                if not pending_segments:
                    continue

                to_embed = []
                for seg in pending_segments:
                    start, end = int(seg.start_sec * sr), int(seg.end_sec * sr)
                    stem_window = stem_audio[start:end]
                    mix_window = audio[start:end]
                    if _has_meaningful_energy(stem_window, mix_window):
                        to_embed.append((seg.id, stem_window))
                    else:
                        embedding_repo.mark_skipped(seg.id, facet_name)

                if not to_embed:
                    continue
                windows = [w for _, w in to_embed]
                vectors = facet.embed_batch(windows, sr)
                for (seg_id, _), vector in zip(to_embed, vectors):
                    embedding_repo.add_to_index(facet_name, seg_id, vector)
                    pending_confirmation[facet_name].append((seg_id, vector.shape[-1]))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            failed_track_ids.append(int(row.track_id))
            if on_error:
                on_error(int(row.track_id), exc)

        if (i + 1) % checkpoint_every == 0:
            checkpoint()
            if on_checkpoint:
                on_checkpoint(i + 1, total)

    checkpoint()  # final flush -- nothing should stay unconfirmed after the run completes
    return failed_track_ids
