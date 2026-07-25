"""Shared, cached "naive metadata baseline vs. real audio similarity"
comparison data -- used by both Overview (the graph comparison + the audio-
playback demo) and Approach (step 1's waveform contrast). Both pages show
the exact same demo pair, by design (a callback across pages, not two
independently-curated examples), which is why this lives in its own leaf
module rather than inside overview_page.py: neither page should import this
from the other, and it isn't really "Overview's" data anymore once Approach
depends on it too -- same reasoning overview_page.py itself documents for
why OVERVIEW_PAGE lives in its own module rather than inside Overview.py.

Demo-pair selection is deliberately dynamic (computed live from whatever
library is loaded), not two hardcoded song titles -- picking fixed titles
was exactly the bug class fixed in pages/1_Methodology.py earlier this
project (examples that don't exist in the smaller deployed subset). See
analysis/network_graph.py's pick_naive_mismatch_pair()/
pick_real_cross_genre_pair() for the actual selection rule."""

import json

import pandas as pd
import streamlit as st

from sonic_explorer.analysis.network_graph import (
    DemoPair,
    SongMetadata,
    build_metadata_similarity_graph,
    build_similarity_graph,
    pick_naive_mismatch_pair,
    pick_real_cross_genre_pair,
)
from sonic_explorer.analysis.taste_map import mean_pool_song_vectors


def _song_metadata(song) -> SongMetadata:
    """genres_all/track_tags are JSON-encoded lists (see
    scripts/enrich_fma_metadata.py); None on any song enrichment hasn't
    touched yet, treated the same as "recovered but empty" -- both mean
    "no signal here," not an error."""
    genres_all = frozenset(json.loads(song.genres_all)) if song.genres_all else frozenset()
    tags = frozenset(json.loads(song.track_tags)) if song.track_tags else frozenset()
    return SongMetadata(genre_top=song.genre_top, genres_all=genres_all, album_id=song.album_id, tags=tags)


@st.cache_data
def build_naive_vs_real_graphs(_song_repo, _embedding_repo, cache_key: int):
    """Same songs, two similarity rules: a combined non-audio metadata score
    (naive) vs. audio-embedding cosine similarity (this project) -- built
    from the same vector set so the comparison isolates the rule, not which
    songs are shown. Also returns the raw vectors and a genre lookup so
    get_demo_pairs() can derive its picks from this exact same cached
    computation rather than redoing mean_pool_song_vectors (not itself
    Streamlit-cached) a second time. cache_key (song count) invalidates the
    cache when the library changes, since _song_repo/_embedding_repo are
    excluded from Streamlit's cache hashing by the leading underscore, same
    convention as Explore's build_explore_graph."""
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)

    real_result = build_similarity_graph(vectors)
    naive_result = build_metadata_similarity_graph({sid: _song_metadata(songs_by_id[sid]) for sid in vectors})

    def _nodes_df(result):
        return pd.DataFrame([
            {
                "song_id": n.song_id, "x": n.x, "y": n.y, "cluster": n.cluster,
                "title": songs_by_id[n.song_id].title, "artist": songs_by_id[n.song_id].artist,
                "genre": songs_by_id[n.song_id].genre_top,
            }
            for n in result.nodes
        ])

    genre_by_song = {sid: songs_by_id[sid].genre_top for sid in vectors}
    return _nodes_df(naive_result), naive_result.edges, _nodes_df(real_result), real_result.edges, vectors, genre_by_song


def get_demo_pairs(_song_repo, _embedding_repo, cache_key: int) -> tuple[DemoPair | None, DemoPair | None]:
    """(naive_mismatch_pair, real_cross_genre_pair) -- either may be None if
    there aren't enough songs/edges to compute one yet. Calling
    build_naive_vs_real_graphs again here is cheap: identical arguments hit
    Streamlit's existing cache entry rather than recomputing anything."""
    _, naive_edges, _, real_edges, vectors, genre_by_song = build_naive_vs_real_graphs(
        _song_repo, _embedding_repo, cache_key
    )
    naive_pair = pick_naive_mismatch_pair(naive_edges, vectors)
    real_pair = pick_real_cross_genre_pair(real_edges, genre_by_song)
    return naive_pair, real_pair
