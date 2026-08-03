"""Minimal cached-resource layer for this app, mirroring streamlit_app/
resources.py's get_repositories() pattern -- not imported from there
(streamlit_app isn't an installed package, see plotting.py's module
docstring for the same reasoning).

Does now carry a small LLM layer (get_agent() + its two DNA-normalization
helpers, added for pages/3_Ask_The_DJ.py per docs/ASK_THE_DJ_HANDOFF.md) --
an earlier version of this docstring said "no LLM layer at all," true back
when the six-independent-facet-tabs Moment Matcher redesign had dropped
rerank/explanation. That reasoning never applied to the DJ agent, which
isn't a rerank/explain client at all but a real tool-calling loop
(sonic_explorer/llm/agent.py's MusicAgent) -- copied here the same
pattern-not-import way as everything else in this file.

Explicitly points at deploy_data/, not sonic_explorer.config.DATA_DIR's
data/-wins-if-present resolution -- this app is meant to demo exactly what a
stranger visiting the deployed link sees, and curated_examples.py's picks
were only validated to exist against deploy_data/ (see that module's
docstring). Pointing at config.DATA_DIR instead would silently show
data/'s different song_id numbering locally and deploy_data's on
Streamlit Cloud -- fma_track_id lookups make that difference harmless
either way, but pinning to deploy_data/ here keeps local dev and the
deployed app showing identically, which matters more for a presentation
demo than for the main exploratory app.

persistent_song_and_moment()'s selection now survives more than a browser
refresh: st.query_params alone (the previous mechanism, still used for
within-session URL-shareable links) is gone the moment the browser tab/URL
is lost, and st.session_state is gone the moment the Streamlit process
exits -- neither survives an actual app/server restart, which a live
presentation needs to tolerate. LOCAL_SIMILARITY_STATE_PATH is the real
fix: a small JSON file on disk, read on every call as the durable fallback
default (behind st.query_params, ahead of curated_examples.DEFAULT_EXAMPLE)
and rewritten whenever a facet's selection actually changes. Deliberately
NOT persisting "the matched result" as its own field -- retrieval
(RetrievalService.query_by_segment, FAISS IndexFlatIP) is fully
deterministic given the same song+segment+facet, so the match is always
exactly reproducible from the persisted query alone; storing it separately
would just be a second copy of derived data that could drift.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.llm.agent import MusicAgent
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.service import RetrievalService

DEPLOY_ROOT = Path(__file__).resolve().parent.parent / "deploy_data"

# Local Similarity's per-facet song+moment selection, written to a real file
# on disk -- survives a full app/server restart, unlike st.session_state
# (gone the instant the process exits) or even st.query_params (gone the
# instant the browser tab/URL is lost). This is deliberately a plain,
# human-readable JSON file, not a new SQLite table: the entire persisted
# state is six small (facet, fma_track_id, segment_index) rows, there's
# nothing here that benefits from a schema, a query engine, or a second
# open connection to sonic_explorer.db, and a JSON file is trivially
# inspectable/editable by hand if a presenter wants to hand-tune the
# rehearsed setup before a live run. Committed to the repo (not gitignored)
# -- see this module's own docstring for why: the point is a durable
# *rehearsed* default that survives a fresh clone/deploy too, not just a
# local scratch file.
LOCAL_SIMILARITY_STATE_PATH = Path(__file__).resolve().parent / "local_similarity_state.json"


@st.cache_resource
def get_repositories():
    conn = init_db(DEPLOY_ROOT / "artifacts" / "sonic_explorer.db")
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=DEPLOY_ROOT / "artifacts")
    # Only the facets this app actually features -- no need to load
    # sound_tags' index here (see curated_examples.py's module docstring
    # for why sound_tags is deliberately excluded from this demo).
    for facet_name in ["sound", "harmony", "vocal", "drums", "bass", "instrumental"]:
        embedding_repo.load_index(facet_name)
    retrieval_service = RetrievalService(song_repo, embedding_repo)
    return song_repo, embedding_repo, retrieval_service


def audio_path_for(song) -> Path:
    """Same resolution sonic_explorer.config.audio_path_for does, pinned to
    DEPLOY_ROOT specifically (see this module's docstring) instead of that
    function's data/-wins-if-present DATA_DIR."""
    candidate = DEPLOY_ROOT / "audio" / f"{song.fma_track_id}.mp3"
    return candidate if candidate.exists() else Path(song.filepath)


def facet_display_name(name: str) -> str:
    return name.replace("_", " ").capitalize()


@st.cache_data(show_spinner=False)
def cached_full_waveform(song_id: int, audio_path: str):
    """Whole-song amplitude envelope, cached per song -- shared across every
    column's query/match waveform (a song that shows up more than once in
    the same session reuses the same cached array). Mirrors streamlit_app/
    pages/7_Explore.py's _cached_full_waveform exactly (same cache key
    shape), copied rather than imported for the same reason as every other
    file in this app -- see plotting.py's module docstring."""
    from sonic_explorer.analysis.waveform_preview import waveform_envelope

    return waveform_envelope(audio_path)


def _load_local_similarity_state() -> dict:
    """{} if the file doesn't exist yet (first-ever run against a fresh
    checkout) or is corrupt (e.g. hand-edited into invalid JSON) -- degrades
    to curated_examples' DEFAULT_EXAMPLE for every facet rather than
    crashing the page over a durability nicety."""
    try:
        return json.loads(LOCAL_SIMILARITY_STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_local_similarity_facet_state(facet: str, fma_track_id: int, segment_index: int) -> None:
    """Read-modify-write the whole file so updating one facet never clobbers
    the other five's already-persisted state. Best-effort: a write failure
    (e.g. a read-only filesystem on some hosting platforms) is swallowed
    rather than crashing the live demo over this."""
    state = _load_local_similarity_state()
    state[facet] = {"fma_track_id": fma_track_id, "segment_index": segment_index}
    try:
        LOCAL_SIMILARITY_STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")
    except OSError:
        pass


def persistent_song_and_moment(facet: str, all_songs: list, song_repo, default_song, default_segment_index: int):
    """One facet column's independent song+moment picker. Three-tier default
    resolution, each tier overriding the last: curated_examples'
    DEFAULT_EXAMPLE (first-ever run, nothing persisted anywhere yet) -> the
    on-disk LOCAL_SIMILARITY_STATE_PATH file (survives a full app/server
    restart -- see this module's own docstring for why st.session_state and
    st.query_params alone aren't enough) -> st.query_params (a specific URL
    visited/reloaded/shared within this browser session, the previous
    mechanism, kept as the innermost override for that narrower case). Every
    facet gets its own pair of query-param keys (song_<facet>, seg_<facet>)
    and its own widget keys, so six columns never collide with or override
    each other's persisted choice.

    Identifies the song by fma_track_id and the moment by segment_index
    (both stable across data/ vs deploy_data/ -- see curated_examples.py's
    module docstring), not by song.id/segment.id, which differ across
    environments. Returns (song, segment)."""
    song_qp, seg_qp = f"song_{facet}", f"seg_{facet}"

    default_fma = default_song.fma_track_id
    default_seg_index = default_segment_index

    persisted = _load_local_similarity_state().get(facet)
    if persisted:
        default_fma = persisted.get("fma_track_id", default_fma)
        default_seg_index = persisted.get("segment_index", default_seg_index)

    if song_qp in st.query_params:
        try:
            default_fma = int(st.query_params[song_qp])
        except ValueError:
            pass
    songs_by_fma = {s.fma_track_id: s for s in all_songs}
    if default_fma not in songs_by_fma:
        default_fma = default_song.fma_track_id
    default_song_idx = next(i for i, s in enumerate(all_songs) if s.fma_track_id == default_fma)

    labels = [f"{s.title} — {s.artist} ({s.genre_top})" for s in all_songs]
    song_idx = st.selectbox(
        "Song", options=range(len(all_songs)), index=default_song_idx, format_func=lambda i: labels[i],
        key=f"song_select_{facet}",
    )
    song = all_songs[song_idx]
    st.query_params[song_qp] = str(song.fma_track_id)

    segments = song_repo.get_segments(song.id)

    if seg_qp in st.query_params:
        try:
            candidate = int(st.query_params[seg_qp])
            if any(s.segment_index == candidate for s in segments):
                default_seg_index = candidate
        except ValueError:
            pass
    if not any(s.segment_index == default_seg_index for s in segments):
        default_seg_index = segments[0].segment_index

    # Same reset-on-song-change trick Explore's own segment pills use --
    # st.pills' default= is only honored on a widget's very first render
    # for a given key, so switching THIS column's song needs an explicit
    # session_state overwrite, scoped to this facet so it doesn't disturb
    # the other five columns' own pills.
    song_change_key = f"_last_song_{facet}"
    if st.session_state.get(song_change_key) != song.id:
        st.session_state[song_change_key] = song.id
        default_segment = next((s for s in segments if s.segment_index == default_seg_index), segments[0])
        st.session_state[f"segment_pills_{facet}"] = default_segment.id

    segment_labels = {s.id: f"{s.start_sec:.1f}–{s.end_sec:.1f}s" for s in segments}
    selected_segment_id = st.pills(
        "Moment", options=[s.id for s in segments], format_func=lambda sid: segment_labels[sid],
        required=True, key=f"segment_pills_{facet}", label_visibility="collapsed",
    )
    # Fallback (not just a bare next()) for the same reason line 192's and
    # line 202's lookups already have one: selected_segment_id comes from
    # st.pills' own widget state, which can transiently disagree with this
    # run's `segments` (e.g. the instant a song switch is being processed) --
    # degrade to this song's first segment rather than crash the page over a
    # one-rerun inconsistency that resolves itself immediately after.
    query_segment = next((s for s in segments if s.id == selected_segment_id), segments[0])
    st.query_params[seg_qp] = str(query_segment.segment_index)

    # Written only when this facet's resolved selection actually differs
    # from what's already on disk -- a plain rerun with no new interaction
    # shouldn't touch the file every single time.
    already_persisted = (
        persisted is not None
        and persisted.get("fma_track_id") == song.fma_track_id
        and persisted.get("segment_index") == query_segment.segment_index
    )
    if not already_persisted:
        _save_local_similarity_facet_state(facet, song.fma_track_id, query_segment.segment_index)

    return song, query_segment


@st.cache_data(show_spinner="Building similarity graph...")
def cached_similarity_graph(_song_repo, _embedding_repo, facet_name: str, index_size: int):
    """Per-song mean-pooled vectors -> k-NN graph (nodes + edges), mirroring
    streamlit_app/pages/7_Explore.py's build_explore_graph exactly (same
    cache-key shape: facet_name + index_size, so this rebuilds only when
    that facet's index actually changes size, not on every rerun/click).
    Measured cold against deploy_data's 233 songs: ~0.4s to mean-pool +
    ~3.0s for build_similarity_graph itself (dominated by networkx's
    spring_layout, roughly O(n^2) per iteration) -- paid once per facet per
    session, not per click; a click only changes which node/edge is
    highlighted, not the graph itself. This is also exactly why this page
    stays pinned to deploy_data rather than sonic_explorer.config.DATA_DIR's
    data/-wins-if-present resolution (same reasoning as DEPLOY_ROOT above,
    now with a second, harder reason: spring_layout's cost would grow
    roughly quadratically against data/'s full ~1400-song library, into the
    tens of seconds -- fine for deploy_data's scale, not for that one)."""
    from sonic_explorer.analysis.network_graph import build_similarity_graph
    from sonic_explorer.analysis.taste_map import mean_pool_song_vectors

    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)
    result = build_similarity_graph(song_vectors)
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    nodes_df = pd.DataFrame([
        {
            "song_id": n.song_id, "x": n.x, "y": n.y, "cluster": n.cluster,
            "title": songs_by_id[n.song_id].title, "artist": songs_by_id[n.song_id].artist,
            "genre": songs_by_id[n.song_id].genre_top,
        }
        for n in result.nodes if n.song_id in songs_by_id
    ])
    return nodes_df, result.edges


@st.cache_data(show_spinner=False)
def cached_taste_map(_song_repo, _embedding_repo, facet_name: str, index_size: int):
    """PCA projection of the same per-song vectors cached_similarity_graph
    uses -- mirrors streamlit_app/pages/7_Explore.py's build_taste_map_points,
    fixed to method="pca" (no ICA toggle -- out of scope for this page, and
    App Walkthrough already covers that comparison in the main app). Cheap:
    ~0.1s cold against deploy_data's 233 songs, sklearn PCA over an already-
    computed matrix, no separate mean-pooling pass (recomputes it here since
    st.cache_data has no way to share an intermediate result between two
    independently-keyed cached functions without a third cache entry, and at
    ~0.4s this isn't worth that complexity)."""
    from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors

    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)
    result = compute_taste_map(song_vectors, method="pca")
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    return pd.DataFrame([
        {
            "song_id": p.song_id, "x": p.x, "y": p.y, "cluster": p.cluster,
            "title": songs_by_id[p.song_id].title, "artist": songs_by_id[p.song_id].artist,
            "genre": songs_by_id[p.song_id].genre_top,
        }
        for p in result.points if p.song_id in songs_by_id
    ])


@st.cache_data
def build_dna_normalizer(_song_repo, cache_key):
    """Same pattern as streamlit_app/resources.py's function of the same
    name -- a [0,1]-per-axis normalizer fit across the whole library's raw
    DNA scalars, needed by MusicAgent's get_song_profile/search_by_mood_
    profile/compare_songs/get_random_song tools. cache_key is just
    len(song_repo.list_songs()) at the call site, the same "cheap proxy for
    the underlying data having changed" trick get_repositories()'s callers
    already use elsewhere in this app."""
    raw_stats = [{axis: getattr(s, axis) for axis in AXES} for s in _song_repo.list_songs()]
    return fit_normalizer(raw_stats)


@st.cache_data
def build_normalized_dna_by_song(_song_repo, _normalizer, cache_key):
    """Every song's DNA pre-normalized into the same [0,1]^5 space
    MusicAgent's tools compare within -- same pattern as streamlit_app/
    resources.py's function of the same name. Skips any song missing one or
    more raw DNA scalars (the structure batch pipeline hasn't run against
    it yet) rather than normalizing a partial/None-containing profile."""
    out = {}
    for s in _song_repo.list_songs():
        raw = {axis: getattr(s, axis) for axis in AXES}
        if all(v is not None for v in raw.values()):
            out[s.id] = _normalizer.normalize(raw)
    return out


def _get_anthropic_api_key() -> str | None:
    """None when unset -- the DJ agent is a value-add, never load-bearing
    for the rest of this app, same requirement CLAUDE.md's Streamlit-
    conventions section states for every LLM-backed feature in the main
    app. Checks st.secrets first (the platform secrets manager once
    deployed), falling back to an env var for local dev -- same order
    streamlit_app/resources.py's version checks in."""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    return api_key or None


@st.cache_resource
def get_agent() -> MusicAgent | None:
    """Stateless orchestrator (sonic_explorer/llm/agent.py) -- safe to share
    across sessions via cache_resource, since conversation history and the
    taste profile both live in each session's own st.session_state, never
    inside the agent itself (see agent.py's own module docstring). None
    when no API key is configured -- pages/3_Ask_The_DJ.py must degrade
    gracefully (st.info + st.stop()), never crash, on that."""
    api_key = _get_anthropic_api_key()
    if not api_key:
        return None

    import anthropic

    song_repo, embedding_repo, retrieval_service = get_repositories()
    dna_normalizer = build_dna_normalizer(song_repo, len(song_repo.list_songs()))
    normalized_dna_by_song = build_normalized_dna_by_song(song_repo, dna_normalizer, len(song_repo.list_songs()))
    return MusicAgent(
        anthropic.Anthropic(api_key=api_key),
        song_repo, embedding_repo, retrieval_service, dna_normalizer, normalized_dna_by_song,
    )
