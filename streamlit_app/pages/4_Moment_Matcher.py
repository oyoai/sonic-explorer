import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS, nearest_songs_by_dna
from sonic_explorer.analysis.taste_map import mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.retrieval.song_level_index import build_song_level_index, query_song_level
from components.plotting import song_dna_radar_overlay
from resources import (
    build_dna_normalizer,
    build_normalized_dna_by_song,
    get_explanation_client,
    get_rerank_client,
    get_repositories,
    show_data_source_banner,
)

FACET_REGISTRY = default_registry()
MAX_LLM_CALLS_PER_SESSION = 60  # simple abuse/cost guardrail for the public deployment (spec section 11)
MAX_DNA_DISTANCE = math.sqrt(len(AXES))  # every axis lives in [0,1], so this is the diagonal of the unit hypercube
RERANK_POOL_SIZE = 15  # stage-1 cosine-similarity over-fetch, reranked down to FINAL_K by the LLM
FINAL_K = 6

st.set_page_config(page_title="Moment Matcher", page_icon="\U0001F3AF")
st.title("Moment Matcher")
st.caption("Pick a moment in a song and find sonically similar moments elsewhere in the library.")
st.page_link("pages/6_Explore.py", label="← Back to Explore", icon="\U0001F310")

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()
llm_client = get_explanation_client()
rerank_client = get_rerank_client()

songs = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
if not songs:
    st.info("No songs in the library yet.")
    st.stop()

# Reached from Song X-Ray's "Find similar moments" button, which stashes the
# clicked song+segment here before switching pages -- popped so it only
# applies once; a plain sidebar/URL visit with no context still defaults to
# index 0 for both the song and the moment.
mm_context = st.session_state.pop("mm_context", None)
default_song_index = 0
if mm_context is not None:
    for i, s in enumerate(songs):
        if s.id == mm_context["song_id"]:
            default_song_index = i
            break

if "llm_calls" not in st.session_state:
    st.session_state.llm_calls = 0


@st.cache_data(show_spinner=False)
def _cached_explanation(
    _client, query_title, query_artist, query_genre, query_start, query_end,
    match_title, match_artist, match_genre, match_start, match_end, facet_name, score,
):
    return _client.generate_explanation(
        query_title=query_title, query_artist=query_artist, query_genre=query_genre,
        query_start_sec=query_start, query_end_sec=query_end,
        match_title=match_title, match_artist=match_artist, match_genre=match_genre,
        match_start_sec=match_start, match_end_sec=match_end,
        facet_name=facet_name, score=score,
    )


def explanation_for_match(client, query_song, query_seg, match) -> str | None:
    """None means "don't show an explanation" -- either no client configured,
    the per-session guardrail tripped, or the API call itself failed. Never
    raises up into the page -- the explanation is a value-add, matches must
    still render without it."""
    if client is None or st.session_state.llm_calls >= MAX_LLM_CALLS_PER_SESSION:
        return None
    st.session_state.llm_calls += 1
    try:
        return _cached_explanation(
            client,
            query_song.title, query_song.artist, query_song.genre_top,
            query_seg.start_sec, query_seg.end_sec,
            match.song.title, match.song.artist, match.song.genre_top,
            match.segment.start_sec, match.segment.end_sec,
            facet_name, max(0.0, match.score),
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _cached_rerank_order(_client, query_title, query_artist, query_genre, facet_name, candidate_tuples, k):
    candidates = [{"title": t, "artist": a, "genre": g, "score": s} for (t, a, g, s) in candidate_tuples]
    return _client.rerank(query_title, query_artist, query_genre, facet_name, candidates, k)


def get_matches(client, query_song, query_seg, facet_name, k=FINAL_K):
    """Stage 1: cosine-similarity retrieval (over-fetched to RERANK_POOL_SIZE
    when a rerank client is available). Stage 2: the LLM reorders that pool
    down to k, best-first. Falls back to plain cosine order -- no client
    configured, session budget exhausted, or the call itself failing -- so
    matches always render either way; reranking is a value-add, not
    load-bearing. Returns (matches, was_reranked)."""
    pool_size = RERANK_POOL_SIZE if client is not None else k
    candidates = retrieval_service.query_by_segment(query_seg.id, facet_name=facet_name, k=pool_size)
    if not candidates:
        return candidates, False

    if client is None or st.session_state.llm_calls >= MAX_LLM_CALLS_PER_SESSION:
        return candidates[:k], False

    st.session_state.llm_calls += 1
    try:
        candidate_tuples = tuple(
            (c.song.title, c.song.artist, c.song.genre_top, max(0.0, c.score)) for c in candidates
        )
        order = _cached_rerank_order(
            client, query_song.title, query_song.artist, query_song.genre_top, facet_name, candidate_tuples, k
        )
        return [candidates[i] for i in order], True
    except Exception:
        return candidates[:k], False


@st.cache_data(show_spinner=False)
def _cached_song_vectors(_song_repo, _embedding_repo, facet_name, index_size):
    return mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)


@st.cache_resource(show_spinner="Building song-level index...")
def _cached_song_level_index(_song_repo, _embedding_repo, facet_name, index_size):
    """A second FAISS index per facet -- see retrieval/song_level_index.py --
    IndexIDMap2 objects aren't picklable the way cache_data expects, so this
    uses cache_resource (same reasoning as get_repositories() caching the
    segment-level indexes), keyed on index_size so it rebuilds when new
    embeddings land."""
    return build_song_level_index(_song_repo, _embedding_repo, facet_name)


dna_normalizer = build_dna_normalizer(song_repo, len(songs))

mode = st.radio(
    "Query by",
    options=["existing_song", "hand_drawn_profile"],
    format_func=lambda m: "Existing song moment" if m == "existing_song" else "Hand-drawn profile",
    horizontal=True,
    help="'Existing song moment' matches on a specific ~5s clip using the selected facet's embeddings. "
         "'Hand-drawn profile' instead lets you sculpt a target (tempo, energy, brightness, etc.) "
         "and finds whole songs closest to that shape.",
)

if mode == "existing_song":
    labels = [f"{s.title} — {s.artist} ({s.genre_top})" for s in songs]
    song_choice = st.selectbox(
        "Song", options=range(len(songs)), index=default_song_index, format_func=lambda i: labels[i]
    )
    song = songs[song_choice]

    granularity = st.radio(
        "Match against",
        options=["moment", "whole_song"],
        format_func=lambda g: "A specific moment" if g == "moment" else "Whole songs",
        horizontal=True,
        help="'A specific moment' ranks individual ~5s clips. 'Whole songs' mean-pools each song's "
             "segments first, then ranks whole songs -- sharper, more decisive rankings library-wide "
             "(see Methodology §7d), at the cost of moment-level precision.",
    )

    segments = song_repo.get_segments(song.id)
    if not segments:
        st.warning("This song has no segments yet.")
        st.stop()

    if granularity == "moment":
        default_moment_index = 0
        if mm_context is not None and mm_context["song_id"] == song.id:
            for i, seg in enumerate(segments):
                if seg.id == mm_context["segment_id"]:
                    default_moment_index = i
                    break

        moment_labels = [f"{seg.start_sec:.1f}s – {seg.end_sec:.1f}s" for seg in segments]
        moment_choice = st.select_slider(
            "Moment", options=range(len(segments)), value=default_moment_index, format_func=lambda i: moment_labels[i]
        )
        query_segment = segments[moment_choice]

        st.markdown(f"**Listening at {query_segment.start_sec:.1f}s – {query_segment.end_sec:.1f}s:**")
        st.audio(str(audio_path_for(song)), start_time=query_segment.start_sec)
    else:
        query_segment = None
        st.markdown("**Listening to the whole song:**")
        st.audio(str(audio_path_for(song)))

    facet_name = st.radio(
        "Match by",
        options=FACET_REGISTRY.names(),
        format_func=lambda f: f.capitalize(),
        horizontal=True,
    )
    st.markdown(f"### Match by: {facet_name.capitalize()}")

    query_raw = {axis: getattr(song, axis) for axis in AXES}
    query_has_dna = all(v is not None for v in query_raw.values())

    if granularity == "moment":
        if embedding_repo.status(query_segment.id, facet_name) != "done":
            st.warning(f"This moment hasn't been embedded for the {facet_name.capitalize()} facet yet.")
            st.stop()

        matches, was_reranked = get_matches(rerank_client, song, query_segment, facet_name)

        if not matches:
            st.info("No matches found elsewhere in the library yet.")
        else:
            if llm_client is None:
                st.caption("Set ANTHROPIC_API_KEY to also get a plain-language explanation for each match.")
            if was_reranked:
                st.caption("Results re-ranked by Claude for overall fit, not just raw similarity.")

            for match in matches:
                pct = max(0.0, match.score) * 100
                st.markdown(f"**{pct:.0f}% {facet_name} match** — {match.song.title} by {match.song.artist} ({match.song.genre_top})")
                st.caption(f"at {match.segment.start_sec:.1f}s – {match.segment.end_sec:.1f}s")
                st.audio(str(audio_path_for(match.song)), start_time=match.segment.start_sec)

                explanation = explanation_for_match(llm_client, song, query_segment, match)
                if explanation:
                    st.markdown(f"\U0001F4AC *{explanation}*")

                match_raw = {axis: getattr(match.song, axis) for axis in AXES}
                if query_has_dna and all(v is not None for v in match_raw.values()):
                    with st.expander("Compare song DNA"):
                        st.caption(
                            "Where the shapes overlap, the songs agree on that quality; where one bulges "
                            "past the other, they diverge."
                        )
                        query_norm = dna_normalizer.normalize(query_raw)
                        match_norm = dna_normalizer.normalize(match_raw)
                        fig = song_dna_radar_overlay(
                            axis_labels=[AXIS_LABELS[a] for a in AXES],
                            values_a=[query_norm[a] for a in AXES], label_a=song.title,
                            values_b=[match_norm[a] for a in AXES], label_b=match.song.title,
                        )
                        st.plotly_chart(fig, width="stretch", key=f"dna_radar_{match.segment.id}")
                st.divider()
    else:  # whole_song
        song_vectors = _cached_song_vectors(song_repo, embedding_repo, facet_name, embedding_repo.index_size(facet_name))
        if song.id not in song_vectors:
            st.warning(f"This song hasn't been embedded for the {facet_name.capitalize()} facet yet.")
            st.stop()

        index = _cached_song_level_index(song_repo, embedding_repo, facet_name, embedding_repo.index_size(facet_name))
        results = query_song_level(index, song_vectors[song.id], k=FINAL_K, exclude_song_id=song.id)

        if not results:
            st.info("No matches found elsewhere in the library yet.")
        else:
            st.caption(
                "Whole-song mode ranks by mean-pooled similarity across each song's segments -- no "
                "LLM re-ranking or explanations here (those need a specific moment); see \"A specific "
                "moment\" above for those."
            )
            songs_by_id = {s.id: s for s in songs}
            for match_song_id, score in results:
                match_song = songs_by_id.get(match_song_id)
                if match_song is None:
                    continue
                pct = max(0.0, score) * 100
                st.markdown(f"**{pct:.0f}% {facet_name} match (whole song)** — {match_song.title} by {match_song.artist} ({match_song.genre_top})")
                st.audio(str(audio_path_for(match_song)))

                match_raw = {axis: getattr(match_song, axis) for axis in AXES}
                if query_has_dna and all(v is not None for v in match_raw.values()):
                    with st.expander("Compare song DNA"):
                        st.caption(
                            "Where the shapes overlap, the songs agree on that quality; where one bulges "
                            "past the other, they diverge."
                        )
                        query_norm = dna_normalizer.normalize(query_raw)
                        match_norm = dna_normalizer.normalize(match_raw)
                        fig = song_dna_radar_overlay(
                            axis_labels=[AXIS_LABELS[a] for a in AXES],
                            values_a=[query_norm[a] for a in AXES], label_a=song.title,
                            values_b=[match_norm[a] for a in AXES], label_b=match_song.title,
                        )
                        st.plotly_chart(fig, width="stretch", key=f"dna_radar_song_{match_song_id}")
                st.divider()

else:  # hand_drawn_profile
    normalized_by_song = build_normalized_dna_by_song(song_repo, dna_normalizer, len(songs))
    if not normalized_by_song:
        st.info("No songs with computed DNA yet -- run the structure batch pipeline first.")
        st.stop()

    st.caption(
        "Sculpt a target profile below -- the search re-runs automatically whenever you release a slider."
    )
    target = {}
    slider_cols = st.columns(len(AXES))
    for axis, col in zip(AXES, slider_cols, strict=False):
        with col:
            target[axis] = st.slider(AXIS_LABELS[axis], min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"dna_slider_{axis}")

    matches = nearest_songs_by_dna(target, normalized_by_song, k=6)
    songs_by_id = {s.id: s for s in songs}

    if not matches:
        st.info("No matches found.")
        st.stop()

    top_match = matches[0]
    top_song = songs_by_id[top_match.song_id]

    st.markdown(f"### Closest match: {top_song.title} — {top_song.artist}")
    st.caption(f"Genre: {top_song.genre_top}")
    st.audio(str(audio_path_for(top_song)))

    fig = song_dna_radar_overlay(
        axis_labels=[AXIS_LABELS[a] for a in AXES],
        values_a=[target[a] for a in AXES], label_a="Your target",
        values_b=[normalized_by_song[top_match.song_id][a] for a in AXES], label_b=top_song.title,
    )
    st.plotly_chart(fig, width="stretch", key="dna_query_radar")

    st.markdown("#### Other close matches")
    for match in matches[1:]:
        song = songs_by_id[match.song_id]
        closeness_pct = max(0.0, 1.0 - match.distance / MAX_DNA_DISTANCE) * 100
        st.markdown(f"**{closeness_pct:.0f}% close** — {song.title} by {song.artist} ({song.genre_top})")
        st.audio(str(audio_path_for(song)))
        st.divider()
