import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS, fit_normalizer, nearest_songs_by_dna
from sonic_explorer.config import audio_path_for
from components.plotting import song_dna_radar_overlay
from resources import get_explanation_client, get_repositories, show_data_source_banner

MAX_EXPLANATIONS_PER_SESSION = 60  # simple abuse/cost guardrail for the public deployment (spec section 11)
MAX_DNA_DISTANCE = math.sqrt(len(AXES))  # every axis lives in [0,1], so this is the diagonal of the unit hypercube

st.set_page_config(page_title="Moment Matcher", page_icon="\U0001F3AF")
st.title("Moment Matcher")
st.caption("Pick a moment in a song and find sonically similar moments elsewhere in the library.")

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()
llm_client = get_explanation_client()

songs = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
if not songs:
    st.info("No songs in the library yet.")
    st.stop()

if "explanation_calls" not in st.session_state:
    st.session_state.explanation_calls = 0


@st.cache_data
def build_dna_normalizer(_song_repo, cache_key):
    raw_stats = [
        {axis: getattr(s, axis) for axis in AXES}
        for s in _song_repo.list_songs()
    ]
    return fit_normalizer(raw_stats)


@st.cache_data
def build_normalized_dna_by_song(_song_repo, _normalizer, cache_key):
    """Every song's DNA, pre-normalized into the same [0,1]^5 space a
    hand-drawn target lives in -- nearest_songs_by_dna() just does distance
    + ranking over this, no new infrastructure (spec 2.3)."""
    out = {}
    for s in _song_repo.list_songs():
        raw = {axis: getattr(s, axis) for axis in AXES}
        if all(v is not None for v in raw.values()):
            out[s.id] = _normalizer.normalize(raw)
    return out


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
    if client is None or st.session_state.explanation_calls >= MAX_EXPLANATIONS_PER_SESSION:
        return None
    st.session_state.explanation_calls += 1
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


dna_normalizer = build_dna_normalizer(song_repo, len(songs))

mode = st.radio(
    "Query by",
    options=["existing_song", "hand_drawn_profile"],
    format_func=lambda m: "Existing song moment" if m == "existing_song" else "Hand-drawn profile",
    horizontal=True,
    help="'Existing song moment' matches on a specific ~5s clip using sound/harmony embeddings. "
         "'Hand-drawn profile' instead lets you sculpt a target (tempo, energy, brightness, etc.) "
         "and finds whole songs closest to that shape.",
)

if mode == "existing_song":
    labels = [f"{s.title} — {s.artist} ({s.genre_top})" for s in songs]
    song_choice = st.selectbox("Song", options=range(len(songs)), format_func=lambda i: labels[i])
    song = songs[song_choice]

    segments = song_repo.get_segments(song.id)
    if not segments:
        st.warning("This song has no segments yet.")
        st.stop()

    moment_labels = [f"{seg.start_sec:.1f}s – {seg.end_sec:.1f}s" for seg in segments]
    moment_choice = st.select_slider("Moment", options=range(len(segments)), format_func=lambda i: moment_labels[i])
    query_segment = segments[moment_choice]

    st.markdown(f"**Listening at {query_segment.start_sec:.1f}s – {query_segment.end_sec:.1f}s:**")
    st.audio(str(audio_path_for(song)), start_time=query_segment.start_sec)

    FACET_LABELS = {"sound": "Sound", "harmony": "Harmony"}
    facet_name = st.radio(
        "Match by",
        options=["sound", "harmony"],
        format_func=lambda f: FACET_LABELS[f],
        horizontal=True,
    )
    st.markdown(f"### Match by: {FACET_LABELS[facet_name]}")

    if embedding_repo.status(query_segment.id, facet_name) != "done":
        st.warning(f"This moment hasn't been embedded for the {FACET_LABELS[facet_name]} facet yet.")
        st.stop()

    matches = retrieval_service.query_by_segment(query_segment.id, facet_name=facet_name, k=6)

    if not matches:
        st.info("No matches found elsewhere in the library yet.")
    else:
        if llm_client is None:
            st.caption("Set ANTHROPIC_API_KEY to also get a plain-language explanation for each match.")

        query_raw = {axis: getattr(song, axis) for axis in AXES}
        query_has_dna = all(v is not None for v in query_raw.values())

        for match in matches:
            pct = max(0.0, match.score) * 100
            match_word = FACET_LABELS[facet_name].lower()
            st.markdown(f"**{pct:.0f}% {match_word} match** — {match.song.title} by {match.song.artist} ({match.song.genre_top})")
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
    for axis, col in zip(AXES, slider_cols):
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
