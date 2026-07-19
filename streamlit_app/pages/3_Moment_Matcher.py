import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS, fit_normalizer
from sonic_explorer.config import audio_path_for
from components.plotting import song_dna_radar_overlay
from resources import get_explanation_client, get_repositories, show_data_source_banner

MAX_EXPLANATIONS_PER_SESSION = 60  # simple abuse/cost guardrail for the public deployment (spec section 11)

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
