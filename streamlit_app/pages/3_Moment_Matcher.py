import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS, fit_normalizer
from sonic_explorer.config import audio_path_for
from components.plotting import song_dna_radar_overlay
from resources import get_repositories, show_data_source_banner

st.set_page_config(page_title="Moment Matcher", page_icon="\U0001F3AF")
st.title("Moment Matcher")
st.caption("Pick a moment in a song and find sonically similar moments elsewhere in the library.")

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()

songs = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
if not songs:
    st.info("No songs in the library yet.")
    st.stop()


@st.cache_data
def build_dna_normalizer(_song_repo, cache_key):
    raw_stats = [
        {axis: getattr(s, axis) for axis in AXES}
        for s in _song_repo.list_songs()
    ]
    return fit_normalizer(raw_stats)


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
    query_raw = {axis: getattr(song, axis) for axis in AXES}
    query_has_dna = all(v is not None for v in query_raw.values())

    for match in matches:
        pct = max(0.0, match.score) * 100
        match_word = FACET_LABELS[facet_name].lower()
        st.markdown(f"**{pct:.0f}% {match_word} match** — {match.song.title} by {match.song.artist} ({match.song.genre_top})")
        st.caption(f"at {match.segment.start_sec:.1f}s – {match.segment.end_sec:.1f}s")
        st.audio(str(audio_path_for(match.song)), start_time=match.segment.start_sec)

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
