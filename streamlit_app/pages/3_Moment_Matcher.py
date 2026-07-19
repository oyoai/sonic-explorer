import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.config import audio_path_for
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

st.markdown("### Match by: Sound")

if embedding_repo.status(query_segment.id, "sound") != "done":
    st.warning("This moment hasn't been embedded yet.")
    st.stop()

matches = retrieval_service.query_by_segment(query_segment.id, facet_name="sound", k=6)

if not matches:
    st.info("No matches found elsewhere in the library yet.")
else:
    for match in matches:
        pct = max(0.0, match.score) * 100
        st.markdown(f"**{pct:.0f}% sonic match** — {match.song.title} by {match.song.artist} ({match.song.genre_top})")
        st.caption(f"at {match.segment.start_sec:.1f}s – {match.segment.end_sec:.1f}s")
        st.audio(str(audio_path_for(match.song)), start_time=match.segment.start_sec)
        st.divider()
