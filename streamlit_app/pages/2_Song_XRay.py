import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from resources import get_repositories, show_data_source_banner

st.set_page_config(page_title="Song X-Ray", page_icon="\U0001F50D")
st.title("Song X-Ray")
st.caption("A song's structural anatomy -- matching colors below mean similar-sounding sections.")

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()

songs = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
if not songs:
    st.info("No songs in the library yet.")
    st.stop()

labels = [f"{s.title} — {s.artist} ({s.genre_top})" for s in songs]
choice = st.selectbox("Pick a song", options=range(len(songs)), format_func=lambda i: labels[i])
song = songs[choice]

st.subheader(f"{song.title} — {song.artist}")
st.caption(f"Genre: {song.genre_top}")
st.audio(str(audio_path_for(song)))

st.markdown("#### Structure timeline")
st.caption(
    "Each colored block is a stretch of the song. Same color = similar-sounding sections "
    "(e.g. a verse repeating later) -- discovered automatically from the audio, not labeled by us."
)
try:
    timeline = embedding_repo.get_structure_timeline(song.id)

    palette = px.colors.qualitative.Set2
    unique_labels = sorted(set(timeline.segment_labels.tolist()))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    durations = timeline.segment_ends - timeline.segment_starts
    hover_text = [f"{s:.1f}s – {e:.1f}s" for s, e in zip(timeline.segment_starts, timeline.segment_ends)]

    timeline_fig = go.Figure(go.Bar(
        x=durations,
        y=["Structure"] * len(durations),
        base=timeline.segment_starts,
        orientation="h",
        marker_color=[color_map[lab] for lab in timeline.segment_labels.tolist()],
        marker_line_width=0,
        hovertext=hover_text,
        hoverinfo="text",
    ))
    timeline_fig.update_layout(
        height=140,
        showlegend=False,
        xaxis_title="Time (s)",
        yaxis=dict(showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=40),
        bargap=0,
    )
    st.plotly_chart(timeline_fig, width="stretch")
except FileNotFoundError:
    st.warning("No structure timeline computed for this song yet.")

st.markdown("#### Position in the Taste Map")


@st.cache_data
def build_taste_map_df(_song_repo, _embedding_repo, cache_key):
    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    result = compute_taste_map(song_vectors)
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    return pd.DataFrame([
        {
            "song_id": p.song_id, "x": p.x, "y": p.y, "cluster": str(p.cluster),
            "title": songs_by_id[p.song_id].title, "genre": songs_by_id[p.song_id].genre_top,
        }
        for p in result.points
        if p.song_id in songs_by_id
    ])


taste_df = build_taste_map_df(song_repo, embedding_repo, embedding_repo.index_size("sound"))
this_song = taste_df[taste_df["song_id"] == song.id]

if not this_song.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=taste_df["x"], y=taste_df["y"], mode="markers",
        marker=dict(size=7, color="lightgray"), name="library", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=this_song["x"], y=this_song["y"], mode="markers",
        marker=dict(size=16, color="crimson", symbol="star"), name=song.title,
    ))
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("Not enough embedded segments yet to place this song on the Taste Map.")

with st.expander("Technical detail: raw self-similarity matrix"):
    st.caption(
        "The matrix the timeline above is derived from. Every moment matches itself perfectly, so the "
        "main diagonal is always brightest; bright parallel stripes off the diagonal mark repeated sections."
    )
    try:
        matrix = embedding_repo.get_structure_matrix(song.id)
        heatmap = px.imshow(
            matrix, color_continuous_scale="Magma", origin="lower",
            labels=dict(x="beat", y="beat", color="similarity"),
        )
        heatmap.update_layout(height=420)
        st.plotly_chart(heatmap, width="stretch")
    except FileNotFoundError:
        st.warning("No structure matrix computed for this song yet.")
