import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from resources import get_repositories

st.set_page_config(page_title="Song X-Ray", page_icon="\U0001F50D")
st.title("Song X-Ray")
st.caption("A song's structural anatomy -- repeated sections show up as bright stripes off the main diagonal.")

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
st.audio(song.filepath)

st.markdown("#### Structure (self-similarity matrix)")
try:
    matrix = embedding_repo.get_structure_matrix(song.id)
    heatmap = px.imshow(
        matrix, color_continuous_scale="Magma", origin="lower",
        labels=dict(x="beat", y="beat", color="similarity"),
    )
    heatmap.update_layout(height=420)
    st.plotly_chart(heatmap, width="stretch")
    st.caption("Bright parallel stripes off the main diagonal mark repeated sections (verse/chorus).")
except FileNotFoundError:
    st.warning("No structure matrix computed for this song yet.")

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
