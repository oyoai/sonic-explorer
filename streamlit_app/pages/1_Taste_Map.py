import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from resources import get_repositories

st.set_page_config(page_title="Taste Map", page_icon="\U0001F5FA️")
st.title("Taste Map")
st.caption("A 2D map of the library clustered by sonic character. Click a point to hear it.")

song_repo, embedding_repo, retrieval_service = get_repositories()


@st.cache_data
def build_taste_map_df(_song_repo, _embedding_repo, cache_key):
    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    result = compute_taste_map(song_vectors)
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    return pd.DataFrame([
        {
            "song_id": p.song_id,
            "x": p.x,
            "y": p.y,
            "cluster": str(p.cluster),
            "title": songs_by_id[p.song_id].title,
            "artist": songs_by_id[p.song_id].artist,
            "genre": songs_by_id[p.song_id].genre_top,
        }
        for p in result.points
        if p.song_id in songs_by_id
    ])


# index_size is a crude but effective cache key -- recompute only when the
# synced embeddings actually change (new batch, more segments done)
df = build_taste_map_df(song_repo, embedding_repo, embedding_repo.index_size("sound"))

if df.empty:
    st.info("No embedded songs yet. Run the batch embedding pipeline first.")
    st.stop()

fig = px.scatter(
    df, x="x", y="y", color="cluster", custom_data=["song_id"],
    hover_data={"title": True, "artist": True, "genre": True, "x": False, "y": False, "cluster": False},
    title=f"{len(df)} songs, clustered by sound",
)
fig.update_traces(marker=dict(size=10))

event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="taste_map_chart")

selected_song_id = None
if event and event.selection and event.selection.points:
    selected_song_id = event.selection.points[0]["customdata"][0]

if selected_song_id is not None:
    song = song_repo.get_song(int(selected_song_id))
    st.subheader(f"{song.title} — {song.artist}")
    st.caption(f"Genre: {song.genre_top}")
    st.audio(song.filepath)
else:
    st.info("Click a point on the map to hear that song.")
