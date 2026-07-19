import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from resources import get_repositories, show_data_source_banner

st.set_page_config(page_title="Taste Map", page_icon="\U0001F5FA️")
st.title("Taste Map")
st.caption("A 2D map of the library clustered by sonic character. Click a point to hear it.")

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()


@st.cache_data
def build_taste_map_df(_song_repo, _embedding_repo, cache_key, method):
    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    result = compute_taste_map(song_vectors, method=method)
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


method = st.radio(
    "Projection", options=["pca", "ica"], format_func=lambda m: m.upper(), horizontal=True,
    help="PCA finds the directions of maximum spread across the library. ICA looks for statistically "
         "independent directions instead -- sometimes these land on more individually-nameable "
         "qualities than PCA's axes, sometimes they're just as hard to name. Use 'Inspect these axes' "
         "below to judge for a given library, don't just take the claim on faith.",
)

# index_size is a crude but effective cache key -- recompute only when the
# synced embeddings actually change (new batch, more segments done)
df = build_taste_map_df(song_repo, embedding_repo, embedding_repo.index_size("sound"), method)

if df.empty:
    st.info("No embedded songs yet. Run the batch embedding pipeline first.")
    st.stop()

color_by = st.radio(
    "Color by", options=["cluster", "genre"], horizontal=True,
    help="'cluster' is what the map discovers on its own (K-means over sound embeddings, no genre "
         "labels involved). 'genre' overlays the known labels -- compare the two to see whether "
         "sonic clusters actually line up with genre or cut across it.",
)

with st.expander(f"Inspect these axes ({method.upper()})"):
    st.caption(
        "The songs at each axis's extremes -- listen to a few from each end and see if you'd name "
        "what that axis represents (e.g. 'quiet/ambient vs. loud/aggressive'), or if it doesn't "
        "resolve into anything nameable. This is the actual test of whether an axis is interpretable, "
        "not just an assumption about the method."
    )
    axis_col1, axis_col2 = st.columns(2)
    for axis, col in [("x", axis_col1), ("y", axis_col2)]:
        with col:
            st.markdown(f"**Axis: {axis}**")
            st.caption("Highest")
            for _, row in df.nlargest(3, axis).iterrows():
                st.caption(f"{row['title']} — {row['artist']} ({row['genre']})")
            st.caption("Lowest")
            for _, row in df.nsmallest(3, axis).iterrows():
                st.caption(f"{row['title']} — {row['artist']} ({row['genre']})")

fig = px.scatter(
    df, x="x", y="y", color=color_by, custom_data=["song_id"],
    hover_data={"title": True, "artist": True, "genre": True, "x": False, "y": False, "cluster": False},
    title=f"{len(df)} songs, colored by {color_by} ({method.upper()} projection)",
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
    st.audio(str(audio_path_for(song)))
else:
    st.info("Click a point on the map to hear that song.")
