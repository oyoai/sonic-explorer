import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from resources import get_repositories

st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7")

st.title("Sonic Explorer")
st.write("Explore your music library by how it actually sounds -- not tags or genre labels.")
st.write("Use the sidebar to open **Taste Map** or **Moment Matcher**.")

song_repo, embedding_repo, retrieval_service = get_repositories()

col1, col2 = st.columns(2)
col1.metric("Songs in library", len(song_repo.list_songs()))
col2.metric("Embedded segments (sound)", embedding_repo.index_size("sound"))
