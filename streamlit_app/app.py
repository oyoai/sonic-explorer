import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from resources import get_repositories, show_data_source_banner

st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7")

st.title("Sonic Explorer")
st.write("Explore your music library by how it actually sounds -- not tags or genre labels.")

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()

col1, col2 = st.columns(2)
col1.metric("Songs in library", len(song_repo.list_songs()))
col2.metric("Embedded segments (sound)", embedding_repo.index_size("sound"))

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link("pages/1_Taste_Map.py", label="**Taste Map**", icon="\U0001F5FA️")
    st.caption("A 2D map of the library clustered by sound. Click a point to hear it.")
with col2:
    st.page_link("pages/2_Song_XRay.py", label="**Song X-Ray**", icon="\U0001F50D")
    st.caption("See a song's structural anatomy -- verse/chorus shape as a self-similarity matrix.")
with col3:
    st.page_link("pages/3_Moment_Matcher.py", label="**Moment Matcher**", icon="\U0001F3AF")
    st.caption("Pick a moment in a song, get ranked sonically-similar moments elsewhere.")
with col4:
    st.page_link("pages/4_Ask_The_DJ.py", label="**Ask the DJ**", icon="\U0001F399️")
    st.caption("Describe what you want in plain language -- a conversational front-end over the above.")
