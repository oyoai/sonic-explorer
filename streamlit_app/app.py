import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from resources import get_repositories, show_data_source_banner

st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7", layout="wide")

st.title("Sonic Explorer")
st.write(
    "Explore your music library by how it actually sounds — not tags or genre labels."
)

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()

if all_songs:
    genres = sorted({s.genre_top for s in all_songs})
    stat_cols = st.columns(3)
    stat_cols[0].metric("Songs in library", len(all_songs))
    stat_cols[1].metric("Genres", len(genres))
    stat_cols[2].metric("Embedded segments (sound facet)", embedding_repo.index_size("sound"))

st.divider()

# ---------------------------------------------------------------------------
# 1. Project introduction
# ---------------------------------------------------------------------------
st.header("1. What this is")
st.write(
    "Most music discovery tools lean on metadata: genre tags, artist graphs, playlist "
    "co-occurrence. That's useful, but it means two songs get called \"similar\" because of "
    "who made them or how a label described them — not because of what they actually sound "
    "like. It also means two songs that sound remarkably alike but sit in different genre "
    "buckets never get connected."
)
st.write(
    "Sonic Explorer starts from the opposite direction: analyze the audio directly. Every "
    "song is broken into several independent **facets** — overall sound/timbre, harmony, "
    "isolated vocals, drums, bass, backing instrumentation, and structural shape — using "
    "pretrained audio embedding models and signal-processing techniques, with genre labels "
    "never entering the similarity computation itself. Genre is kept around only afterward, "
    "as an evaluation yardstick: do a facet's nearest neighbors share a genre more often "
    "than chance would predict? That's a check on whether the audio-based approach is "
    "finding real signal — not the mechanism generating the matches."
)
st.write(
    "From those facets, the app builds several ways to explore a library: per-song "
    "\"DNA\" and visual fingerprints, a 2D map of the whole collection, moment-to-moment "
    "matching on any facet, a conversational front-end over all of it, and free-form "
    "exploration. The rest of this page sets up the problem before the next pages walk "
    "through how it was actually built and how well it works."
)

st.subheader("1.1 The naive approach — and why it falls short")
st.info(
    "**Placeholder.** This subsection should show a concrete naive baseline (e.g. "
    "genre-tag-based \"similarity\" — group songs by their label and call that similar) "
    "running against a couple of real examples, to make the motivation tangible before the "
    "facet-based approach is introduced. Not written yet — needs a real worked example "
    "rather than an asserted claim, so it's left as a stub instead of guessed at.",
    icon="\U0001F6A7",
)

st.subheader("1.2 Related work")
st.info(
    "**Placeholder.** This subsection should briefly place Sonic Explorer against existing "
    "content-based music-information-retrieval approaches and commercial systems (e.g. "
    "audio-feature-based recommendation, embedding-based audio search) — a few citations and "
    "a short contrast, not a full literature review. Not written yet — left as a stub rather "
    "than filled with unverified claims.",
    icon="\U0001F6A7",
)

st.divider()

st.write(
    "Next: **Methodology** walks through how the library was actually analyzed and "
    "improved, with real evidence at each step."
)
st.page_link("pages/0_Methodology.py", label="**Continue to Methodology →**", icon="\U0001F52C")
