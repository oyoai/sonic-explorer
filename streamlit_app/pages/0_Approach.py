"""Approach -- a bridge page between Overview and Methodology answering "why
does this pipeline look the way it does" visually, before Methodology dives
into technical depth and evidence. Every step below is built from real
audio/data already in the library (a real waveform, a real embedding vector,
the real audio-similarity graph) rather than abstract UI-box illustrations,
so the mechanic is shown, not just described.

Animation style: lightweight and Streamlit-native (Plotly + interactive
sliders/buttons the user drives), not custom Canvas/JS. A fully custom,
cinematic animation per step would be real bespoke frontend engineering --
7 steps at even a modest 45-90 min each is 5-10 hours, likely the single
biggest remaining piece of this restructure. This trades some visual polish
for a fraction of the build cost and zero new engineering risk.

Step 3 ("six ways of listening") is a schematic, not literally separated
audio: isolated stem *audio* was never persisted anywhere in this project
(only stem embeddings were computed and stored), and running Demucs live on
page load isn't reasonable. The same real waveform is shown across six
labeled lanes instead -- honest about what it is, not presented as literally
separated stems."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from comparison_data import build_naive_vs_real_graphs, get_demo_pairs
from components.plotting import network_graph_figure, waveform_figure
from resources import get_explanation_client, get_repositories, show_data_source_banner, show_logo
from sonic_explorer.analysis.waveform_preview import waveform_envelope
from sonic_explorer.config import WINDOW_SEC, audio_path_for

st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7", layout="wide")
show_logo()
show_data_source_banner()

st.title("Approach")
st.write(
    "Overview made the case for analyzing audio directly instead of trusting tags. Here's how "
    "that actually works, one step at a time -- real audio and real data at every step, not "
    "abstract diagrams. **Methodology** walks through each of these in full technical depth "
    "afterward, with evidence."
)

song_repo, embedding_repo, _ = get_repositories()
all_songs = song_repo.list_songs()
songs_by_id = {s.id: s for s in all_songs}

if not all_songs:
    st.info("No songs available yet to build this walkthrough.", icon="\U0001F6A7")
    st.stop()

naive_nodes, naive_edges, real_nodes, real_edges, vectors, genre_by_song = build_naive_vs_real_graphs(
    song_repo, embedding_repo, len(all_songs)
)
naive_pair, real_pair = get_demo_pairs(song_repo, embedding_repo, len(all_songs))

st.divider()

# ---------------------------------------------------------------------------
# Step 1: the problem, restated visually
# ---------------------------------------------------------------------------
st.header("1. The problem, restated visually")
st.write("Genre lies. The waveform doesn't. The same two pairs from Overview -- now look at them:")

if naive_pair is not None and real_pair is not None:
    a1, b1 = songs_by_id[naive_pair.song_id_a], songs_by_id[naive_pair.song_id_b]
    a2, b2 = songs_by_id[real_pair.song_id_a], songs_by_id[real_pair.song_id_b]

    st.caption(f"**Same genre tag ({a1.genre_top}), sound different** -- similarity {naive_pair.audio_similarity:.2f}")
    row1 = st.columns(2)
    with row1[0]:
        st.plotly_chart(
            waveform_figure(waveform_envelope(audio_path_for(a1)), title=a1.title, color="rgb(239,85,59)"),
            width="stretch", key="step1_naive_a",
        )
    with row1[1]:
        st.plotly_chart(
            waveform_figure(waveform_envelope(audio_path_for(b1)), title=b1.title, color="rgb(239,85,59)"),
            width="stretch", key="step1_naive_b",
        )

    st.caption(
        f"**Different genres ({a2.genre_top} / {b2.genre_top}), sound similar** -- "
        f"similarity {real_pair.audio_similarity:.2f}"
    )
    row2 = st.columns(2)
    with row2[0]:
        st.plotly_chart(
            waveform_figure(waveform_envelope(audio_path_for(a2)), title=a2.title, color="rgb(99,110,250)"),
            width="stretch", key="step1_real_a",
        )
    with row2[1]:
        st.plotly_chart(
            waveform_figure(waveform_envelope(audio_path_for(b2)), title=b2.title, color="rgb(99,110,250)"),
            width="stretch", key="step1_real_b",
        )
else:
    st.info("Not enough graph data yet to show this comparison.", icon="\U0001F6A7")

st.divider()

# ---------------------------------------------------------------------------
# Step 2: segmentation
# ---------------------------------------------------------------------------
st.header("2. Slicing the track into windows")
seg_song = songs_by_id.get(real_pair.song_id_a) if real_pair is not None else all_songs[0]
duration = seg_song.duration_sec or 30.0
hop_sec = 2.5
st.write(
    f"Every song is sliced into overlapping {WINDOW_SEC:.0f}-second windows, {hop_sec:.1f}s apart -- "
    f"{song_repo.count_segments()} such windows across the whole library right now. Drag to slide "
    f"the window across \"{seg_song.title}\":"
)
max_start = max(0.0, duration - WINDOW_SEC)
window_start = st.slider(
    "Window start (seconds)", min_value=0.0, max_value=float(max_start), value=0.0, step=hop_sec, key="step2_slider"
)
st.plotly_chart(
    waveform_figure(
        waveform_envelope(audio_path_for(seg_song)),
        duration_sec=duration, highlight_range=(window_start, window_start + WINDOW_SEC),
        color="rgb(99,110,250)", height=160,
    ),
    width="stretch", key="step2_waveform",
)

st.divider()

# ---------------------------------------------------------------------------
# Step 3: six ways of listening (facets)
# ---------------------------------------------------------------------------
st.header("3. Six ways of listening")
st.write(
    "Each window gets measured six different ways -- not six separate audio files (isolating "
    "stems for every song was done once, offline, for embeddings only, not kept as playable "
    "audio), but six independent numeric \"lenses\" on the same real audio below:"
)
facet_lanes = [
    ("Sound", "Overall timbre, instrumentation, production character", "rgb(99,110,250)"),
    ("Harmony", "Key, chords, tonal color", "rgb(0,204,150)"),
    ("Vocal", "Isolated voice timbre and delivery", "rgb(239,85,59)"),
    ("Drums", "Isolated drum/percussion pattern and timbre", "rgb(171,99,250)"),
    ("Bass", "Isolated bassline tone and pattern", "rgb(255,161,90)"),
    ("Instrumental", "Backing instrumentation with vocals removed", "rgb(25,211,243)"),
]
lane_song = seg_song
lane_envelope = waveform_envelope(audio_path_for(lane_song))
lane_cols = st.columns(3)
for i, (name, desc, color) in enumerate(facet_lanes):
    with lane_cols[i % 3]:
        st.plotly_chart(
            waveform_figure(lane_envelope, title=name, color=color, height=100),
            width="stretch", key=f"step3_lane_{name}",
        )
        st.caption(desc)
st.caption(
    f"One song (\"{lane_song.title}\"), six independently-computed similarity spaces -- a match on "
    "Vocal doesn't require a match on Drums, and vice versa."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 4: embeddings
# ---------------------------------------------------------------------------
st.header("4. Turning sound into points in space")
embed_song = seg_song
if embed_song.id in vectors:
    vec = vectors[embed_song.id]
    st.write(
        f"\"{embed_song.title}\"'s Sound facet collapses down to **{len(vec)} numbers** -- a single "
        "point in space. Two songs that sound alike land close together; two that don't, land far "
        "apart. That's the entire mechanism behind every comparison on this page."
    )
    preview_dims = min(48, len(vec))
    st.plotly_chart(
        waveform_figure(
            np.abs(vec[:preview_dims]), title=f"First {preview_dims} of {len(vec)} dimensions",
            color="rgb(0,204,150)", height=120,
        ),
        width="stretch", key="step4_vector",
    )
else:
    st.info("No embedding available yet for this song.", icon="\U0001F6A7")

st.divider()

# ---------------------------------------------------------------------------
# Step 5: retrieval
# ---------------------------------------------------------------------------
st.header("5. Finding what's similar")
st.write(
    "Once every song is a point, \"similar\" just means \"nearby.\" Below is the real audio-"
    "similarity graph from Overview, with one song highlighted -- its edges are its actual "
    "nearest neighbors, the same mechanic Moment Matcher uses on any single moment:"
)
if not real_nodes.empty and real_pair is not None:
    st.plotly_chart(
        network_graph_figure(real_nodes, real_edges, selected_song_id=real_pair.song_id_a),
        width="stretch", key="step5_retrieval_graph",
    )
else:
    st.info("Not enough data yet to show retrieval.", icon="\U0001F6A7")

st.divider()

# ---------------------------------------------------------------------------
# Step 6: explaining it in plain language
# ---------------------------------------------------------------------------
st.header("6. Explaining it in plain language")
st.write("A match is a number. Ask the DJ and Explore both turn that number into a sentence:")


@st.cache_data(show_spinner=False)
def _demo_explanation(song_a_id: int, song_b_id: int) -> str | None:
    client = get_explanation_client()
    if client is None:
        return None
    a, b = songs_by_id[song_a_id], songs_by_id[song_b_id]
    duration_a = min(30.0, a.duration_sec or 30.0)
    duration_b = min(30.0, b.duration_sec or 30.0)
    return client.generate_explanation(
        a.title, a.artist, a.genre_top, 0.0, duration_a,
        b.title, b.artist, b.genre_top, 0.0, duration_b,
        facet_name="sound", score=real_pair.audio_similarity if real_pair else 0.0,
    )


if real_pair is not None:
    if st.button("▶ Reveal the explanation", key="step6_reveal"):
        explanation = _demo_explanation(real_pair.song_id_a, real_pair.song_id_b)
        if explanation is None:
            st.info(
                "No API key configured in this environment, so live generation isn't available here "
                "-- this is exactly the graceful fallback the app uses everywhere an LLM feature "
                "isn't load-bearing. In a configured environment, this reveals a real generated "
                "sentence, one character at a time.",
                icon="\U0001F6A7",
            )
        else:
            placeholder = st.empty()
            shown = ""
            for ch in explanation:
                shown += ch
                placeholder.markdown(f"\U0001F4AC *{shown}*")
                time.sleep(0.015)
else:
    st.info("Not enough data yet for a live example.", icon="\U0001F6A7")

st.divider()

# ---------------------------------------------------------------------------
# Step 7: the payoff -- the whole map
# ---------------------------------------------------------------------------
st.header("7. Putting it all on a map you can explore")
st.write(
    "Every song, positioned by real audio similarity, all at once -- this is what Explore's "
    "network view actually shows, built from exactly the mechanic above, repeated across the "
    "whole library:"
)
if not real_nodes.empty:
    st.plotly_chart(network_graph_figure(real_nodes, real_edges), width="stretch", key="step7_full_map")
else:
    st.info("Not enough data yet to show the full map.", icon="\U0001F6A7")

st.divider()

st.write("Next: **Methodology** walks through each of these steps in depth, with real evidence at every stage.")
st.page_link("pages/1_Methodology.py", label="**Continue to Methodology →**", icon="\U0001F52C")
