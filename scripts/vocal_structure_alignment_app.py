"""Temporary, throwaway Streamlit tool -- NOT part of the Sonic Explorer app,
just a visual companion to the structure-alignment check: for each of the 10
blind-listened segments, does the Structure facet's novelty curve show a
transition near where vocals actually started/stopped (per listening notes)?

Run as its own process, its own port:

    streamlit run scripts/vocal_structure_alignment_app.py --server.port 8503
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sonic_explorer.config import ARTIFACTS_DIR, AUDIO_DIR, DB_PATH
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

# (title, genre, window start, window end, human note, score_5s, verdict_5s_correct, score_15s, verdict_15s_correct)
SEGMENTS = [
    ("412", "Hip-Hop", 20.0, 25.0, "vocal throughout", 0.0179, False, 0.0564, True),
    ("Dismissal", "Pop", 5.0, 10.0, "vocal, vague", 0.0171, False, 0.0182, True),
    ("Facing the Sea (Album Version)", "Pop", 5.0, 10.0, "vocal only in last 2s (~8s onward)", 0.0195, True, 0.0147, False),
    ("A Message", "Rock", 12.5, 17.5, "no vocal", 0.0174, True, 0.0031, True),
    ("Requiem for a Small Town", "Folk", 12.5, 17.5, "vocal throughout", 0.0175, False, 0.0089, False),
    ("something brewing", "Hip-Hop", 5.0, 10.0, "no vocal", 0.0028, True, 0.0010, True),
    ("A1 Symphony", "Hip-Hop", 17.5, 22.5, "no vocal", 0.0017, True, 0.0004, True),
    ("Underwater", "Electronic", 2.5, 7.5, "no vocal", 0.0007, True, 0.0002, True),
    ("Ride My Bike", "Instrumental", 7.5, 12.5, "no vocal", 0.0002, True, 0.0001, True),
    ("Thursday & Snow (Reprise)", "Hip-Hop", 2.5, 7.5, "no vocal", 0.0228, False, 0.0106, True),
]

st.set_page_config(page_title="Structure Alignment Check (temporary)", page_icon="\U0001F4C8", layout="wide")
st.title("Structure Alignment Check")
st.caption(
    "Temporary QA tool, not part of the real app. For each blind-listened segment: does the "
    "Structure facet's novelty curve show a transition near where vocals actually started/stopped?"
)

conn = init_db(DB_PATH)
song_repo = SongRepository(conn)
embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
by_title = {s.title: s for s in song_repo.list_songs()}

for title, genre, start, end, note, score_5s, ok_5s, score_15s, ok_15s in SEGMENTS:
    song = by_title.get(title)
    st.divider()
    st.subheader(f"\"{title}\" ({genre})")
    st.caption(f"Sampled window: {start:.1f}-{end:.1f}s — human note: *{note}*")

    if song is None:
        st.warning("Song not found.")
        continue

    audio_path = AUDIO_DIR / f"{song.fma_track_id}.mp3"
    if audio_path.exists():
        st.audio(str(audio_path))

    verdict_cols = st.columns(2)
    with verdict_cols[0]:
        st.metric("5s-window score", f"{score_5s:.4f}", delta="correct" if ok_5s else "wrong",
                   delta_color="normal" if ok_5s else "inverse")
    with verdict_cols[1]:
        st.metric("15s-context score", f"{score_15s:.4f}", delta="correct" if ok_15s else "wrong",
                   delta_color="normal" if ok_15s else "inverse")

    try:
        timeline = embedding_repo.get_structure_timeline(song.id)
    except FileNotFoundError:
        st.warning("No structure timeline computed for this song.")
        continue

    starts = [float(s) for s in timeline.segment_starts]
    ends = [float(e) for e in timeline.segment_ends]
    boundaries = sorted(set(starts + ends) - {0.0, 30.0, float(ends[-1]) if ends else 30.0})
    straddled = [b for b in boundaries if start < b < end]

    fig = go.Figure()
    if timeline.novelty_curve is not None and timeline.novelty_times is not None:
        times = np.asarray(timeline.novelty_times, dtype=float)
        curve = np.asarray(timeline.novelty_curve, dtype=float)
        fig.add_trace(go.Scatter(x=times, y=curve, mode="lines", fill="tozeroy",
                                  line=dict(color="rgb(99,110,250)"), name="novelty"))
        # mark peaks
        peak_x, peak_y = [], []
        for i in range(1, len(curve) - 1):
            if curve[i] > curve[i - 1] and curve[i] > curve[i + 1] and curve[i] > 0.3:
                peak_x.append(float(times[i]))
                peak_y.append(float(curve[i]))
        fig.add_trace(go.Scatter(x=peak_x, y=peak_y, mode="markers",
                                  marker=dict(color="orange", size=8, symbol="triangle-up"), name="peak"))
    for b in boundaries:
        fig.add_vline(x=b, line=dict(color="gray", dash="dot", width=1))
    fig.add_vrect(x0=start, x1=end, fillcolor="crimson", opacity=0.15, line_width=0,
                  annotation_text="sampled window", annotation_position="top left")
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30, b=30), showlegend=False,
        xaxis_title="Time (s)", yaxis=dict(title="novelty", range=[0, 1.05]),
        title=f"Structural confidence: {timeline.structural_confidence:.3f}" if timeline.structural_confidence else None,
    )
    st.plotly_chart(fig, width="stretch", key=f"novelty_{title}")

    if straddled:
        st.info(f"Window **straddles** a structural boundary at {', '.join(f'{b:.1f}s' for b in straddled)}.",
                icon="✂️")
    else:
        st.caption("Window sits entirely within one structural segment -- no boundary straddled.")
