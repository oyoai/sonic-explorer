import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from sonic_explorer.facets.fingerprint import composite_fingerprint, structure_fingerprint
from components.plotting import composite_fingerprint_thumbnail, fingerprint_thumbnail
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
if song.description:
    st.caption(f"\U0001F3B6 *{song.description}*")
st.audio(str(audio_path_for(song)))

# Fetched once, used by both the fingerprint row and the timeline chart below.
try:
    timeline = embedding_repo.get_structure_timeline(song.id)
except FileNotFoundError:
    timeline = None

try:
    matrix = embedding_repo.get_structure_matrix(song.id)
except FileNotFoundError:
    matrix = None

structure_fp = structure_fingerprint(matrix) if matrix is not None else None
sound_fp = timeline.sound_fingerprint if timeline is not None else None
harmony_fp = timeline.harmony_fingerprint if timeline is not None else None

if structure_fp is not None or sound_fp is not None or harmony_fp is not None:
    st.markdown("#### Fingerprint")
    st.caption(
        "A visual identity for the song, derived from its structure, sound, and harmony -- also usable "
        "as an album-art fallback. The composite overlays all three as color channels: where they agree "
        "the image reads bright, where they diverge, distinct color casts appear."
    )
    cols = st.columns(4)
    if structure_fp is not None:
        with cols[0]:
            st.plotly_chart(fingerprint_thumbnail(structure_fp, "Structure"), width="stretch", key="fp_structure")
    if sound_fp is not None:
        with cols[1]:
            st.plotly_chart(fingerprint_thumbnail(sound_fp, "Sound"), width="stretch", key="fp_sound")
    if harmony_fp is not None:
        with cols[2]:
            st.plotly_chart(fingerprint_thumbnail(harmony_fp, "Harmony"), width="stretch", key="fp_harmony")
    if structure_fp is not None and sound_fp is not None and harmony_fp is not None:
        with cols[3]:
            composite = composite_fingerprint(structure_fp, sound_fp, harmony_fp)
            st.plotly_chart(composite_fingerprint_thumbnail(composite), width="stretch", key="fp_composite")

st.markdown("#### Structure timeline")

selected_segment_idx = None
if timeline is None:
    st.warning("No structure timeline computed for this song yet.")
elif not timeline.has_clear_structure:
    # Not every song has clean verse/chorus repetition -- forcing a segmented
    # timeline onto an ambient/through-composed track would show either one
    # meaningless block or noisy fake boundaries. The novelty curve says so
    # honestly instead of hiding the limitation.
    st.caption(
        "This song evolves gradually rather than repeating in clear sections -- shown below as a "
        "continuous curve instead of colored blocks. Peaks are moments that sound most different "
        "from what came just before; none were sharp enough to count as a clear section change."
    )
    if timeline.novelty_curve is not None:
        curve_fig = go.Figure(go.Scatter(
            x=timeline.novelty_times, y=timeline.novelty_curve, mode="lines", fill="tozeroy",
            line=dict(color="rgb(99,110,250)"),
        ))
        curve_fig.update_layout(
            height=180,
            xaxis_title="Time (s)",
            yaxis=dict(title="novelty", showticklabels=False, range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(curve_fig, width="stretch", key="novelty_curve_chart")
else:
    st.caption(
        "Each colored block is a stretch of the song. Same color = similar-sounding sections "
        "(e.g. a verse repeating later) -- discovered automatically from the audio, not labeled by us. "
        "Click a block to loop just that section."
    )
    palette = px.colors.qualitative.Set2
    unique_labels = sorted(set(timeline.segment_labels.tolist()))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    durations = timeline.segment_ends - timeline.segment_starts
    hover_text = [f"{s:.1f}s – {e:.1f}s" for s, e in zip(timeline.segment_starts, timeline.segment_ends, strict=False)]

    timeline_fig = go.Figure(go.Bar(
        x=durations,
        y=["Structure"] * len(durations),
        base=timeline.segment_starts,
        orientation="h",
        marker_color=[color_map[lab] for lab in timeline.segment_labels.tolist()],
        marker_line_width=0,
        customdata=[[i] for i in range(len(durations))],
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
    event = st.plotly_chart(timeline_fig, width="stretch", on_select="rerun", key="structure_timeline_chart")

    if event and event.selection and event.selection.points:
        selected_segment_idx = event.selection.points[0]["customdata"][0]

    if selected_segment_idx is not None:
        seg_start = float(timeline.segment_starts[selected_segment_idx])
        seg_end = float(timeline.segment_ends[selected_segment_idx])
        st.markdown(f"**Looping section:** {seg_start:.1f}s – {seg_end:.1f}s")
        st.audio(str(audio_path_for(song)), start_time=seg_start, end_time=seg_end, loop=True)
    else:
        st.caption("Click a colored block above to loop just that section.")

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
        "The matrix the timeline above is derived from. The main diagonal is deliberately left blank -- "
        "bright parallel stripes off the diagonal are what mark repeated sections."
    )
    if matrix is not None:
        heatmap = px.imshow(
            matrix, color_continuous_scale="Magma", origin="lower",
            labels=dict(x="beat", y="beat", color="similarity"),
        )
        heatmap.update_layout(height=420)
        st.plotly_chart(heatmap, width="stretch")
    else:
        st.warning("No structure matrix computed for this song yet.")
