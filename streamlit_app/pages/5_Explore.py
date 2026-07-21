import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sonic_explorer.analysis.network_graph import build_similarity_graph
from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS
from sonic_explorer.analysis.taste_map import compute_taste_map, correlate_axes_with_features, mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from sonic_explorer.facets.fingerprint import composite_fingerprint, structure_fingerprint
from sonic_explorer.facets.registry import default_registry
from components.plotting import (
    composite_fingerprint_thumbnail,
    extract_selected_song_id,
    fingerprint_thumbnail,
    network_graph_figure,
)
from resources import get_repositories, show_data_source_banner

FACET_REGISTRY = default_registry()
QUEUE_MODE_LABELS = {"random": "Random", "loop": "Loop", "closest_match": "Closest match"}
CORRELATION_NOTABLE_THRESHOLD = 0.4  # |r| above this is presented as "this explains the axis"


def _compute_up_next(song_repo, embedding_repo, retrieval_service, current_song, mode, facet_name, candidate_ids):
    """None means "nothing to suggest" -- e.g. closest-match on a facet this
    song was never embedded for. Random is restricted to the currently
    filtered/visible set of songs; closest match searches the full facet
    index, same as every other retrieval call in the app."""
    if mode == "loop":
        return current_song.id
    if mode == "random":
        candidates = [sid for sid in candidate_ids if sid != current_song.id]
        return random.choice(candidates) if candidates else current_song.id
    if not current_song.segments:
        return None
    query_seg = current_song.segments[len(current_song.segments) // 2]
    if embedding_repo.status(query_seg.id, facet_name) != "done":
        return None
    matches = retrieval_service.query_by_segment(query_seg.id, facet_name=facet_name, k=1)
    return matches[0].song.id if matches else None


st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F310", layout="wide")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()
if not all_songs:
    st.title("Sonic Explorer")
    st.info("No songs in the library yet.")
    st.stop()

show_data_source_banner()

if "explore_selected_song_id" not in st.session_state:
    st.session_state.explore_selected_song_id = None
if "explore_info_dismissed" not in st.session_state:
    st.session_state.explore_info_dismissed = False
if "explore_up_next_cache" not in st.session_state:
    st.session_state.explore_up_next_cache = {}

st.title("Sonic Explorer")

if not st.session_state.explore_info_dismissed:
    with st.container(border=True):
        info_col, dismiss_col = st.columns([6, 1])
        with info_col:
            st.write("Explore your music library by how it actually sounds — not tags or genre labels.")
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Songs in library", len(all_songs))
            stat_col2.metric("Embedded segments", embedding_repo.index_size("sound"))
        with dismiss_col:
            if st.button("Dismiss", key="dismiss_info"):
                st.session_state.explore_info_dismissed = True
                st.rerun()

view_mode = st.radio(
    "View",
    options=["network", "map"],
    format_func=lambda v: "Network graph" if v == "network" else "2D map (PCA/ICA)",
    horizontal=True,
    help="Network graph: force-directed layout of an actual nearest-neighbor graph -- edges are "
         "real connections. 2D map: a PCA/ICA projection instead -- no edges, but the axes "
         "themselves can potentially be interpreted (see 'Inspect these axes' below).",
)

graph_facet = st.radio(
    "Explore by",
    options=FACET_REGISTRY.names(),
    format_func=lambda f: f.capitalize(),
    horizontal=True,
    help="Changes what \"similar\" means, in both views.",
)

with st.expander("Filter / sort / highlight"):
    scope = st.radio(
        "Library", options=["all", "saved"],
        format_func=lambda s: "All songs" if s == "all" else "My Library (saved)",
        horizontal=True,
    )

    genre_options = sorted({s.genre_top for s in all_songs if s.genre_top})
    selected_genres = st.multiselect("Genre", options=genre_options, default=genre_options)

    tempo_values = [s.tempo_bpm for s in all_songs if s.tempo_bpm is not None]
    if tempo_values:
        tempo_lo, tempo_hi = float(min(tempo_values)), float(max(tempo_values)) + 1.0
        tempo_range = st.slider("Tempo (BPM)", min_value=tempo_lo, max_value=tempo_hi, value=(tempo_lo, tempo_hi))
    else:
        tempo_range = None
        st.caption("Tempo filter unavailable -- no songs have computed DNA yet.")

    st.selectbox("Decade", options=["Not available yet -- release year isn't tracked"], disabled=True)

if scope == "saved" and not any(s.is_saved for s in all_songs):
    st.info("You haven't saved any songs yet -- open a song below and use \"Save to My Library.\"")
    st.stop()

filtered_songs = [
    s for s in all_songs
    if (scope != "saved" or s.is_saved)
    and s.genre_top in selected_genres
    and (tempo_range is None or s.tempo_bpm is None or tempo_range[0] <= s.tempo_bpm <= tempo_range[1])
]
target_song_ids = frozenset(s.id for s in filtered_songs)


@st.cache_data
def build_explore_graph(_song_repo, _embedding_repo, facet_name: str, index_size: int, target_song_ids: frozenset):
    target_songs = [s for s in _song_repo.list_songs() if s.id in target_song_ids]
    all_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)
    song_vectors = {sid: vec for sid, vec in all_vectors.items() if sid in target_song_ids}
    result = build_similarity_graph(song_vectors)
    songs_by_id = {s.id: s for s in target_songs}
    nodes_df = pd.DataFrame([
        {
            "song_id": n.song_id, "x": n.x, "y": n.y, "cluster": n.cluster,
            "title": songs_by_id[n.song_id].title, "artist": songs_by_id[n.song_id].artist,
            "genre": songs_by_id[n.song_id].genre_top,
        }
        for n in result.nodes
        if n.song_id in songs_by_id
    ])
    return nodes_df, result.edges


@st.cache_data
def build_map_df(_song_repo, _embedding_repo, facet_name: str, index_size: int, target_song_ids: frozenset, method: str):
    target_songs = [s for s in _song_repo.list_songs() if s.id in target_song_ids]
    all_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)
    song_vectors = {sid: vec for sid, vec in all_vectors.items() if sid in target_song_ids}
    result = compute_taste_map(song_vectors, method=method)
    songs_by_id = {s.id: s for s in target_songs}
    return pd.DataFrame([
        {
            "song_id": p.song_id, "x": p.x, "y": p.y, "cluster": p.cluster,
            "title": songs_by_id[p.song_id].title, "artist": songs_by_id[p.song_id].artist,
            "genre": songs_by_id[p.song_id].genre_top,
        }
        for p in result.points if p.song_id in songs_by_id
    ])


points_df: pd.DataFrame
edges = []

if view_mode == "network":
    points_df, edges = build_explore_graph(
        song_repo, embedding_repo, graph_facet, embedding_repo.index_size(graph_facet), target_song_ids
    )
    if points_df.empty:
        st.info(f"No songs embedded for {graph_facet.capitalize()} in this view yet.")
        st.stop()

    fig = network_graph_figure(points_df, edges, selected_song_id=st.session_state.explore_selected_song_id)
    fig.update_layout(dragmode="pan")
    event = st.plotly_chart(
        fig, width="stretch", on_select="rerun", key="explore_graph_chart", config={"scrollZoom": True}
    )
    st.caption(
        "Position: force-directed layout of a k-nearest-neighbor graph -- pulled toward close "
        "matches, pushed apart from everything else. A line = a real top-neighbor connection, not "
        "every pair. Color = K-means cluster on the same vectors."
    )
else:
    map_cols = st.columns(2)
    with map_cols[0]:
        projection_method = st.radio(
            "Projection", options=["pca", "ica"], format_func=lambda m: m.upper(), horizontal=True,
        )
    with map_cols[1]:
        color_by = st.radio("Color by", options=["cluster", "genre"], horizontal=True)

    points_df = build_map_df(
        song_repo, embedding_repo, graph_facet, embedding_repo.index_size(graph_facet), target_song_ids,
        projection_method,
    )
    if points_df.empty:
        st.info(f"No songs embedded for {graph_facet.capitalize()} in this view yet.")
        st.stop()
    points_df = points_df.assign(cluster_str=points_df["cluster"].astype(str))

    with st.expander(f"Inspect these axes ({projection_method.upper()})"):
        st.caption(
            "**Primary check -- quantitative.** Does this axis actually correlate with an "
            "already-computed, independently-meaningful feature? A clean |r| lets you name the axis "
            "with real evidence, not a guess. |r| ≥ "
            f"{CORRELATION_NOTABLE_THRESHOLD:.1f} is flagged below as a real explanation; anything "
            "weaker means the axis isn't well-explained by these five features."
        )
        dna_by_song = {s.id: {axis: getattr(s, axis) for axis in AXES} for s in filtered_songs}
        has_dna = points_df["song_id"].map(
            lambda sid: sid in dna_by_song and all(v is not None for v in dna_by_song[sid].values())
        )
        dna_points = points_df[has_dna]

        if len(dna_points) < 3:
            st.caption("Not enough songs with computed DNA in this filtered view to correlate.")
        else:
            features = {
                axis: np.array([dna_by_song[sid][axis] for sid in dna_points["song_id"]]) for axis in AXES
            }
            correlations = correlate_axes_with_features(
                dna_points["x"].to_numpy(), dna_points["y"].to_numpy(), features
            )
            for axis_label in ["x", "y"]:
                axis_corrs = sorted(
                    [c for c in correlations if c.axis == axis_label], key=lambda c: -abs(c.r)
                )
                best = axis_corrs[0]
                st.markdown(f"**{axis_label}-axis:**")
                rows = " · ".join(
                    f"{AXIS_LABELS[c.feature]} r={c.r:+.2f}" for c in axis_corrs
                )
                st.caption(rows)
                if abs(best.r) >= CORRELATION_NOTABLE_THRESHOLD:
                    st.markdown(f"→ well-explained by **{AXIS_LABELS[best.feature]}** (r={best.r:+.2f})")
                else:
                    st.markdown("→ no single feature explains this axis cleanly (all |r| < "
                                f"{CORRELATION_NOTABLE_THRESHOLD:.1f}) -- see the qualitative check below.")

        st.caption(
            "**Secondary check -- qualitative.** For axes the correlation table doesn't resolve, "
            "listen to a few songs at each extreme and judge for yourself whether it still resolves "
            "into something nameable that just isn't one of the five DNA scalars, or doesn't resolve "
            "into anything nameable at all -- both are valid outcomes."
        )
        axis_col1, axis_col2 = st.columns(2)
        for axis, col in [("x", axis_col1), ("y", axis_col2)]:
            with col:
                st.markdown(f"**Axis: {axis}**")
                st.caption("Highest")
                for _, row in points_df.nlargest(3, axis).iterrows():
                    st.caption(f"{row['title']} — {row['artist']} ({row['genre']})")
                st.caption("Lowest")
                for _, row in points_df.nsmallest(3, axis).iterrows():
                    st.caption(f"{row['title']} — {row['artist']} ({row['genre']})")

    fig = px.scatter(
        points_df, x="x", y="y", color=color_by if color_by == "genre" else "cluster_str",
        custom_data=["song_id"],
        hover_data={"title": True, "artist": True, "genre": True, "x": False, "y": False, "cluster_str": False},
        title=f"{len(points_df)} songs, colored by {color_by} ({projection_method.upper()} projection)",
    )
    fig.update_traces(marker=dict(size=9))
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="explore_map_chart")
    st.caption(
        "Position: PCA/ICA projection -- these axes are, in principle, nameable (see above). Point "
        "density reflects the real data: crowded regions genuinely have many similar-sounding songs."
    )

if event and event.selection and event.selection.points:
    clicked_song_id = extract_selected_song_id(event.selection.points[0])
    if clicked_song_id is not None:
        st.session_state.explore_selected_song_id = clicked_song_id

st.divider()

selected_id = st.session_state.explore_selected_song_id
if selected_id is None or selected_id not in set(points_df["song_id"]):
    st.info("Click a point above to start listening.")
else:
    song = song_repo.get_song(int(selected_id))
    node_row = points_df[points_df["song_id"] == selected_id].iloc[0]

    st.subheader(song.title)
    st.caption(f"{song.artist} · {song.genre_top}")
    st.audio(str(audio_path_for(song)))

    if song.is_saved:
        if st.button("★ Remove from My Library", key="unsave_btn"):
            song_repo.unsave_song(song.id)
            st.rerun()
    else:
        if st.button("☆ Save to My Library", key="save_btn"):
            song_repo.save_song(song.id)
            st.rerun()

    queue_mode = st.selectbox(
        "Up next", options=["random", "loop", "closest_match"], format_func=lambda m: QUEUE_MODE_LABELS[m],
    )
    queue_facet = graph_facet
    if queue_mode == "closest_match":
        queue_facet = st.selectbox(
            "Closest match by", options=FACET_REGISTRY.names(), format_func=lambda f: f.capitalize(),
            key="queue_facet_select",
            help="Independent from \"Explore by\" above -- just controls what plays next.",
        )

    cache_key = (song.id, queue_mode, queue_facet)
    if cache_key not in st.session_state.explore_up_next_cache:
        st.session_state.explore_up_next_cache[cache_key] = _compute_up_next(
            song_repo, embedding_repo, retrieval_service, song, queue_mode, queue_facet, target_song_ids,
        )
    up_next_id = st.session_state.explore_up_next_cache[cache_key]

    if up_next_id is not None:
        up_next_song = song_repo.get_song(up_next_id)
        st.caption(f"Up next: {up_next_song.title} — {up_next_song.artist}")
        if st.button("▶ Play next", key="play_next_btn"):
            st.session_state.explore_selected_song_id = up_next_id
            st.rerun()
    else:
        st.caption("Up next: nothing available for this mode yet.")

    with st.expander("Song details"):
        tempo_text = f"{song.tempo_bpm:.0f} BPM" if song.tempo_bpm is not None else "Not yet computed"
        st.markdown(f"- Tempo: {tempo_text}")
        st.markdown("- Key: Not yet computed")
        st.markdown(f"- Cluster: {int(node_row['cluster'])}")

        availability = {
            facet_name: any(embedding_repo.status(seg.id, facet_name) == "done" for seg in song.segments)
            for facet_name in FACET_REGISTRY.names()
        }
        facet_text = ", ".join(f"{name.capitalize()} {'✓' if ok else '—'}" for name, ok in availability.items())
        st.markdown(f"- Facets: {facet_text}")

        try:
            matrix = embedding_repo.get_structure_matrix(song.id)
        except FileNotFoundError:
            matrix = None
        try:
            timeline = embedding_repo.get_structure_timeline(song.id)
        except FileNotFoundError:
            timeline = None

        structure_fp = structure_fingerprint(matrix) if matrix is not None else None
        sound_fp = timeline.sound_fingerprint if timeline is not None else None
        harmony_fp = timeline.harmony_fingerprint if timeline is not None else None

        if structure_fp is not None or sound_fp is not None or harmony_fp is not None:
            fp_cols = st.columns(4)
            slot = 0
            if structure_fp is not None:
                with fp_cols[slot]:
                    st.plotly_chart(fingerprint_thumbnail(structure_fp, "Structure"), width="stretch", key="panel_fp_structure")
                slot += 1
            if sound_fp is not None:
                with fp_cols[slot]:
                    st.plotly_chart(fingerprint_thumbnail(sound_fp, "Sound"), width="stretch", key="panel_fp_sound")
                slot += 1
            if harmony_fp is not None:
                with fp_cols[slot]:
                    st.plotly_chart(fingerprint_thumbnail(harmony_fp, "Harmony"), width="stretch", key="panel_fp_harmony")
                slot += 1
            if structure_fp is not None and sound_fp is not None and harmony_fp is not None:
                with fp_cols[slot]:
                    composite = composite_fingerprint(structure_fp, sound_fp, harmony_fp)
                    st.plotly_chart(composite_fingerprint_thumbnail(composite), width="stretch", key="panel_fp_composite")
        else:
            st.caption("No fingerprints computed for this song yet.")
