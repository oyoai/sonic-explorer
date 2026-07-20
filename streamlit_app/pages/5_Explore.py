import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from sonic_explorer.analysis.network_graph import build_similarity_graph
from sonic_explorer.analysis.taste_map import mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from sonic_explorer.facets.fingerprint import composite_fingerprint, structure_fingerprint
from components.plotting import composite_fingerprint_thumbnail, fingerprint_thumbnail, network_graph_figure
from resources import get_repositories, show_data_source_banner

FACETS_TO_CHECK = ["sound", "harmony"]

st.set_page_config(page_title="Explore", page_icon="\U0001F310", layout="wide")
st.title("Explore")
st.caption(
    "Every song as a node; edges connect genuinely similar songs. Click a node to open its "
    "identity card -- no page navigation, everything happens right here."
)

show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()
if not all_songs:
    st.info("No songs in the library yet.")
    st.stop()

if "explore_selected_song_id" not in st.session_state:
    st.session_state.explore_selected_song_id = None


@st.cache_data
def build_explore_graph(_song_repo, _embedding_repo, cache_key, scope):
    """One graph-building path for both Explore (global) and My Library --
    the only difference is which songs' vectors go in (spec: "a filter on the
    same underlying view, not a separate implementation")."""
    target_songs = _song_repo.list_songs(saved_only=(scope == "saved"))
    target_ids = {s.id for s in target_songs}
    all_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    song_vectors = {sid: vec for sid, vec in all_vectors.items() if sid in target_ids}

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


def facet_available(song) -> dict[str, bool]:
    return {
        facet_name: any(embedding_repo.status(seg.id, facet_name) == "done" for seg in song.segments)
        for facet_name in FACETS_TO_CHECK
    }


scope = st.radio(
    "Library", options=["all", "saved"], format_func=lambda s: "Explore (global)" if s == "all" else "My Library",
    horizontal=True,
)

if scope == "saved" and not any(s.is_saved for s in all_songs):
    st.info("You haven't saved any songs yet -- open a song's card below and use \"Save to My Library.\"")
    st.stop()

nodes_df, edges = build_explore_graph(song_repo, embedding_repo, embedding_repo.index_size("sound"), scope)

if nodes_df.empty:
    st.info("Nothing embedded for this view yet.")
    st.stop()

graph_col, panel_col = st.columns([2.2, 1])

with graph_col:
    fig = network_graph_figure(nodes_df, edges, selected_song_id=st.session_state.explore_selected_song_id)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="explore_graph_chart")
    if event and event.selection and event.selection.points:
        st.session_state.explore_selected_song_id = event.selection.points[0]["customdata"][0]

with panel_col:
    selected_id = st.session_state.explore_selected_song_id
    if selected_id is None or selected_id not in set(nodes_df["song_id"]):
        st.info("Click a node on the graph to see its identity card.")
    else:
        song = song_repo.get_song(int(selected_id))
        node_row = nodes_df[nodes_df["song_id"] == selected_id].iloc[0]

        st.subheader(song.title)
        st.caption(f"{song.artist} · {song.genre_top} · Year: —")

        if song.is_saved:
            if st.button("★ Remove from My Library", key="unsave_btn"):
                song_repo.unsave_song(song.id)
                st.rerun()
        else:
            if st.button("☆ Save to My Library", key="save_btn"):
                song_repo.save_song(song.id)
                st.rerun()

        st.audio(str(audio_path_for(song)))

        st.markdown("**Profile**")
        tempo_text = f"{song.tempo_bpm:.0f} BPM" if song.tempo_bpm is not None else "Not yet computed"
        st.markdown(f"- Tempo: {tempo_text}")
        st.markdown("- Key: Not yet computed")
        st.markdown(f"- Cluster: {int(node_row['cluster'])}")

        availability = facet_available(song)
        facet_text = ", ".join(
            f"{name.capitalize()} {'✓' if ok else '—'}" for name, ok in availability.items()
        )
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
            st.markdown("**Fingerprints**")
            fp_cols = st.columns(2)
            slot = 0
            if structure_fp is not None:
                with fp_cols[slot % 2]:
                    st.plotly_chart(fingerprint_thumbnail(structure_fp, "Structure"), width="stretch", key="panel_fp_structure")
                slot += 1
            if sound_fp is not None:
                with fp_cols[slot % 2]:
                    st.plotly_chart(fingerprint_thumbnail(sound_fp, "Sound"), width="stretch", key="panel_fp_sound")
                slot += 1
            if harmony_fp is not None:
                with fp_cols[slot % 2]:
                    st.plotly_chart(fingerprint_thumbnail(harmony_fp, "Harmony"), width="stretch", key="panel_fp_harmony")
                slot += 1
            if structure_fp is not None and sound_fp is not None and harmony_fp is not None:
                with fp_cols[slot % 2]:
                    composite = composite_fingerprint(structure_fp, sound_fp, harmony_fp)
                    st.plotly_chart(composite_fingerprint_thumbnail(composite), width="stretch", key="panel_fp_composite")
        else:
            st.caption("No fingerprints computed for this song yet.")
