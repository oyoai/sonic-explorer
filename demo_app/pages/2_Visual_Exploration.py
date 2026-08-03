"""Audio Space -- the global similarity space, reusing the exact same
network graph streamlit_app/pages/3_App_Walkthrough.py introduces (section
1a there), but as a real, clickable exploration tool here rather than
static illustration text. See Demo.py's module docstring for this app's
three-mode architecture. (Filename kept as 2_Visual_Exploration.py --
st.navigation()'s list order in Demo.py controls sidebar position and
title, not this file's own name/number; see CLAUDE.md's own note on this.)

Facet-specific by design (per the actual decision made on this page, not an
oversight): a similarity graph only means something once you know what
"similar" means, and blending several facets into one graph would average
away exactly the kind of cross-facet divergence Local Similarity (this
app's other page) exists to show. build_blended_similarity_graph() exists
in sonic_explorer.analysis.network_graph and could support a blended mode
later, but isn't wired up here.

Network graph only (this revision) -- an earlier version paired the graph
with a second tab, a PCA 2D map of the same per-song vectors
(taste_map_figure/cached_taste_map, still present in plotting.py/
resources.py but no longer imported here). Both answered the same question
("what does this facet consider similar") from two different projections,
which was judged more redundant than complementary for a live demo's single
page -- one clickable, always-visible graph, not a tab a presenter has to
remember to switch into.

Facet picker is a plain, always-visible st.radio (no expander) -- an
earlier version tucked it inside a collapsed "Similarity facet" expander,
which meant an extra click to even see or change it. For a live demo this
control gets touched constantly, so hiding it behind a collapse was pure
friction with no benefit.

Selection state (visual_selected_song_id) is plain st.session_state, not
st.query_params -- same precedent Explore's own equivalent selection state
(explore_selected_song_id) already follows in the main app. No default
selection: it starts at None (never a curated example's song) and a real
browser refresh drops it back to None too, not just the explicit "Clear
selection" button -- this page's job is to prove the audio space genuinely
holds up under an audience's own, unrehearsed clicks, so it deliberately
doesn't open on a pre-picked "good" example the way Local Similarity's
curated defaults do.

Clicking a node already gives real visual feedback on the chart itself
(halo + gold neighbor rings + labels, see plotting.py's docstrings) --
there's no separate "Selected song" detail panel (audio player, waveform,
ranked neighbor list) restating it in text.

Color-by toggle: network_graph_figure accepts a color_by param ("genre",
"cluster", or "none" -- plotting.py's _node_color()). "None" turns off
category coloring entirely so the ONE thing that still visually pops is
literally which node is selected, not a genre/cluster it happens to belong
to -- the selected node renders in this app's established green
(_SELECTED_NODE_COLOR, the same color match_pill_html's badge uses) instead
of a category color, everything else falls back to the same neutral gray
already used for non-highlighted nodes.

Performance: built from cached per-song vectors (resources.
cached_similarity_graph, keyed on facet name + index size, mirroring
streamlit_app/pages/7_Explore.py's own caching exactly) -- computed once
per facet per session, not on every click/rerun. See that function's own
docstring for real measured timing against deploy_data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from plotting import extract_selected_song_id, filter_to_neighbors, network_graph_figure
from resources import cached_similarity_graph, facet_display_name, get_repositories

FACETS = ["sound", "harmony", "vocal", "drums", "bass", "instrumental"]

st.title("Audio Space")

song_repo, embedding_repo, retrieval_service = get_repositories()

all_songs = song_repo.list_songs()
if not all_songs:
    st.info("No songs in the library yet.")
    st.stop()

if "visual_selected_song_id" not in st.session_state:
    st.session_state.visual_selected_song_id = None
selected_id = st.session_state.visual_selected_song_id  # None until a song is clicked, or after "Clear selection"

facet = st.radio(
    "Similarity facet", options=FACETS, format_func=facet_display_name, horizontal=True,
    key="visual_facet",
    help="Defines what \"similar\" means for the graph below -- position, clustering, and every "
         "edge/neighbor come from this facet's own embeddings, nothing blended.",
)

control_col, clear_col = st.columns([3, 1])
with control_col:
    color_by_label = st.radio(
        "Color nodes by", options=["Genre", "Cluster", "None"], index=2, horizontal=True,
        key="visual_color_by",
        help="Genre is the fixed, human-assigned FMA label; Cluster is the unsupervised K-means "
             "grouping computed from this facet's own embeddings; None turns off category coloring "
             "entirely so the selected node itself is the only thing that stands out (in green).",
    )
color_by = color_by_label.lower()
with clear_col:
    st.write("")
    if st.button("Clear selection", key="visual_clear_selection", disabled=selected_id is None, width="stretch"):
        st.session_state.visual_selected_song_id = None
        st.rerun()

nodes_df, edges = cached_similarity_graph(song_repo, embedding_repo, facet, embedding_repo.index_size(facet))

# The graph's neighbor highlighting -- empty when nothing is selected, or
# when the selected song has no embedded segments for this facet (an honest
# gap: not every song has e.g. a detected vocal stem), not an error.
neighbor_song_ids: set[int] = set()
if selected_id is not None and not nodes_df.empty and selected_id in set(nodes_df["song_id"]):
    _, neighbor_edges = filter_to_neighbors(nodes_df, edges, selected_id)
    for e in neighbor_edges:
        neighbor_song_ids.add(e.song_id_b if e.song_id_a == selected_id else e.song_id_a)

if nodes_df.empty:
    st.info(f"No songs embedded for the {facet_display_name(facet)} facet yet.")
else:
    network_fig = network_graph_figure(
        nodes_df, edges, selected_song_id=selected_id, highlight_song_ids=neighbor_song_ids,
        facet_label=facet_display_name(facet), click_priority=True, height=620, color_by=color_by,
    )
    network_event = st.plotly_chart(
        network_fig, width="stretch", on_select="rerun", key=f"visual_network_{facet}",
        config={"displayModeBar": False, "scrollZoom": True},
    )
    if network_event and network_event.selection and network_event.selection.points:
        clicked_id = extract_selected_song_id(network_event.selection.points[0])
        if clicked_id is not None and clicked_id != selected_id:
            st.session_state.visual_selected_song_id = clicked_id
            st.rerun()
