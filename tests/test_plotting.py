"""AppTest can't fully simulate a real browser's Plotly click-selection
payload, so it missed a real bug: clicking network_graph_figure's graph
crashed with KeyError: 0 on event.selection.points[0]["customdata"][0] --
most likely a click landing on the edges trace (mode="lines", no customdata
configured) rather than a node marker, close enough to a line for Plotly to
register the click there instead. extract_selected_song_id() replaces the
bare index with defensive handling so a click that doesn't carry a real
song_id is a silent no-op instead of crashing the page."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.plotting import (  # noqa: E402
    CHROMA_PITCH_LABELS,
    chord_strip_figure,
    chromagram_figure,
    composite_fingerprint_image_data_uri,
    composite_fingerprint_thumbnail,
    concept_bubble_diagram,
    extract_selected_song_id,
    filter_to_neighbors,
    fingerprint_image_data_uri,
    fingerprint_thumbnail,
    fingerprint_thumbnail_image,
    library_waffle_grid,
    mel_spectrogram_figure,
    network_graph_figure,
    network_hero_figure,
    taste_map_figure,
    waveform_figure,
)
from sonic_explorer.analysis.key_chord import ChordSegment  # noqa: E402


def test_fingerprint_thumbnail_image_returns_a_real_square_rgb_array():
    """Real reported bug: small (~56px) fingerprint thumbnails rendered as
    stretched rectangles via st.plotly_chart(), even with matching width/
    height declared both in the figure and at the call site. Moved off
    Plotly entirely for these small cases -- st.image() has simple,
    well-established exact-pixel sizing a charting library doesn't
    reliably guarantee at this size. This checks the actual array shape
    st.image() will render, not a Plotly figure's declared (but apparently
    unreliable) layout.width/height."""
    fingerprint = np.random.rand(32, 32).astype(np.float32)

    image = fingerprint_thumbnail_image(fingerprint)

    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
    assert image.min() >= 0
    assert image.max() <= 255


def test_fingerprint_thumbnail_image_flips_vertically_to_match_origin_lower():
    """fingerprint_thumbnail's px.imshow(..., origin="lower") puts row 0 at
    the bottom (this app's math/audio convention everywhere else);
    st.image() always puts row 0 at the top -- without an explicit flip,
    the same song's list thumbnail and detail-view fingerprint would
    visibly disagree with each other."""
    fingerprint = np.zeros((4, 4), dtype=np.float32)
    fingerprint[0, 0] = 1.0  # bright cell in the array's first (bottom, post-origin-lower) row

    image = fingerprint_thumbnail_image(fingerprint)

    # After the vertical flip, that bright cell must land in the LAST row
    # of the rendered (top-down) image, not the first.
    assert image[0, 0].sum() < image[-1, 0].sum()


def test_fingerprint_image_data_uri_returns_a_valid_png_data_uri():
    """Explore's Selected Song panel needs a real <img> tag (for a native
    title= hover tooltip -- st.image() has no hover mechanism at all), so
    this must be a real, decodable PNG data URI, not just an array."""
    import base64
    import io

    from PIL import Image

    fingerprint = np.random.rand(32, 32).astype(np.float32)

    uri = fingerprint_image_data_uri(fingerprint)

    assert uri.startswith("data:image/png;base64,")
    encoded = uri.removeprefix("data:image/png;base64,")
    decoded_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded_bytes))
    assert image.format == "PNG"
    assert image.size == (32, 32)


def test_fingerprint_image_data_uri_matches_fingerprint_thumbnail_image_pixels():
    """Both the small list/result thumbnails (fingerprint_thumbnail_image)
    and the Selected Song detail view (fingerprint_image_data_uri) must
    draw from the identical pixel data -- the whole point of unifying onto
    one image pipeline was fixing a real reported bug where the two looked
    visibly different for the same song."""
    import base64
    import io

    from PIL import Image

    fingerprint = np.random.rand(8, 8).astype(np.float32)

    expected_rgb = fingerprint_thumbnail_image(fingerprint)
    uri = fingerprint_image_data_uri(fingerprint)
    decoded_bytes = base64.b64decode(uri.removeprefix("data:image/png;base64,"))
    actual_rgb = np.array(Image.open(io.BytesIO(decoded_bytes)).convert("RGB"))

    np.testing.assert_array_equal(actual_rgb, expected_rgb)


def test_composite_fingerprint_image_data_uri_returns_a_valid_png_data_uri():
    import base64
    import io

    from PIL import Image

    composite = np.random.rand(8, 8, 3).astype(np.float32)

    uri = composite_fingerprint_image_data_uri(composite)

    assert uri.startswith("data:image/png;base64,")
    decoded_bytes = base64.b64decode(uri.removeprefix("data:image/png;base64,"))
    image = Image.open(io.BytesIO(decoded_bytes))
    assert image.format == "PNG"
    assert image.size == (8, 8)


def test_composite_fingerprint_image_data_uri_flips_vertically():
    """Same row-0-at-bottom vs. row-0-at-top mismatch fingerprint_thumbnail_
    image's own flip fixes -- composite is built by stacking structure/
    sound/harmony fingerprints, which share that same convention."""
    import base64
    import io

    from PIL import Image

    composite = np.zeros((4, 4, 3), dtype=np.float32)
    composite[0, 0] = 1.0  # bright cell in the array's first (bottom, pre-flip) row

    uri = composite_fingerprint_image_data_uri(composite)
    decoded_bytes = base64.b64decode(uri.removeprefix("data:image/png;base64,"))
    image = np.array(Image.open(io.BytesIO(decoded_bytes)).convert("RGB"))

    assert image[0, 0].sum() < image[-1, 0].sum()


def test_network_graph_figure_customdata_round_trips_song_ids():
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
        {"song_id": 202, "x": 1.0, "y": 1.0, "cluster": 1, "title": "B", "artist": "Artist B", "genre": "Jazz"},
    ])

    fig = network_graph_figure(nodes_df, edges=[])

    node_trace = fig.data[1]  # data[0] is the edge trace, data[1] is nodes
    customdata = list(node_trace.customdata)
    assert list(customdata[0])[0] == 101
    assert list(customdata[1])[0] == 202


def test_network_graph_figure_uses_qualitative_palette_not_continuous_colorscale():
    """Regression guard for a real reported bug: coloring nominal/categorical
    ids with a sequential colorscale (the old Viridis-on-cluster-id approach)
    made adjacent ids look falsely similar and distant ones falsely opposed,
    reported as "visually messy." Node color must be resolved to explicit
    qualitative hex/rgb strings, one per node, not a numeric array + a
    colorscale -- still true now that color is genre, not cluster."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
        {"song_id": 202, "x": 1.0, "y": 1.0, "cluster": 1, "title": "B", "artist": "Artist B", "genre": "Pop"},
        {"song_id": 303, "x": 2.0, "y": 2.0, "cluster": 0, "title": "C", "artist": "Artist C", "genre": "Rock"},
    ])

    fig = network_graph_figure(nodes_df, edges=[])

    node_trace = fig.data[1]
    assert node_trace.marker.colorscale is None
    colors = list(node_trace.marker.color)
    assert all(isinstance(c, str) for c in colors)
    assert colors[0] == colors[2]  # same genre (Rock) -> same color
    assert colors[0] != colors[1]  # different genre (Rock vs Pop) -> different color


def test_network_graph_figure_colors_by_genre_not_cluster():
    """A real reported preference: cluster coloring read as "insignificant"
    (an unlabeled unsupervised id a viewer can't check against anything);
    genre is a real, checkable label. Two songs in the SAME cluster but
    DIFFERENT genres must get different colors -- the inverse of the old
    cluster-coloring behavior, confirming this really switched sources
    rather than coincidentally matching for same-cluster-same-genre cases."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
        {"song_id": 202, "x": 1.0, "y": 1.0, "cluster": 0, "title": "B", "artist": "Artist B", "genre": "Folk"},
    ])

    fig = network_graph_figure(nodes_df, edges=[])

    colors = list(fig.data[1].marker.color)
    assert colors[0] != colors[1]  # same cluster (0), different genre -> different color


def test_network_graph_figure_genre_color_is_consistent_with_genre_color_map():
    """Same genre must resolve to the exact same color GENRE_COLOR_MAP
    assigns elsewhere (fixed/alphabetical, not derived from this view's own
    song counts) -- the whole point of a fixed map is one color meaning one
    genre everywhere, not just self-consistently within one render."""
    from components.plotting import GENRE_COLOR_MAP

    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Electronic"},
    ])

    fig = network_graph_figure(nodes_df, edges=[])

    assert list(fig.data[1].marker.color)[0] == GENRE_COLOR_MAP["Electronic"]


def test_network_graph_figure_unknown_genre_falls_back_gracefully():
    """A genre value outside the known 8 (shouldn't happen with real
    library data, but defensive against e.g. synthetic dev data) must not
    crash -- falls back to a fixed neutral color rather than a KeyError."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Unknown"},
    ])

    fig = network_graph_figure(nodes_df, edges=[])

    assert list(fig.data[1].marker.color)[0] == "#888888"


def test_network_graph_figure_highlight_song_ids_widen_marker_and_ring():
    """highlight_song_ids (search matches) must render distinctly from an
    unhighlighted node -- larger marker, a visible ring -- separate from
    selected_song_id's own white-ring styling."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
        {"song_id": 202, "x": 1.0, "y": 1.0, "cluster": 1, "title": "B", "artist": "Artist B", "genre": "Jazz"},
    ])

    fig = network_graph_figure(nodes_df, edges=[], highlight_song_ids=[202])

    node_trace = fig.data[1]
    sizes = list(node_trace.marker.size)
    line_widths = list(node_trace.marker.line.width)
    assert sizes[1] > sizes[0]
    assert line_widths[1] > 0
    assert line_widths[0] == 0


def test_network_graph_figure_nodes_have_real_hover_with_title_artist_genre():
    """Reversed from an earlier "no hover, click only" design -- see the
    function's own docstring for why. Edge trace now ALSO has real hover
    (the similarity weight) -- see test_network_graph_figure_edges_show_
    real_similarity_on_hover below, a later real feature request."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A Song", "artist": "Artist A", "genre": "Rock"},
    ])

    fig = network_graph_figure(nodes_df, edges=[])

    node_trace = fig.data[1]
    assert node_trace.hoverinfo == "text"
    assert node_trace.hovertext[0] == "A Song<br>Artist A · Rock"


def test_network_graph_figure_edges_show_real_similarity_on_hover():
    """Real feature request: a viewer could previously see two nodes were
    connected but not by how much (edges were hoverinfo="skip"). Each
    edge's real cosine-similarity weight must now be hoverable."""
    from sonic_explorer.analysis.network_graph import GraphEdge

    nodes_df = pd.DataFrame([
        {"song_id": 1, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
        {"song_id": 2, "x": 1.0, "y": 1.0, "cluster": 0, "title": "B", "artist": "Artist B", "genre": "Rock"},
    ])
    edges = [GraphEdge(song_id_a=1, song_id_b=2, weight=0.876)]

    fig = network_graph_figure(nodes_df, edges=edges)

    edge_trace = fig.data[0]
    assert edge_trace.hoverinfo == "text"
    assert "88% similarity" in list(edge_trace.hovertext)


def test_network_graph_figure_hide_isolated_nodes_drops_disconnected_songs():
    """Off by default -- every node stays visible with zero edges (today's
    existing behavior, unaffected). Opt-in (hide_isolated_nodes=True) drops
    any node with no edge touching it, but the selected song stays visible
    even if it's isolated -- it's the graph's own center, not just another
    node."""
    from sonic_explorer.analysis.network_graph import GraphEdge

    nodes_df = pd.DataFrame([
        {"song_id": 1, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
        {"song_id": 2, "x": 1.0, "y": 1.0, "cluster": 0, "title": "B", "artist": "Artist B", "genre": "Rock"},
        {"song_id": 3, "x": 2.0, "y": 2.0, "cluster": 0, "title": "C", "artist": "Artist C", "genre": "Jazz"},
    ])
    edges = [GraphEdge(song_id_a=1, song_id_b=2, weight=0.9)]  # song 3 has no edge at all

    default_fig = network_graph_figure(nodes_df, edges=edges)
    assert len(default_fig.data[1].customdata) == 3  # song 3 still shown despite zero edges

    hidden_fig = network_graph_figure(nodes_df, edges=edges, hide_isolated_nodes=True)
    remaining_ids = {row[0] for row in hidden_fig.data[1].customdata}
    assert remaining_ids == {1, 2}

    centered_fig = network_graph_figure(nodes_df, edges=edges, selected_song_id=3, hide_isolated_nodes=True)
    centered_ids = {row[0] for row in centered_fig.data[1].customdata}
    assert 3 in centered_ids  # selected song stays visible even though it's isolated


def test_network_graph_figure_click_priority_disables_dragmode():
    """Root cause of a real "clicking a node doesn't work reliably" report:
    Plotly's default pan dragmode swallows a click that has any mouse
    movement in it. click_priority=True must disable dragging outright
    (dragmode=False) without touching hover/click itself; default (False)
    must leave dragmode alone (None -- Plotly's own default) for callers
    that never wire on_select (Results/App Walkthrough's static graphs)."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
    ])

    assert network_graph_figure(nodes_df, edges=[]).layout.dragmode is None
    assert network_graph_figure(nodes_df, edges=[], click_priority=True).layout.dragmode is False


def test_network_graph_figure_center_song_id_sets_axis_range():
    """center_song_id + zoom_radius must zoom the axes to a window around
    that node's position, not leave them auto-ranged over the whole graph."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A", "artist": "Artist A", "genre": "Rock"},
        {"song_id": 202, "x": 10.0, "y": 10.0, "cluster": 1, "title": "B", "artist": "Artist B", "genre": "Jazz"},
    ])

    fig = network_graph_figure(nodes_df, edges=[], center_song_id=202, zoom_radius=1.0)

    assert fig.layout.xaxis.range == (9.0, 11.0)
    assert fig.layout.yaxis.range == (9.0, 11.0)


def test_filter_to_neighbors_keeps_only_center_and_directly_connected_nodes():
    from sonic_explorer.analysis.network_graph import GraphEdge

    nodes_df = pd.DataFrame([
        {"song_id": 1, "x": 0.0, "y": 0.0, "cluster": 0},
        {"song_id": 2, "x": 1.0, "y": 1.0, "cluster": 0},
        {"song_id": 3, "x": 2.0, "y": 2.0, "cluster": 1},
        {"song_id": 4, "x": 3.0, "y": 3.0, "cluster": 1},  # not connected to 1 at all
    ])
    edges = [
        GraphEdge(song_id_a=1, song_id_b=2, weight=0.9),
        GraphEdge(song_id_a=3, song_id_b=1, weight=0.8),  # center on the "b" side
        GraphEdge(song_id_a=2, song_id_b=3, weight=0.7),  # doesn't touch song 1 -- must be dropped
    ]

    filtered_nodes, filtered_edges = filter_to_neighbors(nodes_df, edges, center_song_id=1)

    assert set(filtered_nodes["song_id"]) == {1, 2, 3}
    assert len(filtered_edges) == 2
    assert all(e.song_id_a == 1 or e.song_id_b == 1 for e in filtered_edges)


def test_filter_to_neighbors_falls_back_to_full_graph_for_unknown_song_id():
    nodes_df = pd.DataFrame([{"song_id": 1, "x": 0.0, "y": 0.0, "cluster": 0}])
    filtered_nodes, filtered_edges = filter_to_neighbors(nodes_df, edges=[], center_song_id=999)

    assert len(filtered_nodes) == 1
    assert filtered_edges == []


def test_taste_map_figure_customdata_round_trips_song_ids():
    points_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0},
        {"song_id": 202, "x": 1.0, "y": 1.0, "cluster": 1},
    ])

    fig = taste_map_figure(points_df)

    customdata = list(fig.data[0].customdata)
    assert list(customdata[0])[0] == 101
    assert list(customdata[1])[0] == 202


def test_taste_map_figure_marks_selected_song_with_a_ring():
    points_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0},
        {"song_id": 202, "x": 1.0, "y": 1.0, "cluster": 1},
    ])

    fig = taste_map_figure(points_df, selected_song_id=202)

    widths = list(fig.data[0].marker.line.width)
    assert widths == [0, 2.5]


def test_taste_map_figure_has_no_edge_trace():
    """Unlike network_graph_figure, PCA/ICA position is the real signal --
    there's no k-NN edge data backing this projection, so drawing lines
    would imply graph structure that doesn't exist here."""
    points_df = pd.DataFrame([{"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0}])
    fig = taste_map_figure(points_df)
    assert len(fig.data) == 1


def test_taste_map_figure_hover_is_conditional_on_metadata_columns():
    """title/artist/genre are optional -- a caller without them (e.g. this
    file's other taste_map_figure tests) gets hoverinfo="skip" rather than a
    KeyError; a caller with them (Explore's real usage) gets a real tooltip,
    same format network_graph_figure's node hover uses."""
    bare_df = pd.DataFrame([{"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0}])
    assert taste_map_figure(bare_df).data[0].hoverinfo == "skip"

    meta_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0, "title": "A Song", "artist": "Artist A", "genre": "Rock"},
    ])
    fig = taste_map_figure(meta_df)
    assert fig.data[0].hoverinfo == "text"
    assert fig.data[0].hovertext[0] == "A Song<br>Artist A · Rock"


def test_taste_map_figure_click_priority_disables_dragmode():
    points_df = pd.DataFrame([{"song_id": 101, "x": 0.0, "y": 0.0, "cluster": 0}])
    assert taste_map_figure(points_df).layout.dragmode is None
    assert taste_map_figure(points_df, click_priority=True).layout.dragmode is False


def test_network_hero_figure_has_no_axes_and_is_short():
    """The page-banner variant must render as decoration -- no visible axes,
    no legend, short/thin -- not a second copy of the interactive graph."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 0.0, "y": 0.0},
        {"song_id": 202, "x": 1.0, "y": 1.0},
    ])

    fig = network_hero_figure(nodes_df, edges=[])

    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False
    assert fig.layout.height <= 150
    for trace in fig.data:
        assert trace.showlegend is False


def test_network_hero_figure_uses_real_node_positions_not_placeholders():
    """The banner must actually plot the real x/y positions handed to it --
    confirming this isn't a hardcoded decorative shape unrelated to the
    library's real similarity graph."""
    nodes_df = pd.DataFrame([
        {"song_id": 101, "x": 3.5, "y": -2.0},
        {"song_id": 202, "x": -7.25, "y": 4.0},
    ])

    fig = network_hero_figure(nodes_df, edges=[])

    node_trace = fig.data[1]
    assert list(node_trace.x) == [3.5, -7.25]
    assert list(node_trace.y) == [-2.0, 4.0]


def test_fingerprint_thumbnail_declares_equal_width_and_height():
    """Regression guard for a real reported bug: fingerprints rendered as
    rectangles, not squares. structure_fingerprint() always returns a
    genuinely square array -- px.imshow's own scaleanchor/constrain=domain
    default is known to miscompute after a responsive container resize
    (exactly what Streamlit's width="stretch" triggers), so the figure
    must also self-declare an explicit width=height as a second, more
    robust guarantee. Callers must additionally pass a fixed pixel width
    (not "stretch") to st.plotly_chart() -- this only checks the figure's
    own half of that fix."""
    fig = fingerprint_thumbnail(np.random.rand(32, 32), title="", height=56)
    assert fig.layout.width == fig.layout.height == 56


def test_composite_fingerprint_thumbnail_declares_equal_width_and_height():
    fig = composite_fingerprint_thumbnail(np.random.rand(32, 32, 3), height=260)
    assert fig.layout.width == fig.layout.height == 260


def test_extract_selected_song_id_happy_path():
    point = {"customdata": [101]}
    assert extract_selected_song_id(point) == 101


def test_extract_selected_song_id_missing_customdata_key():
    """The edges trace has no customdata configured at all -- a click
    registering there must not crash."""
    point = {"x": 0.5, "y": 0.5}
    assert extract_selected_song_id(point) is None


def test_extract_selected_song_id_empty_customdata():
    point = {"customdata": []}
    assert extract_selected_song_id(point) is None


def test_extract_selected_song_id_customdata_not_indexable():
    point = {"customdata": None}
    assert extract_selected_song_id(point) is None


def test_extract_selected_song_id_point_not_dict_like():
    assert extract_selected_song_id(object()) is None


def test_library_waffle_grid_one_cell_per_song():
    songs_df = pd.DataFrame([{"title": f"Song {i}", "genre": "Rock"} for i in range(12)])
    fig = library_waffle_grid(songs_df, {"Rock": 12})

    total_cells = sum(len(trace.x) for trace in fig.data)
    assert total_cells == 12


def test_library_waffle_grid_legend_shows_every_genre_and_count():
    songs_df = pd.DataFrame(
        [{"title": f"R{i}", "genre": "Rock"} for i in range(3)]
        + [{"title": f"J{i}", "genre": "Jazz"} for i in range(2)]
    )
    fig = library_waffle_grid(songs_df, {"Rock": 3, "Jazz": 2})

    names = {trace.name for trace in fig.data}
    assert names == {"Rock (3)", "Jazz (2)"}


def test_library_waffle_grid_empty_library_does_not_raise():
    songs_df = pd.DataFrame(columns=["title", "genre"])
    fig = library_waffle_grid(songs_df, {})
    assert len(fig.data) == 0


def test_concept_bubble_diagram_has_one_satellite_marker_per_label():
    fig = concept_bubble_diagram("Center", ["Album", "Artist", "Tags", "Genre", "Year"])

    satellite_trace = fig.data[1]  # data[0] is the connecting-lines trace
    assert list(satellite_trace.text) == ["Album", "Artist", "Tags", "Genre", "Year"]


def test_concept_bubble_diagram_center_bubble_is_separate_trace():
    fig = concept_bubble_diagram("Center label", ["A", "B", "C"])

    center_trace = fig.data[2]
    assert list(center_trace.text) == ["Center label"]
    assert list(center_trace.x) == [0]
    assert list(center_trace.y) == [0]


def test_concept_bubble_diagram_satellites_are_evenly_spaced_around_center():
    fig = concept_bubble_diagram("Center", ["A", "B", "C", "D"])
    satellite_trace = fig.data[1]

    distances = np.hypot(satellite_trace.x, satellite_trace.y)
    assert np.allclose(distances, 1.0)


def test_chromagram_figure_x_axis_is_real_time_not_indices():
    """x values are the real seek time, for display/hover purposes -- but
    NOT for click-to-seek: go.Heatmap doesn't fire a plotly_selected event on
    a plain click the way Bar/Scatter traces do, confirmed as the real cause
    of a live "click the chromagram, nothing happens" bug report. Click-to-
    seek lives on chord_strip_figure's Bar trace instead (see its tests)."""
    chroma = np.random.rand(12, 5)
    times = np.array([0.0, 1.2, 2.4, 3.6, 4.8])

    fig = chromagram_figure(chroma, times)

    heatmap_trace = fig.data[0]
    assert list(heatmap_trace.x) == list(times)


def test_chromagram_figure_y_axis_has_all_twelve_pitch_classes():
    chroma = np.random.rand(12, 5)
    times = np.arange(5, dtype=float)

    fig = chromagram_figure(chroma, times)

    assert list(fig.data[0].y) == CHROMA_PITCH_LABELS
    assert len(CHROMA_PITCH_LABELS) == 12


def test_chromagram_figure_z_values_round_trip_the_chroma_matrix():
    chroma = np.arange(12 * 5, dtype=float).reshape(12, 5)
    times = np.arange(5, dtype=float)

    fig = chromagram_figure(chroma, times)

    assert np.array_equal(np.array(fig.data[0].z), chroma)


def test_chord_strip_figure_one_bar_per_segment():
    segments = [
        ChordSegment(start_sec=0.0, end_sec=2.0, label="C"),
        ChordSegment(start_sec=2.0, end_sec=5.0, label="Am"),
        ChordSegment(start_sec=5.0, end_sec=6.0, label="G"),
    ]

    fig = chord_strip_figure(segments)

    bar_trace = fig.data[0]
    assert list(bar_trace.base) == [0.0, 2.0, 5.0]
    assert list(bar_trace.x) == [2.0, 3.0, 1.0]  # durations


def test_chord_strip_figure_customdata_carries_each_segments_start_time():
    """This is the real click-to-seek control (chromagram_figure's heatmap
    isn't) -- customdata must carry each bar's start_sec so a click event's
    point["customdata"][0] gives a real seek time, same pattern
    network_graph_figure/Song X-Ray's structure timeline already use."""
    segments = [
        ChordSegment(start_sec=0.0, end_sec=2.0, label="C"),
        ChordSegment(start_sec=2.0, end_sec=5.0, label="Am"),
    ]

    fig = chord_strip_figure(segments)

    customdata = list(fig.data[0].customdata)
    assert list(customdata[0])[0] == 0.0
    assert list(customdata[1])[0] == 2.0


def test_chord_strip_figure_short_segments_omit_inline_text():
    """A sub-second segment can't legibly hold a text label -- must not try
    to cram one in and overflow."""
    segments = [
        ChordSegment(start_sec=0.0, end_sec=0.5, label="C"),
        ChordSegment(start_sec=0.5, end_sec=3.0, label="Am"),
    ]

    fig = chord_strip_figure(segments)

    assert list(fig.data[0].text) == ["", "Am"]


def test_chord_strip_figure_empty_segments_does_not_raise():
    fig = chord_strip_figure([])
    assert len(fig.data) == 0


def test_mel_spectrogram_figure_x_axis_is_real_time():
    mel_db = np.random.rand(64, 5) * -80
    times = np.array([0.0, 1.2, 2.4, 3.6, 4.8])

    fig = mel_spectrogram_figure(mel_db, times)

    assert list(fig.data[0].x) == list(times)


def test_mel_spectrogram_figure_y_axis_has_one_frequency_per_mel_bin():
    mel_db = np.random.rand(64, 5) * -80
    times = np.arange(5, dtype=float)

    fig = mel_spectrogram_figure(mel_db, times)

    assert len(fig.data[0].y) == 64
    assert list(fig.data[0].y) == sorted(fig.data[0].y)  # mel frequencies are monotonically increasing


def test_mel_spectrogram_figure_z_values_round_trip_the_mel_matrix():
    mel_db = np.arange(64 * 5, dtype=float).reshape(64, 5)
    times = np.arange(5, dtype=float)

    fig = mel_spectrogram_figure(mel_db, times)

    assert np.array_equal(np.array(fig.data[0].z), mel_db)


def test_waveform_figure_plots_the_real_envelope():
    envelope = np.array([0.1, 0.5, 0.9, 0.3])

    fig = waveform_figure(envelope, duration_sec=4.0)

    assert list(fig.data[0].y) == list(envelope)


def test_waveform_figure_single_highlight_range_adds_one_shaded_region():
    fig = waveform_figure(np.array([0.1, 0.2, 0.3]), duration_sec=3.0, highlight_range=(1.0, 2.0))

    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == 1.0
    assert fig.layout.shapes[0].x1 == 2.0


def test_waveform_figure_highlight_ranges_adds_one_shaded_region_per_segment():
    """Approach Step 1's real use case: each of the four sample window clips
    gets its own shaded region on the full-song waveform, numbered to match
    the players shown below it."""
    envelope = np.zeros(10)
    ranges = [(0.0, 2.5), (5.0, 7.5), (10.0, 12.5), (15.0, 17.5)]

    fig = waveform_figure(envelope, duration_sec=20.0, highlight_ranges=ranges, highlight_labels=["1", "2", "3", "4"])

    assert len(fig.layout.shapes) == 4
    starts = sorted(shape.x0 for shape in fig.layout.shapes)
    assert starts == [0.0, 5.0, 10.0, 15.0]


def test_waveform_figure_highlight_ranges_cycles_colors():
    """More than one region must not all render in the same color, or the
    numbered labels would be the only way to tell them apart."""
    envelope = np.zeros(10)
    ranges = [(0.0, 1.0), (2.0, 3.0)]

    fig = waveform_figure(envelope, duration_sec=10.0, highlight_ranges=ranges)

    colors = {shape.fillcolor for shape in fig.layout.shapes}
    assert len(colors) == 2


def test_waveform_figure_no_highlights_by_default():
    fig = waveform_figure(np.array([0.1, 0.2, 0.3]))
    assert len(fig.layout.shapes) == 0
