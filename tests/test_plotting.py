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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from components.plotting import (
    cluster_density_preview,
    extract_selected_song_id,
    genre_breakdown_bar,
    library_waffle_grid,
    network_graph_figure,
)


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


def test_genre_breakdown_bar_one_trace_per_genre():
    fig = genre_breakdown_bar({"Rock": 10, "Jazz": 5, "Pop": 15})
    assert len(fig.data) == 3


def test_genre_breakdown_bar_sorted_descending_by_count():
    fig = genre_breakdown_bar({"Rock": 10, "Jazz": 5, "Pop": 15})
    assert [trace.name for trace in fig.data] == ["Pop", "Rock", "Jazz"]


def test_genre_breakdown_bar_empty_counts_does_not_raise():
    fig = genre_breakdown_bar({})
    assert len(fig.data) == 0


def test_genre_breakdown_bar_and_waffle_grid_share_colors():
    """The bar and the waffle grid must assign the same color to the same
    genre -- both derive from the same _genre_color_map helper."""
    genre_counts = {"Rock": 10, "Jazz": 5, "Pop": 15}
    songs_df = pd.DataFrame([
        {"title": "A", "genre": "Rock"}, {"title": "B", "genre": "Jazz"}, {"title": "C", "genre": "Pop"},
    ])

    bar_colors = {trace.name: trace.marker.color for trace in genre_breakdown_bar(genre_counts).data}
    waffle_colors = {
        trace.name.rsplit(" (", 1)[0]: trace.marker.color for trace in library_waffle_grid(songs_df, genre_counts).data
    }

    assert bar_colors == waffle_colors


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


def test_cluster_density_preview_plots_every_point():
    points_df = pd.DataFrame([
        {"x": 0.0, "y": 0.0, "cluster": 0},
        {"x": 1.0, "y": 1.0, "cluster": 1},
        {"x": 2.0, "y": -1.0, "cluster": 0},
    ])
    fig = cluster_density_preview(points_df)
    assert len(fig.data[0].x) == 3
