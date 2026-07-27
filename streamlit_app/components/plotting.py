"""Shared Plotly rendering helpers for the interface layer."""

import math

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Perceptually-uniform colormaps (spec 2.6): color intensity maps linearly to
# value, avoiding misread intensity -- an actual data-viz standard, not a
# decorative choice. All three are built into Plotly, no extra dependency.
FINGERPRINT_COLORSCALE = "Magma"

_GENRE_PALETTE = px.colors.qualitative.Set2


def _genre_color_map(genre_counts: dict[str, int]) -> dict[str, str]:
    """Consistent genre->color assignment, sorted largest-genre-first --
    used by Methodology's waffle grid (library_waffle_grid)."""
    ordered = sorted(genre_counts.keys(), key=lambda g: -genre_counts[g])
    return {genre: _GENRE_PALETTE[i % len(_GENRE_PALETTE)] for i, genre in enumerate(ordered)}


def extract_selected_song_id(point):
    """None if this selection-event point doesn't carry a usable song_id --
    e.g. a click that landed on network_graph_figure's non-interactive-by-
    intent edges trace (mode="lines", no customdata) rather than a node
    marker, close enough to a line for Plotly to register the click there
    instead. Real-bug regression guard: AppTest's simulated selection events
    don't reproduce this the way an actual browser click does, so this stays
    defensive (try/except, not a bare index) regardless of the point's exact
    shape rather than assuming customdata is always present and non-empty."""
    try:
        return point["customdata"][0]
    except (KeyError, IndexError, TypeError):
        return None


def fingerprint_thumbnail(fingerprint, title: str) -> go.Figure:
    """A small, axis-free heatmap for a fingerprint array (values in [0, 1])."""
    fig = px.imshow(fingerprint, color_continuous_scale=FINGERPRINT_COLORSCALE, origin="lower")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=180,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_showscale=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def composite_fingerprint_thumbnail(composite, title: str = "Composite") -> go.Figure:
    """The three-facet RGB overlay (structure=red, harmony=green, sound=blue) --
    no color_continuous_scale needed, the array is already RGB in [0, 1]."""
    fig = px.imshow(composite, origin="lower")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=180,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def concept_bubble_diagram(center_label: str, satellite_labels: list[str]) -> go.Figure:
    """A simple radial diagram -- a center concept bubble surrounded by the
    signals that feed it, connected by thin lines. Illustrative only, not
    real computed data (unlike every other chart in this app) -- Overview
    uses this to sketch how existing recommendation paradigms conceptually
    work and to illustrate the facet-based approach, before the real
    metadata-vs-audio network graph (further down the same page) and the
    real audio-based graph (Results) take over as actual evidence."""
    n = len(satellite_labels)
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    sat_x = [math.cos(a) for a in angles]
    sat_y = [math.sin(a) for a in angles]

    line_x, line_y = [], []
    for x, y in zip(sat_x, sat_y, strict=False):
        line_x += [0, x, None]
        line_y += [0, y, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines",
        line=dict(width=1, color="rgba(150,150,150,0.5)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=sat_x, y=sat_y, mode="markers+text",
        marker=dict(size=64, color="rgba(99,110,250,0.30)", line=dict(width=1.5, color="rgb(99,110,250)")),
        text=satellite_labels, textposition="middle center", textfont=dict(size=12),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=120, color="rgba(239,85,59,0.30)", line=dict(width=2, color="rgb(239,85,59)")),
        text=[center_label], textposition="middle center", textfont=dict(size=13),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-1.7, 1.7]),
        yaxis=dict(visible=False, range=[-1.7, 1.7], scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


def waveform_figure(
    envelope, title: str = "", highlight_range: tuple[float, float] | None = None,
    duration_sec: float | None = None, color: str = "rgb(99,110,250)", height: int = 140,
) -> go.Figure:
    """Real amplitude-envelope waveform plot (see analysis/waveform_preview.
    waveform_envelope) -- Approach's step-by-step visuals build on this
    instead of abstract shapes, so the mechanic being explained (slicing,
    isolating, collapsing to a point) is shown against real audio, not a
    generic placeholder. highlight_range (start_sec, end_sec) shades a
    region -- used for the segmentation-window step; duration_sec maps the
    envelope's fixed n_points onto a real time axis, defaulting to sample
    index if not given."""
    n = len(envelope)
    x = np.linspace(0, duration_sec, n) if duration_sec else np.arange(n)
    fig = go.Figure(go.Scatter(x=x, y=envelope, mode="lines", fill="tozeroy", line=dict(color=color, width=1)))
    if highlight_range is not None:
        fig.add_vrect(
            x0=highlight_range[0], x1=highlight_range[1],
            fillcolor="rgba(255,255,255,0.15)", line_width=0,
        )
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30 if title else 5, b=20),
        title=dict(text=title, font=dict(size=13)) if title else None,
        xaxis=dict(title="seconds" if duration_sec else None), yaxis=dict(visible=False),
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def network_graph_figure(nodes_df, edges, selected_song_id=None) -> go.Figure:
    """Song-as-node similarity graph (spec 2.1's network/relationship view) --
    nodes_df needs columns song_id, x, y, cluster, title, artist, genre; edges
    is a list of analysis.network_graph.GraphEdge. Edges render as one line
    trace (None-separated segments -- the standard Plotly technique for
    drawing many disconnected line segments in a single trace) underneath the
    node scatter. Deliberately no hover tooltips -- click is the only way to
    see song info (Plotly tooltips can't render the fingerprint/thumbnail
    imagery well, so this avoids fighting the tool for a payoff click already
    provides via the player section below)."""
    pos = {row.song_id: (row.x, row.y) for row in nodes_df.itertuples()}
    edge_x, edge_y = [], []
    for edge in edges:
        if edge.song_id_a not in pos or edge.song_id_b not in pos:
            continue
        x0, y0 = pos[edge.song_id_a]
        x1, y1 = pos[edge.song_id_b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.6, color="rgba(150,150,150,0.35)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=nodes_df["x"], y=nodes_df["y"], mode="markers",
        marker=dict(
            size=9, color=nodes_df["cluster"], colorscale="Viridis",
            line=dict(
                width=[2.5 if sid == selected_song_id else 0 for sid in nodes_df["song_id"]],
                color="#FFFFFF",
            ),
        ),
        customdata=[[sid] for sid in nodes_df["song_id"]],
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def library_waffle_grid(songs_df, genre_counts: dict[str, int]) -> go.Figure:
    """One small square per song (songs_df needs columns title, genre),
    arranged in contiguous same-genre blocks -- the classic waffle-chart
    convention -- so genre proportions read directly from the grid's shape,
    not just individual colors. One trace per genre gives Plotly's native
    legend (color swatch + name + count) for free, readable without hovering
    anything; hovering a single cell additionally shows that one song's
    title and genre. Grid/cell size adapts to library size so this stays a
    lightweight, roughly-fixed-height element (Methodology's dataset section)
    whether it's rendering a ~200-song deploy subset or the full ~1400-song
    local library."""
    color_map = _genre_color_map(genre_counts)
    ordered_genres = sorted(genre_counts.keys(), key=lambda g: -genre_counts[g])

    songs_by_genre: dict[str, list[str]] = {g: [] for g in ordered_genres}
    for row in songs_df.itertuples():
        if row.genre in songs_by_genre:
            songs_by_genre[row.genre].append(row.title)

    n = len(songs_df)
    cols = max(1, math.ceil(math.sqrt(n * 2.2))) if n else 1
    rows = math.ceil(n / cols) if cols else 1
    cell_px = max(4, min(14, 260 // max(rows, 1)))

    fig = go.Figure()
    position = 0
    for genre in ordered_genres:
        titles = songs_by_genre[genre]
        xs, ys, hover = [], [], []
        for title in titles:
            row_idx, col_idx = divmod(position, cols)
            xs.append(col_idx)
            ys.append(-row_idx)
            hover.append(f"{title} — {genre}")
            position += 1
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=f"{genre} ({len(titles)})",
            marker=dict(symbol="square", size=cell_px, color=color_map[genre], line=dict(width=0)),
            hovertext=hover, hoverinfo="text",
        ))

    fig.update_layout(
        height=rows * cell_px + 110,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[-1, cols]),
        yaxis=dict(visible=False, range=[-rows, 1], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0.5, xanchor="center", font=dict(size=11)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def embedding_strip_figure(vec, n_dims: int = 48, title: str = "") -> go.Figure:
    """A strip of discrete colored cells, one per embedding dimension --
    deliberately NOT a connected line the way waveform_figure renders audio.
    A line plot would visually imply the dimensions have a meaningful order/
    continuity to trace across, which they don't -- a CLAP embedding's
    dimensions are independent coordinates in a space, not a signal over
    time, and discrete cells are the honest way to show that."""
    preview = np.asarray(vec[:n_dims]).reshape(1, -1)
    fig = px.imshow(preview, color_continuous_scale="Viridis", aspect="auto")
    fig.update_layout(
        height=90, margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        title=dict(text=title, font=dict(size=13)) if title else None,
        coloraxis_showscale=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def close_by_illustration() -> go.Figure:
    """A small, illustrative "nearest neighbors in embedding space" sketch --
    NOT the real network graph (that's a later step, and Results/Explore's
    job once the mechanism has actually been explained). Fixed, hand-placed
    points, not real songs or real distances: one query point at the
    center, two genuinely "close" points connected by a line, a few "far"
    points left unconnected -- illustrating that similarity search just
    means distance, nothing more, before the real graph makes the same
    point with actual data."""
    query = (0.0, 0.0)
    near = [(0.35, 0.25), (-0.2, 0.4)]
    far = [(1.3, -0.9), (-1.4, -0.6), (0.9, 1.3)]

    fig = go.Figure()
    for nx, ny in near:
        fig.add_trace(go.Scatter(
            x=[query[0], nx], y=[query[1], ny], mode="lines",
            line=dict(width=1.5, color="rgba(99,110,250,0.6)"), hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[p[0] for p in far], y=[p[1] for p in far], mode="markers",
        marker=dict(size=14, color="rgba(150,150,150,0.5)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[p[0] for p in near], y=[p[1] for p in near], mode="markers",
        marker=dict(size=16, color="rgba(99,110,250,0.85)"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[query[0]], y=[query[1]], mode="markers",
        marker=dict(size=22, color="rgba(239,85,59,0.9)", line=dict(width=2, color="white")),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-2, 2]),
        yaxis=dict(visible=False, range=[-1.5, 1.7], scaleanchor="x", scaleratio=1),
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


def song_dna_bars(axis_labels: list[str], values: list[float], title: str = "") -> go.Figure:
    """One song's normalized ([0,1] per axis) song-DNA profile as a simple
    horizontal bar chart -- unlike song_dna_radar_overlay (built for
    overlaying two songs' shapes against each other), this is for showing a
    single song's profile on its own, e.g. Approach's step-by-step walkthrough
    where only one example song is in play at that point."""
    fig = go.Figure(go.Bar(
        x=values, y=axis_labels, orientation="h",
        marker=dict(color="rgb(0,204,150)"), text=[f"{v:.2f}" for v in values], textposition="auto",
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        title=dict(text=title, font=dict(size=13)) if title else None,
        xaxis=dict(range=[0, 1], title=None), yaxis=dict(autorange="reversed"),
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def song_dna_radar_overlay(
    axis_labels: list[str],
    values_a: list[float],
    label_a: str,
    values_b: list[float],
    label_b: str,
) -> go.Figure:
    """Two songs' normalized ([0,1] per axis) song-DNA profiles overlaid,
    semi-transparent -- where the shapes agree, they overlap; where one bulges
    past the other, they diverge (spec 2.2). Values must already be normalized
    -- see analysis/song_dna.py's DNANormalizer."""
    # Scatterpolar doesn't auto-close the loop -- repeat the first point/label.
    theta = axis_labels + [axis_labels[0]]
    r_a = values_a + [values_a[0]]
    r_b = values_b + [values_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_a, theta=theta, fill="toself", name=label_a,
        line=dict(color="rgb(99,110,250)"), fillcolor="rgba(99,110,250,0.3)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_b, theta=theta, fill="toself", name=label_b,
        line=dict(color="rgb(239,85,59)"), fillcolor="rgba(239,85,59,0.3)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        showlegend=True,
        height=350,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig
