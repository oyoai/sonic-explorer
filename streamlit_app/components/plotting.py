"""Shared Plotly rendering helpers for the interface layer."""

import math

import plotly.express as px
import plotly.graph_objects as go

# Perceptually-uniform colormaps (spec 2.6): color intensity maps linearly to
# value, avoiding misread intensity -- an actual data-viz standard, not a
# decorative choice. All three are built into Plotly, no extra dependency.
FINGERPRINT_COLORSCALE = "Magma"

_GENRE_PALETTE = px.colors.qualitative.Set2


def _genre_color_map(genre_counts: dict[str, int]) -> dict[str, str]:
    """Consistent genre->color assignment, sorted largest-genre-first --
    shared by every genre-colored visual on the Overview page (the waffle
    grid and the composition bar) so the same genre reads as the same color
    in both rather than each picking colors independently."""
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


def genre_breakdown_bar(genre_counts: dict[str, int]) -> go.Figure:
    """A single horizontal stacked bar, one segment per genre sized by its
    share of the library -- a compact "at a glance" composition view for the
    Overview page, in the same small-multiple spirit as the fingerprint
    thumbnails above. Sorted descending so the largest genre anchors the left
    edge, matching how Methodology's own genre bar chart orders things."""
    items = sorted(genre_counts.items(), key=lambda kv: -kv[1])
    total = sum(genre_counts.values()) or 1
    color_map = _genre_color_map(genre_counts)

    fig = go.Figure()
    for genre, count in items:
        fig.add_trace(go.Bar(
            x=[count], y=["Library"], orientation="h", name=genre,
            marker=dict(color=color_map[genre]),
            text=f"{genre} ({count / total:.0%})", textposition="inside", insidetextanchor="middle",
            hovertemplate=f"{genre}: {count} songs (%{{customdata:.0%}})<extra></extra>",
            customdata=[count / total],
        ))
    fig.update_layout(
        barmode="stack",
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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
    lightweight, roughly-fixed-height landing-page element whether it's
    rendering a 200-song deploy subset or the full ~1400-song local library."""
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


def cluster_density_preview(points_df) -> go.Figure:
    """A small, non-interactive preview of the library's sonic clustering --
    points_df needs columns x, y, cluster (see analysis.taste_map.compute_taste_map).
    Deliberately stripped down (no axes, no legend, no hover) since this is a
    teaser for the real, full Explore map, not a substitute for it."""
    fig = go.Figure(go.Scatter(
        x=points_df["x"], y=points_df["y"], mode="markers",
        marker=dict(
            size=5, color=points_df["cluster"], colorscale="Viridis", opacity=0.75,
            line=dict(width=0),
        ),
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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
