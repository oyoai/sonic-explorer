"""Shared Plotly rendering helpers for the interface layer."""

import plotly.express as px
import plotly.graph_objects as go

# Perceptually-uniform colormaps (spec 2.6): color intensity maps linearly to
# value, avoiding misread intensity -- an actual data-viz standard, not a
# decorative choice. All three are built into Plotly, no extra dependency.
FINGERPRINT_COLORSCALE = "Magma"


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
        customdata=nodes_df[["song_id"]],
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
