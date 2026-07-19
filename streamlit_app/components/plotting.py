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
