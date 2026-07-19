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
