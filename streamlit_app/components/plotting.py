"""Shared Plotly rendering helpers for the interface layer."""

import math

import numpy as np
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go

# Perceptually-uniform colormaps (spec 2.6): color intensity maps linearly to
# value, avoiding misread intensity -- an actual data-viz standard, not a
# decorative choice. All three are built into Plotly, no extra dependency.
FINGERPRINT_COLORSCALE = "Magma"

_GENRE_PALETTE = px.colors.qualitative.Set2

# Fixed, alphabetical -- this library's real 8 FMA genres (see spec/CLAUDE.md),
# not derived from any one view's own song counts. network_graph_figure needs
# a genre->color assignment that's IDENTICAL across every call site (Explore's
# mini-graph, its full graph, Results, App Walkthrough) regardless of which
# subset of songs that particular view happens to show -- a count-based
# ordering (like _genre_color_map below) would assign different colors to the
# same genre in a filtered view (e.g. Explore's "selected song + neighbors
# only" mini-graph) than in the full-library view, which defeats the point of
# a legend a viewer can actually learn.
_KNOWN_GENRES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
]
GENRE_COLOR_MAP = {genre: _GENRE_PALETTE[i % len(_GENRE_PALETTE)] for i, genre in enumerate(_KNOWN_GENRES)}


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


def fingerprint_thumbnail(fingerprint, title: str, height: int = 180, hover_caption: str | None = None) -> go.Figure:
    """A small, axis-free heatmap for a fingerprint array (values in [0, 1]).
    height is configurable -- Explore's 50-songs-per-page browsable list and
    Moment Matcher's result cards need a much smaller thumbnail (~60px) than
    the full detail views (~180px) this originally shipped for.

    hover_caption, when given, replaces Plotly's default per-cell "x/y/color
    value" hover tooltip with this fixed text everywhere on the image --
    Explore's Selected Song panel uses this so its explanation of what the
    fingerprint means shows up on hover only, not as permanently-visible
    text taking up space under a full-detail image.

    width=height in the returned figure's own layout, on top of px.imshow's
    default scaleanchor/constrain="domain", plus a matching fixed pixel
    width (not width="stretch") at the st.plotly_chart() call site --
    fingerprint data really is square (structure_fingerprint always returns
    FINGERPRINT_SIZE x FINGERPRINT_SIZE), and this combination is confirmed
    working for the ~260px Selected Song detail view. It was NOT enough at
    ~56px (list rows, result cards) -- those still rendered as a stretched
    rectangle even with matching width/height declared everywhere, most
    likely a Plotly/Streamlit sizing quirk specific to small
    config={"staticPlot": True} charts. Rather than keep chasing that,
    fingerprint_thumbnail_image() below renders those small cases as a real
    image (st.image()) instead of a chart at all -- st.image has simple,
    well-established exact-pixel sizing that a charting library's
    responsive-by-default model doesn't reliably guarantee at this size."""
    fig = px.imshow(fingerprint, color_continuous_scale=FINGERPRINT_COLORSCALE, origin="lower")
    if hover_caption is not None:
        fig.update_traces(hovertemplate=hover_caption + "<extra></extra>")
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=13)),
        height=height,
        width=height,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        coloraxis_showscale=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def fingerprint_thumbnail_image(fingerprint: np.ndarray, colorscale: str = FINGERPRINT_COLORSCALE) -> np.ndarray:
    """Renders a normalized [0, 1] fingerprint array as a real small RGB
    image (uint8, HxWx3) for st.image() -- see fingerprint_thumbnail's own
    docstring for why tiny (~56px) thumbnails moved off st.plotly_chart()
    entirely rather than continuing to chase a Plotly/Streamlit small-chart
    sizing quirk. Uses plotly.colors.sample_colorscale to apply the SAME
    colorscale (Magma) fingerprint_thumbnail's heatmap uses, so a song's
    thumbnail in the browsable list looks identical to its fingerprint in
    the Selected Song detail view -- just a different rendering path, not a
    different visual. No new dependency: plotly.colors is already part of
    the plotly package this app already depends on, unlike e.g. matplotlib,
    which was deliberately NOT reached for here.

    Flips the array vertically before rendering: fingerprint_thumbnail's
    px.imshow(..., origin="lower") puts row 0 at the BOTTOM (the
    math/audio convention this app uses everywhere else), while st.image()
    always renders row 0 at the top (the standard image convention) --
    without the flip, the list's thumbnails and the detail view's
    fingerprint for the same song would visibly disagree with each other."""
    fingerprint = np.flipud(fingerprint)
    flat = np.clip(fingerprint, 0.0, 1.0).ravel()
    sampled = pc.sample_colorscale(colorscale, flat.tolist())
    rgb = np.array([[int(component) for component in color[4:-1].split(", ")] for color in sampled], dtype=np.uint8)
    return rgb.reshape(fingerprint.shape[0], fingerprint.shape[1], 3)


def fingerprint_image_data_uri(fingerprint: np.ndarray, colorscale: str = FINGERPRINT_COLORSCALE) -> str:
    """fingerprint_thumbnail_image() encoded as a base64 PNG data URI, for
    the ONE remaining case that needs a real <img> tag instead of
    st.image(): Explore's Selected Song panel, which wants a hover
    explanation of what the fingerprint means. st.image() has no hover-
    tooltip mechanism at all -- a raw <img title="..."> gets a native
    browser tooltip on hover for free, no JS needed, and (unlike Plotly's
    hovertemplate) doesn't require any interactivity layer to work.

    This is also the fix for a real reported bug: the Selected Song
    panel's fingerprint (previously still rendered via fingerprint_thumbnail(),
    a Plotly chart) looked visibly different from the list/result-card
    thumbnails (fingerprint_thumbnail_image(), a real image) for the SAME
    song, despite both sampling the identical colorscale at identical data
    values (verified directly) -- something about the two different
    rendering pipelines (Plotly heatmap vs. a raw image) still produced a
    different look. Rather than keep reconciling two implementations,
    every fingerprint display in Explore now draws from this one image
    pipeline -- Plotly is no longer involved in fingerprint rendering at
    all, so there's only one visual to keep consistent, by construction."""
    import base64
    import io

    from PIL import Image

    rgb = fingerprint_thumbnail_image(fingerprint, colorscale)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def composite_fingerprint_image_data_uri(composite: np.ndarray) -> str:
    """composite_fingerprint()'s RGB overlay (structure=red, harmony=green,
    sound=blue), base64 PNG-encoded for an ImageColumn cell -- the same
    fingerprint_image_data_uri exists for, just skipping fingerprint_
    thumbnail_image's colorscale-sampling step, since composite is already
    RGB in [0, 1], not a single-channel value needing a colorscale mapped
    onto it. Same flipud as fingerprint_thumbnail_image and for the same
    reason: composite is built by stacking structure/sound/harmony
    fingerprints, which all share their row-0-at-bottom convention, while
    an <img> tag always renders row 0 at the top."""
    import base64
    import io

    from PIL import Image

    rgb = (np.clip(np.flipud(composite), 0.0, 1.0) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def composite_fingerprint_thumbnail(composite, title: str = "Composite", height: int = 180) -> go.Figure:
    """The three-facet RGB overlay (structure=red, harmony=green, sound=blue) --
    no color_continuous_scale needed, the array is already RGB in [0, 1].
    width=height for the same reason fingerprint_thumbnail sets it -- see
    that function's docstring."""
    fig = px.imshow(composite, origin="lower")
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=13)),
        height=height,
        width=height,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
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


_SEGMENT_HIGHLIGHT_COLORS = [
    "rgba(239,85,59,0.28)", "rgba(0,204,150,0.28)", "rgba(255,161,90,0.28)", "rgba(171,99,250,0.28)",
]


def waveform_figure(
    envelope, title: str = "", highlight_range: tuple[float, float] | None = None,
    duration_sec: float | None = None, color: str = "rgb(99,110,250)", height: int = 140,
    highlight_ranges: list[tuple[float, float]] | None = None, highlight_labels: list[str] | None = None,
) -> go.Figure:
    """Real amplitude-envelope waveform plot (see analysis/waveform_preview.
    waveform_envelope) -- Approach's step-by-step visuals build on this
    instead of abstract shapes, so the mechanic being explained (slicing,
    isolating, collapsing to a point) is shown against real audio, not a
    generic placeholder. highlight_range (start_sec, end_sec) shades a
    single region -- used for the segmentation-window step; duration_sec
    maps the envelope's fixed n_points onto a real time axis, defaulting to
    sample index if not given. highlight_ranges shades several regions at
    once, each a distinct color (cycled from _SEGMENT_HIGHLIGHT_COLORS) with
    an optional label from highlight_labels at the same index -- used on
    Step 1's full-song waveform to show exactly which stretches the sample
    window clips below it cover, rather than leaving the connection between
    "the waveform" and "these four playable clips" implicit."""
    n = len(envelope)
    x = np.linspace(0, duration_sec, n) if duration_sec else np.arange(n)
    fig = go.Figure(go.Scatter(x=x, y=envelope, mode="lines", fill="tozeroy", line=dict(color=color, width=1)))
    if highlight_range is not None:
        fig.add_vrect(
            x0=highlight_range[0], x1=highlight_range[1],
            fillcolor="rgba(255,255,255,0.15)", line_width=0,
        )
    if highlight_ranges:
        for i, (start, end) in enumerate(highlight_ranges):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=_SEGMENT_HIGHLIGHT_COLORS[i % len(_SEGMENT_HIGHLIGHT_COLORS)], line_width=0,
                annotation_text=highlight_labels[i] if highlight_labels and i < len(highlight_labels) else None,
                annotation_position="top left", annotation_font_size=10,
            )
    # ticksuffix="s" (0s, 5s, 10s...) instead of a separate "seconds" axis
    # title -- a real reported bug: at small heights (Moment Matcher's
    # result-card waveforms, height=70) there wasn't enough room for both
    # the tick labels AND a title, so the title got visibly clipped. The
    # suffix carries the same unit information inline, with no second line
    # to make room for.
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30 if title else 5, b=20),
        title=dict(text=title or "", font=dict(size=13)),
        xaxis=dict(ticksuffix="s" if duration_sec else None), yaxis=dict(visible=False),
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def filter_to_neighbors(nodes_df, edges, center_song_id):
    """Restricts a full similarity graph down to one song and its direct
    k-NN neighbors -- Explore's "simplified network view": the full-library
    graph (~1400 nodes) reads as a tangled mess, so this answers the
    narrower, more directly useful question "what IS this song actually
    connected to?" instead. A pure filter over already-computed nodes_df/
    edges (analysis/network_graph.py's real k-NN graph is unchanged) -- no
    new graph-construction logic, since GraphEdge already records exactly
    which songs are connected to which; this just keeps the subset touching
    center_song_id. Edges aren't necessarily symmetric (song A's top-k
    neighbors can include B without B's own top-k including A), so this
    keeps every edge touching center_song_id from EITHER side. Returns
    (filtered_nodes_df, filtered_edges); an unrecognized center_song_id
    returns everything unfiltered rather than an empty graph, since that's
    a more useful fallback than a blank chart."""
    if center_song_id not in set(nodes_df["song_id"]):
        return nodes_df, edges

    neighbor_ids = {center_song_id}
    connecting_edges = []
    for edge in edges:
        if edge.song_id_a == center_song_id:
            neighbor_ids.add(edge.song_id_b)
            connecting_edges.append(edge)
        elif edge.song_id_b == center_song_id:
            neighbor_ids.add(edge.song_id_a)
            connecting_edges.append(edge)

    filtered_nodes = nodes_df[nodes_df["song_id"].isin(neighbor_ids)]
    return filtered_nodes, connecting_edges


def taste_map_figure(points_df, selected_song_id=None, height: int = 180, click_priority: bool = False) -> go.Figure:
    """PCA/ICA spatial scatter (analysis.taste_map.compute_taste_map) as a
    real, clickable alternative to the network graph -- points_df needs
    columns song_id, x, y, cluster; title/artist/genre are optional (used
    for a real hover tooltip when present, hoverinfo stays "skip" for a
    caller that doesn't supply them, e.g. this function's own tests). Single
    go.Scatter markers trace (the same click-to-select mechanism
    network_graph_figure's own node trace already uses, and the one Plotly
    mark type this app has consistently gotten reliable clicks from -- see
    segment_waveform_bars_figure's docstring for the real bug history behind
    sticking to a single, unambiguous trace). Cluster coloring uses the same
    qualitative _GENRE_PALETTE network_graph_figure's nodes use and for the
    same reason (cluster ids are categorical, not ordered -- see that
    function's docstring). No edges here -- PCA/ICA position IS the signal,
    unlike the network graph where edges are real k-NN connections; drawing
    invented lines between nearby points would imply graph structure this
    projection doesn't have.

    click_priority: when True, sets dragmode=False -- see
    network_graph_figure's docstring for why (a real reported "clicking a
    node doesn't work reliably" bug, root-caused to Plotly's default pan
    dragmode swallowing clicks that have even a pixel of mouse movement
    during them). Opt-in, not the default, so a caller that never wires
    on_select (there are none for this function today, but network_graph_
    figure has two) doesn't lose free pan/zoom exploration it never needed
    fixing."""
    cluster_colors = [_GENRE_PALETTE[int(c) % len(_GENRE_PALETTE)] for c in points_df["cluster"]]
    has_meta = {"title", "artist", "genre"}.issubset(points_df.columns)
    hover_kwargs = (
        dict(hovertext=[f"{t}<br>{a} · {g}" for t, a, g in zip(
            points_df["title"], points_df["artist"], points_df["genre"], strict=True,
        )], hoverinfo="text")
        if has_meta else dict(hoverinfo="skip")
    )
    fig = go.Figure(go.Scatter(
        x=points_df["x"], y=points_df["y"], mode="markers",
        marker=dict(
            size=9,
            color=cluster_colors,
            line=dict(
                width=[2.5 if sid == selected_song_id else 0 for sid in points_df["song_id"]],
                color="#FFFFFF",
            ),
        ),
        customdata=[[sid] for sid in points_df["song_id"]],
        showlegend=False,
        **hover_kwargs,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode=False if click_priority else None,
    )
    return fig


def network_graph_figure(
    nodes_df, edges, selected_song_id=None, highlight_song_ids=None, center_song_id=None, zoom_radius=None,
    height: int = 560, click_priority: bool = False,
) -> go.Figure:
    """Song-as-node similarity graph (spec 2.1's network/relationship view) --
    nodes_df needs columns song_id, x, y, cluster, title, artist, genre; edges
    is a list of analysis.network_graph.GraphEdge. Edges render as one line
    trace (None-separated segments -- the standard Plotly technique for
    drawing many disconnected line segments in a single trace) underneath the
    node scatter. Nodes carry a real hover tooltip (title/artist/genre) --
    reversed from an earlier "deliberately no hover tooltips, click is the
    only way to see song info" design once hover was explicitly requested as
    a first-class interaction alongside click, not instead of it; the earlier
    reasoning (Plotly tooltips can't render the fingerprint imagery a click's
    own payoff panel does) still holds for WHY a click is needed too, it just
    no longer argues against hover existing at all.

    highlight_song_ids: an iterable of song_ids (e.g. a search's matches) to
    draw with a distinct ring + larger marker, separate from selected_song_id
    (the single clicked/active node, which keeps its own white-ring style).
    center_song_id + zoom_radius: when both given, sets the axis ranges to a
    zoom_radius-sized window around that node's (x, y) position -- "center on
    a focused song" -- instead of the default autoranged full-graph view.

    click_priority: when True, sets dragmode=False. Root cause of a real
    "clicking a node doesn't work reliably" report: Plotly's default pan
    dragmode treats ANY mouse movement between mousedown and mouseup as a
    pan gesture and cancels the click event outright -- and a perfectly
    ordinary click (mouse or trackpad) almost always has a pixel or two of
    movement in it, so clicks were being silently swallowed as micro-pans,
    not lost to any bug in this app's own selection-handling code. Disabling
    drag entirely doesn't touch click detection at all (Plotly fires
    click/select via clickmode, a separate mechanism from dragmode), and
    scroll-wheel zoom (config={"scrollZoom": True} at every call site that
    also sets on_select) keeps working since it isn't gated by dragmode
    either -- only click-and-drag panning is what's given up. Opt-in (not
    the default) because Results/App Walkthrough render this same figure
    purely as static evidence, with no on_select wired up at all -- there's
    no click to protect there, only free pan/zoom exploration to keep."""
    pos = {row.song_id: (row.x, row.y) for row in nodes_df.itertuples()}
    highlight_ids = frozenset(highlight_song_ids) if highlight_song_ids else frozenset()
    edge_x, edge_y = [], []
    for edge in edges:
        if edge.song_id_a not in pos or edge.song_id_b not in pos:
            continue
        x0, y0 = pos[edge.song_id_a]
        x1, y1 = pos[edge.song_id_b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Genre, not the unsupervised K-means cluster id -- a real reported
    # preference (cluster coloring read as "insignificant"/hard to learn
    # from, whereas genre is an immediately meaningful, checkable label).
    # GENRE_COLOR_MAP is fixed/alphabetical (see its own module-level
    # comment) so the same genre gets the same color in every view, not
    # just within one render. The 2D taste map (taste_map_figure) keeps its
    # own cluster-vs-genre "Color by" toggle unchanged -- that toggle IS
    # the deliberate cluster/genre comparison the map exists to support;
    # this graph never had one, so there's no comparison feature to
    # preserve here, only a single fixed choice to make.
    node_colors = [GENRE_COLOR_MAP.get(g, "#888888") for g in nodes_df["genre"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.6, color="rgba(150,150,150,0.35)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=nodes_df["x"], y=nodes_df["y"], mode="markers",
        marker=dict(
            size=[13 if sid in highlight_ids else 9 for sid in nodes_df["song_id"]],
            color=node_colors,
            line=dict(
                width=[
                    2.5 if sid == selected_song_id else (2 if sid in highlight_ids else 0)
                    for sid in nodes_df["song_id"]
                ],
                color=[
                    "#FFFFFF" if sid == selected_song_id else "#FFD54A"
                    for sid in nodes_df["song_id"]
                ],
            ),
        ),
        customdata=[[sid] for sid in nodes_df["song_id"]],
        hovertext=[f"{t}<br>{a} · {g}" for t, a, g in zip(
            nodes_df["title"], nodes_df["artist"], nodes_df["genre"], strict=True,
        )],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode=False if click_priority else None,
    )
    if center_song_id is not None and zoom_radius is not None and center_song_id in pos:
        cx, cy = pos[center_song_id]
        fig.update_xaxes(range=[cx - zoom_radius, cx + zoom_radius])
        fig.update_yaxes(range=[cy - zoom_radius, cy + zoom_radius])
    return fig


def network_hero_figure(nodes_df, edges, height: int = 110) -> go.Figure:
    """A muted, non-interactive banner built from the SAME real per-song
    similarity graph network_graph_figure() renders on Explore -- real node
    positions and real k-NN edges from the library's own embeddings, just
    desaturated and stripped of axes/legend/cluster-color/hover/click so it
    reads as page decoration rather than a second, confusing copy of the
    interactive graph. This is the "no AI-slop hero art" answer: rather than
    a generic decorative image, every page's banner is quietly showing an
    honest (if muted) picture of the actual similarity structure this whole
    project is about. config={"staticPlot": True} at the call site disables
    all interactivity -- callers should never wire on_select to this figure,
    it's not the click target."""
    edge_x, edge_y = [], []
    pos = {row.song_id: (row.x, row.y) for row in nodes_df.itertuples()}
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
        line=dict(width=0.5, color="rgba(200,200,200,0.18)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=nodes_df["x"], y=nodes_df["y"], mode="markers",
        marker=dict(size=4, color="rgba(200,200,200,0.4)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


CHROMA_PITCH_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chromagram_figure(chroma: np.ndarray, times: np.ndarray, title: str = "", height: int = 220) -> go.Figure:
    """Chroma-over-time heatmap (see analysis.waveform_preview.chroma_for_display)
    -- 12 pitch-class rows x real time on the x-axis. Deliberately NOT wired
    for click-to-seek itself: unlike go.Scatter/go.Bar (marker/bar traces,
    which Plotly treats as directly selectable on a plain click), go.Heatmap
    doesn't fire a plotly_selected event on a simple click -- confirmed as
    the real cause of a live "click the chromagram, nothing happens" bug
    report, not a fluke. chord_strip_figure() (a Bar trace, the same proven
    click-to-seek technique Song X-Ray's structure timeline already uses)
    is the real seek control for this pair of visuals; this heatmap is
    display-only."""
    fig = go.Figure(go.Heatmap(
        z=chroma, x=times, y=CHROMA_PITCH_LABELS,
        colorscale="Magma", showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30 if title else 5, b=30),
        title=dict(text=title or "", font=dict(size=13)),
        xaxis=dict(title="seconds"), yaxis=dict(title=None),
    )
    return fig


def mel_spectrogram_figure(mel_db: np.ndarray, times: np.ndarray, height: int = 220) -> go.Figure:
    """Mel-spectrogram-over-time heatmap (see
    analysis.waveform_preview.mel_spectrogram_for_display) -- Sound's
    supporting/contextual visual: the raw acoustic texture (frequency energy
    over time) Sound is computed *from*, not a view of CLAP's actual learned
    embedding space, which has no raw visual proxy the way chroma is
    Harmony's literal input. A go.Heatmap, same as chromagram_figure and for
    the same reason not wired for click-to-seek -- heatmaps don't fire a
    plotly_selected event on a plain click the way Bar/Scatter traces do."""
    n_mels = mel_db.shape[0]
    import librosa

    freqs = librosa.mel_frequencies(n_mels=n_mels)
    fig = go.Figure(go.Heatmap(
        z=mel_db, x=times, y=freqs,
        colorscale="Magma", showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=5, b=30),
        xaxis=dict(title="seconds"), yaxis=dict(title="Hz", type="log"),
    )
    return fig


def chord_strip_figure(segments, height: int = 70) -> go.Figure:
    """Horizontal labeled-block strip of chord segments (see
    analysis.key_chord.estimate_chords) -- one bar per (start_sec, end_sec,
    label) segment, meant to sit directly under chromagram_figure so a
    viewer sees which chord goes with which stretch of the heatmap above it.
    Same horizontal-bar-segment technique Song X-Ray's structure timeline
    already uses (pages/4_Song_XRay.py), reused here rather than inventing a
    new annotation-overlay scheme -- and, unlike chromagram_figure's heatmap,
    this Bar trace genuinely supports click-to-select, so this is the real
    seek control for the pair: customdata carries each segment's start_sec
    for a caller to read back via event.selection.points[0]["customdata"][0]."""
    if not segments:
        return go.Figure()

    durations = [s.end_sec - s.start_sec for s in segments]
    starts = [s.start_sec for s in segments]
    labels = [s.label for s in segments]
    # Text only on bars wide enough to hold it -- a label crammed into a
    # sub-second segment would just overflow illegibly.
    text = [label if dur >= 1.0 else "" for label, dur in zip(labels, durations, strict=False)]

    fig = go.Figure(go.Bar(
        x=durations, y=["Chord"] * len(segments), base=starts, orientation="h",
        text=text, textposition="inside", insidetextanchor="middle",
        customdata=[[s.start_sec] for s in segments],
        hovertext=[f"{lab}: {s.start_sec:.1f}s–{s.end_sec:.1f}s" for lab, s in zip(labels, segments, strict=False)],
        hoverinfo="text", marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.3)",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=5, b=20),
        xaxis=dict(title="seconds"), yaxis=dict(showticklabels=False),
        showlegend=False, bargap=0,
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
        title=dict(text=title or "", font=dict(size=13)),
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
        title=dict(text=title or "", font=dict(size=13)),
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
