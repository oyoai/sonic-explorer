"""Trimmed copies of chart builders from streamlit_app/components/plotting.py
-- copied rather than imported because streamlit_app isn't an installed
package (see CLAUDE.md's "Environment-dependent imports": only sonic_explorer*
is actually pip-installed per pyproject.toml, so anything under streamlit_app/
is only importable locally by accident of sys.path, and would break on
Streamlit Community Cloud). This app is a deliberately separate, isolated
deployable, so it carries its own minimal copies instead of reaching across
into the other app's directory. Trimmed to just the params/branches this
demo actually uses -- e.g. network_graph_figure here drops the original's
center_song_id/zoom_radius/hide_isolated_nodes params (no caller in this app
needs a focused/zoomed sub-view yet) but adds a facet_label param the
original doesn't have, so an edge's hover can say which facet's similarity
it represents (streamlit_app's version never needed this: it only ever
shows one facet's graph per page with the facet named in surrounding page
copy, not per edge).

Also owns the "green pill" match-score badge -- a small, scoped CSS
addition (not streamlit_app/resources.py's full inject_global_styles(),
which also pulls in Google Fonts and page-wide rules this app has no other
use for) mirroring that file's own .match-badge rule, just for this one
element.

And the presenter-only Q/M play-pause keyboard shortcuts for
pages/1_Moment_Matcher.py -- see inject_keyboard_shortcuts()'s own
docstring for the real mechanics (an st.iframe reaching into
window.parent.document, since that's the only way injected JS gets a
document-level keydown listener at all; nothing here goes through
Streamlit's own component protocol or touches Python state).
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_MATCH_PILL_CSS_INJECTED_KEY = "_match_pill_css_injected"
_KEYBOARD_SHORTCUTS_INJECTED_KEY = "_keyboard_shortcuts_injected"

# Same fixed, alphabetical genre->color assignment as streamlit_app/components/
# plotting.py's GENRE_COLOR_MAP -- fixed (not derived from whichever songs a
# given view happens to show) so a genre reads as the same color everywhere
# in this app, not just within one chart. This app's whole library (deploy_data,
# 233 songs) only ever spans these 8 real FMA genres.
_GENRE_PALETTE = px.colors.qualitative.Set2
_KNOWN_GENRES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
]
GENRE_COLOR_MAP = {genre: _GENRE_PALETTE[i % len(_GENRE_PALETTE)] for i, genre in enumerate(_KNOWN_GENRES)}

# pages/1_Moment_Matcher.py's per-facet waveform coloring -- the query
# waveform always renders in QUERY_WAVEFORM_COLOR (a neutral gray) regardless
# of facet, so a viewer's eye has one fixed "this is the query" anchor across
# all six tabs; the MATCH waveform gets its own facet's color from this map,
# so switching tabs also visibly changes the match's color, reinforcing that
# each tab is a genuinely different similarity signal, not just a relabeled
# copy of the same one. Colors are Plotly's own default qualitative sequence
# (already used elsewhere in this app, e.g. song DNA overlays), picked so
# none of them reads as the same gray as the query waveform.
QUERY_WAVEFORM_COLOR = "rgb(140,140,140)"
FACET_WAVEFORM_COLORS = {
    "sound": "rgb(99,110,250)",
    "harmony": "rgb(239,85,59)",
    "vocal": "rgb(0,204,150)",
    "drums": "rgb(255,161,90)",
    "bass": "rgb(171,99,250)",
    "instrumental": "rgb(25,211,243)",
}


def waveform_figure(envelope, title: str = "", duration_sec: float | None = None,
                     highlight_range: tuple[float, float] | None = None,
                     color: str = "rgb(99,110,250)", height: int = 100) -> go.Figure:
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
        title=dict(text=title or "", font=dict(size=13)),
        xaxis=dict(ticksuffix="s" if duration_sec else None), yaxis=dict(visible=False),
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def inject_match_pill_style() -> None:
    """Call once per page render (Demo.py does this near the top) --
    idempotent via session_state, same "inject once, guard with a flag"
    reasoning streamlit_app/resources.py's own inject_global_styles()
    docstring explains for why that one is only ever called from Overview.py."""
    if st.session_state.get(_MATCH_PILL_CSS_INJECTED_KEY):
        return
    st.session_state[_MATCH_PILL_CSS_INJECTED_KEY] = True
    st.markdown(
        """<style>
        .match-pill {
            display: inline-block; border-radius: 12px; padding: 2px 10px;
            font-size: 13px; font-weight: 600; margin: 2px 0;
            background: rgba(0,200,120,0.18); border: 1px solid rgba(0,200,120,0.5); color: #7CFCB4;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def match_pill_html(pct: float) -> str:
    return f'<span class="match-pill">{pct:.0f}% match</span>'


def inject_keyboard_shortcuts() -> None:
    """Q/M play-pause the active tab's query/match clip; switching facet tabs
    stops whatever's playing. Presenter-only convenience (not a general
    accessibility feature) -- there's no on-screen affordance for it, by
    design: it exists to make live A/B comparison during the talk faster
    than reaching for the mouse, not to be discovered by a random visitor.

    Call-once, guarded by session_state -- same pattern as this file's own
    inject_match_pill_style(), and for a stronger reason than just "avoid
    redundant work": an earlier version of this function called st.iframe()
    unconditionally on every rerun, reasoning that a <style> tag is safe to
    stop re-emitting once present but an iframe skipped on a later rerun
    might get torn down along with the listeners its script attached. That
    reasoning was backwards in practice and caused two real, reported bugs
    at once. First, calling st.iframe() every rerun while its neighbor
    inject_match_pill_style() calls st.markdown() only on the FIRST rerun
    means the element immediately before st.tabs() differs in kind between
    run 1 (markdown, then iframe) and every run after (iframe only) --
    exactly the kind of run-to-run structural inconsistency that leads
    Streamlit's frontend to lose track of widget identity for whatever
    comes after, and st.tabs() has no explicit key to pin its identity
    against that: the reported symptom was every song/facet change
    snapping the active tab back to Sound. Second, an iframe that's
    destroyed and recreated on every rerun tears down its own JS realm each
    time -- and a listener function created inside that realm can go dead
    even though it's still nominally registered on window.parent.document
    (a real, if obscure, cross-realm browser behavior), which was the other
    reported symptom: Q/M working once, then going silent after the next
    interaction. Calling this once and never again removes both failure
    modes the same way inject_match_pill_style()'s CSS never having this
    problem already demonstrated in this exact app: the iframe is created
    once, is never torn down, and the element sequence around it never
    shifts.

    window.parent.document, not the iframe's own: st.iframe (like the
    st.components.v1.html it replaces here -- deprecated in this project's
    installed Streamlit version, 1.59.2) always renders into a same-origin
    iframe, a separate document with its own keyboard focus -- a listener on
    the iframe's own document would only ever see keys pressed while that
    invisible 0-height frame happens to have focus, never the presenter's
    actual keypresses on the real page. Same-origin means reaching across
    via window.parent is allowed by the browser; this is the standard
    technique behind most third-party "streamlit keyboard shortcut"
    components.

    Finding the RIGHT pair of <audio> elements -- not just the first two in
    the document -- is the real trick pages/1_Moment_Matcher.py's six-tab
    layout needs handled correctly: st.tabs() force-mounts every tab's
    content (confirmed directly against the installed frontend bundle,
    .venv/Lib/site-packages/streamlit/static/static/js/index.*.js:
    `shouldForceMount:!0`, with the inactive-vs-active panel distinction
    implemented purely as `display: isActive ? undefined : "none"`), so
    all twelve <audio> elements (2 per facet x 6 facets) are in the DOM
    at once, not just the visible tab's two. offsetParent !== null is the
    standard, layout-engine-native way to ask "is this actually rendered
    right now" (false under any display:none ancestor, true otherwise) --
    filtering to only the visible ones and taking them in DOM order
    (query_col's audio is always written before match_col's, in the page's
    own source) reliably gets [query_audio, match_audio] for whichever tab
    is actually on screen, with no need to know Streamlit's own class names
    for which tab is "active."

    Q/M also pause whichever OTHER clip is playing before starting their
    own -- overlapping query+match audio would work directly against the
    stated point of this shortcut (a fast, legible A/B comparison), so this
    treats "at most one clip playing at a time" as part of the same ask,
    not a separate feature.

    Stopping audio on a facet-tab switch is a real DOM event, not a Streamlit
    rerun, to listen for: switching which tab is selected is handled
    entirely client-side (all tabs' Python content already executes on
    every rerun regardless of which is selected -- see this app's own
    AppTest note on that same fact elsewhere in this codebase -- so nothing
    server-side needs to run just to flip the visible panel). The tabs are
    react-aria's real ARIA tabs implementation under the hood (also
    confirmed against the installed bundle: `role:"tab"`, `aria-selected`,
    `role:"tabpanel"`), so a MutationObserver watching for aria-selected
    attribute changes -- filtered to elements whose own role is literally
    "tab", so an unrelated aria-selected flip elsewhere on the page (e.g.
    inside the song selectbox's dropdown) can't misfire this -- catches a
    tab switch by any input method (click or arrow-key navigation within
    the tablist), without needing to pre-locate the tabs container (which
    would be racy against exactly when this script's iframe finishes
    loading relative to the rest of the page)."""
    if st.session_state.get(_KEYBOARD_SHORTCUTS_INJECTED_KEY):
        return
    st.session_state[_KEYBOARD_SHORTCUTS_INJECTED_KEY] = True
    st.iframe(
        """
        <script>
        (function() {
            var doc = window.parent.document;
            if (doc.__momentMatcherShortcutsInstalled) return;
            doc.__momentMatcherShortcutsInstalled = true;

            function visibleAudios() {
                return Array.prototype.filter.call(
                    doc.querySelectorAll("audio"),
                    function(a) { return a.offsetParent !== null; }
                );
            }

            function pauseAll() {
                Array.prototype.forEach.call(doc.querySelectorAll("audio"), function(a) {
                    if (!a.paused) a.pause();
                });
            }

            function toggle(index) {
                var audios = visibleAudios();
                var target = audios[index];
                if (!target) return;
                if (target.paused) {
                    audios.forEach(function(a, i) { if (i !== index && !a.paused) a.pause(); });
                    target.play();
                } else {
                    target.pause();
                }
            }

            doc.addEventListener("keydown", function(e) {
                var t = e.target;
                var tag = (t && t.tagName || "").toLowerCase();
                if (tag === "input" || tag === "textarea" || (t && t.isContentEditable)) return;
                if (e.key === "q" || e.key === "Q") { toggle(0); e.preventDefault(); }
                else if (e.key === "m" || e.key === "M") { toggle(1); e.preventDefault(); }
            });

            new MutationObserver(function(mutations) {
                for (var i = 0; i < mutations.length; i++) {
                    var el = mutations[i].target;
                    if (el && el.getAttribute && el.getAttribute("role") === "tab") {
                        pauseAll();
                        return;
                    }
                }
            }).observe(doc.body, { attributes: true, attributeFilter: ["aria-selected"], subtree: true });
        })();
        </script>
        """,
        height=1,  # st.iframe rejects 0 (StreamlitInvalidHeightError) -- 1px is the smallest
        # allowed value and just as visually negligible; there's nothing to see in this iframe.
    )


def extract_selected_song_id(point):
    """None if this selection-event point doesn't carry a usable song_id --
    e.g. a click that landed on network_graph_figure's non-interactive-by-
    intent edges trace (mode="lines", no customdata) rather than a node
    marker, close enough to a line for Plotly to register the click there
    instead. Real-bug regression guard (see streamlit_app's own copy of this
    function): AppTest's simulated selection events don't reproduce this the
    way an actual browser click does, so this stays defensive (try/except,
    not a bare index) regardless of the point's exact shape."""
    try:
        return point["customdata"][0]
    except (KeyError, IndexError, TypeError):
        return None


def filter_to_neighbors(nodes_df, edges, center_song_id):
    """Restricts a full similarity graph down to one song and its direct
    k-NN neighbors. A pure filter over already-computed nodes_df/edges
    (sonic_explorer.analysis.network_graph's real k-NN graph is unchanged)
    -- no new graph-construction logic, since GraphEdge already records
    exactly which songs are connected to which. Edges aren't necessarily
    symmetric (song A's top-k neighbors can include B without B's own top-k
    including A), so this keeps every edge touching center_song_id from
    EITHER side. Returns (filtered_nodes_df, filtered_edges); an
    unrecognized center_song_id returns everything unfiltered rather than
    an empty graph, since that's a more useful fallback than a blank chart."""
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


_HALO_COLOR = "rgba(255,255,255,0.22)"
_BG_NODE_COLOR = "rgba(120,120,120,0.55)"
_SELECTED_RING_COLOR = "#FFFFFF"
_NEIGHBOR_RING_COLOR = "#FFD54A"
_LABEL_FONT = dict(size=11, color="#FFFFFF")

# "None" color-by's one selected-node fill -- the same green match_pill_html's
# badge (and this app's mint accent elsewhere) already uses for "positive/
# found," reused here rather than a new one-off color.
_SELECTED_NODE_COLOR = "#7CFCB4"


def _label_positions(rows, selected_song_id):
    """One textposition per row: the selected song always labels above its
    marker (a fixed anchor a viewer can rely on); its neighbors alternate
    top/bottom. Direct neighbors are, by construction, close together in
    both charts (that's what "nearest neighbor" means), so several same-
    side labels packed into a small area is a real, reported clutter
    problem -- alternating halves the chance any two adjacent labels
    collide. Not a full label-placement solver (overkill for ~4-8 labels in
    a demo), just cheap enough to meaningfully help."""
    positions = []
    neighbor_i = 0
    for r in rows:
        if r.song_id == selected_song_id:
            positions.append("top center")
        else:
            positions.append("top center" if neighbor_i % 2 == 0 else "bottom center")
            neighbor_i += 1
    return positions


def _node_color(row, color_by: str, selected_song_id=None):
    """Shared by both figures below so "genre" and "cluster" mean the exact
    same color assignment regardless of which chart or which coloring mode
    is active -- a viewer switching the color-by toggle mid-comparison
    should see the SAME song get the SAME color in both charts, not two
    independently-assigned palettes that happen to both be qualitative.

    color_by="none" turns off category coloring entirely: every node renders
    in the same neutral _BG_NODE_COLOR EXCEPT the selected node itself
    (selected_song_id), which renders in _SELECTED_NODE_COLOR -- the point of
    "None" is that the only thing still visually distinguished is literally
    which node is selected, not a genre/cluster it happens to belong to.
    selected_song_id defaults to None (nothing selected), which correctly
    makes every row fall through to the neutral color -- a plain, uniform
    point cloud, matching what "None" + no selection should look like."""
    if color_by == "none":
        return _SELECTED_NODE_COLOR if selected_song_id is not None and row.song_id == selected_song_id else _BG_NODE_COLOR
    if color_by == "cluster":
        return _GENRE_PALETTE[int(row.cluster) % len(_GENRE_PALETTE)]
    return GENRE_COLOR_MAP.get(row.genre, "#888888")


def taste_map_figure(
    points_df, selected_song_id=None, highlight_song_ids=None, height: int = 420, click_priority: bool = False,
    color_by: str = "genre",
) -> go.Figure:
    """PCA spatial scatter (sonic_explorer.analysis.taste_map.compute_taste_map)
    -- points_df needs columns song_id, x, y, cluster, title, artist, genre.

    color_by ("genre" or "cluster") picks which real label drives node color
    -- genre is the fixed, human-assigned FMA label; cluster is the
    unsupervised K-means grouping over this facet's own embeddings. Both
    figures in this module take the same color_by value (Audio Space drives
    them from one shared toggle, not a per-chart control), so a viewer can
    put either chart in either coloring mode and compare cluster boundaries
    against genre boundaries directly, without the two charts being
    permanently locked to different schemes.

    Everything except the selected song and highlight_song_ids (this page
    passes its direct network-graph neighbors, even here where there are no
    edges to derive them from independently -- reusing the same set keeps
    "the highlighted ones" consistent across both charts) renders as a
    muted, unlabeled gray -- a real reported problem with an earlier
    version: a thin ring alone doesn't read as "selected" at a glance
    against ~230 same-sized, same-opacity dots. The selected point also gets
    a soft halo (a big, semi-transparent marker underneath it) for the same
    reason -- unmissable, not just technically distinguishable on close
    inspection. Selected/neighbor points get a visible text label (the
    title), not just a hover tooltip, so both are identifiable without
    mousing over them.

    selected_song_id=None (e.g. right after a "Clear selection" action)
    switches to a plain mode instead: every point in its normal cluster
    color, no gray-out, no halo, no labels -- the full, unfiltered map, for
    a viewer who wants to see the whole space again rather than one song's
    neighborhood.

    click_priority: sets dragmode=False when True -- see network_graph_figure's
    docstring for why (a real reported "clicking a node doesn't work
    reliably" bug, root-caused to Plotly's default pan dragmode swallowing
    clicks that have even a pixel of mouse movement during them)."""
    has_selection = selected_song_id is not None
    highlight_ids = frozenset(highlight_song_ids) if highlight_song_ids else frozenset()
    relevant_ids = highlight_ids | ({selected_song_id} if has_selection else frozenset())

    rows = list(points_df.itertuples())
    bg_rows = [r for r in rows if r.song_id not in relevant_ids]
    hi_rows = [r for r in rows if r.song_id in relevant_ids]

    fig = go.Figure()

    if has_selection:
        sel_rows = [r for r in rows if r.song_id == selected_song_id]
        if sel_rows:
            fig.add_trace(go.Scatter(
                x=[sel_rows[0].x], y=[sel_rows[0].y], mode="markers",
                marker=dict(size=34, color=_HALO_COLOR, line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ))

    if bg_rows:
        bg_colors = (
            _BG_NODE_COLOR if has_selection
            else [_node_color(r, color_by, selected_song_id) for r in bg_rows]
        )
        fig.add_trace(go.Scatter(
            x=[r.x for r in bg_rows], y=[r.y for r in bg_rows], mode="markers",
            marker=dict(size=9 if not has_selection else 8, color=bg_colors, line=dict(width=0)),
            customdata=[[r.song_id] for r in bg_rows],
            hovertext=[f"{r.title}<br>{r.artist} · {r.genre}" for r in bg_rows],
            hoverinfo="text", showlegend=False,
        ))

    if hi_rows and has_selection:
        hi_colors = [_node_color(r, color_by, selected_song_id) for r in hi_rows]
        fig.add_trace(go.Scatter(
            x=[r.x for r in hi_rows], y=[r.y for r in hi_rows], mode="markers+text",
            marker=dict(
                size=[13 if r.song_id == selected_song_id else 10 for r in hi_rows],
                color=hi_colors,
                line=dict(
                    width=[3.5 if r.song_id == selected_song_id else 2.5 for r in hi_rows],
                    color=[
                        _SELECTED_RING_COLOR if r.song_id == selected_song_id else _NEIGHBOR_RING_COLOR
                        for r in hi_rows
                    ],
                ),
            ),
            text=[r.title for r in hi_rows], textposition=_label_positions(hi_rows, selected_song_id),
            textfont=_LABEL_FONT,
            customdata=[[r.song_id] for r in hi_rows],
            hovertext=[f"{r.title}<br>{r.artist} · {r.genre}" for r in hi_rows],
            hoverinfo="text", showlegend=False,
        ))

    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode=False if click_priority else None,
    )
    return fig


def network_graph_figure(
    nodes_df, edges, selected_song_id=None, highlight_song_ids=None, facet_label: str | None = None,
    height: int = 520, click_priority: bool = False, color_by: str = "genre",
) -> go.Figure:
    """Song-as-node similarity graph (sonic_explorer.analysis.network_graph.
    build_similarity_graph) -- nodes_df needs columns song_id, x, y, cluster,
    title, artist, genre; edges is a list of GraphEdge. color_by ("genre" or
    "cluster") -- see taste_map_figure's own docstring above and this
    module's _node_color() for why both figures share the exact same
    color-assignment logic for either mode. Edges render as one
    line trace (None-separated segments -- the standard Plotly technique for
    many disconnected line segments in a single trace) underneath the node
    scatter, with a real per-edge hover showing both the cosine-similarity
    weight AND facet_label (e.g. "78% similarity (Sound)") -- facet_label is
    this page's own addition over streamlit_app's version of this function,
    which never needed it (see this module's own docstring for why).

    highlight_song_ids: an iterable of song_ids (this page passes the
    selected node's direct neighbors) to draw with a distinct gold ring,
    full color, and a visible text label, separate from selected_song_id's
    own white ring + halo. Everything else (every other node, every edge
    not touching the selected song) renders grayed-out/faint -- a real
    reported problem with the previous version: a thin white ring alone
    doesn't read as "selected" at a glance against ~230 same-sized nodes and
    ~700 same-opacity edges. The halo (a big, soft, semi-transparent marker
    underneath the selected node) is the actual fix -- unmissable, not just
    technically distinguishable up close.

    click_priority: sets dragmode=False when True -- root cause of a real
    "clicking a node doesn't work reliably" report upstream: Plotly's default
    pan dragmode treats ANY mouse movement between mousedown/mouseup as a pan
    gesture and cancels the click event outright, and an ordinary click
    almost always has a pixel or two of movement in it. Disabling drag
    doesn't touch click detection (a separate mechanism, clickmode) or
    scroll-wheel zoom (config={"scrollZoom": True} at the call site) --
    only click-and-drag panning is given up.

    selected_song_id=None (e.g. right after a "Clear selection" action)
    switches to a plain mode instead: every node in its normal genre color,
    every edge at normal opacity with real hover, no gray-out, no halo, no
    labels -- the full, unfiltered graph, for a viewer who wants to see the
    whole space again rather than one song's neighborhood."""
    pos = {row.song_id: (row.x, row.y) for row in nodes_df.itertuples()}
    has_selection = selected_song_id is not None
    highlight_ids = frozenset(highlight_song_ids) if highlight_song_ids else frozenset()
    relevant_ids = highlight_ids | ({selected_song_id} if has_selection else frozenset())

    edge_x_bg, edge_y_bg, edge_hover_bg = [], [], []
    edge_x_hi, edge_y_hi, edge_hover_hi = [], [], []
    for edge in edges:
        if edge.song_id_a not in pos or edge.song_id_b not in pos:
            continue
        x0, y0 = pos[edge.song_id_a]
        x1, y1 = pos[edge.song_id_b]
        hover_label = f"{edge.weight:.0%} similarity" + (f" ({facet_label})" if facet_label else "")
        touches_selected = has_selection and selected_song_id in (edge.song_id_a, edge.song_id_b)
        if touches_selected:
            edge_x_hi += [x0, x1, None]
            edge_y_hi += [y0, y1, None]
            edge_hover_hi += [hover_label, hover_label, None]
        else:
            edge_x_bg += [x0, x1, None]
            edge_y_bg += [y0, y1, None]
            edge_hover_bg += [hover_label, hover_label, None]

    rows = list(nodes_df.itertuples())
    bg_rows = [r for r in rows if r.song_id not in relevant_ids]
    hi_rows = [r for r in rows if r.song_id in relevant_ids]

    fig = go.Figure()

    # Background/full edges -- faint + hover-suppressed once something is
    # selected (avoid a hover tooltip war across ~700 muted lines; a
    # similarity value only matters for edges touching the selected song,
    # which the highlighted trace below covers); normal opacity + real
    # hover in plain mode (nothing selected), since these ARE all the edges
    # there are to inspect then.
    fig.add_trace(go.Scatter(
        x=edge_x_bg, y=edge_y_bg, mode="lines",
        line=dict(
            width=0.5 if has_selection else 0.6,
            color="rgba(120,120,120,0.12)" if has_selection else "rgba(150,150,150,0.35)",
        ),
        hovertext=None if has_selection else edge_hover_bg,
        hoverinfo="skip" if has_selection else "text",
        showlegend=False,
    ))

    if has_selection:
        # Highlighted edges -- the selected song's own connections, real hover.
        fig.add_trace(go.Scatter(
            x=edge_x_hi, y=edge_y_hi, mode="lines",
            line=dict(width=1.4, color="rgba(255,255,255,0.55)"),
            hovertext=edge_hover_hi, hoverinfo="text", showlegend=False,
        ))
        if selected_song_id in pos:
            hx, hy = pos[selected_song_id]
            fig.add_trace(go.Scatter(
                x=[hx], y=[hy], mode="markers",
                marker=dict(size=42, color=_HALO_COLOR, line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ))

    if bg_rows:
        bg_colors = (
            _BG_NODE_COLOR if has_selection
            else [_node_color(r, color_by, selected_song_id) for r in bg_rows]
        )
        fig.add_trace(go.Scatter(
            x=[r.x for r in bg_rows], y=[r.y for r in bg_rows], mode="markers",
            marker=dict(size=9 if not has_selection else 8, color=bg_colors, line=dict(width=0)),
            customdata=[[r.song_id] for r in bg_rows],
            hovertext=[f"{r.title}<br>{r.artist} · {r.genre}" for r in bg_rows],
            hoverinfo="text", showlegend=False,
        ))

    if hi_rows and has_selection:
        fig.add_trace(go.Scatter(
            x=[r.x for r in hi_rows], y=[r.y for r in hi_rows], mode="markers+text",
            marker=dict(
                size=[15 if r.song_id == selected_song_id else 12 for r in hi_rows],
                color=[_node_color(r, color_by, selected_song_id) for r in hi_rows],
                line=dict(
                    width=[3.5 if r.song_id == selected_song_id else 2.5 for r in hi_rows],
                    color=[
                        _SELECTED_RING_COLOR if r.song_id == selected_song_id else _NEIGHBOR_RING_COLOR
                        for r in hi_rows
                    ],
                ),
            ),
            text=[r.title for r in hi_rows], textposition=_label_positions(hi_rows, selected_song_id),
            textfont=_LABEL_FONT,
            customdata=[[r.song_id] for r in hi_rows],
            hovertext=[f"{r.title}<br>{r.artist} · {r.genre}" for r in hi_rows],
            hoverinfo="text", showlegend=False,
        ))

    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode=False if click_priority else None,
    )
    return fig
