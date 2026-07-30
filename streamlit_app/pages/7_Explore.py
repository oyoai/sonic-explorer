"""Explore -- three-panel layout: a browsable/searchable song list (left),
the currently selected song's identity card (middle), and an inline Moment
Matcher (right) with two tabs -- "Full Song" (song-level graph matching)
and "Moment" (5s-segment matching against the rest of the library).

Each panel is its own fixed-height container (PANEL_HEIGHT) with internal
scrolling -- the page's own layout (search bar, headers, controls) never
grows past that, only a panel's own overflowing content (the song list, a
growing results list) scrolls inside its own box. No hero banner here
unlike every other page: the three panels already take real vertical space
on their own; a decorative banner on top would just push them down rather
than the layout adapting around it.

The song list is a real st.dataframe(), not a hand-rolled loop of custom
row widgets -- an earlier custom-card version (per-row bordered containers,
a type="tertiary" button per row, manual 50-at-a-time pagination) kept
hitting real, one-off styling bugs (thumbnail aspect ratio, text layout)
that a native, virtualized table sidesteps entirely. The real, explicit
trade: a dataframe cell holds one value, so it can't reproduce that
version's stacked "bold title / artist / genre" card look -- this is a
plain table (Thumbnail/Title/Artist/Genre columns) by deliberate choice,
not an oversight. Selection is native (on_select="rerun", single-row) and
kept in sync with explore_selected_song_id in both directions: a row click
updates it, and external changes to it (search, mini-map click) re-seed the
table's own selection state before it's drawn. All songs render at once
(no pagination) since st.dataframe is properly virtualized -- the real
cost this trades in is computing every visible song's thumbnail AND every
other computed column (DNA scalars, fingerprints, novelty curve) up front;
see _row_thumbnail_data_uri/_cached_facet_fingerprint_data_uri's docstrings
for why that's a one-time, server-wide cost (~78s cold for the four
fingerprint columns alone, measured), not a per-user one.

Moment Matcher's "Full Song" tab is the SAME song-level graph an earlier
version of this page rendered as a standalone "mini-map" in the left
panel, moved here wholesale rather than rebuilt -- a network graph already
IS song-level matching (each node a song, pooled across segments, edges
real k-NN connections), so a second, list-based "whole song" feature would
have been the same computation twice. Toggles between two real, clickable
views, not one fixed graph: "Network" filters the full similarity graph
down to the selected song and its direct k-NN neighbors only (components.
plotting.filter_to_neighbors -- a pure filter over already-computed graph
data, no new graph-construction logic), since the full ~1400-node graph
reads as an unreadable tangle at this scale; "Map" is a real PCA
projection (analysis.taste_map.compute_taste_map, the same mechanism
Methodology/Results/App Walkthrough already use elsewhere) with real pan/
scroll-zoom. Both share the same click-to-select handling
(extract_selected_song_id reading customdata off whichever figure's
on_select event fired) and both update explore_selected_song_id exactly
like the list's own row selection -- clicking a node in this tab jumps the
WHOLE page to that song, the same action as picking it from the list, not
a preview-in-place.

Moment Matcher's "Moment" tab (second of the two, "Full Song" leads) keeps
its segment picker as real st.pills(), NOT a clickable Plotly chart and,
since then, not a manual st.button() grid either -- a first attempt
(components.plotting.segment_waveform_bars_figure, now deleted) kept
failing to click reliably in a live browser despite multiple targeted
fixes (thin bars, then overlapping-trace hit-testing ambiguity); the button
grid that replaced it worked, but stretched every button to its own
column's full width regardless of label length, reading as oversized boxes
around short "0.0-5.0s" text -- st.pills sizes each pill to its own label
and wraps on its own, with required=True keeping exactly one always
selected. Every song in this library has 10-11 overlapping ~5s segments
(WINDOW_SEC=5.0, HOP_SEC=2.5 -- see config.py), so only every other one is
shown (~5s apart) to keep the picker short; there's no long-song edge case
to design around at this library's actual scale. A full-song waveform with
the picked segment highlighted sits alongside the pills -- pills say WHICH
moment, the waveform shows WHERE it sits in the song's overall shape, the
same waveform_figure()+highlight_range combination the result cards below
already use. Segment looping uses st.audio()'s own native start_time/
end_time/loop parameters -- already proven elsewhere in this app (Song
X-Ray's "loop this structural section" feature) -- not a custom HTML5/JS
player or a pre-sliced audio file; neither turned out to be necessary.

Each of Moment Matcher's two tabs has its OWN "Match by" facet selector
now, not a single page-wide one -- an earlier version had exactly one
shared control (top bar) deliberately driving both the mini-map and Moment
Matcher's results, specifically to avoid "two controls that could
disagree about what similar means." That reasoning stopped applying once
the mini-map moved INTO Moment Matcher as its own tab: Full Song and
Moment are now two genuinely different questions ("what does this whole
song match" vs. "what does this specific moment match") that can
legitimately want different facets at the same time, not one page-wide
question two panels used to answer identically.

Album art doesn't exist anywhere in this app's data (no image column in the
schema, no image ingestion in scripts/enrich_fma_metadata.py) -- the
Selected Song panel's Structure fingerprint is the deliberate substitute
(its "why this looks the way it does" explanation lives in the image's own
hover tooltip, not permanently-visible text; rendered at size=128, not
structure_fingerprint's own default of 32, since a 32x32 self-similarity
downsample reads as genuinely blocky at any display size -- a display-only
override, doesn't touch retrieval or any persisted fingerprint elsewhere).
The panel's own top-to-bottom order: the fingerprint + title/artist/genre
side by side, then a media-control strip (icon-only Previous/Next flanking
the audio player, reading as one playback cluster rather than separate
widgets), then Tempo/Key (st.metric with its own help= icon), then the
collapsible Song DNA (which now also holds the "Open full Song X-Ray"
link -- Save is removed for now, so X-Ray no longer had a natural button-
row partner). Previous/Next is always visible and always steps through
whatever the left panel's song list is CURRENTLY showing -- the full
sorted library with no search active, just that search's results with one
active -- not a fixed always-full-library sequence.

Tag chips: the schema has genres_all/track_tags columns for real multi-tag
data, and scripts/enrich_fma_metadata.py exists to populate them, but that
script has never actually been run against this DB (checked directly:
0/1400 songs have either populated) -- so this page shows only the single
genre_top label rather than faking chips the data doesn't support yet.

Search architecture: the top search bar is natural-language-only -- an
earlier version also had a literal title/artist mode behind a toggle, but
that path was dropped once the song list itself became a real
st.dataframe(): its own toolbar ships a genuine find/scroll search (the
magnifying-glass icon), which covers "locate a song I already know the
name of" well enough now that the whole library renders unpaginated (no
more hunting page by page to find it). That toolbar search only
highlights/scrolls to a match, it doesn't filter rows the way the old
literal mode did -- a real, known difference, not an oversight -- but
narrowing the actual list is what natural-language search is for now.
NL search goes through sonic_explorer.llm.search.nl_search(), a dedicated
one-shot function (query in, ranked results out) -- NOT llm.agent.
MusicAgent, the multi-turn conversational engine behind Ask the DJ. An
earlier version of this page called MusicAgent.send_message() directly to
save writing a second search path, which is exactly why a full
conversational DJ reply used to leak into Explore's search results: it
wasn't a display bug, the two features were never actually separated. See
llm/search.py's own docstring for the rest of this reasoning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from resources import (
    badge,
    build_dna_normalizer,
    build_normalized_dna_by_song,
    facet_display_name,
    get_explanation_client,
    get_nl_search_client,
    get_rerank_client,
    get_repositories,
    show_data_source_banner,
    show_logo,
)
from sonic_explorer.analysis.key_chord import estimate_chords, estimate_key
from sonic_explorer.analysis.network_graph import build_similarity_graph
from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS
from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from sonic_explorer.analysis.waveform_preview import beat_times_for_song, chroma_for_display, rms_contour, waveform_envelope
from sonic_explorer.config import album_art_path_for, audio_path_for
from sonic_explorer.facets.fingerprint import composite_fingerprint, structure_fingerprint
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.facets.structure import repetition_rate
from sonic_explorer.llm.search import explanation_for_search_match, nl_search
from sonic_explorer.pipeline.sound_tagging import deserialize_tags
from components.plotting import (
    chord_strip_figure,
    composite_fingerprint_image_data_uri,
    composite_fingerprint_thumbnail,
    extract_selected_song_id,
    filter_to_neighbors,
    fingerprint_image_data_uri,
    fingerprint_thumbnail,
    fingerprint_thumbnail_image,
    network_graph_figure,
    song_dna_bars,
    taste_map_figure,
    waveform_figure,
)
from moment_matching import MAX_LLM_CALLS_PER_SESSION, explanation_for_match, get_matches

FACET_REGISTRY = default_registry()
PANEL_HEIGHT = 850  # st.container()'s own required int height -- a real browser overrides this via the
# page's injected CSS (.st-key-explore_*_panel, below) to a viewport-relative height instead, so this
# value only actually governs rendering where that CSS doesn't apply (AppTest has no CSS engine at all).
# Kept equal across all three panels so, absent the override, the row still reads as capped/equal-height;
# each panel's own overflowing content (the song list, the results list) scrolls inside its own nested
# container instead of growing this one, in both the CSS-driven and this-constant-driven case alike.
SEGMENT_DISPLAY_STRIDE = 2  # show every other real segment -> ~5s-spaced pills instead of every
# 2.5s-hop segment. Display-only: the underlying segmentation/embeddings are untouched, and every
# real segment (including the skipped ones) is still fully queryable -- this just changes which
# already-computed segments the picker renders, not what exists.


def _resolve_nl_search(
    query: str, client, song_repo, embedding_repo, retrieval_service, dna_normalizer, normalized_dna_by_song,
) -> tuple[list[int], str | None, dict[int, str]]:
    """Runs llm.search.nl_search() -- a dedicated one-shot search, not
    MusicAgent -- and returns (matched_song_ids, note_text_or_None,
    explanations). note is only ever a plain, locally-generated status
    string (no key configured, budget exhausted, zero results, a failure) --
    never model-written prose, so nothing conversational ever surfaces on
    this page. explanations maps song_id -> a short, data-grounded "why this
    matched" line (see explanation_for_search_match), for the caller to show
    per result rather than as one combined reply."""
    if client is None:
        return [], "No ANTHROPIC_API_KEY configured for natural-language search.", {}
    if st.session_state.get("llm_calls", 0) >= MAX_LLM_CALLS_PER_SESSION:
        return [], "This session's LLM call budget is used up.", {}

    st.session_state.llm_calls = st.session_state.get("llm_calls", 0) + 1
    try:
        result = nl_search(client, query, song_repo, embedding_repo, retrieval_service, dna_normalizer, normalized_dna_by_song)
    except Exception:
        return [], "The natural-language search failed. Try rephrasing.", {}

    if "error" in result or not result.get("matches"):
        return [], f'No matches found for "{query}".', {}

    tool_name = result.get("tool_name")
    tool_input = result.get("tool_input", {})
    matched_ids, explanations = [], {}
    for match in result["matches"]:
        song_id = match.get("song_id")
        if song_id is None:
            continue
        matched_ids.append(song_id)
        explanation = explanation_for_search_match(tool_name, tool_input, match, normalized_dna_by_song)
        if explanation:
            explanations[song_id] = explanation
    return matched_ids, None, explanations


st.set_page_config(page_title="Sonic Explorer", layout="wide")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()
if not all_songs:
    st.title("Sonic Explorer")
    st.info("No songs in the library yet.")
    st.stop()

show_logo()
show_data_source_banner()

all_songs_sorted = sorted(all_songs, key=lambda s: (s.genre_top, s.title))
songs_by_id = {s.id: s for s in all_songs}
all_song_ids = frozenset(songs_by_id)
dna_normalizer = build_dna_normalizer(song_repo, len(all_songs))
normalized_dna_by_song = build_normalized_dna_by_song(song_repo, dna_normalizer, len(all_songs))

_DEFAULTS = {
    "explore_selected_song_id": all_songs_sorted[0].id,
    "explore_last_query": "",
    "explore_filtered_song_ids": None,  # None = browsing the full unfiltered list; a list = an active search's results
    "explore_search_note": None,
    "explore_search_explanations": {},
    "explore_selected_segment_id": None,
    "explore_selected_segment_song_id": None,
    "llm_calls": 0,
}
for _key, _default in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Page-scoped (not resources.py's shared inject_global_styles -- this sizing
# is specific to Explore's three-panel layout, not something every page
# should inherit). Ties each panel's height to the viewport instead of a
# fixed PANEL_HEIGHT pixel constant, so the outer page itself never scrolls
# on a normal-height screen -- only a panel's own already-existing internal
# scroll (the song list, the results list) should ever kick in. The
# `.st-key-*` selectors are Streamlit's own documented hook for a
# container's `key=` (see st.container's docstring on "CSS class name
# prefixed with st-key-"), not a private/internal implementation detail.
# The 150px subtracted is an estimate of everything ABOVE the panel row
# (top bar, optional search-note caption -- the logo no longer counts here,
# it moved into Streamlit's own header via show_logo() and isn't part of
# this page's own content flow anymore). min-height keeps panels from
# collapsing to nothing on a very short window, at the cost of that case
# still scrolling, which is the lesser evil versus a negative height.
st.markdown(
    """<style>
    .st-key-explore_left_panel, .st-key-explore_mid_panel, .st-key-explore_right_panel {
        height: calc(100vh - 150px) !important;
        min-height: 400px;
    }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def build_explore_graph(_song_repo, _embedding_repo, facet_name: str, index_size: int, song_ids: frozenset):
    songs_by_id_local = {s.id: s for s in _song_repo.list_songs() if s.id in song_ids}
    all_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)
    song_vectors = {sid: vec for sid, vec in all_vectors.items() if sid in song_ids}
    result = build_similarity_graph(song_vectors)
    nodes_df = pd.DataFrame([
        {
            "song_id": n.song_id, "x": n.x, "y": n.y, "cluster": n.cluster,
            "title": songs_by_id_local[n.song_id].title, "artist": songs_by_id_local[n.song_id].artist,
            "genre": songs_by_id_local[n.song_id].genre_top,
        }
        for n in result.nodes
        if n.song_id in songs_by_id_local
    ])
    return nodes_df, result.edges


@st.cache_data
def build_taste_map_points(_song_repo, _embedding_repo, facet_name: str, index_size: int, song_ids: frozenset):
    """PCA projection of the same per-song mean-pooled facet vectors
    build_explore_graph() uses -- compute_taste_map() already exists and is
    already used this way elsewhere (App Walkthrough's own demo), so this
    is a thin cached wrapper, not new projection logic. n_clusters/method
    default to compute_taste_map()'s own Core-tier defaults (8, "pca").
    title/artist/genre included the same way build_explore_graph() includes
    them -- taste_map_figure only renders a real hover tooltip when they're
    present."""
    songs_by_id_local = {s.id: s for s in _song_repo.list_songs() if s.id in song_ids}
    all_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)
    song_vectors = {sid: vec for sid, vec in all_vectors.items() if sid in song_ids}
    result = compute_taste_map(song_vectors, method="pca")
    return pd.DataFrame([
        {
            "song_id": p.song_id, "x": p.x, "y": p.y, "cluster": p.cluster,
            "title": songs_by_id_local[p.song_id].title, "artist": songs_by_id_local[p.song_id].artist,
            "genre": songs_by_id_local[p.song_id].genre_top,
        }
        for p in result.points
        if p.song_id in songs_by_id_local
    ])


@st.cache_data(show_spinner=False)
def _cached_row_fingerprint(_embedding_repo, song_id: int):
    """Precomputed/cached per-song thumbnail IMAGE (not a Plotly figure) for
    the list and result cards -- computed once per song per session
    (Streamlit's cache_data), not regenerated on every page-of-50 render
    across pagination/reruns. Real st.image()-renderable RGB array (see
    fingerprint_thumbnail_image's docstring for why these small thumbnails
    moved off st.plotly_chart() entirely -- a real, confirmed-in-the-
    browser sizing bug at this size, not a hypothetical one). Uses the
    structure fingerprint specifically: cheapest of the three (a pure
    downsample of an already-computed matrix, no extra timeline load) and
    the most broadly available (1399/1400 songs have one)."""
    try:
        matrix = _embedding_repo.get_structure_matrix(song_id)
    except FileNotFoundError:
        return None
    return fingerprint_thumbnail_image(structure_fingerprint(matrix))


@st.cache_data(show_spinner=False)
def _cached_album_art_data_uri(song_id: int, art_path: str) -> str:
    """Real generated album art (deploy_data/album_art/{song_id}.png, see
    Approach step 6) base64-encoded the same way the fingerprint images
    already are -- a raw <img src="..."> tag needs a data URI or a real
    URL, not a bare local file path, to actually render inside markdown.
    art_path is part of the cache key (not just song_id) so a differently-
    resolved path for the same song_id -- shouldn't happen in practice,
    but cheap insurance -- doesn't silently serve a stale cached image."""
    import base64

    return f"data:image/png;base64,{base64.b64encode(Path(art_path).read_bytes()).decode('ascii')}"


def _row_sound_tags(song) -> str:
    """Top 3 AST/AudioSet tags by score, comma-joined -- shared by the
    browsable list's "Sound Tags" column and the Selected Song panel's own
    Sound Tags metric, so both read the identical formatting for the same
    song rather than two independently-drifting implementations."""
    tags = deserialize_tags(song.sound_tags)
    top = sorted(tags, key=lambda t: t[1], reverse=True)[:3]
    return ", ".join(label for label, _score in top)


def _row_thumbnail_data_uri(_embedding_repo, song) -> str:
    """Same album-art-priority, structure-fingerprint-fallback logic the
    Selected Song panel already uses (see its own comment above), applied to
    the browsable list's Thumbnail column too -- real generated album art
    when this song has one, the fingerprint substitute otherwise. Not itself
    cached: both branches dispatch straight to an already-@st.cache_data'd
    function, so there's nothing extra to cache here.

    Fallback reuses _cached_facet_fingerprint_data_uri(..., "structure") --
    the SAME structure fingerprint the dataframe's own "Structure" column
    (below) shows, rather than a separate _cached_thumbnail_data_uri that
    used to compute the identical image a second time under a different
    cache key. Real, measured cost of the four fingerprint columns
    (Structure/Sound/Harmony/Composite) together, cold: ~78s across the
    full ~1400-song library -- de-duplicating this one avoids paying for
    Structure's own downsample twice on top of that."""
    album_art_path = album_art_path_for(song)
    if album_art_path is not None:
        return _cached_album_art_data_uri(song.id, str(album_art_path))
    return _cached_facet_fingerprint_data_uri(_embedding_repo, song.id, "structure") or ""


@st.cache_data(show_spinner=False)
def _cached_structure_extras(_embedding_repo, song_id: int):
    """One structure-matrix load + one structure-timeline load per song,
    shared by every column below that needs them (Structure/Sound/Harmony/
    Composite fingerprints, Repetition rate, Novelty curve) -- reads each
    on-disk artifact once per song, not once per column. Both None (not
    raised) when missing, so callers can check without their own try/except."""
    try:
        matrix = _embedding_repo.get_structure_matrix(song_id)
    except FileNotFoundError:
        matrix = None
    try:
        timeline = _embedding_repo.get_structure_timeline(song_id)
    except FileNotFoundError:
        timeline = None
    return matrix, timeline


@st.cache_data(show_spinner=False)
def _cached_facet_fingerprint_data_uri(_embedding_repo, song_id: int, facet: str) -> str:
    """Structure/Sound/Harmony self-similarity-matrix fingerprints as their
    own list columns -- same cost tier as the existing Thumbnail column
    (Structure) or one extra already-persisted-artifact read (Sound/
    Harmony, both piggyback on the same structure-timeline .npz file), not
    a live audio decode -- see Song DNA's own "Self-similarity matrices"
    section, which shows the identical images at detail-view size."""
    matrix, timeline = _cached_structure_extras(_embedding_repo, song_id)
    if facet == "structure":
        fingerprint = structure_fingerprint(matrix) if matrix is not None else None
    elif facet == "sound":
        fingerprint = timeline.sound_fingerprint if timeline is not None else None
    elif facet == "harmony":
        fingerprint = timeline.harmony_fingerprint if timeline is not None else None
    else:
        raise ValueError(f"Unknown facet {facet!r}")
    if fingerprint is None:
        return ""
    return fingerprint_image_data_uri(fingerprint)


@st.cache_data(show_spinner=False)
def _cached_composite_data_uri(_embedding_repo, song_id: int) -> str:
    matrix, timeline = _cached_structure_extras(_embedding_repo, song_id)
    if matrix is None or timeline is None or timeline.sound_fingerprint is None or timeline.harmony_fingerprint is None:
        return ""
    composite = composite_fingerprint(structure_fingerprint(matrix), timeline.sound_fingerprint, timeline.harmony_fingerprint)
    return composite_fingerprint_image_data_uri(composite)


@st.cache_data(show_spinner=False)
def _cached_row_repetition_rate(_embedding_repo, song_id: int) -> float | None:
    matrix, _timeline = _cached_structure_extras(_embedding_repo, song_id)
    return repetition_rate(matrix) if matrix is not None else None


@st.cache_data(show_spinner=False)
def _cached_row_novelty_curve(_embedding_repo, song_id: int) -> list[float] | None:
    """A list of numbers, not a numpy array -- st.column_config.LineChartColumn
    cells need plain lists (pandas/pyarrow can't round-trip a numpy array
    inside a DataFrame cell the same way). Genuinely cheap despite being a
    per-song curve, not a scalar: novelty_curve is PERSISTED in the same
    structure-timeline .npz file Sound/Harmony's fingerprints already read
    above, not computed live -- unlike Key/Chords/Beats/Loudness, which
    this table deliberately does NOT add as columns (see the comment by
    the Tempo column below: measured at ~4.6s/song for all four together,
    which would mean an ~108-minute cold start across the full library)."""
    _matrix, timeline = _cached_structure_extras(_embedding_repo, song_id)
    if timeline is None or timeline.novelty_curve is None:
        return None
    return [float(v) for v in timeline.novelty_curve]


@st.cache_data(show_spinner=False)
def _cached_full_waveform(song_id: int, audio_path: str):
    return waveform_envelope(audio_path)


@st.cache_data(show_spinner=False)
def _estimated_key_for_song(song_id: int, audio_path: str):
    chroma, _times = chroma_for_display(audio_path)
    return estimate_key(chroma)


@st.cache_data(show_spinner=False)
def _estimated_chords_for_song(song_id: int, audio_path: str):
    """Reuses the exact chroma chroma_for_display() already computes for key
    estimation above -- estimate_chords() takes that same (chroma, times)
    pair (see its own docstring: "no separate computation")."""
    chroma, times = chroma_for_display(audio_path)
    return estimate_chords(chroma, times)


@st.cache_data(show_spinner=False)
def _cached_rms_contour(song_id: int, audio_path: str):
    return rms_contour(audio_path)


@st.cache_data(show_spinner=False)
def _cached_beat_times(song_id: int, audio_path: str):
    return beat_times_for_song(audio_path)


# ---------------------------------------------------------------------------
# Top bar: search, Clear, Ask the DJ -- one compact row. No facet selector
# here anymore -- "Match by" used to live here and drive both the mini-map
# (left panel) and Moment Matcher (right panel) as one shared control. The
# mini-map moved into Moment Matcher's own "Full Song" tab (below), so each
# of Moment Matcher's two tabs now gets its own "Match by" selector scoped
# to just that tab's matching, instead of one page-wide control.
# Search is natural-language-only now (no more literal/NL toggle): the song
# list's own st.dataframe ships a real find/scroll search (the magnifying-
# glass icon in its toolbar) for "locate a song I already know the name of,"
# so a second custom literal-match path was redundant with a native
# Streamlit feature once the list stopped paginating. Clear only renders
# once there's something to clear (a non-empty search box), instead of
# always occupying a column. Ask the DJ sits to the search box's right now
# (moved from the far left) and narrower than before -- it's a secondary
# escape hatch to the full conversational agent, not a peer of the search
# box itself. vertical_alignment="center" keeps the button labels and text
# input visually level without the manual height-spacer markdown divs the
# row used before.
# ---------------------------------------------------------------------------
top_cols = st.columns([2.2, 0.65, 0.65], vertical_alignment="center", gap="small")
with top_cols[0]:
    search_query = st.text_input(
        "Search library", placeholder="mood, sound, instrument…", label_visibility="collapsed",
        key="explore_search_input",
        help="Natural-language search -- describe a mood or a specific sound/instrument. One-shot: "
             "query in, ranked results out, not a conversation (that's Ask the DJ).",
    )
with top_cols[1]:
    if search_query:
        if st.button("Clear", key="explore_clear_search", width="stretch", help="Clear search results"):
            st.session_state.explore_filtered_song_ids = None
            st.session_state.explore_search_note = None
            st.session_state.explore_search_explanations = {}
            st.session_state.explore_last_query = ""
            if "explore_search_input" in st.session_state:
                del st.session_state["explore_search_input"]
            st.rerun()
with top_cols[2]:
    if st.button("Ask the DJ", key="nav_to_dj", width="stretch"):
        st.switch_page("pages/6_Ask_The_DJ.py")

if search_query and search_query != st.session_state.explore_last_query:
    st.session_state.explore_last_query = search_query
    matched_ids, note, explanations = _resolve_nl_search(
        search_query, get_nl_search_client(), song_repo, embedding_repo, retrieval_service,
        dna_normalizer, normalized_dna_by_song,
    )
    st.session_state.explore_filtered_song_ids = matched_ids
    st.session_state.explore_search_note = note
    st.session_state.explore_search_explanations = explanations
    # A zero-result search must not leave a PREVIOUS selection's detail
    # panel/Moment Matcher on screen -- that would look like a stale
    # leftover from before the search, not an honest "nothing matched."
    # Both mid_col and the Moment Matcher panel already have a real
    # song-is-None fallback ("Select a song from the list to see it here.")
    # for exactly this case, so clearing here is safe, not a new code path.
    st.session_state.explore_selected_song_id = matched_ids[0] if matched_ids else None

if st.session_state.explore_search_note:
    st.caption(st.session_state.explore_search_note)

# ---------------------------------------------------------------------------
# Three panels: browsable list + mini map | selected song | Moment Matcher --
# each its own fixed-height (PANEL_HEIGHT) container, so the row is capped
# and equal-height regardless of content differences, with genuinely long
# content (the song list, the results list) scrolling in its own nested
# container instead of growing the panel or the page.
# ---------------------------------------------------------------------------
left_col, mid_col, right_col = st.columns([3, 3, 4])

with left_col, st.container(height=PANEL_HEIGHT, border=False, gap="xxsmall", key="explore_left_panel"):
    active_ids = st.session_state.explore_filtered_song_ids
    active_list = all_songs_sorted if active_ids is None else [songs_by_id[sid] for sid in active_ids if sid in songs_by_id]
    row_explanations = st.session_state.explore_search_explanations

    if active_ids is None:
        st.caption(f"{len(active_list)} songs")
    else:
        st.caption(f"{len(active_list)} result{'s' if len(active_list) != 1 else ''} for your search")

    # st.dataframe instead of a hand-rolled row loop + manual pagination --
    # real virtualization (all songs at once, no "page 1 of 28" clicking),
    # native row selection, no more custom-widget styling to maintain for
    # this specific list. The real tradeoff: a dataframe cell holds one
    # value, so this can't do a stacked "bold title / artist / genre" card
    # look, just a plain table -- a deliberate, discussed trade for
    # reliability over that specific visual.
    def _build_row(s):
        repetition_value = _cached_row_repetition_rate(embedding_repo, s.id)
        return {
            "song_id": s.id,
            "Thumbnail": _row_thumbnail_data_uri(embedding_repo, s),
            "Title": s.title,
            "Artist": s.artist,
            "Genre": s.genre_top,
            # Tempo (song.tempo_bpm) is a real DB column, not recomputed
            # here -- free to show for all songs, like every other DNA/
            # metadata column below it. Key, Chord progression, Beats
            # detected, and Loudness contour are deliberately NOT columns
            # here (Novelty curve, below, IS -- it's persisted, these
            # aren't): none of the four are persisted anywhere, each is only
            # ever estimated live from real audio on demand
            # (_estimated_key_for_song etc.), and a real benchmark of all
            # four together measured ~4.6s/song -- ~108 minutes to compute
            # across the full ~1400-song library on a cold render. Hiding a
            # column by default (see column_order below) only solves visual
            # clutter; the dataframe still has to compute every column for
            # every row regardless of which are visible, so hiding these
            # wouldn't avoid that cost -- this table simply doesn't have
            # them at all, a known, deliberate gap (see Methodology) rather
            # than a slow page.
            "Tempo": f"{s.tempo_bpm:.0f} BPM" if s.tempo_bpm is not None else "—",
            "Energy": f"{s.energy:.2f}" if s.energy is not None else "—",
            "Brightness": f"{s.brightness:.2f}" if s.brightness is not None else "—",
            "Harmonic Complexity": f"{s.harmonic_complexity:.2f}" if s.harmonic_complexity is not None else "—",
            "Rhythmic Density": f"{s.rhythmic_density:.2f}" if s.rhythmic_density is not None else "—",
            "Repetition Rate": f"{repetition_value:.2f}" if repetition_value is not None else "—",
            "Novelty Curve": _cached_row_novelty_curve(embedding_repo, s.id) or [],
            "Structure": _cached_facet_fingerprint_data_uri(embedding_repo, s.id, "structure"),
            "Sound": _cached_facet_fingerprint_data_uri(embedding_repo, s.id, "sound"),
            "Harmony": _cached_facet_fingerprint_data_uri(embedding_repo, s.id, "harmony"),
            "Composite": _cached_composite_data_uri(embedding_repo, s.id),
            "Description": s.description or "",
            "Sound Tags": _row_sound_tags(s),
            "Album": s.album_title or "",
            "Why it matched": row_explanations.get(s.id, ""),
        }

    with st.spinner("Loading thumbnails…", show_time=False):
        list_df = pd.DataFrame([_build_row(s) for s in active_list])

    # Only this default set shows without the viewer doing anything --
    # everything else (DNA scalars, fingerprints, novelty curve,
    # description/tags/album) is a real column in list_df, just hidden by
    # default (see st.dataframe's own column_order docs: columns omitted
    # here stay fully available via the column-visibility menu in the
    # table's own toolbar, not deleted from the data).
    column_order = ["Thumbnail", "Title", "Artist", "Genre", "Tempo"]
    if row_explanations:
        column_order.append("Why it matched")

    # Keep the table's own selection in sync when explore_selected_song_id
    # changed via some OTHER mechanism (search, mini-map click) -- dataframe
    # selection is otherwise one-directional (a user's own click updates
    # session_state; it doesn't automatically reflect external changes to
    # that same song id). Pre-seeding the widget's session_state before
    # it's instantiated is the standard way to control a Streamlit widget
    # programmatically.
    #
    # Two things this has to get right, both confirmed by direct testing of
    # Streamlit's own rerun timing, not assumed:
    #
    # 1. A stale row index left over from a LARGER previous list_df (e.g.
    #    before a search narrowed -- or emptied -- the results) is out of
    #    range for this run's list_df and crashes Streamlit's own dataframe
    #    widget with a pandas .iloc IndexError when it tries to restore that
    #    selection -- desired_rows is recomputed fresh against THIS run's
    #    list_df every time, never trusting a carried-forward index.
    #
    # 2. On the very rerun a real click fires, session_state["explore_song_
    #    table"] already holds that fresh click BEFORE this script body (and
    #    this sync block) executes -- confirmed directly: Streamlit applies a
    #    widget's new frontend value to session_state ahead of the rerun, not
    #    after. explore_selected_song_id, in contrast, only gets updated
    #    further below (after reading event.selection), so on this same run
    #    it's still the OLD value. Naively overwriting session_state["explore
    #    _song_table"] whenever it disagrees with desired_rows (computed from
    #    that stale explore_selected_song_id) would silently discard the
    #    user's own just-landed click every single time -- a real, easy-to-
    #    miss bug, not a hypothetical one. The fix: only force the table's
    #    selection back to desired_rows when the table's current selection
    #    still matches what THIS code itself set it to last run (i.e.
    #    nothing new came from the frontend since); otherwise leave it alone
    #    and let the fresh click flow through to the event.selection handling
    #    below instead.
    selected_id = st.session_state.explore_selected_song_id
    target_row = None
    if not list_df.empty and selected_id in list_df["song_id"].values:
        target_row = int(list_df.index[list_df["song_id"] == selected_id][0])
    desired_rows = [target_row] if target_row is not None else []

    current = st.session_state.get("explore_song_table")
    current_rows = current["selection"]["rows"] if current is not None else None
    last_synced_rows = st.session_state.get("_explore_song_table_synced_rows", "unset")
    if (current_rows is None or current_rows == last_synced_rows) and current_rows != desired_rows:
        st.session_state["explore_song_table"] = {"selection": {"rows": desired_rows, "columns": [], "cells": []}}
        current_rows = desired_rows
    st.session_state["_explore_song_table_synced_rows"] = current_rows

    event = st.dataframe(
        list_df,
        column_order=column_order,
        column_config={
            "Thumbnail": st.column_config.ImageColumn(""),
            "Structure": st.column_config.ImageColumn(),
            "Sound": st.column_config.ImageColumn(),
            "Harmony": st.column_config.ImageColumn(),
            "Composite": st.column_config.ImageColumn(),
            # Cells need a plain list of numbers (see _cached_row_novelty_
            # curve's docstring) -- LineChartColumn renders that as a real
            # per-row sparkline, not a single scalar.
            "Novelty Curve": st.column_config.LineChartColumn(),
        },
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="explore_song_table",
        height=480,
        # st.dataframe's cell/header text is canvas-rendered (glide-data-grid),
        # not real DOM text -- confirmed by inspecting the shipped frontend
        # bundle, which bakes a fixed "13px" font size straight into its
        # canvas draw calls -- so CSS can't touch it, and there's no font_size
        # kwarg on st.dataframe itself. row_height is the only actual lever:
        # tightening it (from an earlier 56, sized for the thumbnail) reads
        # as a denser, smaller-feeling table even though the glyphs
        # themselves are unchanged size.
        row_height=40,
    )

    if event.selection and event.selection.rows:
        clicked_song_id = int(list_df.iloc[event.selection.rows[0]]["song_id"])
        if clicked_song_id != selected_id:
            st.session_state.explore_selected_song_id = clicked_song_id
            st.rerun()

with mid_col, st.container(height=PANEL_HEIGHT, border=False, gap="xxsmall", key="explore_mid_panel"):
    selected_id = st.session_state.explore_selected_song_id
    song = song_repo.get_song(int(selected_id)) if selected_id in all_song_ids else None

    if song is None:
        st.caption("Selected song")
        st.info("Select a song from the list to see it here.")
    else:
        if st.session_state.explore_selected_segment_song_id != song.id:
            st.session_state.explore_selected_segment_id = song.segments[0].id if song.segments else None
            st.session_state.explore_selected_segment_song_id = song.id
            # Reset the segment-picker pills' own widget state too (below,
            # right panel) -- st.pills' default= is only honored on a
            # widget's very first render for a given key, so switching songs
            # needs this explicit overwrite to actually move the picker to
            # the new song's first segment rather than silently keeping the
            # old song's pill selected.
            if "explore_segment_pills" in st.session_state:
                st.session_state["explore_segment_pills"] = st.session_state.explore_selected_segment_id

        st.caption("Selected song")

        try:
            matrix = embedding_repo.get_structure_matrix(song.id)
        except FileNotFoundError:
            matrix = None

        # -- Thumbnail + identity side by side, not stacked -- width:100% +
        # aspect-ratio:1/1 + object-fit:fill (see the img's own inline style
        # below) reliably fills whatever the column's real rendered width
        # is, confirmed via direct browser inspection across several earlier
        # sizing bugs. size=128 (up from structure_fingerprint's own default
        # of 32, FINGERPRINT_SIZE) is the real fix for "the art looks bad" --
        # a 32x32 self-similarity downsample stays genuinely blocky no
        # matter how it's upscaled or interpolated at display time, since
        # there's no more real structural detail to recover once it's been
        # block-averaged that coarse. size is a per-call override (not a
        # change to FINGERPRINT_SIZE itself), so this only affects THIS
        # display -- it doesn't touch retrieval, embeddings, or any other
        # already-persisted fingerprint use elsewhere in the app. ----------
        # Real AI-generated album art (Approach step 6: real per-song audio
        # features -> deterministic prompt -> SD-Turbo, see
        # analysis/album_art_prompt.py) takes priority over the structure-
        # fingerprint substitute whenever it actually exists for this
        # song -- album_art_path_for() returns None (not a guessed path)
        # for any song it wasn't generated for, so this degrades to the
        # exact previous fingerprint-only behavior for those songs, not a
        # broken image.
        album_art_path = album_art_path_for(song)
        # gap="xsmall" (0.5rem, down from st.columns' own "small"/1rem
        # default) -- the art and the title/artist/genre block next to it
        # read as two unrelated pieces of content at the default gap,
        # rather than one grouped identity card.
        thumb_col, meta_col = st.columns([1, 1], gap="xsmall")
        with thumb_col:
            if album_art_path is not None:
                art_data_uri = _cached_album_art_data_uri(song.id, str(album_art_path))
                art_hover_text = (
                    "AI-generated album art -- built entirely from this song's own real detected "
                    "audio features (brightness, energy, key, tempo, sound tags), not a generic "
                    "prompt. See Approach step 6 for exactly how."
                )
                st.markdown(
                    f'<img src="{art_data_uri}" title="{art_hover_text}" '
                    'style="width:50%; aspect-ratio:1/1; object-fit:fill; '
                    'border-radius:4px; display:block;" />',
                    unsafe_allow_html=True,
                )
            elif matrix is not None:
                fp_data_uri = fingerprint_image_data_uri(structure_fingerprint(matrix, size=128))
                fp_hover_text = (
                    "Structure fingerprint — brighter cells mean two moments in the song sound more alike. "
                    "Standing in for album art -- see Approach step 6 for the (separate) AI-generated "
                    "album art pipeline, not yet run for every song in this library."
                )
                # 50% of thumb_col's own width, not the full column -- an
                # explicit request to shrink this down from its previous
                # full-column size; aspect-ratio:1/1 still keeps it square.
                st.markdown(
                    f'<img src="{fp_data_uri}" title="{fp_hover_text}" '
                    'style="width:50%; aspect-ratio:1/1; object-fit:fill; '
                    'border-radius:4px; display:block;" />',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No fingerprint yet.")
        with meta_col:
            # One combined block instead of three separate st.markdown/
            # st.caption calls -- each of those adds its own default
            # block-level margin, which is what made title/artist/genre
            # read as loosely spread out rather than grouped with the
            # thumbnail next to them. Title's font matches the global
            # h1-h4 Playfair Display rule (inject_global_styles) manually,
            # since a plain <div> doesn't inherit that selector -- just at
            # a smaller size/tighter margins than a real #### heading.
            # Explicit size hierarchy: title > artist > genre (genre's own
            # badge() pill is already the smallest thing here, 12px per
            # inject_global_styles' .tag-chip rule -- unchanged).
            st.markdown(
                '<div style="margin-top:2px; line-height:1.3;">'
                f'<div style="font-family:\'Playfair Display\', serif; font-weight:700; '
                f'font-size:1.45rem;">{song.title}</div>'
                f'<div style="color:rgba(255,255,255,0.6); font-size:0.95rem; margin:2px 0 6px 0;">'
                f'{song.artist}</div>'
                f'{badge(song.genre_top, kind="tag")}'
                '</div>',
                unsafe_allow_html=True,
            )

        # -- Previous/Next merged into one media-control strip with the
        # audio player, icon-only (⏮/⏭, not "◀ previous"/"next ▶") -- reads
        # as a single playback control cluster the way a real media player's
        # skip-back/skip-forward buttons flank its scrubber, rather than two
        # unrelated widget rows. Always visible (an earlier restriction --
        # hidden without an active search -- was explicitly lifted per
        # direct request) and always steps through whatever the left panel's
        # song list is CURRENTLY showing: the full sorted library with no
        # search active, just that search's results with one active -- the
        # same active_ids/all_songs_sorted logic the list itself uses,
        # recomputed here rather than reaching across panels for its local
        # variable. has_nav only goes False in a defensive edge case (e.g. a
        # search narrows to zero results); in normal use it's always True,
        # so the icons render enabled in practice, not just the player. ----
        active_ids = st.session_state.explore_filtered_song_ids
        nav_song_ids = (
            [s.id for s in all_songs_sorted] if active_ids is None
            else [sid for sid in active_ids if sid in songs_by_id]
        )
        has_nav = bool(nav_song_ids and song.id in nav_song_ids)
        nav_idx = nav_song_ids.index(song.id) if has_nav else 0

        media_cols = st.columns([1, 9, 1], vertical_alignment="center")
        with media_cols[0]:
            # Real Material Symbols icon, not the ⏮ emoji -- the emoji
            # glyph itself bakes in uneven internal padding that varies by
            # font/platform, which flexbox centering can't fix since it
            # centers the character CELL, not the visible ink inside it (a
            # real reported "icon isn't centered" issue). A Material icon
            # is a real vector glyph Streamlit's own button already centers
            # correctly, same mechanism as every other icon= button in this
            # app (see e.g. Ask the DJ/Engineering nav buttons).
            if st.button(
                "", icon=":material/skip_previous:", disabled=not has_nav or nav_idx == 0,
                key="mid_prev_result", help="Previous song", width="stretch",
            ):
                st.session_state.explore_selected_song_id = nav_song_ids[nav_idx - 1]
                st.rerun()
        with media_cols[1]:
            st.audio(str(audio_path_for(song)))
        with media_cols[2]:
            if st.button(
                "", icon=":material/skip_next:", disabled=not has_nav or nav_idx == len(nav_song_ids) - 1,
                key="mid_next_result", help="Next song", width="stretch",
            ):
                st.session_state.explore_selected_song_id = nav_song_ids[nav_idx + 1]
                st.rerun()

        tempo_text = f"{song.tempo_bpm:.0f} BPM" if song.tempo_bpm is not None else "Not yet computed"
        try:
            key_estimate = _estimated_key_for_song(song.id, str(audio_path_for(song)))
            key_text = f"{key_estimate.tonic} {key_estimate.mode}"
        except Exception:
            key_text = "Could not be estimated"

        # st.metric's own help= icon replaces the badge+st.popover("ℹ️")
        # pair -- a native tooltip affordance instead of a hand-rolled one.
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric(
                "Tempo", tempo_text,
                help="Estimated via beat-tracking (librosa.beat.beat_track): an onset-strength envelope "
                     "feeds a dynamic-programming beat tracker that finds the tempo whose beat grid best "
                     "explains the detected onsets, computed directly from the audio -- not read from any "
                     "metadata tag.",
            )
        with metric_cols[1]:
            st.metric(
                "Key", key_text,
                help="Estimated via Krumhansl-Schmuckler key-finding: the song's aggregate chroma vector "
                     "(12 pitch-class energies, averaged over time) is correlated against 24 rotated "
                     "major/minor key profile templates -- the tonic + mode with the highest correlation "
                     "wins. Same chord/key-detection approach used on the Approach page's Harmony card.",
            )

        # Same st.metric format as Tempo/Key above, not a differently-styled
        # row -- _row_sound_tags is the same top-3-by-score formatting the
        # browsable list's own "Sound Tags" column uses (see that function's
        # docstring), so a song reads identically here and in the list.
        sound_tags_text = _row_sound_tags(song) or "Not yet tagged"
        st.metric(
            "Sound Tags", sound_tags_text,
            help="Top 3 AST/AudioSet tags by confidence score, from a real per-segment audio "
                 "classification pass (pipeline/sound_tagging.py) -- not read from any metadata tag.",
        )

        # Open full Song X-Ray -> now lives inside Song DNA (moved from its
        # own Save/X-Ray button row -- Save is gone for now, see below, so
        # X-Ray no longer had a natural partner up there). Song DNA is where
        # someone reaching for "more detail on this song" already is.
        normalized_dna = normalized_dna_by_song.get(song.id)
        if normalized_dna:
            with st.expander("Song DNA"):
                st.plotly_chart(
                    song_dna_bars([AXIS_LABELS[a] for a in AXES], [normalized_dna[a] for a in AXES]),
                    width="stretch", key="mid_dna_bars",
                )
                st.caption(
                    "Each axis normalized to [0, 1] across the library -- 1.0 means the highest value "
                    "for that axis library-wide, not an absolute unit."
                )

                # -- Five additional views below the radar, none of them new
                # radar axes: the radar's 5 scalars are corpus-normalized
                # (DNANormalizer needs every song's raw value to fit a
                # min/max range), which chord progressions, curves, and
                # beat positions aren't -- they're per-song shapes/events,
                # not single comparable numbers. Each reuses data this app
                # already computes/persists elsewhere rather than a new
                # signal-processing pipeline (see this session's feature
                # audit): chords/beats are computed live the same way
                # Tempo/Key above already are; novelty curve and repetition
                # rate are read straight from the structure artifact
                # Song X-Ray already persists per song, not recomputed. ----
                song_audio_path = str(audio_path_for(song))

                try:
                    structure_timeline = embedding_repo.get_structure_timeline(song.id)
                except FileNotFoundError:
                    structure_timeline = None

                # -- Self-similarity-matrix fingerprints -- moved here from
                # the Selected Song thumbnail above (which now prioritizes
                # real album art, with the structure fingerprint only as its
                # own fallback when no art exists -- see _row_thumbnail_
                # data_uri/the thumb_col comment above). Song DNA is where
                # someone already wants technical detail, so this is where
                # the full set (not just Structure) belongs: each facet's
                # own self-similarity matrix, downsampled into a small
                # brighter-means-more-alike heatmap, plus the composite RGB
                # overlay (structure=red, harmony=green, sound=blue) the
                # App Walkthrough page introduces this same visual language
                # through -- reused here directly, not reinvented. sound_fp/
                # harmony_fp piggyback on the same structure-timeline .npz
                # file structure_fp's own matrix load doesn't touch (see
                # StructureTimeline's docstring), so this costs one extra
                # file read, not three separate matrix computations.
                if matrix is not None:
                    st.markdown("###### Self-similarity matrices")
                    structure_fp = structure_fingerprint(matrix)
                    sound_fp = structure_timeline.sound_fingerprint if structure_timeline is not None else None
                    harmony_fp = structure_timeline.harmony_fingerprint if structure_timeline is not None else None

                    fp_cols = st.columns(4)
                    with fp_cols[0]:
                        st.plotly_chart(
                            fingerprint_thumbnail(structure_fp, "Structure", height=110),
                            width="stretch", key="dna_fp_structure",
                        )
                    if sound_fp is not None:
                        with fp_cols[1]:
                            st.plotly_chart(
                                fingerprint_thumbnail(sound_fp, "Sound", height=110),
                                width="stretch", key="dna_fp_sound",
                            )
                    if harmony_fp is not None:
                        with fp_cols[2]:
                            st.plotly_chart(
                                fingerprint_thumbnail(harmony_fp, "Harmony", height=110),
                                width="stretch", key="dna_fp_harmony",
                            )
                    if sound_fp is not None and harmony_fp is not None:
                        with fp_cols[3]:
                            composite = composite_fingerprint(structure_fp, sound_fp, harmony_fp)
                            st.plotly_chart(
                                composite_fingerprint_thumbnail(composite, "Composite", height=110),
                                width="stretch", key="dna_fp_composite",
                            )
                    st.caption(
                        "Each cell compares two moments in the song against each other on that facet -- "
                        "brighter means more alike. Composite overlays all three as RGB (structure=red, "
                        "harmony=green, sound=blue)."
                    )

                st.markdown("###### Chord progression")
                chord_segments = _estimated_chords_for_song(song.id, song_audio_path)
                if chord_segments:
                    st.plotly_chart(
                        chord_strip_figure(chord_segments), width="stretch", key="dna_chord_strip",
                    )
                else:
                    st.caption("No chords detected (near-silent or unpitched audio).")

                novelty_cols = st.columns(2)
                with novelty_cols[0]:
                    st.markdown("###### Novelty curve")
                    if structure_timeline is not None and structure_timeline.novelty_curve is not None:
                        novelty_fig = waveform_figure(
                            structure_timeline.novelty_curve, duration_sec=song.duration_sec,
                            height=90, color="rgb(255,161,90)",
                        )
                        st.plotly_chart(
                            novelty_fig, width="stretch", key="dna_novelty_curve",
                            config={"staticPlot": True},
                        )
                        st.caption(
                            "Higher = more likely a structural boundary (Foote 2000 checkerboard-kernel "
                            "novelty), not a loudness or energy measure."
                        )
                    else:
                        st.caption("Not yet computed for this song -- see Song X-Ray's structure pipeline.")
                with novelty_cols[1]:
                    st.markdown("###### Loudness contour")
                    loudness_envelope = _cached_rms_contour(song.id, song_audio_path)
                    loudness_fig = waveform_figure(
                        loudness_envelope, duration_sec=song.duration_sec, height=90, color="rgb(0,204,150)",
                    )
                    st.plotly_chart(
                        loudness_fig, width="stretch", key="dna_loudness_contour",
                        config={"staticPlot": True},
                    )
                    st.caption("RMS energy over time -- a perceptual-loudness proxy, not raw amplitude.")

                metric_cols_2 = st.columns(2)
                with metric_cols_2[0]:
                    try:
                        structure_matrix = embedding_repo.get_structure_matrix(song.id)
                        repetition_value = repetition_rate(structure_matrix)
                        st.metric(
                            "Repetition rate", f"{repetition_value:.2f}",
                            help="Mean similarity between all non-identical moments in the song (from the same "
                                 "self-similarity matrix Song X-Ray's structure timeline uses) -- higher means "
                                 "the song revisits similar-sounding material more, not necessarily 'more "
                                 "repetitive' in a musical-form sense.",
                        )
                    except FileNotFoundError:
                        st.caption("Repetition rate: not yet computed for this song.")
                with metric_cols_2[1]:
                    beat_times = _cached_beat_times(song.id, song_audio_path)
                    if len(beat_times) >= 2:
                        avg_interval = float(np.mean(np.diff(beat_times)))
                        st.metric(
                            "Beats detected", str(len(beat_times)),
                            help=f"Beat-level positions only (librosa.beat.beat_track), not bar/measure "
                                 f"downbeats -- this app has no downbeat detector. Average interval between "
                                 f"beats: {avg_interval:.2f}s.",
                        )
                    else:
                        st.caption("Beats detected: not enough beats found to report.")

                if st.button("Open full Song X-Ray →", key="open_xray_btn", width="stretch"):
                    st.session_state["xray_context_song_id"] = song.id
                    st.switch_page("pages/4_Song_XRay.py")

with right_col, st.container(height=PANEL_HEIGHT, border=False, gap="xxsmall", key="explore_right_panel"):
    st.markdown("### Moment Matcher")

    if song is None:
        st.info("Select a song to start matching moments.")
    else:
        # -- Two tabs, not two separate features: both are "find similar
        # songs," just at different granularities. "Full Song" reuses the
        # SAME song-level graph the old standalone mini-map used to render
        # in the left panel (build_explore_graph/filter_to_neighbors/
        # network_graph_figure, taste_map_figure for the PCA alternative) --
        # moved here wholesale, not rebuilt, since a network graph IS
        # already song-level matching (each node a song, pooled across
        # segments). "Moment" is the existing 5s-segment matcher, unchanged
        # apart from getting its own facet control. Full Song first: this
        # is the primary "what does this song as a whole match" question;
        # Moment is the more specific follow-up. Each tab now has its OWN
        # "Match by" selector (moved out of the shared page-level top bar)
        # since the two tabs can legitimately want different facets at
        # once -- there's no single graph_facet variable anymore. ---------
        full_song_tab, moment_tab = st.tabs(["Full Song", "Moment"])
        selected_id = st.session_state.explore_selected_song_id

        with full_song_tab:
            full_song_facet = st.selectbox(
                "Match by", options=FACET_REGISTRY.names(), format_func=lambda f: facet_display_name(f),
                key="explore_full_song_facet_select", width=160,
                help="Which facet's embeddings determine this graph's layout and edges.",
            )
            mini_points_df, mini_edges = build_explore_graph(
                song_repo, embedding_repo, full_song_facet, embedding_repo.index_size(full_song_facet), all_song_ids
            )

            if mini_points_df.empty:
                st.caption(f"No songs embedded for {facet_display_name(full_song_facet)} yet.")
            else:
                mini_graph_mode = st.segmented_control(
                    "Mini-graph view", options=["network", "map"],
                    format_func=lambda m: {"network": "Network", "map": "Map"}[m],
                    default="network", required=True, key="explore_mini_graph_mode", label_visibility="collapsed",
                )

                mini_event = None
                if mini_graph_mode == "network":
                    neighbor_nodes, neighbor_edges = filter_to_neighbors(mini_points_df, mini_edges, selected_id)

                    # Edge weight is the real cosine similarity that decided
                    # this connection in the first place (analysis/
                    # network_graph.py's GraphEdge.weight, clamped to
                    # [0, 1]) -- the graph itself is still built the same
                    # fixed-k-NN way (DEFAULT_K_NEIGHBORS=4 edges per node,
                    # unchanged), this slider is a display-time filter on
                    # top of those already-computed edges, not a different
                    # graph-construction strategy. Default 0.0 (show every
                    # k-NN edge, today's behavior) so this is purely additive.
                    threshold = st.slider(
                        "Similarity threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                        key="explore_network_threshold",
                        help="Hides edges below this cosine-similarity weight -- tighten toward 1.0 to see only "
                             "the strongest connections, loosen toward 0.0 to see every k-NN edge this graph "
                             "computed for the selected song.",
                    )
                    displayed_edges = [e for e in neighbor_edges if e.weight >= threshold]
                    mini_fig = network_graph_figure(
                        neighbor_nodes, displayed_edges, selected_song_id=selected_id, height=260,
                        click_priority=True,
                    )
                    mini_event = st.plotly_chart(
                        mini_fig, width="stretch", on_select="rerun", key="explore_mini_network",
                        config={"displayModeBar": False, "scrollZoom": True},
                    )
                    st.caption(
                        f"Selected song + its direct {facet_display_name(full_song_facet)} neighbors only -- "
                        f"showing {len(displayed_edges)} of {len(neighbor_edges)} edges at ≥{threshold:.2f} "
                        "similarity. Click a node to select that song. Color = genre."
                    )
                else:
                    taste_points_df = build_taste_map_points(
                        song_repo, embedding_repo, full_song_facet, embedding_repo.index_size(full_song_facet),
                        all_song_ids,
                    )
                    mini_fig = taste_map_figure(
                        taste_points_df, selected_song_id=selected_id, height=260, click_priority=True,
                    )
                    mini_event = st.plotly_chart(
                        mini_fig, width="stretch", on_select="rerun", key="explore_mini_map",
                        config={"displayModeBar": False, "scrollZoom": True},
                    )
                    st.caption(
                        "PCA projection -- pan/scroll-zoom to explore, click a point to select that song. "
                        "Color = unsupervised K-means cluster, not genre."
                    )

                if mini_event and mini_event.selection and mini_event.selection.points:
                    clicked_id = extract_selected_song_id(mini_event.selection.points[0])
                    if clicked_id is not None and clicked_id != selected_id:
                        st.session_state.explore_selected_song_id = clicked_id
                        st.rerun()

        with moment_tab:
            if not song.segments:
                st.caption("This song has no segments yet.")
            else:
                moment_facet = st.selectbox(
                    "Match by", options=FACET_REGISTRY.names(), format_func=lambda f: facet_display_name(f),
                    key="explore_moment_facet_select", width=160,
                    help="Which facet's embeddings determine these results.",
                )

                segments = song.segments
                segment_ids = {s.id for s in segments}
                selected_segment_id = st.session_state.explore_selected_segment_id
                if selected_segment_id not in segment_ids:
                    selected_segment_id = segments[0].id

                # -- segment picker: st.pills, not a button grid -- the
                # earlier button-grid version stretched every button to its
                # column's full width regardless of label length, reading as
                # oversized boxes around short "0.0-5.0s" text. st.pills
                # sizes each pill to its own label (width="content" is its
                # default) and wraps into new rows on its own, both of which
                # a manual st.columns(len(row)) grid can't do -- no more
                # SEGMENT_BUTTONS_PER_ROW to hand-tune either. Only every
                # other real segment is shown (~5s apart instead of every
                # 2.5s hop) so the picker stays short and readable; the full
                # segment list (and its real embeddings) is untouched --
                # query_segment below is still looked up from `segments`,
                # not `displayed_segments`, so nothing about matching itself
                # changes. required=True keeps exactly one pill always
                # selected (st.pills otherwise allows clicking the active
                # pill again to deselect down to nothing, which this UI has
                # no sensible "no moment picked" state for). ---------------
                displayed_segments = segments[::SEGMENT_DISPLAY_STRIDE]
                segment_pill_labels = {
                    seg.id: f"{seg.start_sec:.1f}–{seg.end_sec:.1f}s" for seg in displayed_segments
                }
                st.caption("Pick a moment to match:")
                selected_segment_id = st.pills(
                    "Pick a moment to match", options=[seg.id for seg in displayed_segments],
                    format_func=lambda sid: segment_pill_labels[sid], default=selected_segment_id,
                    required=True, key="explore_segment_pills", label_visibility="collapsed",
                )
                st.session_state.explore_selected_segment_id = selected_segment_id

                query_segment = next(s for s in segments if s.id == selected_segment_id)

                # -- Full-song waveform with the picked segment highlighted,
                # on top of (not instead of) the pill picker -- the pills
                # say WHICH moment is selected, this shows WHERE it sits in
                # the song's overall shape. Same waveform_figure()+
                # highlight_range combination the result cards below
                # already use, just for the query song itself; reuses
                # _cached_full_waveform (song_id, path)-keyed cache, already
                # warm for any song that's shown up as a match elsewhere
                # this session, cold otherwise -- same cost profile as
                # every other per-song waveform in this app, nothing new.
                query_full_envelope = _cached_full_waveform(song.id, str(audio_path_for(song)))
                query_wave_fig = waveform_figure(
                    query_full_envelope, duration_sec=song.duration_sec,
                    highlight_range=(query_segment.start_sec, query_segment.end_sec),
                    height=70, color="rgb(0,204,150)",
                )
                st.plotly_chart(
                    query_wave_fig, width="stretch", key="query_full_waveform",
                    config={"staticPlot": True},
                )

                st.audio(
                    str(audio_path_for(song)), start_time=query_segment.start_sec, end_time=query_segment.end_sec,
                    loop=True,
                )
                st.caption(
                    f"Looping {query_segment.start_sec:.1f}s–{query_segment.end_sec:.1f}s, "
                    f"by **{facet_display_name(moment_facet)}**."
                )

                if embedding_repo.status(query_segment.id, moment_facet) != "done":
                    st.warning(
                        f"This moment hasn't been embedded for the {facet_display_name(moment_facet)} facet yet."
                    )
                else:
                    llm_client = get_explanation_client()
                    rerank_client = get_rerank_client()
                    matches, was_reranked = get_matches(
                        retrieval_service, rerank_client, song, query_segment, moment_facet
                    )

                    if not matches:
                        st.info("No matches found elsewhere in the library yet.")
                    else:
                        st.markdown(f"#### Top {len(matches)} results")
                        if was_reranked:
                            st.caption(
                                "Re-ranked for overall fit, not just raw similarity.",
                                help="An LLM re-ordered these candidates using its own judgment of the "
                                     "title/artist/genre pairing, not just the stage-1 cosine-similarity score -- "
                                     "a lower-scored candidate can outrank a higher-scored one if it's a more "
                                     "sensible match, and vice versa. It sees only text metadata for this, not "
                                     "additional audio analysis.",
                            )

                        with st.container(height=480):
                            for match in matches:
                                with st.container(border=True):
                                    card_cols = st.columns([1, 3])
                                    with card_cols[0]:
                                        thumb = _cached_row_fingerprint(embedding_repo, match.song.id)
                                        if thumb is not None:
                                            st.image(thumb, width=56)
                                    with card_cols[1]:
                                        pct = max(0.0, match.score) * 100
                                        st.markdown(
                                            f"**{match.song.title}**  \n{match.song.artist} · "
                                            f"{match.song.genre_top}  \n" + badge(f"{pct:.0f}% match", kind="match"),
                                            unsafe_allow_html=True,
                                        )

                                    result_envelope = _cached_full_waveform(
                                        match.song.id, str(audio_path_for(match.song))
                                    )
                                    result_fig = waveform_figure(
                                        result_envelope, duration_sec=match.song.duration_sec,
                                        highlight_range=(match.segment.start_sec, match.segment.end_sec),
                                        height=70, color="rgb(99,110,250)",
                                    )
                                    st.plotly_chart(
                                        result_fig, width="stretch", key=f"result_wave_{match.segment.id}",
                                        config={"staticPlot": True},
                                    )
                                    st.caption("Play moment (loops):")
                                    st.audio(
                                        str(audio_path_for(match.song)), start_time=match.segment.start_sec,
                                        end_time=match.segment.end_sec, loop=True,
                                    )

                                    explanation = explanation_for_match(
                                        llm_client, song, query_segment, match, moment_facet
                                    )
                                    if explanation:
                                        st.caption(f"*{explanation}*")
