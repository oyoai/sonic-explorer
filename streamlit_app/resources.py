"""Shared cached resources -- the interface layer's only entry point into the
core package. Every page calls get_repositories() instead of constructing its own."""

import os
from pathlib import Path

import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.config import ARTIFACTS_DIR, DATA_DIR, DB_PATH, DEV_DATA_MARKER
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.llm.agent import MusicAgent
from sonic_explorer.llm.explain import ExplanationClient
from sonic_explorer.llm.rerank import RerankClient
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.service import RetrievalService

# logo_transparent.png is a derived asset (white wordmark, transparent
# background) generated from the user-provided static/logo.png, which has a
# solid white background -- the app is dark-theme-only (.streamlit/config.toml
# sets base="dark"), so the original would render as a white box in the
# sidebar. static/logo.png is kept as the untouched source asset.
LOGO_PATH = Path(__file__).resolve().parent / "static" / "logo.svg"


def show_logo() -> None:
    """Renders the wordmark in the sidebar/upper-left corner -- call once per
    page, near the top, same as show_data_source_banner()."""
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH), size="large")
        # st.sidebar.caption("AI-powered music exploration")


def inject_global_styles() -> None:
    """Custom font/CSS styling, called once from Overview.py (the real entry
    point every page's script execution routes through via st.navigation()
    + pg.run()) so it applies everywhere -- previously three near-identical
    copies of this same block were pasted into overview_page.py, pages/
    0_Approach.py, and pages/7_Explore.py separately (with real drift between
    them: different font choices, different .pull-quote weights), which is
    exactly the kind of duplication a single shared call sidesteps. Content
    is Overview's own version (Libre Baskerville pull-quotes, Montserrat
    body) since that was the most recently revised of the three."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@200;400;500;600&family=Playfair+Display:wght@700&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700;
    }

    input, p, .stMarkdown, body {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 200;
    }

    [data-testid="stTextInput"] input {
        border-radius: 20px !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }

    /* Streamlit's default primaryColor (#FF4B4B, its brand red -- this app's
       config.toml never overrides it) is what drives the red focus/hover
       ring browsers show by default on a focused input; overriding it here
       to a neutral white keeps the rest of the theme (buttons, sliders)
       using the real primaryColor untouched, rather than changing that
       theme-wide. */
    [data-testid="stTextInput"] input:hover,
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(255,255,255,0.8) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.8) !important;
    }

    /* Was .st-emotion-cache-1bk7tlj -- an Emotion CSS-in-JS-generated class
       name, not a stable public selector (these hashes can and do change on
       any Streamlit version bump, since they're implementation details, not
       an API). [data-testid="stButton"] is Streamlit's own documented,
       stable hook for targeting buttons. */
    [data-testid="stButton"] button {
        border-radius: 20px !important;
    }

    [data-testid="stButton"] button:hover {
        border-radius: 25px !important;
    }

    .pull-quote {
        font-family: 'Libre Baskerville', serif !important;
        font-style: italic;
        font-weight: 400;
        font-size: 27px;
        border-left: 3.5px solid #A1A1A1;
        padding-left: 19px;
        margin: 0px 5px 10px 20px;
    }

    div[data-testid="stAlert"] {
        font-size: 14px !important;
        padding: 8px 12px !important;
    }

    div[data-testid="stAlert"] p {
        font-size: 12px !important;
    }

    /* Streamlit reserves a large default top gap in .block-container (room
       for its own toolbar -- the hamburger menu/Deploy button visible at
       the top right, a fixed-position overlay, not part of normal document
       flow) -- real dead space on content-dense pages like Explore, where
       every panel is already fighting for vertical room. Content starts at
       the actual top of the viewport now; the toolbar overlays on top of
       it rather than reserving its own flow space, by explicit request.
       Left/right trimmed the same way and for the same reason -- applied
       globally (not just Explore) since that's the existing precedent this
       rule already set for padding-top, and every page benefits from the
       extra usable width, not just the panel-dense one that prompted it.

       Two nested elements both apply their own padding/max-width, confirmed
       directly in Streamlit's own shipped frontend bundle (not guessed):
       an outer wrapper (data-testid="stMain", Emotion-styled with its own
       dynamic isWideMode/showPadding-driven width/padding/max-width), and
       the inner div.block-container[data-testid="stMainBlockContainer"]
       (a separate styled() component, also with its own padding/max-width).
       An earlier version of this rule only targeted the inner one -- looked
       identical in devtools at a glance (both show padding/width/max-width
       properties) but is a genuinely different DOM node, so overriding it
       alone left the outer wrapper's own padding fully intact. Both call
       through Emotion's styled() API, not plain CSS classes, but neither
       compiles its own rule with !important (confirmed by grepping the
       bundle) -- so !important here is sufficient to win regardless of
       specificity or injection order, no special selector boosting needed,
       once both real nodes are actually targeted. */
    [data-testid="stMain"],
    .block-container {
        padding-top: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: initial !important;
        min-width: auto !important;
        width: 100% !important;
    }

    /* Explore's search box shows a red focus ring by default -- Streamlit's
       default primaryColor (#FF4B4B, this app's .streamlit/config.toml
       never overrides it) is what colors a focused input's border app-wide.
       Scoped to just this one input via its own st-key- class rather than
       changing primaryColor globally -- that would also recolor every
       type="primary" button, slider, checkbox, etc. across the whole app,
       a much bigger change than "make the search box's border white." */
    .st-key-explore_search_input input:focus {
        border-color: white !important;
        box-shadow: 0 0 0 1px white !important;
    }

    /* Small inline pill badges -- Explore's Tempo/Key readouts
       (.metric-badge), its one-genre-label-for-now tag (.tag-chip, see
       CLAUDE.md-adjacent decision: real multi-genre/tag data exists in the
       schema but was never backfilled into this DB, so this stays a single
       label rather than faking chips the data doesn't support yet), and
       Moment Matcher's "% match" readout (.match-badge, a distinct green so
       it reads as a score rather than another neutral metadata fact). */
    .metric-badge, .tag-chip, .match-badge {
        display: inline-block;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 500;
        margin: 2px 4px 2px 0;
    }

    .metric-badge {
        background: rgba(99,110,250,0.18);
        border: 1px solid rgba(99,110,250,0.45);
        color: #c7cbff;
    }

    .tag-chip {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.25);
        color: rgba(255,255,255,0.85);
    }

    .match-badge {
        background: rgba(0,200,120,0.18);
        border: 1px solid rgba(0,200,120,0.5);
        color: #7CFCB4;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _hero_graph_data(_song_repo, _embedding_repo, index_size: int):
    """Real per-song positions + edges for the hero banner -- same
    similarity-graph construction Explore's interactive graph uses (see
    analysis/network_graph.py), computed once per session and shared by
    every page's banner rather than recomputed per page load. Uses the
    "sound" facet specifically since it's the one facet every song in the
    library is guaranteed to have been embedded for (see spec's facet
    rollout order), so the banner never silently comes up near-empty on a
    partially-embedded library the way a rarer facet like sound_tags could."""
    from sonic_explorer.analysis.network_graph import build_similarity_graph
    from sonic_explorer.analysis.taste_map import mean_pool_song_vectors

    vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name="sound")
    return build_similarity_graph(vectors)


def hero_banner(key: str) -> None:
    """Real-data page banner -- a muted, non-interactive rendering of the
    library's actual sound-similarity graph (components.plotting.
    network_hero_figure), called once near the top of every page. This
    exists specifically as the "no generic AI-slop hero art" answer: rather
    than a decorative stock image, every page quietly shows an honest (if
    desaturated) picture of the real similarity structure the whole project
    is about. Fails silently (renders nothing) rather than crashing a page
    if the library has no songs/embeddings yet -- purely decorative, never
    load-bearing. key must be unique per page (Streamlit requires a unique
    key per plotly_chart on the same script run)."""
    from components.plotting import network_hero_figure

    song_repo, embedding_repo, _ = get_repositories()
    try:
        result = _hero_graph_data(song_repo, embedding_repo, embedding_repo.index_size("sound"))
    except Exception:
        return
    if not result.nodes:
        return

    import pandas as pd

    nodes_df = pd.DataFrame([{"song_id": n.song_id, "x": n.x, "y": n.y} for n in result.nodes])
    fig = network_hero_figure(nodes_df, result.edges)
    st.plotly_chart(fig, width="stretch", key=f"hero_{key}", config={"staticPlot": True})


def facet_display_name(name: str) -> str:
    """Human-readable facet name for display -- plain str.capitalize() alone
    turns "sound_tags" into "Sound_tags" (capitalize() only touches the
    first character, it doesn't know about underscores), a real, live bug
    across every page that lists facet names this way. Replaces underscores
    with spaces first, so "sound_tags" -> "Sound tags"."""
    return name.replace("_", " ").capitalize()


def badge(text: str, kind: str = "metric") -> str:
    """HTML for one inline pill badge (see inject_global_styles' .metric-badge/
    .tag-chip/.match-badge) -- returns markup for the caller to compose
    several badges on one line via a single st.markdown(..., unsafe_allow_html=True)
    call, rather than rendering (and this doesn't call st.markdown itself,
    since Streamlit puts each st.markdown call on its own line/block)."""
    css_class = {"metric": "metric-badge", "tag": "tag-chip", "match": "match-badge"}[kind]
    return f'<span class="{css_class}">{text}</span>'


def big_quote(text: str) -> None:
    """A short, large, visually-set-apart line -- used for the handful of
    moments across the app meant to land as a single idea, not a paragraph
    to read. Shared (was three separate _big_quote() copies, only one of
    which -- overview_page.py's -- was actually ever called)."""
    st.markdown(
        f'<div class="pull-quote">{text}</div>',
        unsafe_allow_html=True,
    )


# Streamlit's dark-theme default background -- this app's .streamlit/config.toml
# only sets base="dark" (no explicit backgroundColor override), so
# st.get_option("theme.backgroundColor") reads back None (it only reflects an
# explicit override, not Streamlit's own resolved default). Confirmed directly
# against this installed Streamlit version's own frontend bundle
# (static/js/utils.*.js's color palette: gray100 = "#0e1117", the shade the
# dark theme's page background actually uses) rather than assumed from memory.
_DEFAULT_DARK_BACKGROUND = "#0e1117"


def sticky_header(key: str):
    """Returns a context manager: wrap a page's title/header content in it
    (`with sticky_header("approach_header"): st.title(...)`) to pin that
    content to the top of the scrolling main content area, standard
    sticky-header behavior -- everything else on the page scrolls beneath
    it. Separate from the sidebar (st.logo() already stays fixed there
    natively, no CSS needed) -- this is specifically for main-content
    titles, which otherwise scroll away like any other element.

    Targets st.container(key=...)'s documented, stable `.st-key-<key>`
    class -- Streamlit's own supported hook for styling a specific
    container via CSS, not an internal data-testid or emotion-cache class
    name that isn't part of the public API and can change on a Streamlit
    version bump. `key` must be unique per page (reused across pages would
    make every page's header share one CSS rule harmlessly, but reused
    within the same page would collide).

    Background color: reads the real configured theme.backgroundColor if
    this app's config.toml ever sets one explicitly, falling back to the
    verified current dark-theme default otherwise -- not a hardcoded value
    with no path to staying correct if the theme changes."""
    background = st.get_option("theme.backgroundColor") or _DEFAULT_DARK_BACKGROUND
    st.markdown(
        f"""
        <style>
        .st-key-{key} {{
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: {background};
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(250, 250, 250, 0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return st.container(key=key)


@st.cache_resource
def get_repositories():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    for facet_name in default_registry().names():
        embedding_repo.load_index(facet_name)
    retrieval_service = RetrievalService(song_repo, embedding_repo)
    return song_repo, embedding_repo, retrieval_service


@st.cache_data
def build_dna_normalizer(_song_repo, cache_key):
    raw_stats = [{axis: getattr(s, axis) for axis in AXES} for s in _song_repo.list_songs()]
    return fit_normalizer(raw_stats)


@st.cache_data
def build_normalized_dna_by_song(_song_repo, _normalizer, cache_key):
    """Every song's DNA, pre-normalized into the same [0,1]^5 space a
    hand-drawn target (or an agent-picked mood profile) lives in -- shared by
    Moment Matcher's radar-chart-as-query and the agent's search_by_mood_profile
    tool, both just doing nearest-neighbor search over this same precomputed
    dict (spec 2.3)."""
    out = {}
    for s in _song_repo.list_songs():
        raw = {axis: getattr(s, axis) for axis in AXES}
        if all(v is not None for v in raw.values()):
            out[s.id] = _normalizer.normalize(raw)
    return out


def _get_anthropic_api_key() -> str | None:
    """None when no key is configured -- every LLM feature is a value-add, not
    load-bearing, so pages must degrade gracefully rather than crash when
    ANTHROPIC_API_KEY isn't set. Checks st.secrets first (the platform
    secrets manager once deployed -- spec section 11), falling back to an env
    var for local dev convenience."""
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    return api_key or None


@st.cache_resource
def get_explanation_client() -> ExplanationClient | None:
    api_key = _get_anthropic_api_key()
    if not api_key:
        return None

    import anthropic

    return ExplanationClient(anthropic.Anthropic(api_key=api_key))


@st.cache_resource
def get_nl_search_client():
    """Raw anthropic.Anthropic client for llm/search.py's one-shot nl_search()
    -- deliberately NOT get_agent()'s MusicAgent. Explore's search bar and
    Ask the DJ are separate features (one-shot query-in/results-out vs. a
    multi-turn conversation) that used to share MusicAgent directly, which
    is exactly why a full conversational DJ reply used to leak into
    Explore's search results. Same per-client-per-feature pattern
    get_explanation_client()/get_rerank_client() already use -- no shared
    client instance across features here either."""
    api_key = _get_anthropic_api_key()
    if not api_key:
        return None

    import anthropic

    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource
def get_rerank_client() -> RerankClient | None:
    api_key = _get_anthropic_api_key()
    if not api_key:
        return None

    import anthropic

    return RerankClient(anthropic.Anthropic(api_key=api_key))


@st.cache_resource
def get_agent() -> MusicAgent | None:
    """Stateless orchestrator (see llm/agent.py) -- safe to share across
    sessions via cache_resource, since conversation history lives in each
    page's own st.session_state, not inside the agent."""
    api_key = _get_anthropic_api_key()
    if not api_key:
        return None

    import anthropic

    song_repo, embedding_repo, retrieval_service = get_repositories()
    dna_normalizer = build_dna_normalizer(song_repo, len(song_repo.list_songs()))
    normalized_dna_by_song = build_normalized_dna_by_song(song_repo, dna_normalizer, len(song_repo.list_songs()))
    return MusicAgent(
        anthropic.Anthropic(api_key=api_key),
        song_repo, embedding_repo, retrieval_service, dna_normalizer, normalized_dna_by_song,
    )


def is_dev_data() -> bool:
    return DEV_DATA_MARKER.exists()


def is_deploy_subset() -> bool:
    """True when sonic_explorer.config resolved to deploy_data/ (the small,
    committed, stratified sample Streamlit Cloud actually runs against)
    rather than data/ (the full, gitignored local library) -- distinct from
    is_dev_data(), which only flags synthetic placeholder audio. Any page
    whose copy asserts something about the *shape* of the loaded library
    (genre distribution, segment counts, etc.) needs this, not is_dev_data(),
    since both the full library and the deploy subset are real data."""
    return DATA_DIR.name == "deploy_data"


def show_data_source_banner() -> None:
    if is_dev_data():
        st.warning("Using synthetic dev data (sine-wave placeholder audio) -- not the real library yet.")


def nav_button(label: str, target_page: str, key: str) -> None:
    """Real, clickable button that navigates to another page -- st.page_link
    renders as an underlined link, not a button, which is why every
    cross-page navigation call in this app goes through this helper instead.
    st.switch_page() halts script execution and redirects internally (it
    raises its own control-flow exception), so nothing after this call ever
    runs once the button's been clicked -- no return value needed."""
    if st.button(label, key=key):
        st.switch_page(target_page)
