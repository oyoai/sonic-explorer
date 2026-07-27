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
LOGO_PATH = Path(__file__).resolve().parent / "static" / "logo_transparent.png"


def show_logo() -> None:
    """Renders the wordmark in the sidebar/upper-left corner -- call once per
    page, near the top, same as show_data_source_banner()."""
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH), size="large")


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
