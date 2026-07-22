"""Shared cached resources -- the interface layer's only entry point into the
core package. Every page calls get_repositories() instead of constructing its own."""

import os
from pathlib import Path

import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH, DEV_DATA_MARKER
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


def show_data_source_banner() -> None:
    if is_dev_data():
        st.warning("Using synthetic dev data (sine-wave placeholder audio) -- not the real library yet.", icon="\U0001F9EA")
