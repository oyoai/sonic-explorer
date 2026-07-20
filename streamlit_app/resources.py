"""Shared cached resources -- the interface layer's only entry point into the
core package. Every page calls get_repositories() instead of constructing its own."""

import os

import streamlit as st

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH, DEV_DATA_MARKER
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.llm.explain import ExplanationClient
from sonic_explorer.llm.rerank import RerankClient
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.service import RetrievalService


@st.cache_resource
def get_repositories():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    for facet_name in default_registry().names():
        embedding_repo.load_index(facet_name)
    retrieval_service = RetrievalService(song_repo, embedding_repo)
    return song_repo, embedding_repo, retrieval_service


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


def is_dev_data() -> bool:
    return DEV_DATA_MARKER.exists()


def show_data_source_banner() -> None:
    if is_dev_data():
        st.warning("Using synthetic dev data (sine-wave placeholder audio) -- not the real library yet.", icon="\U0001F9EA")
