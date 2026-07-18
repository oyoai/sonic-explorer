"""Shared cached resources -- the interface layer's only entry point into the
core package. Every page calls get_repositories() instead of constructing its own."""

import streamlit as st

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.service import RetrievalService


@st.cache_resource
def get_repositories():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    embedding_repo.load_index("sound")
    retrieval_service = RetrievalService(song_repo, embedding_repo)
    return song_repo, embedding_repo, retrieval_service
