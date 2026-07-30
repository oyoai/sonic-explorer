"""Shared moment-matching logic (retrieve + LLM rerank + explain) -- used by
both Moment Matcher's own full page and Explore's sidebar section, so the two
call the identical matching pipeline rather than one drifting from the other.
Streamlit-coupled (st.cache_data, st.session_state) so this lives in
streamlit_app/, not sonic_explorer/ -- see CLAUDE.md's core/interface split.

st.session_state.llm_calls is a single, shared per-session budget
(MAX_LLM_CALLS_PER_SESSION) -- Moment Matcher's own page and Explore's
embedded matching draw from the same pool, not two independent budgets,
since both are the same underlying cost (real Claude API calls)."""

import streamlit as st

MAX_LLM_CALLS_PER_SESSION = 60  # simple abuse/cost guardrail for the public deployment (spec section 11)
RERANK_POOL_SIZE = 15  # stage-1 cosine-similarity over-fetch, reranked down to FINAL_K by the LLM
FINAL_K = 10  # Explore's Moment Matcher panel shows "Top 10 results" -- RERANK_POOL_SIZE still has headroom above this


@st.cache_data(show_spinner=False)
def _cached_explanation(
    _client, query_title, query_artist, query_genre, query_start, query_end,
    match_title, match_artist, match_genre, match_start, match_end, facet_name, score,
):
    return _client.generate_explanation(
        query_title=query_title, query_artist=query_artist, query_genre=query_genre,
        query_start_sec=query_start, query_end_sec=query_end,
        match_title=match_title, match_artist=match_artist, match_genre=match_genre,
        match_start_sec=match_start, match_end_sec=match_end,
        facet_name=facet_name, score=score,
    )


def explanation_for_match(client, query_song, query_seg, match, facet_name) -> str | None:
    """None means "don't show an explanation" -- either no client configured,
    the per-session guardrail tripped, or the API call itself failed. Never
    raises up into the page -- the explanation is a value-add, matches must
    still render without it."""
    if client is None or st.session_state.get("llm_calls", 0) >= MAX_LLM_CALLS_PER_SESSION:
        return None
    st.session_state.llm_calls = st.session_state.get("llm_calls", 0) + 1
    try:
        return _cached_explanation(
            client,
            query_song.title, query_song.artist, query_song.genre_top,
            query_seg.start_sec, query_seg.end_sec,
            match.song.title, match.song.artist, match.song.genre_top,
            match.segment.start_sec, match.segment.end_sec,
            facet_name, max(0.0, match.score),
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _cached_rerank_order(_client, query_title, query_artist, query_genre, facet_name, candidate_tuples, k):
    candidates = [{"title": t, "artist": a, "genre": g, "score": s} for (t, a, g, s) in candidate_tuples]
    return _client.rerank(query_title, query_artist, query_genre, facet_name, candidates, k)


def get_matches(retrieval_service, client, query_song, query_seg, facet_name, k=FINAL_K):
    """Stage 1: cosine-similarity retrieval (over-fetched to RERANK_POOL_SIZE
    when a rerank client is available). Stage 2: the LLM reorders that pool
    down to k, best-first. Falls back to plain cosine order -- no client
    configured, session budget exhausted, or the call itself failing -- so
    matches always render either way; reranking is a value-add, not
    load-bearing. Returns (matches, was_reranked)."""
    pool_size = RERANK_POOL_SIZE if client is not None else k
    candidates = retrieval_service.query_by_segment(query_seg.id, facet_name=facet_name, k=pool_size)
    if not candidates:
        return candidates, False

    if client is None or st.session_state.get("llm_calls", 0) >= MAX_LLM_CALLS_PER_SESSION:
        return candidates[:k], False

    st.session_state.llm_calls = st.session_state.get("llm_calls", 0) + 1
    try:
        candidate_tuples = tuple(
            (c.song.title, c.song.artist, c.song.genre_top, max(0.0, c.score)) for c in candidates
        )
        order = _cached_rerank_order(
            client, query_song.title, query_song.artist, query_song.genre_top, facet_name, candidate_tuples, k
        )
        return [candidates[i] for i in order], True
    except Exception:
        return candidates[:k], False
