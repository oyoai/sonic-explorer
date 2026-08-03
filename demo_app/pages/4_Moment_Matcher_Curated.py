"""Local Similarity -- Curated: a static, presentation-safe alternative to
Local Similarity (pages/1_Moment_Matcher.py) for a live talk where picking
a song/moment on stage and waiting on retrieval would be a reliability
risk. That page is completely untouched by this one -- this is a new,
separate page added alongside it in Demo.py's navigation, not a
replacement or a redesign of how it works.

CURATED_PAIRS is six fixed query/match pairs, one per facet, identified by
song_id + segment_id (not song titles -- exact, unambiguous, and immune to
any future title/artist metadata edits) so real Song/Segment rows are
looked up fresh via song_repo.get_song()/get_segment() at render time,
keeping playback (waveform + audio player) genuinely real. What's fixed is
only the RETRIEVAL step: no retrieval_service.query_by_segment() call
happens anywhere on this page, so there is no live-query latency and no
run-to-run variance -- always the same six pairs.

match_pct per pair is a REAL number, not invented to look plausible: the
actual cosine similarity between that exact pair's two segment embeddings
for that facet, computed once offline (L2-normalized dot product between
embedding_repo.get_vector() for each segment -- the identical computation
EmbeddingRepository.search()/RetrievalService use internally, just called
directly on two specific, pre-chosen segments instead of as a k-NN search)
and hardcoded here afterward. These pairs were hand-picked, not each
facet's literal top-1 nearest neighbor by construction, so the percentages
run lower (48-70%) than the near-100% pairs a real top-1 search tends to
surface -- an honest number for the pair actually shown, not a cherry-picked
high score."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from plotting import FACET_WAVEFORM_COLORS, QUERY_WAVEFORM_COLOR, inject_match_pill_style, match_pill_html, waveform_figure
from resources import audio_path_for, cached_full_waveform, facet_display_name, get_repositories

inject_match_pill_style()

FACETS = ["sound", "harmony", "vocal", "drums", "bass", "instrumental"]

# facet -> fixed query/match identity + real, precomputed match percentage.
# See this file's module docstring for exactly how match_pct was computed.
CURATED_PAIRS: dict[str, dict] = {
    "sound": {
        "query_song_id": 5, "query_segment_id": 43,      # Sunshine -- Fancy Mike, 0.0s
        "match_song_id": 14, "match_segment_id": 141,    # Her breath -- arizono kazuhiro, 17.5s
        "match_pct": 63.5,
    },
    "harmony": {
        "query_song_id": 22, "query_segment_id": 219,    # ruby cactus -- I, Cactus, 5.0s
        "match_song_id": 149, "match_segment_id": 1553,  # Tsymbaly Solo -- Koliadnyky of Kryvorivnia, 17.5s
        "match_pct": 50.3,
    },
    "vocal": {
        "query_song_id": 145, "query_segment_id": 1510,  # Malka Moma -- Black Sea Hotel, 15.0s
        "match_song_id": 66, "match_segment_id": 687,    # I Don't Care -- Mary Lorson, 15.0s
        "match_pct": 56.5,
    },
    "drums": {
        "query_song_id": 218, "query_segment_id": 2274,  # Inspiration -- Abunai!, 10.0s
        "match_song_id": 212, "match_segment_id": 2210,  # Lovedropper -- Boy Friend, 10.0s
        "match_pct": 55.8,
    },
    "bass": {
        "query_song_id": 108, "query_segment_id": 1130,  # Hots For Brooklyn Instrumental -- Ryan Cullinane, 22.5s
        "match_song_id": 18, "match_segment_id": 184,    # Easy -- SPCZ, 17.5s
        "match_pct": 69.9,
    },
    "instrumental": {
        "query_song_id": 136, "query_segment_id": 1410,  # Piece de Tarita -- Eastern Watershed Klezmer Quartet, 0.0s
        "match_song_id": 141, "match_segment_id": 1472,  # Digital system -- Garage firm, 22.5s
        "match_pct": 47.9,
    },
}

st.title("Local Similarity — Curated")

song_repo, embedding_repo, retrieval_service = get_repositories()

facet_tabs = st.tabs([facet_display_name(f) for f in FACETS])
for facet, tab in zip(FACETS, facet_tabs, strict=False):
    with tab:
        pair = CURATED_PAIRS[facet]
        query_song = song_repo.get_song(pair["query_song_id"])
        query_segment = song_repo.get_segment(pair["query_segment_id"])
        match_song = song_repo.get_song(pair["match_song_id"])
        match_segment = song_repo.get_segment(pair["match_segment_id"])

        query_col, match_col = st.columns(2)

        with query_col:
            st.caption("Query")
            st.markdown(f"**{query_song.title}** — {query_song.artist} · {query_song.genre_top}")
            query_envelope = cached_full_waveform(query_song.id, str(audio_path_for(query_song)))
            st.plotly_chart(
                waveform_figure(
                    query_envelope, duration_sec=query_song.duration_sec,
                    highlight_range=(query_segment.start_sec, query_segment.end_sec), height=90,
                    color=QUERY_WAVEFORM_COLOR,
                ),
                width="stretch", key=f"curated_query_wave_{facet}",
            )
            st.audio(
                str(audio_path_for(query_song)), start_time=query_segment.start_sec,
                end_time=query_segment.end_sec,
            )

        with match_col:
            st.caption("Best match")
            st.markdown(
                match_pill_html(pair["match_pct"])
                + f" {match_song.title} — {match_song.artist} · {match_song.genre_top}",
                unsafe_allow_html=True,
            )
            match_envelope = cached_full_waveform(match_song.id, str(audio_path_for(match_song)))
            st.plotly_chart(
                waveform_figure(
                    match_envelope, duration_sec=match_song.duration_sec,
                    highlight_range=(match_segment.start_sec, match_segment.end_sec), height=90,
                    color=FACET_WAVEFORM_COLORS[facet],
                ),
                width="stretch", key=f"curated_match_wave_{facet}",
            )
            st.audio(
                str(audio_path_for(match_song)), start_time=match_segment.start_sec,
                end_time=match_segment.end_sec,
            )
