"""Moment Matcher (nav title in Demo.py) -- a static, presentation-safe
alternative to the live, pick-your-own-song Moment Matcher
(pages/1_Moment_Matcher.py) for a live talk where picking a song/moment on
stage and waiting on retrieval would be a reliability risk. That page is
completely untouched by this one -- it used to sit alongside this page in
Demo.py's navigation for comparison, but was pulled from the router (not
deleted) once this curated version became the one actually used on stage;
see Demo.py's own module docstring for that decision. Filename kept as
4_Moment_Matcher_Curated.py -- the nav title, not the filename, is what a
viewer actually sees, same convention every other page in this app follows.

CURATED_PAIRS is six fixed query/match pairs, one per facet, identified by
song_id + segment_id (not song titles -- exact, unambiguous, and immune to
any future title/artist metadata edits) so real Song/Segment rows are
looked up fresh via song_repo.get_song()/get_segment() at render time,
keeping playback (waveform + audio player) genuinely real. What's fixed is
only the RETRIEVAL step: no retrieval_service.query_by_segment() call
happens anywhere on this page ITSELF, so there is no live-query latency
and no run-to-run variance -- always the same six pairs.

Each match IS each query's real top-1 nearest neighbor, not an arbitrary
paired song -- an earlier version of this page used six hand-picked pairs
instead (given as a fixed list of two songs per facet, paired first-with-
second), and a real, reported problem followed directly from that: the
displayed "match" didn't agree with what the LIVE Moment Matcher
(pages/1_Moment_Matcher.py) showed for the same query, since a hand-picked
pair isn't necessarily what real retrieval would surface (those six pairs
scored a real but unremarkable 48-70% -- nowhere near the 88%+ a genuine
top-1 typically scores). Fixed by actually calling
retrieval_service.query_by_segment(query_segment_id, facet_name=facet,
k=1) once per facet, offline, at authoring time, and hardcoding THAT
result -- the exact same call the live page makes on every interaction,
just made once here instead of on every page load. match_pct is that
call's real .score (max(0.0, score) * 100, same convention as every other
match-percentage display in this codebase), not invented to look
plausible.

LISTEN_FOR gives each tab one line on what to actually pay attention to --
a match percentage alone doesn't tell a listener what the comparison is
even about. Phrased in real musical vocabulary (tonal color, phrasing,
groove) rather than restating the facet name back at the listener ("for
bass, listen for the bass" says nothing a title doesn't already). For the
four stem facets (vocal/drums/bass/instrumental), it also says plainly
that playback is still the full mix, not an isolated track -- no isolated-
stem audio is ever persisted anywhere in this project (see
pages/1_Moment_Matcher.py's own module docstring for why), so "listen for
the voice" could otherwise read as a promise of an a cappella clip that
isn't actually coming."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from plotting import (
    FACET_WAVEFORM_COLORS,
    QUERY_WAVEFORM_COLOR,
    inject_keyboard_shortcuts,
    inject_match_pill_style,
    match_pill_html,
    waveform_figure,
)
from resources import audio_path_for, cached_full_waveform, facet_display_name, get_repositories

inject_match_pill_style()
# Same Q/M play-pause-query/match, stop-on-tab-switch shortcuts
# pages/1_Moment_Matcher.py has -- see inject_keyboard_shortcuts()'s own
# docstring for the full mechanics. Applies unchanged here: each facet tab
# below renders query audio before match audio in the same DOM order that
# function's offsetParent-visibility filtering relies on, so the identical
# implementation works without modification.
inject_keyboard_shortcuts()

FACETS = ["sound", "harmony", "vocal", "drums", "bass", "instrumental"]

# facet -> fixed query/match identity + real, precomputed match percentage.
# See this file's module docstring for exactly how match_pct was computed.
CURATED_PAIRS: dict[str, dict] = {
    "sound": {
        "query_song_id": 5, "query_segment_id": 43,        # Sunshine -- Fancy Mike, 0.0s
        "match_song_id": 13, "match_segment_id": 124,      # Another Boring Lunchtime (edit) -- Psychadelik Pedestrian, 0.0s
        "match_pct": 88.4,
    },
    "harmony": {
        "query_song_id": 22, "query_segment_id": 219,      # ruby cactus -- I, Cactus, 5.0s
        "match_song_id": 231, "match_segment_id": 2417,    # Softer Place To Fall -- Mark Fosson, 15.0s
        "match_pct": 91.8,
    },
    "vocal": {
        "query_song_id": 145, "query_segment_id": 1510,    # Malka Moma -- Black Sea Hotel, 15.0s
        "match_song_id": 143, "match_segment_id": 1485,    # Ibish Aga -- Black Sea Hotel, 5.0s
        "match_pct": 96.3,
    },
    "drums": {
        "query_song_id": 218, "query_segment_id": 2274,    # Inspiration -- Abunai!, 10.0s
        "match_song_id": 123, "match_segment_id": 1283,    # Hope -- Joao Picoito, 15.0s
        "match_pct": 94.0,
    },
    "bass": {
        "query_song_id": 108, "query_segment_id": 1130,    # Hots For Brooklyn Instrumental -- Ryan Cullinane, 22.5s
        "match_song_id": 87, "match_segment_id": 903,      # Domino's -- Alaclair Ensemble, 2.5s
        "match_pct": 93.6,
    },
    "instrumental": {
        "query_song_id": 136, "query_segment_id": 1410,    # Piece de Tarita -- Eastern Watershed Klezmer Quartet, 0.0s
        "match_song_id": 147, "match_segment_id": 1527,    # Ne Si Jo Prodavaj Chiflikot -- Gogofski, 5.0s
        "match_pct": 89.7,
    },
}

# One line per facet on what the match is actually grounded in, since the
# match percentage alone doesn't tell a listener what to pay attention to.
# Sound/harmony run on the full mix natively, so no caveat needed; vocal/
# drums/bass/instrumental run on an isolated Demucs stem's embedding, but
# no isolated-stem audio is ever persisted (see pages/1_Moment_Matcher.py's
# own module docstring for the real reason), so playback is still the full
# mix for those four -- worth saying plainly rather than letting "vocal"
# sound like you're about to hear an a cappella track.
LISTEN_FOR: dict[str, str] = {
    "sound": "Overall texture and production character -- the full mix's timbre, not any single element.",
    "harmony": "Chord movement and tonal color, underneath the rhythm and melody.",
    "vocal": "The voice's phrasing and tone -- matched from the isolated vocal, though you're hearing the full track.",
    "drums": "Rhythm and hit character -- matched from the isolated drum stem, though you're hearing the full track.",
    "bass": "The bassline's movement and tone -- matched from the isolated bass stem, though you're hearing the full track.",
    "instrumental": (
        "The backing instrumentation -- everything but vocals, drums, and bass -- matched from that isolated stem."
    ),
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

        st.caption(f"🎧 What to listen for: {LISTEN_FOR[facet]}")

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
