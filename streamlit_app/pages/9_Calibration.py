"""Calibration rating tool -- blind human XAB similarity judgments (the
dataset section 9's blend-weight regression depends on) plus a separate
"do I like this" taste judgment, both blind (no title/artist/algorithm score
shown -- judging what you actually hear, not what you recognize).

Replaces the old scripts/calibration_rating_app.py, previously its own
separate process on port 8504 -- now a normal (if hidden-from-sidebar, see
Overview.py's visibility="hidden") page in the main app, so rating no longer
requires a second local process, and multiple people can rate without either
of them running anything themselves.

Multiple rating profiles: real calibration participants get their own named
profile (REAL_RATER_PROFILES below), plus a fixed "Guest / Test" profile for
anyone using the app who isn't a real participant. Guest ratings write to a
physically separate table (calibration_ratings_guest / taste_ratings_guest,
see db.py's schema comment) so they can never leak into the real blend-
weight regression. The profile list itself is environment-aware
(resources.is_deploy_subset()): locally, every real profile plus Guest/Test
is selectable; on the deployed app, only Guest/Test appears at all -- a real
participant's name/progress is never exposed to a random visitor.

Similarity and taste ratings share this one page/section for convenience,
but run as sequential batches (SIMILARITY_BATCH_SIZE, then TASTE_BATCH_SIZE,
then back to similarity) rather than side-by-side questions about the same
clip -- answering "is this similar" and "do I like this" back to back for
the same triplet risked one judgment priming the other.

Methodological note, to carry into Methodology whenever the taste flow
gets written up with real numbers: taste is rated on the same 5-second
segments similarity uses (matching the rest of the calibration/retrieval
pipeline, and simplest to build under real time constraints), NOT whole
songs. A segment-level "do I like this" judgment is a proxy for whole-song
taste, not identical to it -- a 5-second clip can't capture a song's
structure, build, or payoff the way hearing the whole thing can. Whatever
this data ends up showing should be reported as liking of isolated
5-second clips, not silently generalized to liking of songs.

Known limitation, also to carry into Methodology whenever it's written up:
the main similarity pool's triplets are all sampled using SOUND-facet
retrieval results (see calibration_triplets.py's generate_calibration_
triplets -- band candidates come from the reference's own real Sound
nearest-neighbors). That's a real coverage bias: a pair where, say, only
Harmony places two segments close together (independent of whether Sound
does) may never get sampled into the main pool at all, since the pool only
ever explores the space Sound's own similarity ranking considers "high/
medium/random." The small opt-in Harmony sanity-check pool below exists
specifically to spot-check whether that bias changes anything in practice,
without a full rotation rebuild of the main 350-triplet pool -- see
HARMONY_CHECK_FACET below. Ratings from it still go into the real
calibration_ratings table (this is real data, not a guest-style
exclusion), tagged via sampling_facet so they stay distinguishable from
the main Sound-sampled pool in any later analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from resources import (
    get_calibration_repositories,
    get_repositories,
    hero_banner,
    is_deploy_subset,
    show_data_source_banner,
    show_logo,
)
from sonic_explorer.config import audio_path_for
from sonic_explorer.evaluation.calibration_triplets import generate_calibration_triplets, generate_taste_candidates

REAL_RATER_PROFILES = ["profile1", "profile2"]
GUEST_PROFILE = "Guest / Test"
SIMILARITY_BATCH_SIZE = 5
TASTE_BATCH_SIZE = 5
N_TRIPLETS = 350
MAIN_SAMPLING_FACET = "sound"
# Small, opt-in supplemental pool sampled from a different facet's retrieval
# results, to sanity-check MAIN_SAMPLING_FACET's known coverage bias (see
# module docstring) without a full rotation rebuild of the main pool.
HARMONY_CHECK_FACET = "harmony"
HARMONY_CHECK_N_TRIPLETS = 15


def _available_profiles() -> list[str]:
    if is_deploy_subset():
        return [GUEST_PROFILE]
    return [*REAL_RATER_PROFILES, GUEST_PROFILE]


def _similarity_key(triplet) -> tuple[int, int, int]:
    return (triplet.segment_x_id, min(triplet.segment_a_id, triplet.segment_b_id),
            max(triplet.segment_a_id, triplet.segment_b_id))


def _advance_phase() -> None:
    st.session_state.calibration_phase_count += 1
    limit = SIMILARITY_BATCH_SIZE if st.session_state.calibration_phase == "similarity" else TASTE_BATCH_SIZE
    if st.session_state.calibration_phase_count >= limit:
        st.session_state.calibration_phase = (
            "taste" if st.session_state.calibration_phase == "similarity" else "similarity"
        )
        st.session_state.calibration_phase_count = 0


@st.cache_data(show_spinner="Generating the triplet set...")
def _cached_triplets(_song_repo, _embedding_repo, index_size):
    return generate_calibration_triplets(
        _song_repo, _embedding_repo, facet_name=MAIN_SAMPLING_FACET, n_triplets=N_TRIPLETS
    )


@st.cache_data(show_spinner="Generating the Harmony sanity-check pool...")
def _cached_harmony_check_triplets(_song_repo, _embedding_repo, index_size):
    return generate_calibration_triplets(
        _song_repo, _embedding_repo, facet_name=HARMONY_CHECK_FACET, n_triplets=HARMONY_CHECK_N_TRIPLETS
    )


@st.cache_data(show_spinner="Generating the taste-rating queue...")
def _cached_taste_candidates(_song_repo, _embedding_repo, index_size):
    return generate_taste_candidates(_song_repo, _embedding_repo, facet_name="sound")


st.set_page_config(page_title="Calibration Rating")
hero_banner("calibration")
st.title("Calibration Rating")
show_logo()
show_data_source_banner()
st.caption(
    "Blind judgments -- no titles, no artist names, no algorithm score. Similarity rounds ask "
    "which of Clip A or Clip B sounds more like the Reference; taste rounds ask whether you like "
    "what you're hearing. Progress saves after every choice, so it's safe to close this and come "
    "back later."
)

song_repo, embedding_repo, _ = get_repositories()
calibration_repo, calibration_guest_repo, taste_repo, taste_guest_repo = get_calibration_repositories()

profiles = _available_profiles()
selected_profile = st.selectbox("Rating profile", profiles, key="calibration_profile")
is_guest = selected_profile == GUEST_PROFILE
active_calibration_repo = calibration_guest_repo if is_guest else calibration_repo
active_taste_repo = taste_guest_repo if is_guest else taste_repo
if is_guest:
    st.caption("Guest ratings are kept separate and never used in the real calibration numbers.")

if "calibration_phase" not in st.session_state:
    st.session_state.calibration_phase = "similarity"
    st.session_state.calibration_phase_count = 0
if "calibration_skipped" not in st.session_state:
    st.session_state.calibration_skipped = set()  # not persisted -- reappear next session
if "taste_skipped" not in st.session_state:
    st.session_state.taste_skipped = set()

all_triplets = _cached_triplets(song_repo, embedding_repo, embedding_repo.index_size("sound"))
all_taste_candidates = _cached_taste_candidates(song_repo, embedding_repo, embedding_repo.index_size("sound"))

if not all_triplets or not all_taste_candidates:
    st.warning("No embedded segments found -- run the sound-facet batch pipeline first.")
    st.stop()

st.divider()

if st.session_state.calibration_phase == "similarity":
    st.subheader(f"Similarity round ({st.session_state.calibration_phase_count + 1}/{SIMILARITY_BATCH_SIZE})")

    rated_keys = active_calibration_repo.rated_triplet_keys(rater=selected_profile)
    remaining = [
        t for t in all_triplets
        if _similarity_key(t) not in rated_keys and _similarity_key(t) not in st.session_state.calibration_skipped
    ]

    if not remaining:
        st.success(f"All {len(all_triplets)} similarity triplets rated for {selected_profile}.")
    else:
        triplet = remaining[0]
        seg_x = song_repo.get_segment(triplet.segment_x_id)
        seg_a = song_repo.get_segment(triplet.segment_a_id)
        seg_b = song_repo.get_segment(triplet.segment_b_id)

        st.write("**Reference**")
        song_x = song_repo.get_song(seg_x.song_id)
        st.audio(str(audio_path_for(song_x)), start_time=seg_x.start_sec, end_time=seg_x.end_sec, loop=True)

        st.write("Which of these two clips sounds more similar to the Reference?")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Clip A**")
            song_a = song_repo.get_song(seg_a.song_id)
            st.audio(str(audio_path_for(song_a)), start_time=seg_a.start_sec, end_time=seg_a.end_sec, loop=True)
        with col2:
            st.write("**Clip B**")
            song_b = song_repo.get_song(seg_b.song_id)
            st.audio(str(audio_path_for(song_b)), start_time=seg_b.start_sec, end_time=seg_b.end_sec, loop=True)

        choice_cols = st.columns(3)
        with choice_cols[0]:
            if st.button("A is more similar", width="stretch", type="primary", key="calib_choice_a"):
                active_calibration_repo.add_choice(
                    triplet.segment_x_id, triplet.segment_a_id, triplet.segment_b_id, "a",
                    rater=selected_profile, sampling_facet=MAIN_SAMPLING_FACET,
                )
                _advance_phase()
                st.rerun()
        with choice_cols[1]:
            if st.button("B is more similar", width="stretch", type="primary", key="calib_choice_b"):
                active_calibration_repo.add_choice(
                    triplet.segment_x_id, triplet.segment_a_id, triplet.segment_b_id, "b",
                    rater=selected_profile, sampling_facet=MAIN_SAMPLING_FACET,
                )
                _advance_phase()
                st.rerun()
        with choice_cols[2]:
            if st.button("Skip (don't rate)", width="stretch", key="calib_skip"):
                st.session_state.calibration_skipped.add(_similarity_key(triplet))
                st.rerun()
else:
    st.subheader(f"Taste round ({st.session_state.calibration_phase_count + 1}/{TASTE_BATCH_SIZE})")

    rated_seg_ids = active_taste_repo.rated_segment_ids(rater=selected_profile)
    remaining_taste = [
        sid for sid in all_taste_candidates
        if sid not in rated_seg_ids and sid not in st.session_state.taste_skipped
    ]

    if not remaining_taste:
        st.success(f"All {len(all_taste_candidates)} taste clips rated for {selected_profile}.")
    else:
        seg_id = remaining_taste[0]
        seg = song_repo.get_segment(seg_id)
        song = song_repo.get_song(seg.song_id)

        st.write("**Clip**")
        st.audio(str(audio_path_for(song)), start_time=seg.start_sec, end_time=seg.end_sec, loop=True)
        st.write("Do you like this?")

        taste_cols = st.columns(3)
        with taste_cols[0]:
            if st.button("\U0001F44D Like", width="stretch", type="primary", key="taste_like"):
                active_taste_repo.add_rating(seg_id, liked=True, rater=selected_profile)
                _advance_phase()
                st.rerun()
        with taste_cols[1]:
            if st.button("\U0001F44E Dislike", width="stretch", type="primary", key="taste_dislike"):
                active_taste_repo.add_rating(seg_id, liked=False, rater=selected_profile)
                _advance_phase()
                st.rerun()
        with taste_cols[2]:
            if st.button("Skip (don't rate)", width="stretch", key="taste_skip"):
                st.session_state.taste_skipped.add(seg_id)
                st.rerun()

st.divider()

with st.expander(f"\U0001F3B5 Harmony sanity-check pool (optional, {HARMONY_CHECK_N_TRIPLETS} triplets)"):
    st.caption(
        "Same XAB format as above, but candidates are sampled using Harmony-facet retrieval instead "
        "of Sound -- a small, separate check for whether the main pool's Sound-only sampling misses "
        "pairs that another facet would have surfaced. Entirely optional, doesn't count toward the "
        "similarity/taste batches above."
    )

    if "harmony_check_skipped" not in st.session_state:
        st.session_state.harmony_check_skipped = set()

    harmony_triplets = _cached_harmony_check_triplets(song_repo, embedding_repo, embedding_repo.index_size("harmony"))
    if not harmony_triplets:
        st.info("No Harmony-embedded segments found -- run the harmony-facet batch pipeline first.")
    else:
        harmony_rated_keys = active_calibration_repo.rated_triplet_keys(rater=selected_profile)
        harmony_remaining = [
            t for t in harmony_triplets
            if _similarity_key(t) not in harmony_rated_keys
            and _similarity_key(t) not in st.session_state.harmony_check_skipped
        ]

        if not harmony_remaining:
            st.success(f"All {len(harmony_triplets)} Harmony sanity-check triplets rated for {selected_profile}.")
        else:
            h_triplet = harmony_remaining[0]
            h_seg_x = song_repo.get_segment(h_triplet.segment_x_id)
            h_seg_a = song_repo.get_segment(h_triplet.segment_a_id)
            h_seg_b = song_repo.get_segment(h_triplet.segment_b_id)

            st.caption(
                f"{len(harmony_triplets) - len(harmony_remaining)}/{len(harmony_triplets)} rated"
            )
            st.write("**Reference**")
            h_song_x = song_repo.get_song(h_seg_x.song_id)
            st.audio(
                str(audio_path_for(h_song_x)), start_time=h_seg_x.start_sec, end_time=h_seg_x.end_sec, loop=True,
            )

            st.write("Which of these two clips sounds more similar to the Reference?")
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.write("**Clip A**")
                h_song_a = song_repo.get_song(h_seg_a.song_id)
                st.audio(
                    str(audio_path_for(h_song_a)), start_time=h_seg_a.start_sec, end_time=h_seg_a.end_sec, loop=True,
                )
            with h_col2:
                st.write("**Clip B**")
                h_song_b = song_repo.get_song(h_seg_b.song_id)
                st.audio(
                    str(audio_path_for(h_song_b)), start_time=h_seg_b.start_sec, end_time=h_seg_b.end_sec, loop=True,
                )

            h_choice_cols = st.columns(3)
            with h_choice_cols[0]:
                if st.button("A is more similar", width="stretch", key="harmony_check_a"):
                    active_calibration_repo.add_choice(
                        h_triplet.segment_x_id, h_triplet.segment_a_id, h_triplet.segment_b_id, "a",
                        rater=selected_profile, sampling_facet=HARMONY_CHECK_FACET,
                    )
                    st.rerun()
            with h_choice_cols[1]:
                if st.button("B is more similar", width="stretch", key="harmony_check_b"):
                    active_calibration_repo.add_choice(
                        h_triplet.segment_x_id, h_triplet.segment_a_id, h_triplet.segment_b_id, "b",
                        rater=selected_profile, sampling_facet=HARMONY_CHECK_FACET,
                    )
                    st.rerun()
            with h_choice_cols[2]:
                if st.button("Skip (don't rate)", width="stretch", key="harmony_check_skip"):
                    st.session_state.harmony_check_skipped.add(_similarity_key(h_triplet))
                    st.rerun()
