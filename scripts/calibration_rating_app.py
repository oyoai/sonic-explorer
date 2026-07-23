"""Calibration-rating tool: blind human XAB similarity judgments, the
dataset section 9's blend-weight regression and (conditionally) CLAP
fine-tuning depend on. Given a reference clip (X) and two candidates (A, B),
the rater picks which candidate sounds more similar to X -- a forced binary
discrimination, more rigorous and less subjective than a raw 1-5 scale (see
Overview §1.2's Related Work: Vohra & Akama (2026) validate exactly this
ABX-preference methodology for stem-wise perceptual alignment).
Multi-session by design -- progress is persisted to the real
calibration_ratings table (not session state), so closing the tab and
coming back tomorrow picks up exactly where you left off.

Run as its own process, its own port:

    streamlit run scripts/calibration_rating_app.py --server.port 8504

Deliberately blind: no song title/artist shown, no algorithm similarity
score, just "Reference" / "Clip A" / "Clip B" -- judging what you actually
hear, not what you recognize. Candidates are drawn from three similarity
bands (see evaluation/calibration_triplets.py) so the resulting choices
actually span the similarity range, but which band a candidate came from is
never shown either.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH, audio_path_for
from sonic_explorer.evaluation.calibration_triplets import generate_calibration_triplets
from sonic_explorer.repository.calibration_repository import CalibrationRepository
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

RATER_NAME = "offi"  # single-rater project -- see spec section 1's "no real auth needed" precedent
N_TRIPLETS = 350

st.set_page_config(page_title="Calibration Rating", page_icon="\U0001F3B7")
st.title("Calibration Rating")
st.caption(
    "Blind XAB judgment -- no titles, no artist names, no algorithm score. Which of Clip A or "
    "Clip B sounds more similar to the Reference? Progress saves after every choice, so it's "
    "safe to close this and come back later."
)

conn = init_db(DB_PATH)
song_repo = SongRepository(conn)
embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
embedding_repo.load_index("sound")
calibration_repo = CalibrationRepository(conn)


@st.cache_data(show_spinner="Generating the triplet set...")
def _cached_triplets(_song_repo, _embedding_repo, index_size):
    return generate_calibration_triplets(_song_repo, _embedding_repo, facet_name="sound", n_triplets=N_TRIPLETS)


all_triplets = _cached_triplets(song_repo, embedding_repo, embedding_repo.index_size("sound"))
total = len(all_triplets)

if total == 0:
    st.warning("No embedded segments found -- run the sound-facet batch pipeline first.")
    st.stop()

if "skipped_this_session" not in st.session_state:
    st.session_state.skipped_this_session = set()  # not persisted -- these triplets reappear next session


def _key(t):
    return (t.segment_x_id, min(t.segment_a_id, t.segment_b_id), max(t.segment_a_id, t.segment_b_id))


rated_keys = calibration_repo.rated_triplet_keys()
n_rated = sum(1 for t in all_triplets if _key(t) in rated_keys)
remaining = [
    t for t in all_triplets
    if _key(t) not in rated_keys
    and _key(t) not in st.session_state.skipped_this_session
]

st.progress(n_rated / total, text=f"{n_rated}/{total} rated ({len(remaining)} remaining)")

if not remaining:
    st.success(f"All {total} triplets rated. Thank you -- this dataset is ready for the blend-weight regression.")
    st.stop()

triplet = remaining[0]
seg_x = song_repo.get_segment(triplet.segment_x_id)
seg_a = song_repo.get_segment(triplet.segment_a_id)
seg_b = song_repo.get_segment(triplet.segment_b_id)

st.subheader("Reference")
song_x = song_repo.get_song(seg_x.song_id)
st.audio(str(audio_path_for(song_x)), start_time=seg_x.start_sec, end_time=seg_x.end_sec, loop=True)

st.write("Which of these two clips sounds more similar to the Reference?")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Clip A")
    song_a = song_repo.get_song(seg_a.song_id)
    st.audio(str(audio_path_for(song_a)), start_time=seg_a.start_sec, end_time=seg_a.end_sec, loop=True)
with col2:
    st.subheader("Clip B")
    song_b = song_repo.get_song(seg_b.song_id)
    st.audio(str(audio_path_for(song_b)), start_time=seg_b.start_sec, end_time=seg_b.end_sec, loop=True)

choice_cols = st.columns(3)
with choice_cols[0]:
    if st.button("A is more similar", width="stretch", type="primary"):
        calibration_repo.add_choice(
            triplet.segment_x_id, triplet.segment_a_id, triplet.segment_b_id, "a", rater=RATER_NAME
        )
        st.rerun()
with choice_cols[1]:
    if st.button("B is more similar", width="stretch", type="primary"):
        calibration_repo.add_choice(
            triplet.segment_x_id, triplet.segment_a_id, triplet.segment_b_id, "b", rater=RATER_NAME
        )
        st.rerun()
with choice_cols[2]:
    if st.button("Skip (don't rate)", width="stretch"):
        # Not persisted -- only skipped for the rest of *this* session, so it
        # reappears next time the tool is opened. Fine for occasional skips
        # (e.g. a clip that fails to load); queue order is otherwise stable
        # across sessions since triplet generation is seeded.
        st.session_state.skipped_this_session.add(_key(triplet))
        st.rerun()
