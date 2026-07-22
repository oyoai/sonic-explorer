"""Calibration-rating tool: blind human similarity ratings on segment pairs,
the dataset section 9's blend-weight regression and (conditionally) CLAP
fine-tuning depend on. Multi-session by design -- progress is persisted to
the real calibration_ratings table (not session state), so closing the tab
and coming back tomorrow picks up exactly where you left off.

Run as its own process, its own port:

    streamlit run scripts/calibration_rating_app.py --server.port 8504

Deliberately blind: no song title/artist shown, no algorithm similarity
score, just "Clip A" / "Clip B" -- rating what you actually hear, not what
you recognize. Pairs are drawn from three similarity bands (see
evaluation/calibration_pairs.py) so the resulting ratings actually span the
similarity range, but which band a pair came from is never shown either.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH, audio_path_for
from sonic_explorer.evaluation.calibration_pairs import generate_calibration_pairs
from sonic_explorer.repository.calibration_repository import CalibrationRepository
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

RATER_NAME = "offi"  # single-rater project -- see spec section 1's "no real auth needed" precedent
N_HIGH, N_MEDIUM, N_RANDOM = 120, 110, 120

st.set_page_config(page_title="Calibration Rating", page_icon="\U0001F3B7")
st.title("Calibration Rating")
st.caption(
    "Blind similarity rating -- no titles, no artist names, no algorithm score. Just how similar "
    "the two clips actually sound to you. Progress saves after every rating, so it's safe to close "
    "this and come back later."
)

conn = init_db(DB_PATH)
song_repo = SongRepository(conn)
embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
embedding_repo.load_index("sound")
calibration_repo = CalibrationRepository(conn)


@st.cache_data(show_spinner="Generating the pair set...")
def _cached_pairs(_song_repo, _embedding_repo, index_size):
    return generate_calibration_pairs(
        _song_repo, _embedding_repo, facet_name="sound", n_high=N_HIGH, n_medium=N_MEDIUM, n_random=N_RANDOM,
    )


all_pairs = _cached_pairs(song_repo, embedding_repo, embedding_repo.index_size("sound"))
total = len(all_pairs)

if total == 0:
    st.warning("No embedded segments found -- run the sound-facet batch pipeline first.")
    st.stop()

if "skipped_this_session" not in st.session_state:
    st.session_state.skipped_this_session = set()  # not persisted -- these pairs reappear next session

rated_pairs = calibration_repo.rated_pair_ids()
n_rated = sum(1 for p in all_pairs if (p.segment_a_id, p.segment_b_id) in rated_pairs)
remaining = [
    p for p in all_pairs
    if (p.segment_a_id, p.segment_b_id) not in rated_pairs
    and (p.segment_a_id, p.segment_b_id) not in st.session_state.skipped_this_session
]

st.progress(n_rated / total, text=f"{n_rated}/{total} rated ({len(remaining)} remaining)")

if not remaining:
    st.success(f"All {total} pairs rated. Thank you -- this dataset is ready for the blend-weight regression.")
    st.stop()

pair = remaining[0]
seg_a = song_repo.get_segment(pair.segment_a_id)
seg_b = song_repo.get_segment(pair.segment_b_id)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Clip A")
    song_a = song_repo.get_song(seg_a.song_id)
    st.audio(str(audio_path_for(song_a)), start_time=seg_a.start_sec, end_time=seg_a.end_sec, loop=True)
with col2:
    st.subheader("Clip B")
    song_b = song_repo.get_song(seg_b.song_id)
    st.audio(str(audio_path_for(song_b)), start_time=seg_b.start_sec, end_time=seg_b.end_sec, loop=True)

st.write("How similar do these two clips sound to you?")
rating = st.select_slider(
    "Similarity",
    options=[1, 2, 3, 4, 5],
    value=3,
    format_func=lambda r: {
        1: "1 — Not similar at all", 2: "2 — A little similar", 3: "3 — Somewhat similar",
        4: "4 — Very similar", 5: "5 — Extremely similar",
    }[r],
    key=f"rating_{pair.segment_a_id}_{pair.segment_b_id}",
)

btn_cols = st.columns(2)
with btn_cols[0]:
    if st.button("Submit rating", width="stretch", type="primary"):
        calibration_repo.add_rating(pair.segment_a_id, pair.segment_b_id, float(rating), rater=RATER_NAME)
        st.rerun()
with btn_cols[1]:
    if st.button("Skip (don't rate)", width="stretch"):
        # Not persisted -- only skipped for the rest of *this* session, so it
        # reappears next time the tool is opened. Fine for occasional skips
        # (e.g. a clip that fails to load); queue order is otherwise stable
        # across sessions since pair generation is seeded.
        st.session_state.skipped_this_session.add((pair.segment_a_id, pair.segment_b_id))
        st.rerun()
