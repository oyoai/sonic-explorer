"""Temporary, throwaway Streamlit tool -- NOT part of the Sonic Explorer app
itself, just a blind-listening aid for spot-checking the AST vocal-presence
prevalence sample (scripts/sample_vocal_segment_prevalence.py). Run it as its
own process on its own port so it doesn't touch the real app's navigation:

    streamlit run scripts/vocal_spotcheck_app.py --server.port 8502

Plays each curated segment at its exact 5s window (looped), asks for a blind
vocal/instrumental judgment with no score/label shown, then reveals the
model's verdict for comparison once all segments are answered. Answers are
saved to scripts/vocal_spotcheck_results.csv as you go, so progress survives
a refresh.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from sonic_explorer.config import AUDIO_DIR

RESULTS_CSV = Path(__file__).resolve().parent / "vocal_spotcheck_results.csv"

# The curated 10-segment list -- title/genre for context, real file + timestamp
# to play, and the model's score/label kept hidden until the reveal screen.
SEGMENTS = [
    {"title": "412", "genre": "Hip-Hop", "track_id": 75755, "start": 20.0, "end": 25.0,
     "score": 0.0179, "label": "Rapping", "note": "Most ambiguous -- 0.0001 below threshold"},
    {"title": "Dismissal", "genre": "Pop", "track_id": 41568, "start": 5.0, "end": 10.0,
     "score": 0.0171, "label": "Singing", "note": "Near-threshold flagged"},
    {"title": "Facing the Sea (Album Version)", "genre": "Pop", "track_id": 36959, "start": 5.0, "end": 10.0,
     "score": 0.0195, "label": "Speech", "note": "Kept, just above threshold -- contrast for Dismissal"},
    {"title": "A Message", "genre": "Rock", "track_id": 122066, "start": 12.5, "end": 17.5,
     "score": 0.0174, "label": "Speech", "note": "Near-threshold flagged"},
    {"title": "Requiem for a Small Town", "genre": "Folk", "track_id": 89484, "start": 12.5, "end": 17.5,
     "score": 0.0175, "label": "Singing", "note": "Near-threshold flagged"},
    {"title": "something brewing", "genre": "Hip-Hop", "track_id": 121318, "start": 5.0, "end": 10.0,
     "score": 0.0028, "label": "Speech", "note": "From the earlier 9-song validation"},
    {"title": "A1 Symphony", "genre": "Hip-Hop", "track_id": 154309, "start": 17.5, "end": 22.5,
     "score": 0.0017, "label": "Speech", "note": "Clearly-low, vocal-heavy genre"},
    {"title": "Underwater", "genre": "Electronic", "track_id": 63064, "start": 2.5, "end": 7.5,
     "score": 0.0007, "label": "Speech", "note": "Baseline true-negative check"},
    {"title": "Ride My Bike", "genre": "Instrumental", "track_id": 139537, "start": 7.5, "end": 12.5,
     "score": 0.0002, "label": "Speech", "note": "Baseline true-negative check"},
    {"title": "Thursday & Snow (Reprise)", "genre": "Hip-Hop", "track_id": 64625, "start": 2.5, "end": 7.5,
     "score": 0.0228, "label": "Speech", "note": "Kept, comfortably above threshold -- positive control"},
]

st.set_page_config(page_title="Vocal Spot-Check (temporary)", page_icon="\U0001F442")
st.title("Vocal Spot-Check")
st.caption(
    "Temporary QA tool, not part of the real app. Listen to each ~5s clip and judge blind -- "
    "the model's score/label are hidden until you've answered all 10."
)

if "spotcheck_index" not in st.session_state:
    st.session_state.spotcheck_index = 0
if "spotcheck_answers" not in st.session_state:
    st.session_state.spotcheck_answers = {}


def save_results():
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "genre", "start", "end", "user_verdict", "model_score", "model_label", "agree"])
        for i, seg in enumerate(SEGMENTS):
            verdict = st.session_state.spotcheck_answers.get(i)
            if verdict is None:
                continue
            model_says_vocal = seg["score"] >= 0.018
            agree = (verdict == "vocal") == model_says_vocal
            writer.writerow([seg["title"], seg["genre"], seg["start"], seg["end"], verdict,
                              seg["score"], seg["label"], agree])


idx = st.session_state.spotcheck_index
total = len(SEGMENTS)

if idx < total:
    seg = SEGMENTS[idx]
    st.progress(idx / total, text=f"Segment {idx + 1} of {total}")
    st.subheader(f"\"{seg['title']}\" ({seg['genre']})")
    st.caption(f"{seg['start']:.1f}s – {seg['end']:.1f}s")

    audio_path = AUDIO_DIR / f"{seg['track_id']}.mp3"
    if audio_path.exists():
        st.audio(str(audio_path), start_time=seg["start"], end_time=seg["end"], loop=True)
    else:
        st.error(f"Audio file not found: {audio_path}")

    st.write("What do you hear?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001F3A4 Vocal", width="stretch", key=f"vocal_{idx}"):
            st.session_state.spotcheck_answers[idx] = "vocal"
            st.session_state.spotcheck_index += 1
            save_results()
            st.rerun()
    with col2:
        if st.button("\U0001F3B9 Instrumental", width="stretch", key=f"instrumental_{idx}"):
            st.session_state.spotcheck_answers[idx] = "instrumental"
            st.session_state.spotcheck_index += 1
            save_results()
            st.rerun()

    if idx > 0:
        if st.button("← Back to previous"):
            st.session_state.spotcheck_index -= 1
            st.rerun()
else:
    st.success("All 10 segments answered. Here's the reveal:")
    save_results()

    n_agree = 0
    for i, seg in enumerate(SEGMENTS):
        verdict = st.session_state.spotcheck_answers.get(i, "?")
        model_says = "vocal" if seg["score"] >= 0.018 else "instrumental"
        agree = verdict == model_says
        n_agree += int(agree)
        icon = "✅" if agree else "❌"
        st.markdown(
            f"{icon} **\"{seg['title']}\"** ({seg['genre']}, {seg['start']:.1f}-{seg['end']:.1f}s) -- "
            f"you said **{verdict}**, model score **{seg['score']:.4f}** ({seg['label']}) -> "
            f"model says **{model_says}**"
        )
        st.caption(seg["note"])

    st.metric("Agreement with model", f"{n_agree}/{total}")
    st.caption(f"Results saved to {RESULTS_CSV}")

    if st.button("Start over"):
        st.session_state.spotcheck_index = 0
        st.session_state.spotcheck_answers = {}
        st.rerun()
