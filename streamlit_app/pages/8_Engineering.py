import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from engineering_data import CNN_PER_GENRE_METRICS, CNN_RESULTS, RED_TEAM_CASES, predict_genre
from resources import get_agent, get_repositories, hero_banner, nav_button, show_data_source_banner, show_logo

CI_WORKFLOW_URL = "https://github.com/oyoai/sonic-explorer/actions/workflows/ci.yml"
CI_BADGE_URL = "https://github.com/oyoai/sonic-explorer/actions/workflows/ci.yml/badge.svg"

st.set_page_config(page_title="Sonic Explorer", layout="wide")
hero_banner("engineering")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()

st.title("Engineering")
st.write(
    "Methodology and Results cover whether the approach works. This page covers whether it's "
    "built responsibly -- and rather than just asserting that, each claim below is something "
    "you can actually try against the live, deployed system yourself."
)
nav_button("← Back to Methodology", "pages/1_Methodology.py", key="nav_engineering_to_methodology")

show_logo()
show_data_source_banner()

st.divider()

# ---------------------------------------------------------------------------
# 0. Agent architecture: planning pattern + memory
# ---------------------------------------------------------------------------
st.header("Agent architecture: planning pattern + memory")
st.write(
    "Ask the DJ's loop (`llm/agent.py`'s `MusicAgent.send_message`) is a **ReAct-style tool-calling "
    "loop**: the model reasons about a request, calls a tool, sees the real result, and either "
    "calls another tool or replies -- up to 5 rounds per turn (`DEFAULT_MAX_TOOL_ITERATIONS`) "
    "before falling back to an honest \"couldn't finish that\" reply rather than looping forever. "
    "This is the native Anthropic tool-use loop (structured `tool_use`/`tool_result` blocks), not a "
    "hand-authored `Thought:`/`Action:`/`Observation:` text template -- functionally the same "
    "interleaved reasoning-and-acting pattern ReAct describes, implemented through the API's "
    "built-in tool-calling rather than a textual scaffold."
)
st.write(
    "**Memory model: session-scoped only, no long-term memory.** `MusicAgent` itself is stateless "
    "(safe to share across users via `st.cache_resource` -- see `resources.py`); the actual "
    "conversation history lives in each browser session's `st.session_state.agent_history` and is "
    "passed in and handed back on every call. That means the DJ remembers earlier turns *within* "
    "one open session (it can resolve \"that one\" or \"the second song you mentioned\"), but "
    "nothing persists across a page refresh or between different users -- no long-term memory, no "
    "user profile, no cross-session recall."
)

st.divider()

# ---------------------------------------------------------------------------
# 1. Red-teaming, interactive
# ---------------------------------------------------------------------------
st.header("1. Red-teaming, interactive")
n_pass = sum(1 for _, _, verdict, _ in RED_TEAM_CASES if verdict == "PASS")
st.write(
    f"**{n_pass}/{len(RED_TEAM_CASES)} adversarial prompts passed** -- graded by hand against "
    "each category's own success criteria (no system-prompt leak, no compliance with an injected/"
    "override instruction, no fabricated specifics presented as fact, no attempt at an "
    "out-of-scope action). No automated grader exists for this; the verdicts below are a human "
    "judgment call, not a generated score. Click any prompt to send it live to the deployed DJ "
    "and see its real response -- the same live agent Ask the DJ uses, not a canned replay."
)

if "engineering_red_team_log" not in st.session_state:
    st.session_state.engineering_red_team_log = {}  # prompt -> live reply

agent = get_agent()
if agent is None:
    st.info(
        "Set ANTHROPIC_API_KEY to try these live. The graded verdicts below still reflect real, "
        "previously-run transcripts (scripts/red_team_findings.md)."
    )

current_category = None
for i, (category, prompt, verdict, note) in enumerate(RED_TEAM_CASES):
    if category != current_category:
        st.subheader(category)
        current_category = category

    cols = st.columns([3, 1])
    with cols[0]:
        clicked = st.button(f"\"{prompt}\"", key=f"redteam_{i}", disabled=agent is None)
    with cols[1]:
        st.markdown(f"**{verdict}**")
    st.caption(note)

    if clicked and agent is not None:
        with st.spinner("Sending live to the DJ..."):
            try:
                reply, _ = agent.send_message([], prompt)
            except Exception as exc:
                reply = f"[EXCEPTION: {exc}]"
        st.session_state.engineering_red_team_log[prompt] = reply

    if prompt in st.session_state.engineering_red_team_log:
        st.info(f"**Live reply just now:** {st.session_state.engineering_red_team_log[prompt]}")

st.divider()

# ---------------------------------------------------------------------------
# 2. CI / Docker
# ---------------------------------------------------------------------------
st.header("2. CI / Docker")
st.write(
    "Every push and pull request against `main` runs ruff (lint) and the full pytest suite, plus a "
    "Docker build-verification job -- real, live GitHub Actions runs, not a static claim:"
)
st.markdown(f"[![CI]({CI_BADGE_URL})]({CI_WORKFLOW_URL})")
st.caption(f"Click through to the actual workflow run history: {CI_WORKFLOW_URL}")
st.write(
    "A `Dockerfile` at the repo root also builds the app in a container for local/portability "
    "use -- the CI `docker-build` job verifies the image actually builds on every change. This "
    "isn't how the live app is actually served, though: the deployed instance runs directly from "
    "GitHub via Streamlit Community Cloud, not this container."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. CNN classifier, live
# ---------------------------------------------------------------------------
st.header("3. CNN classifier, live")
st.write(
    "A small CNN trained directly on log-mel spectrograms to predict genre "
    f"({CNN_RESULTS['n_classes']} classes, stratified "
    f"{CNN_RESULTS['n_train']}/{CNN_RESULTS['n_val']}/{CNN_RESULTS['n_test']} train/val/test split "
    "on the real library) -- a parallel, independently-trained baseline to the hand-crafted/"
    "pretrained-embedding facets used everywhere else in this project. Pick any song below and "
    "the model runs real inference on its actual audio, live, right now -- not a cached or "
    "precomputed prediction."
)

cnn_cols = st.columns(3)
cnn_cols[0].metric("Test accuracy", f"{CNN_RESULTS['test_accuracy']:.1%}")
cnn_cols[1].metric("Random baseline", f"{CNN_RESULTS['random_baseline']:.1%}")
cnn_cols[2].metric("Lift over random", f"{(CNN_RESULTS['test_accuracy'] / CNN_RESULTS['random_baseline']):.1f}x")
st.caption(
    "47.2% test accuracy against a 12.5% random baseline (8 balanced classes) -- a real, "
    "non-trivial signal from spectrograms alone, with no pretrained model and no facet "
    "engineering. These summary numbers are a supplement to the live picker below, not a "
    "replacement for it -- try it yourself rather than taking the percentage on faith."
)

st.write(
    "**Per-genre precision/recall/F1**, not just overall accuracy -- the test split is perfectly "
    "class-balanced (27 songs/genre), so there's no support imbalance, but overall accuracy still "
    "hides a real decision-boundary imbalance the model learned:"
)
per_genre_fig = go.Figure()
per_genre_fig.add_trace(go.Bar(
    name="Precision", x=[g for g, *_ in CNN_PER_GENRE_METRICS], y=[p for _, p, _, _, _ in CNN_PER_GENRE_METRICS],
))
per_genre_fig.add_trace(go.Bar(
    name="Recall", x=[g for g, *_ in CNN_PER_GENRE_METRICS], y=[r for _, _, r, _, _ in CNN_PER_GENRE_METRICS],
))
per_genre_fig.add_trace(go.Bar(
    name="F1", x=[g for g, *_ in CNN_PER_GENRE_METRICS], y=[f for _, _, _, f, _ in CNN_PER_GENRE_METRICS],
))
per_genre_fig.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10), barmode="group", yaxis_title="score", yaxis_range=[0, 1],
)
st.plotly_chart(per_genre_fig, width="stretch", key="cnn_per_genre_chart")
st.caption(
    "The model over-predicts **Hip-Hop** (0.481 precision but 0.926 recall -- its default guess "
    "when unsure) while badly under-recalling **Pop** (0.667 precision but only 0.148 recall -- "
    "correct when it does say Pop, but rarely says it). Recomputed directly from the real cached "
    "test-split features and the real saved model weights (not retrained, not estimated) -- "
    "recomputed overall accuracy matched the summary number above exactly, confirming this "
    "reproduction is faithful to the original training run."
)
st.info(
    "**Real gap, not fixed here:** this model was trained with a single stratified train/val/test "
    "split and fixed hyperparameters (Adam, lr=1e-3, batch=16, 20 epochs) -- no cross-validation "
    "and no hyperparameter search were run. On ~1400 real audio files, a proper k-fold CV or even "
    "a small grid over learning rate/batch size is feasible on CPU (each full run takes minutes), "
    "just not done yet -- the single-split number above should be read as a real but "
    "not-fully-validated estimate, not a rigorously tuned result."
)

if not all_songs:
    st.info("No songs in the library yet.")
else:
    song_options = sorted(all_songs, key=lambda s: s.title)
    song_labels = [f"{s.title} — {s.artist} ({s.genre_top})" for s in song_options]
    picked_label = st.selectbox("Pick a song", options=song_labels, key="cnn_song_picker")
    picked_song = song_options[song_labels.index(picked_label)]

    if st.button("Predict genre", key="cnn_predict_button"):
        with st.spinner("Running the CNN on the real audio..."):
            try:
                predictions = predict_genre(picked_song)
            except Exception as exc:
                predictions = None
                st.error(f"Inference failed: {exc}")

        if predictions is not None:
            top_genre, top_prob = predictions[0]
            correct = top_genre == picked_song.genre_top
            st.metric(
                "Predicted genre" + (" ✓" if correct else " ✗"),
                top_genre,
                delta=f"actual: {picked_song.genre_top}",
                delta_color="normal" if correct else "inverse",
            )
            pred_fig = go.Figure(go.Bar(
                x=[p for _, p in predictions], y=[g for g, _ in predictions],
                orientation="h", text=[f"{p:.1%}" for _, p in predictions], textposition="auto",
            ))
            pred_fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="predicted probability", yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(pred_fig, width="stretch", key="cnn_prediction_chart")
            if not correct:
                st.caption(
                    f"Wrong this time -- at 47.2% test accuracy across {CNN_RESULTS['n_classes']} classes, "
                    "misses like this are expected, not a sign something's broken."
                )

st.divider()

st.write("Next: **Results** reports the outcome numbers this rigor was measured against.")
nav_button("Continue to Results →", "pages/2_Results.py", key="nav_engineering_to_results")
