import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from comparison_data import build_naive_vs_real_graphs, get_demo_pairs
from components.plotting import network_graph_figure
from resources import get_repositories, nav_button, show_data_source_banner, show_logo
from sonic_explorer.analysis.network_graph import cross_genre_edge_fraction
from sonic_explorer.config import audio_path_for

# ---------------------------------------------------------------------------
# Curated evidence, embedded directly rather than loaded from data/artifacts/
# or scripts/ at runtime -- same rationale as Methodology.py's top-of-file
# comment: those files are either gitignored or one-off script outputs, and
# this is hand-picked presentation content that should stay stable regardless
# of whether the underlying scripts get re-run later.
#
# Source of truth:
#   GENRE_COHESION_RESULTS -- scripts/run_evaluation.py -> data/artifacts/genre_cohesion_results.json
#   CNN_RESULTS -- scripts/train_genre_cnn.py -> scripts/genre_cnn_results.json (real run, see commit history)
# ---------------------------------------------------------------------------

GENRE_COHESION_RESULTS = {
    "k": 10,
    "sample_size": 500,
    "facets": [
        {"facet_name": "sound", "n_queries": 500, "observed_pct": 54.4, "random_baseline_pct": 11.9},
        {"facet_name": "harmony", "n_queries": 500, "observed_pct": 21.4, "random_baseline_pct": 11.7},
        {"facet_name": "vocal", "n_queries": 500, "observed_pct": 38.1, "random_baseline_pct": 13.9},
        {"facet_name": "drums", "n_queries": 500, "observed_pct": 37.4, "random_baseline_pct": 13.1},
        {"facet_name": "bass", "n_queries": 500, "observed_pct": 26.7, "random_baseline_pct": 12.5},
        {"facet_name": "instrumental", "n_queries": 500, "observed_pct": 41.5, "random_baseline_pct": 11.8},
    ],
}

CNN_RESULTS = {
    "n_train": 976, "n_val": 208, "n_test": 216, "n_classes": 8,
    "test_accuracy": 0.472, "random_baseline": 0.125, "best_val_accuracy": 0.486,
}

st.set_page_config(page_title="Sonic Explorer", layout="wide")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()
songs_by_id = {s.id: s for s in all_songs}

st.title("Results")
st.write(
    "**Methodology** (previous page) covered how the library was analyzed and improved, "
    "including the honest failures along the way. This page reports the outcome: the "
    "quantitative evaluation numbers those improvements were measured against -- starting "
    "with the question Overview raised but deliberately didn't answer yet: is the audio-based "
    "approach actually better than the existing paradigms it's being compared against?"
)
nav_button("← Back to Methodology", "pages/1_Methodology.py", key="nav_results_to_methodology")

show_logo()
show_data_source_banner()

st.divider()

# ---------------------------------------------------------------------------
# 1. Naive baseline vs. real audio similarity
# ---------------------------------------------------------------------------
st.header("1. Naive baseline vs. real audio similarity")
st.write(
    "Overview showed the strongest non-audio baseline this library's metadata can build (genre "
    "+ genre hierarchy + album + tags) and the tautology built into it: a metadata-only edge "
    "rule can only ever connect songs that already share a label, so its clean clusters aren't "
    "evidence of anything. Now that Approach and Methodology have covered the mechanism, here's "
    "the other half of that comparison -- the same songs, connected by real audio-embedding "
    "similarity instead:"
)

if all_songs:
    naive_nodes, naive_edges, real_nodes, real_edges, vectors, genre_by_song = build_naive_vs_real_graphs(
        song_repo, embedding_repo, len(all_songs)
    )
    if not real_nodes.empty:
        st.caption("**Audio embeddings.** Edges come from what the audio actually sounds like.")
        st.plotly_chart(
            network_graph_figure(real_nodes, real_edges), width="stretch", key="results_real_graph"
        )

        naive_cross_pct = cross_genre_edge_fraction(naive_edges, genre_by_song)
        real_cross_pct = cross_genre_edge_fraction(real_edges, genre_by_song)
        st.write(
            f"**{real_cross_pct:.0%}** of this graph's edges cross a genre boundary, versus "
            f"**{naive_cross_pct:.0%}** for the metadata-only version in Overview -- and every one "
            "of those crossings here is a connection no amount of catalog metadata could have "
            "found, since genre never enters this computation at all."
        )

        st.write("**Hear it, don't just read it:**")
        naive_pair, real_pair = get_demo_pairs(song_repo, embedding_repo, len(all_songs))
        demo_cols = st.columns(2)
        with demo_cols[0]:
            if naive_pair is not None:
                a, b = songs_by_id[naive_pair.song_id_a], songs_by_id[naive_pair.song_id_b]
                st.caption(
                    f"**Naive calls these \"similar\":** both tagged **{a.genre_top}** — but "
                    f"their real audio similarity is only {naive_pair.audio_similarity:.2f}, "
                    f"near the bottom of this library. Judge for yourself:"
                )
                st.write(f"\"{a.title}\" — {a.artist}")
                st.audio(str(audio_path_for(a)))
                st.write(f"\"{b.title}\" — {b.artist}")
                st.audio(str(audio_path_for(b)))
            else:
                st.info("Not enough naive-graph edges yet to pick a demo pair.")
        with demo_cols[1]:
            if real_pair is not None:
                a, b = songs_by_id[real_pair.song_id_a], songs_by_id[real_pair.song_id_b]
                st.caption(
                    f"**Audio calls these \"similar\":** **{a.genre_top}** and **{b.genre_top}** "
                    f"— different genres, no shared metadata needed — at a real similarity of "
                    f"{real_pair.audio_similarity:.2f}. Listen for yourself:"
                )
                st.write(f"\"{a.title}\" — {a.artist}")
                st.audio(str(audio_path_for(a)))
                st.write(f"\"{b.title}\" — {b.artist}")
                st.audio(str(audio_path_for(b)))
            else:
                st.info("Not enough cross-genre audio edges yet to pick a demo pair.")
    else:
        st.info("No embedded songs available yet to build this comparison.")
else:
    st.info("No songs available yet to build this comparison.")

st.divider()

# ---------------------------------------------------------------------------
# 2. Genre-cohesion evaluation
# ---------------------------------------------------------------------------
st.header("2. Genre-cohesion evaluation")
st.write(
    f"Quantitative check: do a facet's nearest neighbors actually share genre more often than "
    f"chance, at k={GENRE_COHESION_RESULTS['k']} (sampled over {GENRE_COHESION_RESULTS['facets'][0]['n_queries']} "
    f"queries per facet)? Genre is a proxy, not ground truth for \"sounds similar\" -- but a facet "
    f"that shows no lift over random would be a red flag."
)

facets_data = GENRE_COHESION_RESULTS["facets"]
eval_fig = go.Figure(data=[
    go.Bar(name="Observed", x=[f["facet_name"].capitalize() for f in facets_data],
           y=[f["observed_pct"] for f in facets_data],
           text=[f"{f['observed_pct']:.1f}%" for f in facets_data], textposition="auto"),
    go.Bar(name="Random baseline", x=[f["facet_name"].capitalize() for f in facets_data],
           y=[f["random_baseline_pct"] for f in facets_data],
           text=[f"{f['random_baseline_pct']:.1f}%" for f in facets_data], textposition="auto"),
])
eval_fig.update_layout(
    height=400, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
    yaxis_title="% of neighbors sharing genre",
)
st.plotly_chart(eval_fig, width="stretch", key="genre_cohesion_chart")

st.caption(
    "Every facet clears its random baseline by a wide margin -- Sound strongest (54.4% vs. 11.9%), "
    "Instrumental/Vocal/Drums in the middle (37-42%), Bass and Harmony weakest but still clearly "
    "above chance (21-27%). The ablation-style finding: facets genuinely diverge from each other "
    "rather than one just riding Sound's coattails -- Harmony in particular captures something "
    "clearly different (and on this metric, weaker) than the full mix."
)
st.info(
    "These numbers reflect the completed stem-facet reprocessing pass (fixed a data-quality issue "
    "where a handful of near-silent isolated stems were being indexed as if meaningful -- Sound and "
    "Harmony were never affected, since neither depends on stem separation). The shift was real but "
    "modest: Vocal/Drums/Instrumental all moved a couple points, and interestingly the *random-pair "
    "baseline* moved too (11.7% -> 12.5-13.9% for the four stem facets) -- removing near-silent noise "
    "from the index changed what a \"random\" pair typically looks like, not just what a \"good\" match "
    "looks like. Harmony's number here is also its pre-whitening baseline; see Methodology §7c for "
    "the whitening experiment's own before/after measurement."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Genre classifier baseline (CNN)
# ---------------------------------------------------------------------------
st.header("3. Genre classifier baseline (CNN)")
st.write(
    "A separate, independent check: a small CNN trained directly on log-mel spectrograms to "
    f"predict genre ({CNN_RESULTS['n_classes']} classes, stratified "
    f"{CNN_RESULTS['n_train']}/{CNN_RESULTS['n_val']}/{CNN_RESULTS['n_test']} train/val/test split "
    "on the real library). This isn't part of the retrieval pipeline -- it's a trained-model "
    "baseline, a parallel story to the hand-crafted-features-vs-CLAP comparison: does a model "
    "that only ever sees spectrograms, with no pretrained audio-embedding knowledge, learn "
    "anything real about genre from this library?"
)
cnn_cols = st.columns(3)
cnn_cols[0].metric("Test accuracy", f"{CNN_RESULTS['test_accuracy']:.1%}")
cnn_cols[1].metric("Random baseline", f"{CNN_RESULTS['random_baseline']:.1%}")
cnn_cols[2].metric(
    "Lift over random",
    f"{(CNN_RESULTS['test_accuracy'] / CNN_RESULTS['random_baseline']):.1f}x",
)
st.caption(
    "47.2% test accuracy against a 12.5% random baseline (8 balanced classes) -- a real, "
    "non-trivial signal from spectrograms alone, with no pretrained model and no facet "
    "engineering. Not competitive with the Sound facet's 54.4% genre-cohesion figure above (a "
    "different metric on a different task, so not directly comparable), but it confirms the "
    "library's genre labels correlate with something audible in the raw spectrogram, "
    "independent of CLAP or any of the other pretrained models the rest of the app leans on."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Calibration study & blend-weight regression
# ---------------------------------------------------------------------------
st.header("4. Calibration study & blend-weight regression")
st.write(
    "A genre-cohesion lift is necessary but not sufficient -- it doesn't say whether a match "
    "*feels* right to an actual listener. This study collects blind pairwise similarity "
    "ratings (no title, artist, or score shown) across a mix of high-scoring, medium-scoring, "
    "and random song pairs, then regresses facet blend weights against those human judgments."
)
st.warning(
    "**In progress -- no results yet.** The rating tool is live and collecting data; this "
    "section will report the regression's findings once enough pairs are rated. Shown as an "
    "honest \"not done\" rather than filled with placeholder numbers."
)

st.divider()

st.write("Next: **App Walkthrough** picks up from here, with the live interactive pages themselves.")
nav_button("Continue to the App Walkthrough →", "pages/3_App_Walkthrough.py", key="nav_results_to_walkthrough")
