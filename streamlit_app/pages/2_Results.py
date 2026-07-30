import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comparison_data import build_metadata_vs_real_graphs, get_demo_pairs
from components.plotting import network_graph_figure
from resources import facet_display_name, get_agent, get_repositories, hero_banner, nav_button, show_data_source_banner, show_logo
from sonic_explorer.analysis.network_graph import cross_genre_edge_fraction
from sonic_explorer.analysis.song_dna import AXIS_LABELS
from sonic_explorer.config import audio_path_for
from sonic_explorer.evaluation.retrieval_diagnostics import top1_score_distribution
from sonic_explorer.llm.agent import extract_mentioned_song_ids

# ---------------------------------------------------------------------------
# Curated evidence, embedded directly rather than loaded from data/artifacts/
# or scripts/ at runtime -- same rationale as Methodology.py's top-of-file
# comment: those files are either gitignored or one-off script outputs, and
# this is hand-picked presentation content that should stay stable regardless
# of whether the underlying scripts get re-run later.
#
# Source of truth:
#   GENRE_COHESION_RESULTS -- scripts/run_evaluation.py -> data/artifacts/genre_cohesion_results.json
#
# The CNN classifier's numbers live in engineering_data.py, not here -- see
# the Engineering page, which exercises the CNN live rather than just
# reporting a static number.
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

FACET_ORDER = ["sound", "harmony", "vocal", "drums", "bass", "instrumental"]

# Same real measurement Methodology §7f documents (that page's own copy is
# the technique write-up; this is the same real numbers, duplicated here per
# this file's own "curated evidence embedded directly" convention above,
# not loaded from Methodology at runtime -- pages don't import curated
# numbers from each other in this codebase).
# Source: scripts/measure_clap_gain_sensitivity.py, 30 real segments
# sampled from the library (seed=42), pure multiplicative gain scaling.
CLAP_GAIN_SENSITIVITY_RESULTS = [
    # (gain_db, mean_sim, min_sim, max_sim)
    (-12.0, 0.9086, 0.8201, 0.9704),
    (-6.0, 0.9759, 0.9560, 0.9920),
    (-3.0, 0.9940, 0.9894, 0.9980),
    (3.0, 0.9932, 0.9754, 0.9981),
    (6.0, 0.9729, 0.9046, 0.9924),
    (12.0, 0.8914, 0.6997, 0.9712),
]

# Source: scripts/measure_genre_free_clustering_and_probing.py, run for real
# against deploy_data (233 songs, 8 genres) -- see Methodology §7h for the
# technique (KMeans k=8 over mean-pooled Sound/CLAP embeddings, scored
# against genre_top via Adjusted Rand Index; genre_top never touches the
# clustering step itself).
GENRE_FREE_CLUSTERING_RESULT = {
    "n_songs": 233,
    "n_clusters": 8,
    "adjusted_rand_index": 0.2197,
    "genre_sizes": {
        "Electronic": 31, "Experimental": 27, "Folk": 29, "Hip-Hop": 31,
        "Instrumental": 26, "International": 26, "Pop": 33, "Rock": 30,
    },
    "cluster_sizes": [48, 35, 39, 20, 30, 30, 23, 8],
}

# Source: same script as above -- see Methodology §7i for the technique
# (Ridge regression, 5-fold cross-validated R^2, predicting each raw DNA
# scalar from the same per-song CLAP embeddings the clustering above uses).
LINEAR_PROBING_RESULTS = [
    # (axis, n_songs, r2_mean, r2_std)
    ("tempo_bpm", 233, -0.0367, 0.0251),
    ("energy", 233, 0.3371, 0.0262),
    ("brightness", 233, 0.3006, 0.0423),
    ("harmonic_complexity", 233, 0.2805, 0.0315),
    ("rhythmic_density", 233, 0.2044, 0.0513),
]

# Sound-recognition-specific queries sit alongside typical mood/genre ones --
# the DJ's search_by_sound_content tool (agent_tools.py) is otherwise easy to
# miss since it only fires for named-sound requests, not mood language.
DJ_GALLERY_QUERIES = [
    "Find me something high-energy and bright-sounding",
    "What's moodier and more stripped-back in this library?",
    "Any songs with crow sounds?",
    "Something that sounds like a fart",
]

st.set_page_config(page_title="Sonic Explorer", layout="wide")
hero_banner("results")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()
songs_by_id = {s.id: s for s in all_songs}

st.title("Results")
st.write(
    "**Methodology** covered how the library was analyzed and improved, including the honest "
    "failures along the way, and **Engineering** (previous page) covered the rigor/safety side. "
    "This page reports the outcome: the quantitative evaluation numbers those improvements were "
    "measured against."
)
nav_button("← Back to Engineering", "pages/8_Engineering.py", key="nav_results_to_engineering")

show_logo()
show_data_source_banner()

st.divider()

tab_facet, tab_calibration, tab_dj_gallery, tab_comparison = st.tabs([
    "Facet Evaluation", "Calibration & Blend-Weights", "Ask the DJ Gallery", "Metadata baseline vs. Real approach",
])

# ---------------------------------------------------------------------------
# Facet Evaluation
# ---------------------------------------------------------------------------
with tab_facet:
    st.header("Genre-cohesion outcome")
    st.write(
        f"Do a facet's nearest neighbors actually share genre more often than chance, at "
        f"k={GENRE_COHESION_RESULTS['k']} (sampled over {GENRE_COHESION_RESULTS['facets'][0]['n_queries']} "
        f"queries per facet)? Genre is a proxy, not ground truth for \"sounds similar\" -- but a facet "
        f"that shows no lift over random would be a red flag. See Methodology §3/§7 for how this is "
        f"computed."
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

    st.header("Score distributions across the whole library")
    st.write(
        "A facet can beat the random baseline on genre-sharing while still producing a nearly flat "
        "score landscape underneath, where the \"best\" match isn't meaningfully better than the "
        "10th. This samples real queries per facet and compares the actual top-1 match score against "
        "a random-pair baseline, both drawn from the live index -- not simulated. See Methodology §3a "
        "for why this check matters."
    )

    @st.cache_data(show_spinner="Sampling retrieval scores...")
    def _score_distribution(_song_repo, _embedding_repo, facet_name, index_size, sample_size=200):
        result = top1_score_distribution(_song_repo, _embedding_repo, facet_name=facet_name, sample_size=sample_size)
        return result.n_queries, result.top1_scores, result.random_pair_scores

    score_dist_cols = st.columns(3)
    for i, facet_name in enumerate(FACET_ORDER):
        n_queries, top1_scores, random_scores = _score_distribution(
            song_repo, embedding_repo, facet_name, embedding_repo.index_size(facet_name)
        )
        with score_dist_cols[i % 3]:
            if n_queries == 0:
                st.caption(f"{facet_display_name(facet_name)}: no embedded segments yet.")
                continue
            dist_fig = go.Figure()
            dist_fig.add_trace(go.Histogram(x=top1_scores, name="top-1 match", opacity=0.7, nbinsx=20))
            dist_fig.add_trace(go.Histogram(x=random_scores, name="random pair", opacity=0.7, nbinsx=20))
            dist_fig.update_layout(
                height=240, margin=dict(l=10, r=10, t=30, b=10), barmode="overlay",
                title=f"{facet_display_name(facet_name)} (n={n_queries})", showlegend=(i == 0),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(dist_fig, width="stretch", key=f"score_dist_{facet_name}")

    st.caption(
        "Every facet's top-1 distribution sits clearly to the right of its random-pair distribution -- "
        "real signal, not noise. But look closely at **Harmony**: its random-pair scores already cluster "
        "up near 0.85-0.95, so even a \"real\" top-1 match only pulls a little further right -- the "
        "12-dim chroma embedding space itself has very little natural spread, which is the mechanistic "
        "reason harmony scored weakest on genre-cohesion too. **Sound, Vocal, Drums, Bass, and "
        "Instrumental** all show a much wider gap between the two distributions -- a clearly discriminative "
        "space, even where genre-cohesion alone made a facet look mediocre. Separately (not visible in "
        f"these histograms): the gap between the top-1 and 2nd-best score is small for every facet "
        f"(typically <0.01) -- with {song_repo.count_segments()} segments and no more than a few hundred "
        "per genre, there's usually a long plateau of near-tied candidates rather than one sharply-best "
        "match."
    )

    st.divider()

    st.header("CLAP gain sensitivity")
    st.write(
        "Measured before building a planned perturbation/robustness test on top of this pipeline -- "
        "see Methodology §7f for the full reasoning and the loudness-normalization decision it led "
        "to. 30 real segments sampled from the library (seed=42), gain-shifted by a pure "
        "multiplicative factor (clipped to the valid sample range, so this measures loudness alone, "
        "not loudness plus clipping distortion), cosine similarity between each clip's original and "
        "gain-shifted CLAP embedding."
    )
    _gain_df = pd.DataFrame(CLAP_GAIN_SENSITIVITY_RESULTS, columns=["gain_db", "mean_sim", "min_sim", "max_sim"])
    _results_gain_fig = go.Figure(go.Scatter(
        x=_gain_df["gain_db"], y=_gain_df["mean_sim"], mode="markers+lines",
        error_y=dict(
            type="data", symmetric=False,
            array=_gain_df["max_sim"] - _gain_df["mean_sim"], arrayminus=_gain_df["mean_sim"] - _gain_df["min_sim"],
        ),
        marker=dict(size=9, color="rgb(99,110,250)"), line=dict(color="rgb(99,110,250)"),
    ))
    _results_gain_fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="gain (dB)", yaxis_title="cosine similarity (original vs. gain-shifted)",
        yaxis=dict(range=[0.6, 1.02]),
    )
    st.plotly_chart(_results_gain_fig, width="stretch", key="results_clap_gain_sensitivity_chart")
    st.caption(
        "Markers = mean cosine similarity across the 30 sampled clips; error bars = observed min/max "
        "range at that gain level. **Essentially loudness-invariant at ±3dB** (mean 0.993-0.994, worst "
        "case still 0.989). **Modest, real drift at ±6dB** (mean 0.973-0.976, worst case 0.905). "
        "**Clear drift at ±12dB** (mean 0.891-0.909, worst-case similarity down to 0.70) -- CLAP is "
        "not fully loudness-invariant at larger gain swings, which is why the planned perturbation "
        "suite loudness-normalizes both sides of every comparison rather than assuming this away."
    )

    st.divider()

    st.header("Genre-free clustering: real supporting evidence for the core thesis")
    st.write(
        "This app argues throughout Approach and Methodology that genre_top is a metadata proxy, "
        "not a measurement of what a song actually sounds like. This is the quantitative test of "
        "that claim -- see Methodology §7h for the full method (KMeans, k=8, over per-song "
        "mean-pooled Sound/CLAP embeddings; genre_top never touches the clustering step itself, "
        "only used afterward to score it)."
    )
    _gfc = GENRE_FREE_CLUSTERING_RESULT
    st.metric(
        "Adjusted Rand Index (audio-only clusters vs. real genre labels)",
        f"{_gfc['adjusted_rand_index']:.3f}",
        help="1.0 = clusters perfectly reproduce genre. ~0.0 = no better than chance agreement. "
             "Negative = worse than chance.",
    )
    _genre_cluster_cols = st.columns(2)
    with _genre_cluster_cols[0]:
        st.markdown(f"**Real genre distribution** (n={_gfc['n_songs']})")
        st.bar_chart(pd.Series(_gfc["genre_sizes"], name="songs"))
    with _genre_cluster_cols[1]:
        st.markdown(f"**Audio-only KMeans cluster sizes** (k={_gfc['n_clusters']})")
        st.bar_chart(pd.Series(
            _gfc["cluster_sizes"], index=[f"cluster {i}" for i in range(len(_gfc["cluster_sizes"]))], name="songs",
        ))
    st.info(
        f"**ARI = {_gfc['adjusted_rand_index']:.2f}: a real, moderate, positive signal -- clearly above "
        "chance agreement, clearly far short of genre fully explaining the audio-based clusters.** "
        "If genre fully captured what these songs sound like, ARI would sit close to 1.0; it doesn't. "
        "If audio similarity had nothing to do with genre at all, ARI would sit close to 0; it "
        "doesn't do that either. The honest reading: genre correlates with audio-based similarity "
        "more than chance, but most of the structure KMeans finds in the raw audio embeddings isn't "
        "explained by genre alone -- real, quantitative support for treating genre as a lossy proxy "
        "rather than ground truth for \"sounds similar,\" the framing this app uses throughout, not "
        "an assumption going unchecked."
    )

    st.divider()

    st.header("Linear probing: what CLAP's embedding encodes about Song DNA")
    st.write(
        "Lighter-weight, complementary evidence to the clustering result above -- not about genre, "
        "but about what the Sound facet's CLAP embedding geometrically encodes on its own. See "
        "Methodology §7i for the full method (Ridge regression, 5-fold cross-validated R², "
        "predicting each raw DNA scalar from the same per-song CLAP embeddings the clustering above "
        "uses -- CLAP was never trained on any of these targets)."
    )
    _probe_df = pd.DataFrame(LINEAR_PROBING_RESULTS, columns=["axis", "n_songs", "r2_mean", "r2_std"])
    _probe_fig = go.Figure(go.Bar(
        x=[AXIS_LABELS[a] for a in _probe_df["axis"]], y=_probe_df["r2_mean"],
        error_y=dict(type="data", array=_probe_df["r2_std"]),
        text=[f"{v:.2f}" for v in _probe_df["r2_mean"]], textposition="auto",
        marker_color="rgb(0,204,150)",
    ))
    _probe_fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="cross-validated R²",
    )
    st.plotly_chart(_probe_fig, width="stretch", key="linear_probing_chart")
    st.caption(
        "Error bars = std across the 5 folds. **Energy, brightness, and harmonic complexity probe "
        "moderately well (R² 0.28-0.34)** -- CLAP's embedding encodes real, roughly-linear structure "
        "for these. **Rhythmic density probes more weakly but still clearly above zero (R² 0.20).** "
        "**Tempo does not probe at all (R² -0.04, indistinguishable from a model that just predicts "
        "the mean)** -- a real, honest null result, not a data problem: CLAP is a general-purpose "
        "audio-TEXT alignment model, not trained for periodicity/BPM estimation, so a linear probe "
        "finding no recoverable tempo signal is exactly what that architecture would predict, not a "
        "surprise to paper over."
    )

# ---------------------------------------------------------------------------
# Calibration & Blend-Weights
# ---------------------------------------------------------------------------
with tab_calibration:
    st.header("Calibration study & blend-weight regression")
    st.write(
        "A genre-cohesion lift is necessary but not sufficient -- it doesn't say whether a match "
        "*feels* right to an actual listener. This study collects blind XAB pairwise similarity "
        "ratings (no title, artist, or score shown) across a mix of high-scoring, medium-scoring, "
        "and random song pairs, then regresses facet blend weights against those human judgments -- "
        "see Methodology §8 for the full XAB design."
    )
    st.warning(
        "**In progress -- no results yet.** The rating tool is live and collecting data; this "
        "section will report the XAB outcome, the blend-weight regression result, and the resulting "
        "fine-tuning go/no-go decision once enough pairs are rated. Shown as an honest \"not done\" "
        "rather than filled with placeholder numbers."
    )

# ---------------------------------------------------------------------------
# Ask the DJ Gallery
# ---------------------------------------------------------------------------
with tab_dj_gallery:
    st.header("Ask the DJ Gallery")
    st.write(
        "Click any example below to send it live to the deployed DJ and see its real response -- "
        "including the sound-recognition-specific queries that only search_by_sound_content can "
        "answer, alongside typical mood/genre requests."
    )

    if "results_dj_gallery" not in st.session_state:
        st.session_state.results_dj_gallery = {}  # query -> (reply, song_ids)

    agent = get_agent()
    if agent is None:
        st.info("Set ANTHROPIC_API_KEY to try these live.")

    gallery_cols = st.columns(2)
    for i, query in enumerate(DJ_GALLERY_QUERIES):
        with gallery_cols[i % 2]:
            if st.button(f"\"{query}\"", key=f"gallery_query_{i}", disabled=agent is None):
                with st.spinner("Asking the DJ..."):
                    try:
                        reply, history = agent.send_message([], query)
                        song_ids = extract_mentioned_song_ids(history, 0)
                    except Exception:
                        reply = "Something went wrong on my end -- try again."
                        song_ids = []
                st.session_state.results_dj_gallery[query] = (reply, song_ids)

    for query, (reply, song_ids) in st.session_state.results_dj_gallery.items():
        st.divider()
        st.markdown(f"**You asked:** \"{query}\"")
        st.markdown(reply)
        for song_id in song_ids:
            song = songs_by_id.get(song_id)
            if song is not None:
                st.caption(f"\"{song.title}\" — {song.artist}")
                st.audio(str(audio_path_for(song)))

# ---------------------------------------------------------------------------
# Metadata baseline vs. Real approach
# ---------------------------------------------------------------------------
with tab_comparison:
    st.header("Metadata baseline vs. real audio similarity")
    st.write(
        "Overview showed the strongest non-audio baseline this library's metadata can build (genre "
        "+ genre hierarchy + album + tags) and the tautology built into it: a metadata-only edge "
        "rule can only ever connect songs that already share a label, so its clean clusters aren't "
        "evidence of anything. Now that Approach and Methodology have covered the mechanism, here's "
        "the other half of that comparison -- the same songs, connected by real audio-embedding "
        "similarity instead:"
    )

    if all_songs:
        metadata_nodes, metadata_edges, real_nodes, real_edges, vectors, genre_by_song = build_metadata_vs_real_graphs(
            song_repo, embedding_repo, len(all_songs)
        )
        if not real_nodes.empty:
            st.caption("**Audio embeddings.** Edges come from what the audio actually sounds like.")
            st.plotly_chart(
                network_graph_figure(real_nodes, real_edges), width="stretch", key="results_real_graph"
            )

            metadata_cross_pct = cross_genre_edge_fraction(metadata_edges, genre_by_song)
            real_cross_pct = cross_genre_edge_fraction(real_edges, genre_by_song)
            st.write(
                f"**{real_cross_pct:.0%}** of this graph's edges cross a genre boundary, versus "
                f"**{metadata_cross_pct:.0%}** for the metadata-only version in Overview -- and every one "
                "of those crossings here is a connection no amount of catalog metadata could have "
                "found, since genre never enters this computation at all."
            )

            st.write("**Hear it, don't just read it:**")
            metadata_pair, real_pair = get_demo_pairs(song_repo, embedding_repo, len(all_songs))
            demo_cols = st.columns(2)
            with demo_cols[0]:
                if metadata_pair is not None:
                    a, b = songs_by_id[metadata_pair.song_id_a], songs_by_id[metadata_pair.song_id_b]
                    st.caption(
                        f"**Metadata baseline calls these \"similar\":** both tagged **{a.genre_top}** — but "
                        f"their real audio similarity is only {metadata_pair.audio_similarity:.2f}, "
                        f"near the bottom of this library. Judge for yourself:"
                    )
                    st.write(f"\"{a.title}\" — {a.artist}")
                    st.audio(str(audio_path_for(a)))
                    st.write(f"\"{b.title}\" — {b.artist}")
                    st.audio(str(audio_path_for(b)))
                else:
                    st.info("Not enough metadata-graph edges yet to pick a demo pair.")
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

st.info(
    "**Future work.** This project treats audio-based similarity as a full replacement for the two "
    "existing paradigms from Overview's landscape (metadata matching and collaborative filtering) -- "
    "but an ideal fuller system would combine all three signal types (metadata, audio-based "
    "similarity, and user-level behavioral data) rather than picking just one. This library has no "
    "user-level listen/favorite data to build that third signal from (see Overview), so it's an "
    "honest gap in scope here, not a decision that audio alone is sufficient."
)

st.write("Next: **App Walkthrough** picks up from here, with the live interactive pages themselves.")
nav_button("Continue to the App Walkthrough →", "pages/3_App_Walkthrough.py", key="nav_results_to_walkthrough")
