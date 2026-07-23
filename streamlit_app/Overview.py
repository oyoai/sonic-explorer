import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from components.animated_stats import animated_stat_row
from components.plotting import cluster_density_preview, genre_breakdown_bar
from resources import LOGO_PATH, get_repositories, show_data_source_banner, show_logo


@st.cache_data
def _overview_cluster_preview_df(_song_repo, _embedding_repo, cache_key):
    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    result = compute_taste_map(song_vectors)
    return pd.DataFrame([{"x": p.x, "y": p.y, "cluster": p.cluster} for p in result.points])


def render_overview() -> None:
    st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7", layout="wide")

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=420)
    else:
        st.title("Sonic Explorer")
    st.write(
        "Explore your music library by how it actually sounds — not tags or genre labels."
    )

    show_logo()
    show_data_source_banner()

    song_repo, embedding_repo, retrieval_service = get_repositories()
    all_songs = song_repo.list_songs()

    if all_songs:
        genres = sorted({s.genre_top for s in all_songs})
        genre_counts: dict[str, int] = {}
        for s in all_songs:
            genre_counts[s.genre_top] = genre_counts.get(s.genre_top, 0) + 1

        st.iframe(
            animated_stat_row([
                ("Songs in library", len(all_songs)),
                ("Genres", len(genres)),
                ("Embedded segments (sound facet)", embedding_repo.index_size("sound")),
            ]),
            height=100,
        )

        viz_cols = st.columns([3, 2])
        with viz_cols[0]:
            st.caption("Genre composition")
            st.plotly_chart(genre_breakdown_bar(genre_counts), width="stretch", key="overview_genre_bar")
        with viz_cols[1]:
            st.caption("Sonic clusters, discovered from audio alone — full map in **Explore**")
            preview_df = _overview_cluster_preview_df(song_repo, embedding_repo, embedding_repo.index_size("sound"))
            if not preview_df.empty:
                st.plotly_chart(
                    cluster_density_preview(preview_df), width="stretch", key="overview_cluster_preview"
                )

    st.divider()

    # -----------------------------------------------------------------------
    # 1. Project introduction
    # -----------------------------------------------------------------------
    st.header("1. What this is")
    st.write(
        "Most music discovery tools lean on metadata: genre tags, artist graphs, playlist "
        "co-occurrence. That's useful, but it means two songs get called \"similar\" because of "
        "who made them or how a label described them — not because of what they actually sound "
        "like. It also means two songs that sound remarkably alike but sit in different genre "
        "buckets never get connected."
    )
    st.write(
        "Sonic Explorer starts from the opposite direction: analyze the audio directly. Every "
        "song is broken into several independent **facets** — overall sound/timbre, harmony, "
        "isolated vocals, drums, bass, backing instrumentation, and structural shape — using "
        "pretrained audio embedding models and signal-processing techniques, with genre labels "
        "never entering the similarity computation itself. Genre is kept around only afterward, "
        "as an evaluation yardstick: do a facet's nearest neighbors share a genre more often "
        "than chance would predict? That's a check on whether the audio-based approach is "
        "finding real signal — not the mechanism generating the matches."
    )
    st.write(
        "From those facets, the app builds several ways to explore a library: per-song "
        "\"DNA\" and visual fingerprints, a 2D map of the whole collection, moment-to-moment "
        "matching on any facet, a conversational front-end over all of it, and free-form "
        "exploration. **Explore is the hub** for all of this -- Song X-Ray, Moment Matcher, and "
        "Ask the DJ are reached by interacting with it (selecting a song, then a moment), not "
        "separate destinations. The rest of this page sets up the problem before the next pages "
        "walk through how it was actually built and how well it works."
    )

    st.subheader("1.1 The naive approach — and why it falls short")
    st.info(
        "**Placeholder.** This subsection should show a concrete naive baseline (e.g. "
        "genre-tag-based \"similarity\" — group songs by their label and call that similar) "
        "running against a couple of real examples, to make the motivation tangible before the "
        "facet-based approach is introduced. Not written yet — needs a real worked example "
        "rather than an asserted claim, so it's left as a stub instead of guessed at.",
        icon="\U0001F6A7",
    )

    st.subheader("1.2 Related work")
    st.caption(
        "This project's approach was developed independently, before any literature search "
        "happened. A subsequent search found it aligns with several current research "
        "directions -- convergent, not derivative."
    )
    st.markdown(
        "- **Tovstogan, Serra & Bogdanov (2022), \"Visualization of deep audio embeddings for "
        "music exploration and rediscovery\"** (SMC 2022) -- the closest academic precedent to "
        "this project's Explore/Taste Map: a web interface visualizing personal music "
        "collections via audio embeddings and 2D projections. This project differs by adding "
        "moment-level (not just song-level) matching, several independently-computed facets "
        "with LLM explanations per match, and a conversational agent layer.\n"
        "- **Vohra & Akama (2026), \"Interpretable and Perceptually-Aligned Music Similarity with "
        "Pretrained Embeddings\"** -- directly parallels this project's stem-separated facets "
        "and calibration-to-blend-weights plan: source separation (their work; Demucs here too) "
        "plus linear optimization against human ABX preference judgments, yielding interpretable, "
        "instrument-wise contributions to perceived similarity.\n"
        "- **VidTune (CHI 2026)** -- uses CLAP + t-SNE for a \"Music Map,\" and explicitly frames "
        "its layout as an approximate similarity space rather than individually interpretable "
        "axes -- directly relevant precedent for this project's own PCA/ICA axis-interpretability "
        "findings (Methodology §4b), whatever they turn out to show for a given projection."
    )
    st.warning(
        "**Verification status:** these three citations were checked (title, authors, venue, "
        "and that the described finding is actually what the paper says) via web search against "
        "the papers' own listings and, for Vohra & Akama, the arXiv abstract directly -- not a "
        "full read of each paper. A fourth candidate citation (a commercial tool, initially "
        "described as clustering catalogs by acoustic similarity via CLAP + UMAP/t-SNE) was "
        "checked the same way and **dropped**: the company is real, but its own site makes no "
        "mention of any such product, and the technical description couldn't be confirmed "
        "against an actual source. Treat this section as checked-but-not-final until read in full.",
        icon="\U0001F6A7",
    )

    st.divider()

    st.write(
        "Next: **Methodology** walks through how the library was actually analyzed and "
        "improved, with real evidence at each step."
    )
    st.page_link("pages/0_Methodology.py", label="**Continue to Methodology →**", icon="\U0001F52C")


# ---------------------------------------------------------------------------
# Navigation: Song X-Ray, Moment Matcher, and Ask the DJ are drill-down /
# companion states reached FROM Explore (select a song -> Song X-Ray; select
# a moment within it -> Moment Matcher; a persistent companion link -> Ask
# the DJ) -- not independent top-level destinations. visibility="hidden"
# keeps them fully reachable via st.switch_page/st.page_link (which Explore
# and App Walkthrough both use) while excluding them from the sidebar, so
# nobody lands on a page with no context about which song it's showing.
# ---------------------------------------------------------------------------
pg = st.navigation([
    st.Page(render_overview, title="Overview", url_path="", default=True),
    st.Page("pages/0_Methodology.py", title="Methodology"),
    st.Page("pages/1_Results.py", title="Results"),
    st.Page("pages/2_App_Walkthrough.py", title="App Walkthrough"),
    st.Page("pages/3_Song_XRay.py", title="Song X-Ray", visibility="hidden"),
    st.Page("pages/4_Moment_Matcher.py", title="Moment Matcher", visibility="hidden"),
    st.Page("pages/5_Ask_The_DJ.py", title="Ask the DJ", visibility="hidden"),
    st.Page("pages/6_Explore.py", title="Explore"),
])
pg.run()
