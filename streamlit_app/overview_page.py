"""The Overview/landing page's actual content, plus the StreamlitPage object
that registers it in Overview.py's st.navigation() call.

Why this lives in its own module rather than directly in Overview.py: a page
registered via a callable (st.Page(render_overview, ...)) gets script_path=""
in Streamlit's internal page registry -- Streamlit only derives a real
script_path for file-based pages (see commands/navigation.py's
`script_path = str(page._page) if isinstance(page._page, Path) else ""`).
That means a string-based st.page_link("Overview.py", ...) can never
successfully match against this page (its registered script_path is never
"Overview.py", it's always empty) and always raises
StreamlitPageNotFoundError in a real running app -- confirmed by reading
Streamlit's own source, not something AppTest's page-link handling catches,
since AppTest resolves pages through a different (less strict) fallback path
than the real server.

The fix: st.page_link() also accepts an actual StreamlitPage object, which
resolves via url_path's hash instead of string path-matching, sidestepping
the empty-script_path issue entirely. That requires the exact same
StreamlitPage instance (or at least one with a matching url_path) to be
importable from any page that wants to link back to Overview -- hence
OVERVIEW_PAGE living here, in a leaf module both Overview.py (for
st.navigation()) and pages/0_Methodology.py (for the back-link) import from,
rather than inside Overview.py itself where importing it back would be
circular."""

import json

import pandas as pd
import streamlit as st

from components.plotting import library_waffle_grid, network_graph_figure
from resources import LOGO_PATH, get_repositories, show_data_source_banner, show_logo
from sonic_explorer.analysis.network_graph import SongMetadata, build_metadata_similarity_graph, build_similarity_graph
from sonic_explorer.analysis.taste_map import mean_pool_song_vectors


def _song_metadata(song) -> SongMetadata:
    """genres_all/track_tags are JSON-encoded lists (see
    scripts/enrich_fma_metadata.py); None on any song enrichment hasn't
    touched yet, treated the same as "recovered but empty" -- both mean
    "no signal here," not an error."""
    genres_all = frozenset(json.loads(song.genres_all)) if song.genres_all else frozenset()
    tags = frozenset(json.loads(song.track_tags)) if song.track_tags else frozenset()
    return SongMetadata(genre_top=song.genre_top, genres_all=genres_all, album_id=song.album_id, tags=tags)


@st.cache_data
def _build_naive_vs_real_graphs(_song_repo, _embedding_repo, cache_key: int):
    """Same songs, two similarity rules: a combined non-audio metadata score
    (naive) vs. audio-embedding cosine similarity (this project) -- built
    from the same vector set so the comparison isolates the rule, not which
    songs are shown. The naive side combines genre, genre hierarchy, album,
    and free-text tags (see build_metadata_similarity_graph) rather than
    genre alone, so the audio graph is compared against the strongest
    reasonable non-audio baseline, not a strawman. cache_key (song count)
    invalidates the cache when the library changes, since
    _song_repo/_embedding_repo are excluded from Streamlit's cache hashing by
    the leading underscore, same convention as Explore's build_explore_graph."""
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)

    real_result = build_similarity_graph(vectors)
    naive_result = build_metadata_similarity_graph({sid: _song_metadata(songs_by_id[sid]) for sid in vectors})

    def _nodes_df(result):
        return pd.DataFrame([
            {
                "song_id": n.song_id, "x": n.x, "y": n.y, "cluster": n.cluster,
                "title": songs_by_id[n.song_id].title, "artist": songs_by_id[n.song_id].artist,
                "genre": songs_by_id[n.song_id].genre_top,
            }
            for n in result.nodes
        ])

    return _nodes_df(naive_result), naive_result.edges, _nodes_df(real_result), real_result.edges


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

    song_repo, embedding_repo, _ = get_repositories()
    all_songs = song_repo.list_songs()

    if all_songs:
        genre_counts: dict[str, int] = {}
        for s in all_songs:
            genre_counts[s.genre_top] = genre_counts.get(s.genre_top, 0) + 1

        st.caption(f"{len(all_songs)} songs across {len(genre_counts)} genres — one square per song")
        songs_df = pd.DataFrame([{"title": s.title, "genre": s.genre_top} for s in all_songs])
        st.plotly_chart(
            library_waffle_grid(songs_df, genre_counts), width="stretch", key="overview_waffle_grid"
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
    st.write(
        "A fair comparison needs the strongest non-audio baseline reasonably available, not a "
        "strawman — so the naive graph below combines every relevant catalog signal this "
        "library has: genre tag, FMA's fuller genre-hierarchy overlap, shared album, and "
        "free-text tags. No audio is analyzed anywhere in it. Below are the same songs, laid "
        "out two ways: **left**, an edge is drawn from that combined metadata score; **right**, "
        "an edge is drawn if two songs' audio embeddings are actually close. Same songs, same "
        "number of edges per song — the only thing that changes between the two graphs is what "
        "counts as \"similar.\""
    )

    if all_songs:
        naive_nodes, naive_edges, real_nodes, real_edges = _build_naive_vs_real_graphs(
            song_repo, embedding_repo, len(all_songs)
        )
        if not real_nodes.empty:
            col_naive, col_real = st.columns(2)
            with col_naive:
                st.caption(
                    "**Naive — genre + genre hierarchy + album + tags.** Catalog metadata only, "
                    "nothing heard."
                )
                st.plotly_chart(
                    network_graph_figure(naive_nodes, naive_edges), width="stretch", key="overview_naive_graph"
                )
            with col_real:
                st.caption("**This project — audio embeddings.** Edges come from what the audio sounds like.")
                st.plotly_chart(
                    network_graph_figure(real_nodes, real_edges), width="stretch", key="overview_real_graph"
                )
            st.caption(
                "The naive graph can and does cross genres occasionally — a shared album or tag is "
                "real signal, not a coincidence — but it's still blind to anything only audible in "
                "the audio itself. The audio graph's cross-genre edges are the connections no amount "
                "of catalog metadata could have found."
            )
        else:
            st.info("No embedded songs available yet to build this comparison.", icon="\U0001F6A7")
    else:
        st.info("No songs available yet to build this comparison.", icon="\U0001F6A7")

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


OVERVIEW_PAGE = st.Page(render_overview, title="Overview", url_path="", default=True)
