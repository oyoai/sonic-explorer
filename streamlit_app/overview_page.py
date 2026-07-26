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
st.navigation()) and pages/1_Methodology.py (for the back-link) import from,
rather than inside Overview.py itself where importing it back would be
circular."""

import streamlit as st

from comparison_data import build_naive_vs_real_graphs
from components.plotting import concept_bubble_diagram, network_graph_figure
from resources import LOGO_PATH, get_repositories, nav_button, show_data_source_banner, show_logo
from sonic_explorer.analysis.network_graph import cross_genre_edge_fraction
from sonic_explorer.config import audio_path_for


def render_overview() -> None:
    st.set_page_config(page_title="Sonic Explorer", layout="wide")

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=420)
    else:
        st.title("Sonic Explorer")

    # Header wording is explicitly not locked yet -- "or similar" per the
    # current restructure plan, this is a close variant kept the same
    # digging-beneath-the-surface metaphor, made a bit more explanatory.
    st.header("Unearth your style")
    st.info(
        "**Placeholder.** The pitch line for this header hasn't been drafted yet -- left "
        "intentionally open rather than forced, until more of the page/project narrative "
        "exists to draft it against."
    )

    show_logo()
    show_data_source_banner()

    song_repo, embedding_repo, _ = get_repositories()
    all_songs = song_repo.list_songs()
    songs_by_id = {s.id: s for s in all_songs}

    naive_nodes, naive_edges, real_nodes, real_edges, vectors, genre_by_song = (
        build_naive_vs_real_graphs(song_repo, embedding_repo, len(all_songs)) if all_songs
        else (None, [], None, [], {}, {})
    )

    st.divider()

    # -----------------------------------------------------------------------
    # 1. Problem
    # -----------------------------------------------------------------------
    st.header("1. Problem")
    st.write(
        "**First-draft copy -- needs your real specifics, not left as-is.** Something like: "
        "every so often a song hits exactly right -- not just \"good,\" but *this specific "
        "thing* about how it sounds. The obvious next move is finding more like it. That's "
        "where it falls apart. Search engines want a genre or an artist name. Streaming "
        "recommendations lean on what other people who liked this also liked -- a popularity "
        "signal, not a sound one. Even asking an AI chatbot gets an answer built from what's "
        "written *about* music -- reviews, tags, genre history -- not from what's actually in "
        "the recording. Every option points back to the same shallow signals: genre, metadata, "
        "who-else-liked-this. None of them ever actually listened to the song that started it."
    )

    naive_neighbors_by_song: dict[int, list[int]] = {}
    for e in naive_edges:
        naive_neighbors_by_song.setdefault(e.song_id_a, []).append(e.song_id_b)
        naive_neighbors_by_song.setdefault(e.song_id_b, []).append(e.song_id_a)
    example_song = next((s for s in all_songs if naive_neighbors_by_song.get(s.id)), None)

    if example_song is not None:
        st.write("Here's what that actually looks like in practice:")
        example_cols = st.columns(2)
        with example_cols[0]:
            st.caption("**You loved this song:**")
            st.write(f"\"{example_song.title}\" — {example_song.artist} ({example_song.genre_top})")
            st.audio(str(audio_path_for(example_song)))
        with example_cols[1]:
            st.caption("**A typical existing system recommends:**")
            for nid in naive_neighbors_by_song[example_song.id]:
                n = songs_by_id[nid]
                st.write(f"\"{n.title}\" — {n.artist} ({n.genre_top})")

    st.divider()

    # -----------------------------------------------------------------------
    # 2. Existing solutions
    # -----------------------------------------------------------------------
    st.header("2. Existing solutions")
    st.write(
        "Before proposing anything: how do existing recommendation systems actually work, and "
        "is there really room for an audio-based approach to do better? Two dominant paradigms "
        "cover most of what's out there today:"
    )

    concept_cols = st.columns(2)
    with concept_cols[0]:
        st.caption("**Metadata-based matching**")
        st.plotly_chart(
            concept_bubble_diagram(
                "Songs similar<br>to this song...", ["Album", "Artist", "Tags", "Genre", "Year"]
            ),
            width="stretch", key="concept_metadata",
        )
    with concept_cols[1]:
        st.caption("**Collaborative filtering**")
        st.plotly_chart(
            concept_bubble_diagram(
                "People who liked<br>this also liked...",
                ["Other listeners", "Play history", "Ratings", "Purchases"],
            ),
            width="stretch", key="concept_collaborative",
        )

    st.write(
        "Our approach goes further — past similarity metrics entirely, into what the audio "
        "actually contains. Whether that's actually *better* is the open question the rest of "
        "this project works through, not something to assert here."
    )

    if all_songs and naive_nodes is not None and not naive_nodes.empty:
        st.write(
            "To be fair to those two paradigms: comparing against a genre-tag strawman would be "
            "an easy win. So the version tested here is the strongest non-audio baseline "
            "reasonably achievable from this library's real metadata -- genre tag, FMA's fuller "
            "genre-hierarchy overlap, shared album, and free-text tags, combined -- not genre "
            "alone. This is the same kind of graph as the two diagrams above, except every edge "
            "is now real, computed data, not an illustration:"
        )
        st.caption(
            "**Combined metadata baseline — genre + genre hierarchy + album + tags.** Catalog "
            "metadata only, nothing heard."
        )
        st.plotly_chart(
            network_graph_figure(naive_nodes, naive_edges), width="stretch", key="overview_naive_graph"
        )

        naive_cross_pct = cross_genre_edge_fraction(naive_edges, genre_by_song)
        real_cross_pct = cross_genre_edge_fraction(real_edges, genre_by_song)
        st.warning(
            f"**Read this carefully before concluding anything from the shape above:** its "
            f"clean, single-color islands are not evidence this approach \"worked\" -- its edges "
            f"are *defined* as \"shares a metadata signal,\" so a same-genre-looking graph is "
            f"guaranteed by construction, not earned. Only **{naive_cross_pct:.0%}** of its edges "
            f"cross a genre boundary (the album/tag signals occasionally do this). For context, "
            f"the real audio-based graph -- covered later, once the mechanism actually makes "
            f"sense -- crosses genre boundaries **{real_cross_pct:.0%}** of the time."
        )
    else:
        st.info("No songs available yet to build this comparison.")

    st.divider()

    # -----------------------------------------------------------------------
    # 3. Proposed solution
    # -----------------------------------------------------------------------
    st.header("3. Proposed solution")
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
        "separate destinations."
    )

    st.divider()

    st.write(
        "Next: **Approach** walks through how this actually works, step by step, before "
        "Methodology dives into the technical depth and evidence."
    )
    nav_button("See how it works →", "pages/0_Approach.py", key="nav_overview_to_approach")


OVERVIEW_PAGE = st.Page(render_overview, title="Overview", url_path="", default=True)
