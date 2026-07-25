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

from comparison_data import build_naive_vs_real_graphs, get_demo_pairs
from components.plotting import network_graph_figure
from resources import LOGO_PATH, get_repositories, show_data_source_banner, show_logo
from sonic_explorer.analysis.network_graph import cross_genre_edge_fraction
from sonic_explorer.config import audio_path_for


def render_overview() -> None:
    st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7", layout="wide")

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=420)
    else:
        st.title("Sonic Explorer")

    # Header wording is explicitly not locked yet (restructure plan lists
    # "Unearth" / "Unearth Music" / "Unearth the Sound" as options) -- using
    # the plan's own first-listed form as a placeholder, trivially swappable
    # once the final wording is picked.
    st.header("Unearth")
    st.info(
        "**Placeholder.** The pitch line for this header hasn't been drafted yet -- left "
        "intentionally open rather than forced, per the current restructure plan, until more of "
        "the page/project narrative exists to draft it against.",
        icon="\U0001F6A7",
    )

    show_logo()
    show_data_source_banner()

    song_repo, embedding_repo, _ = get_repositories()
    all_songs = song_repo.list_songs()
    songs_by_id = {s.id: s for s in all_songs}

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

    st.divider()

    # -----------------------------------------------------------------------
    # 2. Existing solutions
    # -----------------------------------------------------------------------
    st.header("2. Existing solutions")
    st.write(
        "To be fair to those tools: comparing against a genre-tag strawman would be an easy "
        "win. So the \"naive\" side of this comparison is the strongest non-audio baseline "
        "reasonably achievable from this library's real metadata -- genre tag, FMA's fuller "
        "genre-hierarchy overlap, shared album, and free-text tags, combined -- not genre alone. "
        "Popularity signals (listens/favorites) were considered and left out: on this "
        "genre-balanced library they're too sparse and low-signal to add anything real, and "
        "padding the baseline with a noisy signal wouldn't make this a fairer comparison, just a "
        "murkier one."
    )
    st.write(
        "Below are the same songs, laid out two ways: **left**, an edge is drawn from that "
        "combined metadata score; **right**, an edge is drawn if two songs' audio embeddings "
        "are actually close. Same songs, same number of edges per song — the only thing that "
        "changes between the two graphs is what counts as \"similar.\""
    )

    if all_songs:
        naive_nodes, naive_edges, real_nodes, real_edges, vectors, genre_by_song = build_naive_vs_real_graphs(
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

            naive_cross_pct = cross_genre_edge_fraction(naive_edges, genre_by_song)
            real_cross_pct = cross_genre_edge_fraction(real_edges, genre_by_song)
            st.warning(
                f"**Read the shapes above carefully:** the naive graph's clean, single-color "
                f"islands are not evidence it \"worked\" — its edges are *defined* as \"shares a "
                f"metadata signal,\" so a same-genre-looking graph is guaranteed by construction, "
                f"not earned. Only **{naive_cross_pct:.0%}** of its edges cross a genre boundary "
                f"(the album/tag signals occasionally do this), versus **{real_cross_pct:.0%}** of "
                f"the audio graph's edges — and every one of those audio cross-genre edges is a "
                f"connection no amount of catalog metadata could have found. The real graph's "
                f"color-bleed is the finding here, not noise.",
                icon="⚠️",
            )

            st.markdown("**Hear it, don't just read it:**")
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
                    st.info("Not enough naive-graph edges yet to pick a demo pair.", icon="\U0001F6A7")
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
                    st.info("Not enough cross-genre audio edges yet to pick a demo pair.", icon="\U0001F6A7")
        else:
            st.info("No embedded songs available yet to build this comparison.", icon="\U0001F6A7")
    else:
        st.info("No songs available yet to build this comparison.", icon="\U0001F6A7")

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
    st.page_link("pages/0_Approach.py", label="**See how it works →**", icon="\U0001F9E9")


OVERVIEW_PAGE = st.Page(render_overview, title="Overview", url_path="", default=True)
