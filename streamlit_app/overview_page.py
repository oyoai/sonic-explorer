"""The Overview/landing page's actual content, plus the StreamlitPage object
that registers it in Overview.py's st.navigation() call.

Content follows a real content spec (Problem / Existing solutions / Research
question) authored directly against this project's real data -- notably the
two real metadata-vs-real similarity pairs in "Existing solutions," which
used to live on Approach's step 1 and moved here since they're the concrete
evidence for exactly the point this section makes (shared metadata doesn't
guarantee shared sound). Approach now starts from segmentation instead.

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

from comparison_data import get_demo_pairs
from components.plotting import concept_bubble_diagram, waveform_figure
from resources import LOGO_PATH, get_repositories, nav_button, show_data_source_banner, show_logo
from sonic_explorer.analysis.waveform_preview import waveform_envelope
from sonic_explorer.config import audio_path_for


def _big_quote(text: str) -> None:
    """A short, large, visually-set-apart line -- used for the two moments
    on this page meant to land as a single idea, not a paragraph to read."""
    st.markdown(f"## _{text}_")


def render_overview() -> None:
    st.set_page_config(page_title="Sonic Explorer", layout="wide")

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=420)
    else:
        st.title("Sonic Explorer")

    show_logo()
    show_data_source_banner()

    song_repo, embedding_repo, _ = get_repositories()
    all_songs = song_repo.list_songs()
    songs_by_id = {s.id: s for s in all_songs}

    metadata_pair, real_pair = (
        get_demo_pairs(song_repo, embedding_repo, len(all_songs)) if all_songs else (None, None)
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Problem
    # -----------------------------------------------------------------------
    st.header("Problem")
    st.subheader("The motivation behind this project came from a recurring experience:")
    _big_quote('"I love this song, find me more like it."')
    st.write(
        "But the recommendations rarely captured the qualities that made the original song "
        "appealing to me."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Existing solutions
    # -----------------------------------------------------------------------
    st.header("Existing solutions")
    st.subheader("How do current systems identify related songs?")
    st.write(
        "Existing music discovery systems generally rely on two sources of information to "
        "estimate similarity:"
    )

    concept_cols = st.columns(2)
    with concept_cols[0]:
        st.plotly_chart(
            concept_bubble_diagram(
                "Songs similar<br>to this song...", ["Album", "Artist", "Tags", "Genre", "Year"]
            ),
            width="stretch", key="concept_metadata",
        )
        st.caption(
            "**Metadata-based approaches:** similarity determined using explicit song "
            "attributes: genre, artist, tags, release year/era, album information."
        )
    with concept_cols[1]:
        st.plotly_chart(
            concept_bubble_diagram(
                "People who liked<br>this also liked...",
                ["Other listeners", "Play history", "Ratings", "Purchases"],
            ),
            width="stretch", key="concept_collaborative",
        )
        st.caption(
            "**Collaborative filtering:** similarity inferred from listener behavior: "
            "listening history, ratings/likes, playlists, similar users' preferences."
        )
        st.info(
            "**Honest gap:** this library has no user-level listen/favorite/interaction data at "
            "all -- collaborative filtering is described here for a complete picture of the "
            "landscape, not something this project can build or compare against directly."
        )

    st.write("But shared metadata doesn't guarantee shared sound.")

    if metadata_pair is not None and real_pair is not None:
        st.caption(
            "Below are two real pairs from this library: two songs tagged with the same genre "
            "that sound nothing alike, and two songs from different genres that sound "
            "remarkably similar."
        )

        a1, b1 = songs_by_id[metadata_pair.song_id_a], songs_by_id[metadata_pair.song_id_b]
        st.markdown(
            f"**Same genre tag ({a1.genre_top}), sound different** -- real audio similarity "
            f"**{metadata_pair.audio_similarity:.2f}**"
        )
        row1 = st.columns(2)
        with row1[0]:
            st.plotly_chart(
                waveform_figure(waveform_envelope(audio_path_for(a1)), title=a1.title, color="rgb(239,85,59)"),
                width="stretch", key="overview_pair_metadata_a",
            )
            st.audio(str(audio_path_for(a1)))
        with row1[1]:
            st.plotly_chart(
                waveform_figure(waveform_envelope(audio_path_for(b1)), title=b1.title, color="rgb(239,85,59)"),
                width="stretch", key="overview_pair_metadata_b",
            )
            st.audio(str(audio_path_for(b1)))

        a2, b2 = songs_by_id[real_pair.song_id_a], songs_by_id[real_pair.song_id_b]
        st.markdown(
            f"**Different genres ({a2.genre_top} / {b2.genre_top}), sound similar** -- real "
            f"audio similarity **{real_pair.audio_similarity:.2f}**"
        )
        row2 = st.columns(2)
        with row2[0]:
            st.plotly_chart(
                waveform_figure(waveform_envelope(audio_path_for(a2)), title=a2.title, color="rgb(99,110,250)"),
                width="stretch", key="overview_pair_real_a",
            )
            st.audio(str(audio_path_for(a2)))
        with row2[1]:
            st.plotly_chart(
                waveform_figure(waveform_envelope(audio_path_for(b2)), title=b2.title, color="rgb(99,110,250)"),
                width="stretch", key="overview_pair_real_b",
            )
            st.audio(str(audio_path_for(b2)))
    else:
        st.info("Not enough data yet to show these two real pairs.")

    st.divider()

    # -----------------------------------------------------------------------
    # Research question
    # -----------------------------------------------------------------------
    st.header("Research question")
    _big_quote("Can we find songs that are actually similar to one another by sound?")

    st.divider()

    nav_button("See how it works →", "pages/0_Approach.py", key="nav_overview_to_approach")


OVERVIEW_PAGE = st.Page(render_overview, title="Overview", url_path="", default=True)
