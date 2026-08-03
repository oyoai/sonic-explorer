"""Local Similarity -- local, facet-level similarity: pick a song and a
moment, and see what each facet independently retrieves for it. See
Demo.py's module docstring for the app-wide navigation/architecture note.
(Filename kept as 1_Moment_Matcher.py -- st.navigation()'s list order in
Demo.py controls sidebar position and title, not this file's own name/
number; see CLAUDE.md's own note on this, and pages/2_Visual_Exploration.py's
docstring for the same point made there.)

Redesigned around independent facet views -- there is no longer one shared
query song feeding six facets' results. Each of the six tabs (Sound,
Harmony, Vocal, Drums, Bass, Instrumental) is a fully self-contained mini
matcher: its own song picker, its own moment picker, its own full-song
waveform with the selected moment highlighted, its own query audio player,
and then that facet's own single best match elsewhere in the library (green
match-score pill, the match song's waveform highlighted at the matched
moment, its own audio player). A viewer can point two different tabs at two
completely different songs if they want to -- the point is that each facet
independently determines similarity from its own representation, so nothing
here should imply they have to move in lockstep.

Query vs. match ownership -- a real reported ambiguity, now fixed
structurally, not just by proximity: the song/moment picker used to sit
full-width ABOVE the two-column split, which visually overflowed into the
match column's space and left it unclear which side the pills actually
controlled. Both columns now open with an identically-sized box
(_PICKER_BOX_HEIGHT tall) immediately under a one-line caption naming what
that column is: query_col's box is bordered and holds the real, interactive
song+moment picker ("Query -- pick a song and a moment"); match_col's box
is an unbordered, textless spacer of the SAME height ("Best match -- found
automatically" needed no border or explanatory caption of its own once the
one-line label above it already says there's nothing to pick here -- see
the old revision this replaced for why it briefly had both). Matching
height (not matching border style) is what keeps this from just being a
labeling fix -- it's what makes the title line, waveform, and audio player
below land at the identical vertical position in both columns, the same
alignment property an earlier version achieved by moving the picker out of
query_col entirely. This is the structural answer to "query moment is
manually selected; matched moment is retrieved, never manually selected" --
visible in the layout itself, not just stated in prose.

Waveform color: the query waveform is always QUERY_WAVEFORM_COLOR (a
neutral gray, plotting.py), regardless of facet -- one fixed "this is the
query" visual anchor across all six tabs. The match waveform uses that
facet's own color from FACET_WAVEFORM_COLORS, so switching tabs visibly
changes the match's color too, reinforcing that each tab is a genuinely
different similarity signal, not a relabeled copy of the same one.

Playback: both the query and match audio players play ONLY the selected
~5s segment and then stop -- no loop=True. Confirmed directly against
Streamlit's own frontend source (installed package, static/js/Audio.*.js):
with end_time set and loop left at its default of False, the player's own
timeupdate handler pauses exactly once when playback reaches end_time
(loop=True instead seeks back to start_time and replays) -- this is
already-built, native Streamlit behavior, not something added here; the
only change from an earlier version was deleting loop=True from both
st.audio() calls below.

Tabs, not side-by-side columns: six columns narrow enough to fit on screen
at once left each one too cramped for a song picker + pills + two waveforms
+ two audio players to read comfortably; a tab gets the full page width
instead, at the cost of only seeing one facet at a time. st.tabs() doesn't
lazy-load (per this project's own testing conventions -- every tab's body
still executes on every rerun, exactly like the six columns did), so this
is a pure display change, not a performance one.

"Structure" is deliberately not a seventh tab: sonic_explorer's facet
registry (facets/registry.py) never registers a StructureFacet against the
FAISS-backed retrieval path RetrievalService.query_by_segment() uses --
Structure exists only as a self-similarity matrix consumed by Song X-Ray
(EmbeddingRepository.get_structure_matrix/get_structure_timeline), a
genuinely different mechanism, not a segment-embedding facet with an index
to query. Adding a real "structure similarity" tab would need a
different comparison method (e.g. matrix-to-matrix distance) that doesn't
exist in this codebase yet -- an honest gap, not an oversight.

Playback (stems): no isolated stem audio (vocal/drums/bass/instrumental) is
ever persisted to disk anywhere in this project -- sonic_explorer/pipeline/
separation.py's Demucs separation happens in-memory, purely to feed the
stem facets' embeddings (notebooks/03_stem_separation_and_embed.ipynb), and
is never written back out as its own audio file. So for every facet here,
query and match playback both use the one audio file that actually exists
-- the original full mix -- while retrieval itself still runs on that
facet's own isolated-stem embedding. Each stem-facet tab says this
explicitly rather than silently playing something that looks like it might
be isolated audio but isn't.

Song/moment selection now survives more than a browser refresh -- see
resources.py's persistent_song_and_moment() and its module docstring for
the on-disk state file this reads/writes, needed so a full app/server
restart mid-presentation doesn't lose the prepared setup.

Nothing in streamlit_app/ or sonic_explorer/ was touched to build this.
Every match and score below is a real, live call into the same
RetrievalService the main app uses (see resources.py) -- curated_examples.py
only seeds a strong shared starting point before any tab's own persisted
choice takes over."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from curated_examples import DEFAULT_EXAMPLE
from plotting import (
    FACET_WAVEFORM_COLORS,
    QUERY_WAVEFORM_COLOR,
    inject_keyboard_shortcuts,
    inject_match_pill_style,
    match_pill_html,
    waveform_figure,
)
from resources import (
    audio_path_for,
    cached_full_waveform,
    facet_display_name,
    get_repositories,
    persistent_song_and_moment,
)

# Called here, not from Demo.py -- see that file's module docstring for the
# real, verified reason: anything Demo.py itself renders before pg.run()
# never reaches the DOM (st.navigation's page-switch machinery appears to
# discard it), so CSS injection has to happen from inside whichever page
# actually uses match_pill_html. Idempotent (session_state-guarded), so
# this is still cheap even called from just this one page.
inject_match_pill_style()
inject_keyboard_shortcuts()

# sound_tags deliberately excluded -- see curated_examples.py's module
# docstring on why it's kept out of this demo's facet list specifically.
FACETS = ["sound", "harmony", "vocal", "drums", "bass", "instrumental"]

# These four run on an isolated Demucs stem's embedding, but (per this
# file's own module docstring) there's no persisted isolated-stem audio
# file anywhere in the project to actually play -- Sound and Harmony run on
# the full mix natively, so there's no such gap to disclose for them.
STEM_FACETS = {"vocal", "drums", "bass", "instrumental"}

# Comfortably fits a song selectbox + moment pills wrapped across up to 3
# rows at half the page's width (query_col is narrower than the picker's
# old full-width home, so the same 10-11 pills every song has -- deploy_data's
# real min/max, checked directly -- wrap more, and wrap inconsistently
# between 2 and 3 rows depending on the exact song title/pill-label widths
# in play at render time). Sized with real headroom after a real browser
# render showed a 3-row wrap clipping against a tighter value, rather than
# tuned to the exact pixel for one specific song. Shared by match_col's
# placeholder box so both columns commit to the identical height regardless
# of which one's real content is taller.
_PICKER_BOX_HEIGHT = 240

st.title("Does this AI actually understand musical similarity?")

song_repo, embedding_repo, retrieval_service = get_repositories()

all_songs = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
if not all_songs:
    st.info("No songs in the library yet.")
    st.stop()

default_song = song_repo.get_song_by_fma_track_id(DEFAULT_EXAMPLE.fma_track_id) or all_songs[0]

st.divider()

facet_tabs = st.tabs([facet_display_name(f) for f in FACETS])
for facet, tab in zip(FACETS, facet_tabs, strict=False):
    with tab:
        query_col, match_col = st.columns(2)

        with query_col:
            st.caption("Query — pick a song and a moment")
            with st.container(height=_PICKER_BOX_HEIGHT, border=True):
                song, query_segment = persistent_song_and_moment(
                    facet, all_songs, song_repo, default_song, DEFAULT_EXAMPLE.segment_index,
                )

        with match_col:
            st.caption("Best match — found automatically")
            with st.container(height=_PICKER_BOX_HEIGHT, border=False):
                pass  # unbordered, empty spacer -- see module docstring: only here to keep this
                # column's height matched to query_col's real (bordered) picker box above, so the
                # title/waveform/player below still land at the same vertical position on both sides.

        with query_col:
            st.markdown(f"**{song.title}** — {song.artist} · {song.genre_top}")

            query_envelope = cached_full_waveform(song.id, str(audio_path_for(song)))
            st.plotly_chart(
                waveform_figure(
                    query_envelope, duration_sec=song.duration_sec,
                    highlight_range=(query_segment.start_sec, query_segment.end_sec), height=90,
                    color=QUERY_WAVEFORM_COLOR,
                ),
                width="stretch", key=f"query_wave_{facet}",
            )
            st.audio(
                str(audio_path_for(song)), start_time=query_segment.start_sec,
                end_time=query_segment.end_sec,
            )
            if facet in STEM_FACETS:
                st.caption(
                    "Playback uses the full mix -- isolated stem audio isn't persisted, only used to "
                    "compute the embedding.",
                )

        with match_col:
            if embedding_repo.status(query_segment.id, facet) != "done":
                st.caption("Not embedded for this moment yet.")
            else:
                matches = retrieval_service.query_by_segment(query_segment.id, facet_name=facet, k=1)
                if not matches:
                    st.caption("No match found elsewhere in the library yet.")
                else:
                    match = matches[0]
                    pct = max(0.0, match.score) * 100
                    st.markdown(
                        match_pill_html(pct)
                        + f" {match.song.title} — {match.song.artist} · {match.song.genre_top}",
                        unsafe_allow_html=True,
                    )

                    match_envelope = cached_full_waveform(match.song.id, str(audio_path_for(match.song)))
                    st.plotly_chart(
                        waveform_figure(
                            match_envelope, duration_sec=match.song.duration_sec,
                            highlight_range=(match.segment.start_sec, match.segment.end_sec), height=90,
                            color=FACET_WAVEFORM_COLORS[facet],
                        ),
                        width="stretch", key=f"match_wave_{facet}",
                    )
                    st.audio(
                        str(audio_path_for(match.song)), start_time=match.segment.start_sec,
                        end_time=match.segment.end_sec,
                    )
                    if facet in STEM_FACETS:
                        st.caption(
                            "Playback uses the full mix -- isolated stem audio isn't persisted, only "
                            "used to compute the embedding.",
                        )
