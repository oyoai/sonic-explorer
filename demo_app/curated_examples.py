"""Fixed set of demo query points -- what gets curated here is WHICH moment
to query, never the results themselves. Every match and score the demo app
shows is computed live at runtime against the real FAISS indexes (see
resources.py/Demo.py) -- nothing here is a baked-in result, only a pointer
to a good starting question to ask the real system.

Usage in Demo.py: this is NOT a fixed radio of 5 choices -- the app's real
song pickers cover the whole library, one independent picker per facet
column, rather than restricting choice. CURATED_EXAMPLES now serves one
narrower role: DEFAULT_EXAMPLE seeds every column's starting song/moment
(before a viewer's own persisted choice, if any, takes over -- see
resources.py's persistent_song_and_moment), so the app doesn't open on an
arbitrary, possibly dull segment. The per-example `headline` field is kept
as real documentation of why each of these five was interesting (each was
originally found via a genuine cross-facet divergence on the SAME query --
see the selection process below) even though the current six-independent-
column design has no single shared query left to hang that comparison off
of in the UI itself; it's reference material for whoever's presenting, not
dead code the app executes.

Selection process (see scripts/select_demo_examples.py, run against
deploy_data -- the same small stratified subset this app and the main
streamlit_app both actually deploy against): for every segment in the
library, compare each stem/harmony facet's top-1 match against the
full-mix "sound" facet's top-1 match. A candidate only counts as a real
"divergence" if the diverging facet's own top-1 score clears that facet's
own empirical median (raw cosine magnitude alone isn't a reliable
confidence signal in this system -- see that script's own comment, and
GENRE_COHESION_RESULTS in streamlit_app/pages/2_Results.py for the actual
validated evidence that these facets outperform random). Candidates were
ranked by a composite of (confident divergences, capped at 2) + average
score, then the first candidate per distinct genre was kept, in ranked
order -- a deterministic tie-break for genre variety, not a hand-pick.

fma_track_id (not song.id) identifies the song -- song.id numbering differs
between data/ and deploy_data/ (see sonic_explorer/config.py's
album_art_path_for docstring), so fma_track_id is the only identifier
that's actually stable regardless of which DATA_DIR resolves in a given
environment. segment_index is exact and grid-aligned (WINDOW_SEC=5.0,
HOP_SEC=2.5), unlike matching on float start_sec.

featured_facets: ordered, first is the default shown on load. Only facets
confirmed embedded for this exact segment at selection time are listed --
an unlisted facet for a given example is an honest gap (that stem simply
wasn't computed for this segment), not an oversight.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CuratedExample:
    fma_track_id: int
    segment_index: int
    featured_facets: list[str]
    title: str  # for a fast human-readable label before the DB lookup even happens
    artist: str
    genre: str
    headline: str  # the one-sentence "why this example was picked" shown in the UI


CURATED_EXAMPLES: list[CuratedExample] = [
    CuratedExample(
        fma_track_id=131454,
        segment_index=9,
        featured_facets=["sound", "drums", "instrumental", "vocal", "bass"],
        title="B08 Mario",
        artist="Mom Jeans",
        genre="Pop",
        headline=(
            "The full mix, the isolated drums, AND the isolated backing track all independently "
            "converge on \"B07 Ya Dumb Bitch\" -- the same artist's other track -- purely from audio, "
            "with no title or artist field involved in the search. Switch to Vocal or Bass and the top "
            "match changes to a completely different song: agreement isn't automatic, so when three "
            "facets DO agree, it means something."
        ),
    ),
    CuratedExample(
        fma_track_id=41018,
        segment_index=10,
        featured_facets=["sound", "harmony", "vocal", "drums", "bass"],
        title="Au royaume de la money",
        artist="Unity Vibration",
        genre="International",
        headline=(
            "The full-mix \"Sound\" facet's top match is a Hip-Hop track -- but Harmony, Vocal, Drums, "
            "and Bass all instead point to \"Digital system\" by Garage firm, a fellow International "
            "track (and the match runs both ways: querying that song's own moments finds this one back). "
            "The whole-mix embedding misses a pairing that four independent, targeted facets agree on."
        ),
    ),
    CuratedExample(
        fma_track_id=122358,
        segment_index=7,
        featured_facets=["sound", "vocal", "bass", "instrumental", "harmony", "drums"],
        title="Night Without Sleep",
        artist="David Mumford",
        genre="Folk",
        headline=(
            "Sound and Vocal agree with each other (both find \"Elektra\"), so the voice is doing a lot "
            "of the work in the full-mix match here. Strip the vocals away and listen to just the Bass "
            "or Instrumental backing instead, and the closest match becomes a different song entirely -- "
            "still a confident match, just not the same one, because it's answering a different question."
        ),
    ),
    CuratedExample(
        fma_track_id=5268,
        segment_index=3,
        featured_facets=["sound", "vocal", "drums", "harmony", "bass"],
        title="3 Rocks Blessed",
        artist="Dälek",
        genre="Hip-Hop",
        headline=(
            "Sound, Vocal, and Drums each pick a DIFFERENT specific Hip-Hop track as the closest match -- "
            "all three stay within genre, but land on three distinct songs. The facets aren't just "
            "rediscovering genre and calling it a day; they're separating individual tracks within it."
        ),
    ),
    CuratedExample(
        fma_track_id=108961,
        segment_index=2,
        featured_facets=["sound", "drums", "bass", "instrumental", "harmony"],
        title="Mr. Person",
        artist="The Mystery Artist",
        genre="Rock",
        headline=(
            "The full mix finds another confident same-genre Rock match. Isolate just the drums, "
            "though, and the closest match jumps to a Pop track; isolate the bass and it jumps again, "
            "to an International track -- the starkest cross-genre split in this set, and a reminder "
            "that a drum pattern or basstone can genuinely resemble something outside its own genre "
            "even when the full mix doesn't. (This segment has no isolated-vocal match at all -- an "
            "honest gap, not a hidden result: that stem simply wasn't computed for this moment.)"
        ),
    ),
]

DEFAULT_EXAMPLE = CURATED_EXAMPLES[0]  # "B08 Mario" -- the clearest single story (3 facets agree on
# the same artist's companion track), so a first-time visitor's very first load already shows something
# convincing rather than an arbitrary song.
