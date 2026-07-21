import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from sonic_explorer.config import audio_path_for
from sonic_explorer.facets.fingerprint import structure_fingerprint
from components.plotting import fingerprint_thumbnail
from resources import get_repositories, show_data_source_banner

# ---------------------------------------------------------------------------
# Curated evidence, embedded directly rather than loaded from data/artifacts/
# at runtime -- those files are gitignored (derived outputs, not source) and
# won't exist once this is deployed or once someone else clones the repo.
# This is hand-picked presentation content; it should stay stable regardless
# of whether the local evaluation/example-generation scripts get re-run
# later, not silently change under a live audience.
#
# Source of truth for how these were produced:
#   scripts/run_evaluation.py -> data/artifacts/genre_cohesion_results.json
#   (nearest-neighbor examples generated via a one-off script using the real
#   RetrievalService + the real ExplanationClient -- see conversation/commit
#   history, not reproduced as a committed script since it's a one-time
#   curation pass, not a reusable pipeline step)
# ---------------------------------------------------------------------------

GENRE_COHESION_RESULTS = {
    "k": 10,
    "sample_size": 500,
    "facets": [
        {"facet_name": "sound", "n_queries": 500, "observed_pct": 54.4, "random_baseline_pct": 11.9},
        {"facet_name": "harmony", "n_queries": 500, "observed_pct": 21.2, "random_baseline_pct": 11.7},
        {"facet_name": "vocal", "n_queries": 500, "observed_pct": 36.1, "random_baseline_pct": 11.7},
        {"facet_name": "drums", "n_queries": 500, "observed_pct": 36.6, "random_baseline_pct": 11.7},
        {"facet_name": "bass", "n_queries": 500, "observed_pct": 27.2, "random_baseline_pct": 11.7},
        {"facet_name": "instrumental", "n_queries": 500, "observed_pct": 40.6, "random_baseline_pct": 11.7},
    ],
}

NN_EXAMPLES = [
    {"facet": "sound", "query": {"title": "Terminally in Love With You", "artist": "Shy Kids", "genre": "Pop"},
     "match": {"title": "Ave", "artist": "PC-ONE", "genre": "Experimental"}, "score_pct": 90.1,
     "explanation": "Both tracks feature a sparse, intimate vocal delivery layered with subtle synth textures and minimal production that creates a delicate, understated atmosphere."},
    {"facet": "sound", "query": {"title": "Elektra (You Were Such Fun)", "artist": "Red Crickets", "genre": "Pop"},
     "match": {"title": "Mr. Person", "artist": "The Mystery Artist", "genre": "Rock"}, "score_pct": 92.5,
     "explanation": "Both tracks share a bright, clean production style with crisp vocals and punchy instrumentation that gives them an almost identical modern pop-rock sheen."},
    {"facet": "harmony", "query": {"title": "Ordinary Girl", "artist": "The Pink Tiles", "genre": "Rock"},
     "match": {"title": "Plasma", "artist": "Redmann", "genre": "Electronic"}, "score_pct": 98.7,
     "explanation": "Both moments use nearly identical chord progressions and tonal colors, creating the same harmonic foundation despite their different musical styles."},
    {"facet": "harmony", "query": {"title": "Mad Honey", "artist": "DubRaJah", "genre": "International"},
     "match": {"title": "This is based upon a true story", "artist": "Plusplus", "genre": "Instrumental"}, "score_pct": 97.1,
     "explanation": "Both tracks use nearly identical warm, minor-key chords that create a contemplative and slightly melancholic tonal atmosphere."},
    {"facet": "vocal", "query": {"title": "A Friendly Noose", "artist": "Big Blood", "genre": "Folk"},
     "match": {"title": "300 Days In July", "artist": "Pete Galub", "genre": "Pop"}, "score_pct": 96.8,
     "explanation": "Both singers use a similarly raw, intimate vocal tone that feels personal and unpolished, almost like you're hearing them speak-sing directly to you."},
    {"facet": "vocal", "query": {"title": "something brewing", "artist": "Coin Locker Kid", "genre": "Hip-Hop"},
     "match": {"title": "Unless", "artist": "Nisi Period", "genre": "Rock"}, "score_pct": 85.3,
     "explanation": "Both singers deliver their lines with a similar raspy, slightly strained vocal texture that creates an intense, raw emotional quality."},
    {"facet": "drums", "query": {"title": "Lovedropper", "artist": "Boy Friend", "genre": "Rock"},
     "match": {"title": "western chow yun-fat", "artist": "This One", "genre": "Hip-Hop"}, "score_pct": 91.2,
     "explanation": "Both tracks layer crisp, punchy kicks and snares with nearly identical timing and attack, creating that same sharp, defined percussion sound despite their different genres."},
    {"facet": "drums", "query": {"title": "It's Okay, Roseanne", "artist": "The Parish of Little Clifton", "genre": "Pop"},
     "match": {"title": "Sillable", "artist": "UncleBibby", "genre": "Electronic"}, "score_pct": 93.3,
     "explanation": "Both tracks use a crisp, rhythmic drum pattern with similar snappy snare hits and tight percussion timing that stands out clearly in the mix."},
    {"facet": "bass", "query": {"title": "The Drop (Gung Who Version)", "artist": "Tickle", "genre": "Hip-Hop"},
     "match": {"title": "The Beast Is A Computer In Luxemburg", "artist": "Fierbinteanu", "genre": "Pop"}, "score_pct": 97.9,
     "explanation": "Both clips feature a nearly identical low-end bassline with the same deep, resonant tone and rhythmic pattern driving underneath."},
    {"facet": "bass", "query": {"title": "Inspiration", "artist": "Abunai!", "genre": "Rock"},
     "match": {"title": "All I Am", "artist": "Pete Galub", "genre": "Pop"}, "score_pct": 95.0,
     "explanation": "Both songs use a deep, steady bass line with nearly identical rhythmic phrasing that anchors their respective grooves."},
    {"facet": "instrumental", "query": {"title": "Sam's Song", "artist": "NaDa BaBa", "genre": "Folk"},
     "match": {"title": "Spot Rockers", "artist": "Cassette Tape Bandits", "genre": "Hip-Hop"}, "score_pct": 81.7,
     "explanation": "Both tracks have a similar underlying groove and percussion pattern that drives the rhythm, even though their genres sound completely different on the surface."},
    {"facet": "instrumental", "query": {"title": "Squinting at the Sun (radio edit)", "artist": "Lee Rosevere", "genre": "Electronic"},
     "match": {"title": "Do Easy", "artist": "Tasseomancy", "genre": "Pop"}, "score_pct": 85.5,
     "explanation": "Both tracks have nearly identical stripped-down instrumental textures with spacious, airy synth pads underneath the vocal melody."},
]

FACET_ORDER = ["sound", "harmony", "vocal", "drums", "bass", "instrumental"]

DNA_EXAMPLES = {
    "slow": {"title": "three lullabies 1", "artist": "Eva-Maria Houben", "genre": "Experimental",
             "tempo_bpm": 40.4, "energy": 0.0054, "brightness": 945.1, "harmonic_complexity": 0.891, "rhythmic_density": 0.57},
    "fast": {"title": "A ninja among culturachippers", "artist": "Rolemusic", "genre": "Electronic",
             "tempo_bpm": 152.0, "energy": 0.5433, "brightness": 1804.2, "harmonic_complexity": 0.994, "rhythmic_density": 7.07},
}

STRUCTURE_EXAMPLE_TITLE = "Cipralex (c/ Pulso)"

st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7", layout="wide")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()

st.title("Sonic Explorer")
st.write("Explore your music library by how it actually sounds — not tags or genre labels.")
st.caption(
    "This page walks through the actual methodology, with real evidence at each step -- not just "
    "asserted -- before opening into the live interactive app at the end."
)

show_data_source_banner()

if all_songs:
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric("Songs in library", len(all_songs))
    stat_col2.metric("Embedded segments (sound facet)", embedding_repo.index_size("sound"))

st.divider()

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
st.header("1. Data")
st.write(
    "The library is a curated subset of the Free Music Archive (FMA) -- Creative Commons-licensed "
    "tracks spanning 8 genres. Every clip is a **30-second preview**, not a full track -- worth "
    "keeping in mind for the structure section below, since it genuinely limits what \"structure\" "
    "can mean here."
)

if all_songs:
    genre_counts: dict[str, int] = {}
    for s in all_songs:
        genre_counts[s.genre_top] = genre_counts.get(s.genre_top, 0) + 1
    genre_items = sorted(genre_counts.items(), key=lambda kv: -kv[1])

    fig = go.Figure(go.Bar(
        x=[g for g, _ in genre_items], y=[c for _, c in genre_items],
        text=[c for _, c in genre_items], textposition="auto",
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="songs")
    st.plotly_chart(fig, width="stretch", key="genre_breakdown_chart")

st.divider()

# ---------------------------------------------------------------------------
# 2. Facets
# ---------------------------------------------------------------------------
st.header("2. Facets")
st.write(
    "Similarity isn't one thing. Instead of a single blended score, the library is embedded along "
    "several independent facets -- each captures a genuinely different aspect of how a song sounds."
)
st.markdown("""
| Facet | What it captures | How it's computed |
|---|---|---|
| **Sound** | Overall timbre, instrumentation, production character | CLAP (pretrained audio-text embedding model) |
| **Harmony** | Key, chords, tonal color | Chroma features |
| **Vocal** | Isolated voice timbre and delivery | Demucs source separation + CLAP on the isolated stem |
| **Drums** | Isolated drum/percussion pattern and timbre | Demucs source separation + CLAP |
| **Bass** | Isolated bassline tone and pattern | Demucs source separation + CLAP |
| **Instrumental** | Backing instrumentation with vocals removed | Demucs source separation + CLAP |
| **Structure** | The song's verse/chorus-style shape | Self-similarity matrix -- *visualized, not embedded/searchable* |
""")
st.caption(
    "Structure is deliberately different from the other six: it's a song-level visualization "
    "(a self-similarity matrix and the fingerprints derived from it), not a per-segment vector in a "
    "FAISS index. That's why it's excluded from the retrieval/evaluation numbers further down -- "
    "genre-cohesion@k measures nearest-neighbor retrieval, which doesn't apply the same way to "
    "something that's visualized rather than searched."
)

st.divider()

# ---------------------------------------------------------------------------
# 3. Song DNA & Fingerprints
# ---------------------------------------------------------------------------
st.header("3. Song DNA & fingerprints")
st.write(
    "Each song also gets a handful of aggregate scalar stats (\"song DNA\") -- tempo, energy, "
    "brightness, harmonic complexity, rhythmic density -- and a small visual \"fingerprint\" per "
    "facet, both cheap byproducts of the same audio analysis."
)

st.subheader("Song DNA -- does it actually track fast/energetic vs. slow/calm?")
st.write("Picked the two songs at opposite ends of a combined tempo+energy+rhythmic-density ranking:")
dna_cols = st.columns(2)
for col, key, label in [(dna_cols[0], "slow", "Slowest / calmest"), (dna_cols[1], "fast", "Fastest / most energetic")]:
    d = DNA_EXAMPLES[key]
    with col:
        st.markdown(f"**{label}: \"{d['title']}\"** — {d['artist']} ({d['genre']})")
        st.markdown(f"""
- Tempo: **{d['tempo_bpm']:.1f} BPM**
- Energy: **{d['energy']:.4f}**
- Brightness: **{d['brightness']:.0f} Hz**
- Harmonic complexity: **{d['harmonic_complexity']:.3f}**
- Rhythmic density: **{d['rhythmic_density']:.2f}**
""")
st.caption(
    "All five axes point the same direction for both songs -- a clean, internally consistent "
    "illustration. Worth flagging honestly: tempo and energy are nearly uncorrelated across the "
    "whole library (r ≈ 0.05) -- the songs with the single fastest tempo values weren't "
    "high-energy at all, most likely librosa tempo-octave errors (locking onto a doubled/halved "
    "tempo), a known failure mode especially on complex-rhythm genres. That's why these two "
    "examples were picked by combining three axes rather than trusting tempo alone."
)

st.subheader("Structure fingerprint -- does it track something real?")
st.write(
    "Checked all 1394 songs with detected structural boundaries for any segment whose label "
    "repeats later in the timeline (genuine \"the verse comes back\" recurrence). **Zero** songs "
    "show this. Every clearly-structured song's segments are uniquely labeled -- a monotonic "
    "evolution across the clip, not a loop. That tracks: 30 seconds usually isn't long enough to "
    "play out a full section and return to an earlier one. So the honest framing is that the "
    "structure facet shows *how a song's texture evolves across its first 30 seconds*, not "
    "full-song verse/chorus form -- that would need complete tracks."
)

structure_song = next((s for s in all_songs if s.title.startswith(STRUCTURE_EXAMPLE_TITLE)), None)
if structure_song is not None:
    try:
        matrix = embedding_repo.get_structure_matrix(structure_song.id)
        timeline = embedding_repo.get_structure_timeline(structure_song.id)
    except FileNotFoundError:
        matrix, timeline = None, None

    if matrix is not None and timeline is not None:
        fp_col, info_col = st.columns([1, 2])
        with fp_col:
            st.plotly_chart(
                fingerprint_thumbnail(structure_fingerprint(matrix), "Structure"),
                width="stretch", key="walkthrough_structure_fp",
            )
        with info_col:
            st.markdown(f"**\"{structure_song.title}\"** — {structure_song.artist} ({structure_song.genre_top})")
            segs = ", ".join(
                f"{s:.1f}-{e:.1f}s" for s, e in zip(timeline.segment_starts, timeline.segment_ends)
            )
            st.caption(f"5 uniquely-labeled segments: {segs}")
            st.caption(f"structural_confidence = {timeline.structural_confidence:.3f} (among the highest in the library)")
            st.audio(str(audio_path_for(structure_song)))
        st.caption(
            "The novelty curve's two strongest peaks land right near the very start (~2s) and very "
            "end (~28s) of the clip -- plausible for a 30-second hip-hop preview (a beat/vocal "
            "entrance shortly after the intro, and a hard cutoff/fade at the preview's edge). "
            "**Honest limitation:** confirming this by ear needs a human listener -- no audio "
            "playback/listening capability here. What the underlying algorithm *has* been validated "
            "against: synthetic audio with known ground truth (a pure tone produces zero detected "
            "peaks; two audibly distinct halves produce exactly one peak, accurate to within 0.01s "
            "of the true midpoint). Use the player above for the last-mile check yourself."
        )

st.divider()

# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------
st.header("4. Retrieval")
st.write(
    "Every facet gets its own FAISS index of segment embeddings. Finding \"similar\" moments is "
    "cosine-similarity nearest-neighbor search within one facet's index -- picking a different "
    "facet changes what \"similar\" means, using the exact same mechanism. A few real matches, "
    "picked from the live index, with a plain-language explanation generated by the same LLM "
    "explanation layer used throughout the app (not written by hand):"
)

for facet in FACET_ORDER:
    examples = [e for e in NN_EXAMPLES if e["facet"] == facet]
    if not examples:
        continue
    st.subheader(facet.capitalize())
    for ex in examples:
        st.markdown(
            f"**{ex['score_pct']:.0f}% match** — "
            f"\"{ex['query']['title']}\" by {ex['query']['artist']} ({ex['query']['genre']}) "
            f"↔ \"{ex['match']['title']}\" by {ex['match']['artist']} ({ex['match']['genre']})"
        )
        st.caption(f"\U0001F4AC *{ex['explanation']}*")

st.divider()

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
st.header("5. Evaluation")
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
    "Instrumental/Drums/Vocal in the middle (36-41%), Bass and Harmony weakest but still clearly "
    "above chance (21-27%). The ablation-style finding: facets genuinely diverge from each other "
    "rather than one just riding Sound's coattails -- Harmony in particular captures something "
    "clearly different (and on this metric, weaker) than the full mix."
)
st.info(
    "These numbers were captured before a stem-facet reprocessing pass currently underway (fixing "
    "a data-quality issue where a handful of near-silent isolated stems were being indexed as if "
    "meaningful). Expect Vocal/Drums/Bass/Instrumental numbers to shift slightly once that "
    "finishes -- Sound and Harmony are unaffected.",
    icon="\U0001F6A7",
)

st.divider()

# ---------------------------------------------------------------------------
# 6. Explore the live app
# ---------------------------------------------------------------------------
st.header("6. Explore the live app")
st.write("Everything above is the methodology. Here's where it actually lives, interactively:")

cta_cols = st.columns(5)
with cta_cols[0]:
    st.page_link("pages/5_Explore.py", label="**Explore**", icon="\U0001F310")
    st.caption("Every song as a node in a network graph. Click one to open the player.")
with cta_cols[1]:
    st.page_link("pages/1_Taste_Map.py", label="**Taste Map**", icon="\U0001F5FA️")
    st.caption("A 2D map of the library clustered by sound. Click a point to hear it.")
with cta_cols[2]:
    st.page_link("pages/2_Song_XRay.py", label="**Song X-Ray**", icon="\U0001F50D")
    st.caption("A song's structural anatomy, fingerprints, and DNA in one place.")
with cta_cols[3]:
    st.page_link("pages/3_Moment_Matcher.py", label="**Moment Matcher**", icon="\U0001F3AF")
    st.caption("Pick a moment, get ranked matches on any facet, with explanations.")
with cta_cols[4]:
    st.page_link("pages/4_Ask_The_DJ.py", label="**Ask the DJ**", icon="\U0001F399️")
    st.caption("Describe what you want in plain language -- a conversational front-end over it all.")
