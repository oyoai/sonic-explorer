import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS
from sonic_explorer.analysis.taste_map import compute_taste_map, correlate_axes_with_features, mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from sonic_explorer.evaluation.retrieval_diagnostics import top1_score_distribution
from sonic_explorer.facets.fingerprint import composite_fingerprint, structure_fingerprint
from components.plotting import composite_fingerprint_thumbnail, fingerprint_thumbnail, song_dna_radar_overlay
from resources import build_dna_normalizer, build_normalized_dna_by_song, get_repositories, show_data_source_banner

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

DNA_EXAMPLE_TITLES = {
    "slow": "three lullabies 1",
    "fast": "A ninja among culturachippers",
}

# A handful of songs spanning the structural-confidence range, so "scroll
# through a few examples" has real variety to scroll through rather than one
# fixed case -- picked from earlier candidate-gathering, not exhaustive.
FINGERPRINT_EXAMPLE_TITLES = [
    "Cipralex (c/ Pulso)",
    "three lullabies 1",
    "A ninja among culturachippers",
    "Kodak Ghosts",
    "OST 05 Go Go Go",
]

st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F3A7", layout="wide")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()
songs_by_title = {s.title: s for s in all_songs}


def _find_song(title: str):
    """Exact match first (curated titles were pulled verbatim from the DB);
    startswith fallback covers any that got truncated/edited in curation."""
    if title in songs_by_title:
        return songs_by_title[title]
    for s in all_songs:
        if s.title.startswith(title[:20]):
            return s
    return None


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
    st.subheader("Genre")
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
    st.caption(
        "8 genres, not evenly represented -- International and Electronic lead, several genres sit "
        "well under half that count. Worth keeping in mind when reading genre-cohesion numbers "
        "later: a facet has an easier time on an over-represented genre."
    )

    st.subheader("Track duration")
    durations = [s.duration_sec for s in all_songs if s.duration_sec is not None]
    dur_fig = go.Figure(go.Histogram(x=durations, nbinsx=20))
    dur_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="duration (s)", yaxis_title="songs")
    st.plotly_chart(dur_fig, width="stretch", key="duration_histogram")
    n_exactly_30 = sum(1 for d in durations if abs(d - 30.0) < 0.05)
    st.caption(
        f"{n_exactly_30}/{len(durations)} songs ({n_exactly_30 / len(durations):.0%}) are within "
        "0.05s of exactly 30.0 seconds -- not an approximation, every clip in this library really is "
        "a uniform 30-second preview. This is the empirical basis for the structure-facet limitation "
        "discussed in §3."
    )

    st.subheader("Artists")
    artist_counts: dict[str, int] = {}
    for s in all_songs:
        artist_counts[s.artist] = artist_counts.get(s.artist, 0) + 1
    n_unique_artists = len(artist_counts)
    top_artists = sorted(artist_counts.items(), key=lambda kv: -kv[1])[:10]
    artist_col1, artist_col2 = st.columns([1, 2])
    with artist_col1:
        st.metric("Unique artists", n_unique_artists)
        st.metric("Songs per artist (median)", f"{np.median(list(artist_counts.values())):.0f}")
    with artist_col2:
        artist_fig = go.Figure(go.Bar(
            x=[c for _, c in top_artists], y=[a for a, _ in top_artists], orientation="h",
            text=[c for _, c in top_artists], textposition="auto",
        ))
        artist_fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="songs",
            yaxis=dict(autorange="reversed"), title="Top 10 artists by track count",
        )
        st.plotly_chart(artist_fig, width="stretch", key="artist_breakdown_chart")
    st.caption(
        f"{n_unique_artists} unique artists across {len(all_songs)} songs -- most contribute a "
        "handful of tracks each, not a library dominated by a small number of prolific artists. "
        "Worth checking for retrieval: a facet that just learns to recognize a specific artist's "
        "signature production style would inflate genre-cohesion if that artist is genre-concentrated, "
        "without actually learning anything general about the genre."
    )

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

st.subheader("3a. Song DNA -- does it actually track fast/energetic vs. slow/calm?")
st.write("Picked the two songs at opposite ends of a combined tempo+energy+rhythmic-density ranking:")

slow_song = _find_song(DNA_EXAMPLE_TITLES["slow"])
fast_song = _find_song(DNA_EXAMPLE_TITLES["fast"])

if slow_song is not None and fast_song is not None:
    dna_cols = st.columns(2)
    with dna_cols[0]:
        st.markdown(f"**Slowest / calmest: \"{slow_song.title}\"** — {slow_song.artist} ({slow_song.genre_top})")
        st.audio(str(audio_path_for(slow_song)))
        st.markdown(f"""
- Tempo: **{slow_song.tempo_bpm:.1f} BPM**
- Energy: **{slow_song.energy:.4f}**
- Brightness: **{slow_song.brightness:.0f} Hz**
- Harmonic complexity: **{slow_song.harmonic_complexity:.3f}**
- Rhythmic density: **{slow_song.rhythmic_density:.2f}**
""")
    with dna_cols[1]:
        st.markdown(f"**Fastest / most energetic: \"{fast_song.title}\"** — {fast_song.artist} ({fast_song.genre_top})")
        st.audio(str(audio_path_for(fast_song)))
        st.markdown(f"""
- Tempo: **{fast_song.tempo_bpm:.1f} BPM**
- Energy: **{fast_song.energy:.4f}**
- Brightness: **{fast_song.brightness:.0f} Hz**
- Harmonic complexity: **{fast_song.harmonic_complexity:.3f}**
- Rhythmic density: **{fast_song.rhythmic_density:.2f}**
""")

    dna_normalizer = build_dna_normalizer(song_repo, len(all_songs))
    normalized_dna_by_song = build_normalized_dna_by_song(song_repo, dna_normalizer, len(all_songs))
    if slow_song.id in normalized_dna_by_song and fast_song.id in normalized_dna_by_song:
        axis_labels = [AXIS_LABELS[a] for a in AXES]
        slow_norm = normalized_dna_by_song[slow_song.id]
        fast_norm = normalized_dna_by_song[fast_song.id]
        st.plotly_chart(
            song_dna_radar_overlay(
                axis_labels,
                [slow_norm[a] for a in AXES], slow_song.title,
                [fast_norm[a] for a in AXES], fast_song.title,
            ),
            width="stretch", key="walkthrough_dna_radar",
        )
        st.caption(
            "Same normalizer the live app uses -- each axis scaled to where this song sits within "
            "the *library's* actual range, not an absolute scale. The two shapes barely overlap, "
            "which is exactly what a clean fast/slow contrast should look like."
        )
else:
    st.warning("DNA example songs not found in the current library.")

st.caption(
    "All five axes point the same direction for both songs -- a clean, internally consistent "
    "illustration. Worth flagging honestly: tempo and energy are nearly uncorrelated across the "
    "whole library (r ≈ 0.05) -- the songs with the single fastest tempo values weren't "
    "high-energy at all, most likely librosa tempo-octave detection errors (locking onto a doubled/"
    "halved tempo), a known failure mode especially on complex-rhythm genres. That's why these two "
    "examples were picked by combining three axes rather than trusting tempo alone."
)

st.markdown("**Full-library distributions** -- where do these two examples sit against everyone else?")
dna_songs_with_values = [s for s in all_songs if all(getattr(s, axis) is not None for axis in AXES)]
if dna_songs_with_values:
    dna_hist_cols = st.columns(3)
    for i, axis in enumerate(AXES):
        values = [getattr(s, axis) for s in dna_songs_with_values]
        hist_fig = go.Figure(go.Histogram(x=values, nbinsx=30))
        marker_lines = []
        if slow_song is not None:
            marker_lines.append(("slow example", getattr(slow_song, axis), "rgb(99,110,250)"))
        if fast_song is not None:
            marker_lines.append(("fast example", getattr(fast_song, axis), "rgb(239,85,59)"))
        for label, val, color in marker_lines:
            hist_fig.add_vline(x=val, line=dict(color=color, dash="dash", width=2))
        hist_fig.update_layout(
            height=220, margin=dict(l=10, r=10, t=30, b=10), title=AXIS_LABELS[axis],
            xaxis_title=None, yaxis_title="songs" if i == 0 else None, showlegend=False,
        )
        with dna_hist_cols[i % 3]:
            st.plotly_chart(hist_fig, width="stretch", key=f"dna_hist_{axis}")
    st.caption(
        f"n={len(dna_songs_with_values)} songs with fully-computed DNA. Dashed lines mark the two "
        "curated examples above (blue = slow, red = fast) -- both sit at genuine, non-cherry-picked "
        "extremes of their respective distributions, not just relative to each other."
    )

st.subheader("3b. Fingerprints -- structure, sound, harmony, and how they combine")
st.write(
    "Every song gets a small visual fingerprint per facet, plus a composite that overlays three of "
    "them as color channels (structure = red, sound = green, harmony = blue) -- where the channels "
    "agree the image reads bright and neutral, where they diverge it casts a color. Pick a song to "
    "see all of them together, exactly as the live Song X-Ray page renders them:"
)

fp_candidates = [t for t in FINGERPRINT_EXAMPLE_TITLES if _find_song(t) is not None]
fp_choice = st.selectbox("Pick a song", options=fp_candidates, key="walkthrough_fp_picker")
fp_song = _find_song(fp_choice) if fp_choice else None

if fp_song is not None:
    st.markdown(f"**{fp_song.title}** — {fp_song.artist} ({fp_song.genre_top})")
    st.audio(str(audio_path_for(fp_song)))

    try:
        matrix = embedding_repo.get_structure_matrix(fp_song.id)
    except FileNotFoundError:
        matrix = None
    try:
        timeline = embedding_repo.get_structure_timeline(fp_song.id)
    except FileNotFoundError:
        timeline = None

    structure_fp = structure_fingerprint(matrix) if matrix is not None else None
    sound_fp = timeline.sound_fingerprint if timeline is not None else None
    harmony_fp = timeline.harmony_fingerprint if timeline is not None else None

    fp_cols = st.columns(4)
    if structure_fp is not None:
        with fp_cols[0]:
            st.plotly_chart(fingerprint_thumbnail(structure_fp, "Structure"), width="stretch", key="wt_fp_structure")
    if sound_fp is not None:
        with fp_cols[1]:
            st.plotly_chart(fingerprint_thumbnail(sound_fp, "Sound"), width="stretch", key="wt_fp_sound")
    if harmony_fp is not None:
        with fp_cols[2]:
            st.plotly_chart(fingerprint_thumbnail(harmony_fp, "Harmony"), width="stretch", key="wt_fp_harmony")
    if structure_fp is not None and sound_fp is not None and harmony_fp is not None:
        with fp_cols[3]:
            composite = composite_fingerprint(structure_fp, sound_fp, harmony_fp)
            st.plotly_chart(composite_fingerprint_thumbnail(composite), width="stretch", key="wt_fp_composite")

    if timeline is not None and timeline.has_clear_structure:
        st.caption(
            "Each colored block below is a stretch of the song; same color = similar-sounding "
            "sections, discovered automatically from the audio, no manual labeling."
        )
        palette = px.colors.qualitative.Set2
        unique_labels = sorted(set(timeline.segment_labels.tolist()))
        color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}
        durations = timeline.segment_ends - timeline.segment_starts
        hover_text = [f"{s:.1f}s – {e:.1f}s" for s, e in zip(timeline.segment_starts, timeline.segment_ends)]
        timeline_fig = go.Figure(go.Bar(
            x=durations, y=["Structure"] * len(durations), base=timeline.segment_starts, orientation="h",
            marker_color=[color_map[lab] for lab in timeline.segment_labels.tolist()],
            marker_line_width=0, hovertext=hover_text, hoverinfo="text",
        ))
        timeline_fig.update_layout(
            height=120, showlegend=False, xaxis_title="Time (s)",
            yaxis=dict(showticklabels=False), margin=dict(l=10, r=10, t=10, b=40), bargap=0,
        )
        st.plotly_chart(timeline_fig, width="stretch", key="wt_structure_timeline")
    elif timeline is not None and timeline.novelty_curve is not None:
        st.caption(
            "This song evolves gradually rather than repeating in clear sections -- shown as a "
            "continuous novelty curve instead of colored blocks."
        )
        curve_fig = go.Figure(go.Scatter(
            x=timeline.novelty_times, y=timeline.novelty_curve, mode="lines", fill="tozeroy",
            line=dict(color="rgb(99,110,250)"),
        ))
        curve_fig.update_layout(
            height=140, xaxis_title="Time (s)",
            yaxis=dict(title="novelty", showticklabels=False, range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(curve_fig, width="stretch", key="wt_novelty_curve")

    with st.expander("Zoom in further: raw self-similarity matrix"):
        st.caption(
            "The matrix everything above is derived from -- the main diagonal is deliberately left "
            "blank; bright parallel stripes off the diagonal are what mark repeated sections."
        )
        if matrix is not None:
            heatmap = px.imshow(
                matrix, color_continuous_scale="Magma", origin="lower",
                labels=dict(x="beat", y="beat", color="similarity"),
            )
            heatmap.update_layout(height=420)
            st.plotly_chart(heatmap, width="stretch", key="wt_ssm_matrix")
        else:
            st.warning("No structure matrix computed for this song yet.")

st.write(
    "Checked all 1394 songs with detected structural boundaries for any segment whose label "
    "repeats later in the timeline (genuine \"the verse comes back\" recurrence). **Zero** songs "
    "show this. Every clearly-structured song's segments are uniquely labeled -- a monotonic "
    "evolution across the clip, not a loop. That tracks: 30 seconds usually isn't long enough to "
    "play out a full section and return to an earlier one. So the honest framing is that the "
    "structure facet shows *how a song's texture evolves across its first 30 seconds*, not "
    "full-song verse/chorus form -- that would need complete tracks."
)
st.caption(
    "**Honest limitation on the algorithm itself:** validated against synthetic audio with known "
    "ground truth (a pure tone produces zero detected peaks; two audibly distinct halves produce "
    "exactly one peak, accurate to within 0.01s of the true midpoint) -- but not re-confirmed by "
    "ear against every real recording shown above. Use the players above for your own last-mile check."
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Taste Map
# ---------------------------------------------------------------------------
st.header("4. Taste Map -- the whole library at once")
st.caption(
    "This is the methodology behind the projection -- the live, interactive, clickable version of "
    "this same map lives in **Explore** (\"2D map\" view), not on this page."
)
st.subheader("4a. Projecting the library")
st.write(
    "Mean-pool every song's sound-facet segment embeddings into one vector per song, then project "
    "the whole library down to 2D (PCA) and cluster it (K-means) -- entirely from audio, no genre "
    "labels involved in either step. Coloring the same map by the *known* genre labels afterward is "
    "a direct visual test of whether sonic clusters actually line up with genre, or cut across it."
)


@st.cache_data
def _walkthrough_taste_map_df(_song_repo, _embedding_repo, cache_key):
    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    result = compute_taste_map(song_vectors, method="pca")
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    return pd.DataFrame([
        {
            "song_id": p.song_id, "x": p.x, "y": p.y, "cluster": str(p.cluster),
            "title": songs_by_id[p.song_id].title, "artist": songs_by_id[p.song_id].artist,
            "genre": songs_by_id[p.song_id].genre_top,
        }
        for p in result.points if p.song_id in songs_by_id
    ])


taste_df = _walkthrough_taste_map_df(song_repo, embedding_repo, embedding_repo.index_size("sound"))

if not taste_df.empty:
    map_cols = st.columns(2)
    with map_cols[0]:
        cluster_fig = px.scatter(
            taste_df, x="x", y="y", color="cluster",
            hover_data={"title": True, "artist": True, "genre": True, "x": False, "y": False, "cluster": False},
            title="Colored by discovered cluster (K-means, sound-only)",
        )
        cluster_fig.update_traces(marker=dict(size=7))
        cluster_fig.update_layout(height=440)
        st.plotly_chart(cluster_fig, width="stretch", key="wt_taste_map_cluster")
    with map_cols[1]:
        genre_fig = px.scatter(
            taste_df, x="x", y="y", color="genre",
            hover_data={"title": True, "artist": True, "genre": True, "x": False, "y": False, "cluster": False},
            title="Colored by known genre label",
        )
        genre_fig.update_traces(marker=dict(size=7))
        genre_fig.update_layout(height=440)
        st.plotly_chart(genre_fig, width="stretch", key="wt_taste_map_genre")
    st.caption(
        "The two maps share the same layout (same x/y for every point) -- only the coloring "
        "differs. Where cluster boundaries roughly track genre boundaries, sonic similarity and "
        "genre agree; where a cluster spans multiple genre colors (or a genre splits across "
        "clusters), the audio is telling you something the label doesn't."
    )

    axis_hist_cols = st.columns(2)
    with axis_hist_cols[0]:
        x_hist = go.Figure(go.Histogram(x=taste_df["x"], nbinsx=30))
        x_hist.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title="x-axis value distribution")
        st.plotly_chart(x_hist, width="stretch", key="taste_map_x_hist")
    with axis_hist_cols[1]:
        y_hist = go.Figure(go.Histogram(x=taste_df["y"], nbinsx=30))
        y_hist.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title="y-axis value distribution")
        st.plotly_chart(y_hist, width="stretch", key="taste_map_y_hist")
    st.caption(
        "Both axes are roughly unimodal with a single dense core and thinner tails -- there's no "
        "obvious multi-cluster gap in either axis alone (multi-modality only emerges from the "
        "combination of both, which is what the K-means clustering above is actually finding)."
    )
else:
    st.info("No embedded songs yet -- Taste Map needs the sound facet's segment embeddings.")

st.subheader("4b. Are the axes interpretable? A rigorous check, not a guess")
st.write(
    "\"Inspect the songs at each axis's extremes and see if you'd name it\" is a real technique, but "
    "it's qualitative and subjective on its own. The rigorous version: correlate each axis against "
    "features that are *already independently computed and meaningful* (tempo, energy, brightness, "
    "harmonic complexity, rhythmic density) -- a clean correlation coefficient is checkable evidence "
    "for what an axis represents; a clean *absence* of correlation is itself a valid, honest finding, "
    "not a failure to report."
)

if not taste_df.empty:
    dna_by_song = {s.id: {axis: getattr(s, axis) for axis in AXES} for s in all_songs}
    has_dna_ids = {sid for sid, raw in dna_by_song.items() if all(v is not None for v in raw.values())}

    for method in ["pca", "ica"]:
        method_vectors = mean_pool_song_vectors(song_repo, embedding_repo)
        method_result = compute_taste_map(method_vectors, method=method)
        pts = [p for p in method_result.points if p.song_id in has_dna_ids]
        if len(pts) < 3:
            continue
        x = np.array([p.x for p in pts])
        y = np.array([p.y for p in pts])
        features = {axis: np.array([dna_by_song[p.song_id][axis] for p in pts]) for axis in AXES}
        correlations = correlate_axes_with_features(x, y, features)

        st.markdown(f"**{method.upper()}** (n={len(pts)} songs with full DNA)")
        corr_cols = st.columns(2)
        for axis_label, col in [("x", corr_cols[0]), ("y", corr_cols[1])]:
            axis_corrs = sorted([c for c in correlations if c.axis == axis_label], key=lambda c: -abs(c.r))
            with col:
                st.caption(f"{axis_label}-axis")
                for c in axis_corrs:
                    flag = " ← strongest" if c is axis_corrs[0] and abs(c.r) >= 0.4 else ""
                    st.markdown(f"- {AXIS_LABELS[c.feature]}: r={c.r:+.3f}{flag}")

    st.caption(
        "**What this actually found:** PCA's y-axis correlates strongly with energy (r≈-0.54), "
        "brightness (r≈-0.46), and harmonic complexity (r≈-0.45) *simultaneously* -- these three "
        "properties move together in this library, so the y-axis reads as a genuine, checkable "
        "\"sparse/calm\" vs. \"dense/bright/complex\" continuum. PCA's x-axis, by contrast, shows no "
        "correlation above noise (all |r| < 0.1) with any of the five features -- it isn't explained "
        "by them, and that's reported honestly rather than papered over with a qualitative guess. "
        "ICA doesn't do meaningfully better at isolating single-feature axes here: its strongest axis "
        "bundles the same three features PCA's y-axis does, just less cleanly -- suggesting these "
        "three genuinely correlate with each other in this library rather than PCA specifically "
        "failing to separate them. For any axis a correlation doesn't explain, the qualitative "
        "\"songs at the extremes\" check in the live Explore page is the honest fallback -- and for "
        "PCA's x-axis, that fallback is genuinely needed."
    )

st.divider()

# ---------------------------------------------------------------------------
# 5. Retrieval
# ---------------------------------------------------------------------------
st.header("5. Retrieval")
st.write(
    "Every facet gets its own FAISS index of segment embeddings. Finding \"similar\" moments is "
    "cosine-similarity nearest-neighbor search within one facet's index -- picking a different "
    "facet changes what \"similar\" means, using the exact same mechanism."
)

st.subheader("5a. Score distributions across the whole library")
st.write(
    "Genre-cohesion (§6) asks whether a facet's neighbors share a label. That's not the whole "
    "story -- a facet can beat the random baseline on label-sharing while still producing a nearly "
    "flat score landscape underneath, where the \"best\" match isn't meaningfully better than the "
    "10th. This samples real queries per facet and compares the actual top-1 match score against a "
    "random-pair baseline, both drawn from the live index -- not simulated."
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
            st.caption(f"{facet_name.capitalize()}: no embedded segments yet.")
            continue
        dist_fig = go.Figure()
        dist_fig.add_trace(go.Histogram(x=top1_scores, name="top-1 match", opacity=0.7, nbinsx=20))
        dist_fig.add_trace(go.Histogram(x=random_scores, name="random pair", opacity=0.7, nbinsx=20))
        dist_fig.update_layout(
            height=240, margin=dict(l=10, r=10, t=30, b=10), barmode="overlay",
            title=f"{facet_name.capitalize()} (n={n_queries})", showlegend=(i == 0),
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
    "these histograms): the gap between the top-1 and 2nd-best score is small for every facet "
    "(typically <0.01) -- with ~14,600 segments and no more than a few hundred per genre, there's "
    "usually a long plateau of near-tied candidates rather than one sharply-best match."
)

st.subheader("5b. Curated examples")
st.write(
    "A few real matches, picked from the live index, with a plain-language explanation generated by "
    "the same LLM explanation layer used throughout the app (not written by hand). Play both clips "
    "below each one to hear the match for yourself:"
)

for facet in FACET_ORDER:
    examples = [e for e in NN_EXAMPLES if e["facet"] == facet]
    if not examples:
        continue
    st.subheader(facet.capitalize())
    for ex in examples:
        query_song = _find_song(ex["query"]["title"])
        match_song = _find_song(ex["match"]["title"])
        with st.expander(
            f"{ex['score_pct']:.0f}% match — \"{ex['query']['title']}\" ↔ \"{ex['match']['title']}\"",
            expanded=False,
        ):
            listen_cols = st.columns(2)
            with listen_cols[0]:
                st.caption(f"Query: \"{ex['query']['title']}\" — {ex['query']['artist']} ({ex['query']['genre']})")
                if query_song is not None:
                    st.audio(str(audio_path_for(query_song)))
            with listen_cols[1]:
                st.caption(f"Match: \"{ex['match']['title']}\" — {ex['match']['artist']} ({ex['match']['genre']})")
                if match_song is not None:
                    st.audio(str(audio_path_for(match_song)))
            st.caption(f"\U0001F4AC *{ex['explanation']}*")

st.divider()

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
st.header("6. Evaluation")
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
# 7. Next: see it in the app
# ---------------------------------------------------------------------------
st.header("7. Next: see it in the app")
st.write(
    "That's the methodology. The **App Walkthrough** picks up from here -- a guided pass through "
    "the live interactive pages themselves, explaining what you're looking at as you go (what the "
    "point cloud's shape means, what an edge between two nodes represents, how to read the "
    "clusters) before you drive it yourself."
)
st.page_link("pages/1_App_Walkthrough.py", label="**Continue to the App Walkthrough →**", icon="\U0001F9ED")
