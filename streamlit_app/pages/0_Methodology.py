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

# ---------------------------------------------------------------------------
# Section 7's case-study evidence: real results from one-time experiments
# (scripts/filter_vocal_facet_by_ast.py's validation runs, scripts/
# whiten_harmony_index.py, scripts/compare_song_level_retrieval.py) --
# embedded as literals rather than recomputed live, since a "before" state
# for an already-applied change (e.g. the harmony index, now permanently
# whitened) no longer exists to recompute against. Same rationale as
# GENRE_COHESION_RESULTS/NN_EXAMPLES above: real numbers, captured once, not
# fabricated for presentation.
# ---------------------------------------------------------------------------

# 7a: whole-clip AST scoring (the FAILED first design) vs. per-segment max
# scoring (the working redesign), on the "Speech" label specifically for the
# whole-clip case and the best-matching vocal-keyword label for per-segment.
VOCAL_GATE_WHOLE_CLIP_SCORES = [
    # (title, expected, "Speech" score) -- expected is what SHOULD happen
    ("3rd Chair", "EXCLUDE (bleed case)", 0.00196),
    ("something brewing", "KEEP (real vocals)", 0.00100),
    ("Bridgewater Triangle", "EXCLUDE (no vocals)", 0.00085),
    ("Sam's Song", "KEEP (real vocals)", 0.00070),
    ("That Horse Ithica", "EXCLUDE (no vocals)", 0.00062),
    ("Pavement Hack", "EXCLUDE (no vocals)", 0.00048),
]
VOCAL_GATE_PER_SEGMENT_SCORES = [
    # (title, expected, max per-segment score across the song's real segments)
    ("Cipralex (c/ Pulso)", "KEEP (real vocals)", 0.154),
    ("A Friendly Noose", "KEEP (real vocals)", 0.103),
    ("Terminally in Love With You", "KEEP (real vocals)", 0.100),
    ("Sam's Song", "KEEP (real vocals)", 0.049),
    ("something brewing", "KEEP (real vocals)", 0.020),
    ("3rd Chair", "EXCLUDE (bleed case)", 0.016),
    ("That Horse Ithica", "EXCLUDE (no vocals)", 0.012),
    ("Bridgewater Triangle", "EXCLUDE (no vocals)", 0.006),
    ("Pavement Hack", "EXCLUDE (no vocals)", 0.004),
]
VOCAL_GATE_THRESHOLD = 0.018

# The per-segment redesign's 9-song validation above was checked against
# *assumed* labels (genre + curated-example status), not real listening.
# A prevalence sample (400 random segments, whole library) found 56.2%
# scoring below threshold -- too high to explain as normal instrumental
# stretches alone -- which prompted an actual blind human-listening
# spot-check: 10 segments, judged with no score/label shown, then compared.
VOCAL_GATE_PREVALENCE_SAMPLE = {"n_segments": 400, "pct_below_threshold": 56.2}
VOCAL_GATE_HUMAN_SPOTCHECK = [
    # (title, genre, model_score, model_verdict, human_verdict, correct)
    ("412", "Hip-Hop", 0.0179, "no vocal", "vocal", False),
    ("Dismissal", "Pop", 0.0171, "no vocal", "vocal", False),
    ("Facing the Sea (Album Version)", "Pop", 0.0195, "vocal", "vocal (last 2s only)", True),
    ("A Message", "Rock", 0.0174, "no vocal", "no vocal", True),
    ("Requiem for a Small Town", "Folk", 0.0175, "no vocal", "vocal", False),
    ("something brewing", "Hip-Hop", 0.0028, "no vocal", "no vocal", True),
    ("A1 Symphony", "Hip-Hop", 0.0017, "no vocal", "no vocal", True),
    ("Underwater", "Electronic", 0.0007, "no vocal", "no vocal", True),
    ("Ride My Bike", "Instrumental", 0.0002, "no vocal", "no vocal", True),
    ("Thursday & Snow (Reprise)", "Hip-Hop", 0.0228, "vocal", "no vocal", False),
]

# 7b: real AST/AudioSet tag output, curated for variety (instrumental with
# specific-instrument tags, ambient/textural, soundtrack, vocal genres).
AST_CAPABILITY_EXAMPLES = [
    {"title": "3rd Chair", "genre": "Instrumental",
     "tags": [("Cello", 0.251), ("Bowed string instrument", 0.090), ("Violin, fiddle", 0.066), ("Double bass", 0.036)]},
    {"title": "Bridgewater Triangle", "genre": "Instrumental",
     "tags": [("Gong", 0.523), ("Ambient music", 0.047), ("Timpani", 0.011), ("Singing bowl", 0.006)]},
    {"title": "OST 05 Go Go Go", "genre": "Electronic",
     "tags": [("Video game music", 0.043), ("Soundtrack music", 0.015), ("Funny music", 0.013), ("Theme music", 0.005)]},
    {"title": "A Friendly Noose", "genre": "Folk",
     "tags": [("Female singing", 0.083), ("Singing", 0.034), ("Guitar", 0.014), ("Country", 0.014)]},
    {"title": "Cipralex (c/ Pulso)", "genre": "Hip-Hop",
     "tags": [("Mantra", 0.085), ("Chant", 0.036), ("Speech", 0.011), ("Electronic music", 0.014)]},
]

# 7c: harmony whitening before/after (k=10, sample_size=300, seed=42).
HARMONY_WHITENING_RESULTS = {
    "before": {"top1_mean": 0.983, "random_mean": 0.847, "margin_mean": 0.0027, "cohesion_pct": 20.7, "baseline_pct": 11.5},
    "after": {"top1_mean": 0.865, "random_mean": -0.016, "margin_mean": 0.0187, "cohesion_pct": 20.1, "baseline_pct": 11.5},
}

# 7d: segment-level vs. song-level retrieval, all six facets (k=10,
# sample_size=300, seed=42) -- harmony's numbers here are measured on the
# already-whitened index (7c ran first).
SONG_LEVEL_COMPARISON = [
    {"facet": "sound", "seg_margin": 0.0080, "song_margin": 0.0185, "seg_cohesion": 55.4, "song_cohesion": 52.5},
    {"facet": "harmony", "seg_margin": 0.0187, "song_margin": 0.0326, "seg_cohesion": 20.1, "song_cohesion": 21.8},
    {"facet": "vocal", "seg_margin": 0.0093, "song_margin": 0.0147, "seg_cohesion": 34.4, "song_cohesion": 38.3},
    {"facet": "drums", "seg_margin": 0.0088, "song_margin": 0.0152, "seg_cohesion": 33.5, "song_cohesion": 36.5},
    {"facet": "bass", "seg_margin": 0.0087, "song_margin": 0.0114, "seg_cohesion": 25.7, "song_cohesion": 29.8},
    {"facet": "instrumental", "seg_margin": 0.0105, "song_margin": 0.0194, "seg_cohesion": 38.5, "song_cohesion": 44.1},
]

# 7e: does fixed-window segmentation explain 7a's vocal-gate errors? Checked
# against the Structure facet's already-computed novelty detection for the
# same 10 blind-listened segments -- no new audio processing, a pure
# correlation check against existing data.
STRUCTURE_ALIGNMENT_HIT = {
    "title": "Facing the Sea (Album Version)", "human_transition_sec": 8.0,
    "novelty_peak_sec": 8.96, "novelty_peak_strength": 0.58, "segment_boundary_sec": 9.0,
}
STRUCTURE_ALIGNMENT_STRADDLE_TABLE = [
    # (title, straddles a structural boundary?, was this segment an error?)
    ("412", True, "fixed by 15s context"),
    ("Dismissal", True, "fixed by 15s context"),
    ("Facing the Sea (Album Version)", True, "explained -- see above"),
    ("A Message", True, "no error"),
    ("Requiem for a Small Town", False, "persistent error -- unexplained"),
    ("something brewing", True, "no error"),
    ("A1 Symphony", False, "no error"),
    ("Underwater", True, "no error"),
    ("Ride My Bike", True, "no error"),
    ("Thursday & Snow (Reprise)", False, "persistent error -- unexplained"),
]
# Quick, cheap follow-up (reused already-computed song DNA + structural
# confidence, zero new processing): do the two unexplained errors share
# anything? n=2, suggestive not conclusive.
UNEXPLAINED_ERROR_DNA_COMPARISON = [
    {"title": "Thursday & Snow (Reprise)", "structural_confidence": 0.1562, "rhythmic_density": 6.44, "rank_confidence": "lowest of 10", "rank_density": "highest of 10"},
    {"title": "Requiem for a Small Town", "structural_confidence": 0.1806, "rhythmic_density": 5.20, "rank_confidence": "2nd-lowest of 10", "rank_density": "2nd-highest of 10"},
]
REST_OF_SAMPLE_STRUCTURAL_CONFIDENCE_RANGE = (0.1889, 0.2593)
REST_OF_SAMPLE_RHYTHMIC_DENSITY_RANGE = (2.97, 4.74)

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
        for _label, val, color in marker_lines:
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
        hover_text = [f"{s:.1f}s – {e:.1f}s" for s, e in zip(timeline.segment_starts, timeline.segment_ends, strict=False)]
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
    "finishes -- Sound is unaffected. Harmony's number here is also its pre-whitening baseline; "
    "see §7c for the whitening experiment's own before/after measurement.",
    icon="\U0001F6A7",
)

st.divider()

# ---------------------------------------------------------------------------
# 7. Model improvement case studies
# ---------------------------------------------------------------------------
st.header("7. Model improvement case studies")
st.write(
    "§6 established real weaknesses per facet, not just aggregate scores. This section documents "
    "concrete attempts to fix or explain them -- each follows the same discipline: state a "
    "hypothesis, test it against the real library, report the honest result, whether or not it "
    "fully worked. §4b's axis-interpretability check (correlate first, qualitative-listen only "
    "where correlation doesn't resolve it) already followed this same pattern -- it belongs to this "
    "same family of case studies, just located earlier in the narrative."
)

st.subheader("7a. Vocal-facet cross-check: hypothesis, failure, redesign, validation")
st.write(
    "§3b noted the vocal facet's honest limitation: Demucs' \"vocal\" stem can carry real energy "
    "from non-vocal content (confirmed case: \"3rd Chair\", a cello/violin piece, scored 0.58 "
    "stem-to-mix energy ratio -- well above the energy gate's 0.05 threshold -- despite having no "
    "real vocals). **Hypothesis:** a pretrained AudioSet tagger (AST) could independently check "
    "whether a song actually contains singing/speech, catching what the energy gate can't."
)
st.markdown("**First attempt (failed): score the whole 30-second clip at once**")
whole_clip_df = pd.DataFrame(VOCAL_GATE_WHOLE_CLIP_SCORES, columns=["Song", "Expected", "\"Speech\" score"])
st.dataframe(whole_clip_df, hide_index=True, width="stretch")
st.caption(
    "There is no threshold that sorts this correctly -- \"3rd Chair\" (the exact bleed case this "
    "was supposed to catch) scores *higher* than two real-vocal songs it must not exclude. "
    "**Diagnosis:** AST's output over a full 30s clip is a continuous distribution across all 527 "
    "AudioSet classes, not a sparse detector -- dominant instrumental/percussive content in the mix "
    "swamps genuinely-present-but-quieter vocals into the same tiny-probability noise floor that "
    "residual background \"vocal\" mass sits at in truly instrumental tracks."
)
st.markdown("**Redesign: score each ~5s segment individually, take the max**")
per_segment_df = pd.DataFrame(VOCAL_GATE_PER_SEGMENT_SCORES, columns=["Song", "Expected", "Max segment score"])
st.dataframe(per_segment_df, hide_index=True, width="stretch")
st.caption(
    f"Clean separation this time: every \"keep\" song scores ≥ 0.020, every \"exclude\" song scores "
    f"≤ 0.016 -- a threshold around **{VOCAL_GATE_THRESHOLD}** sorts all 9 confirmed cases "
    "correctly, \"3rd Chair\" included. A shorter window has less competing instrumental content, "
    "so a real vocal moment doesn't get drowned out the way it did over the full clip."
)
st.markdown("**Reality check: does the 9-song validation hold up against real listening?**")
st.write(
    "That 9-song validation was checked against *assumed* labels (genre + curated-example status), "
    "not actual listening. Before trusting it at library scale, a 400-segment random sample across "
    "the whole library (not restricted to any genre) found "
    f"**{VOCAL_GATE_PREVALENCE_SAMPLE['pct_below_threshold']:.1f}% of segments scoring below "
    f"threshold** -- far too high to explain as normal instrumental intros/bridges alone. That "
    "prompted an actual blind human-listening spot-check: 10 segments, judged with no score or "
    "label shown, compared afterward."
)
spotcheck_df = pd.DataFrame(
    VOCAL_GATE_HUMAN_SPOTCHECK,
    columns=["Song", "Genre", "Model score", "Model said", "Human heard", "Agree?"],
)
st.dataframe(spotcheck_df, hide_index=True, width="stretch")
n_correct = sum(1 for row in VOCAL_GATE_HUMAN_SPOTCHECK if row[5])
st.caption(f"**{n_correct}/{len(VOCAL_GATE_HUMAN_SPOTCHECK)} agreed with human judgment.**")
st.error(
    "**This kills the threshold-based approach entirely -- not just this cutoff.** Three false "
    "negatives (real vocals the model missed) scored 0.017-0.0179; one false positive (confidently "
    "scored as vocal, no real vocals present) scored 0.0228 -- *higher* than every false negative. "
    "Fixing the false negatives means lowering the threshold below ~0.017; fixing the false positive "
    "means raising it above ~0.023. Those requirements contradict each other -- there's no threshold "
    "that satisfies both. This isn't a calibration problem: the underlying keyword-max score doesn't "
    "reliably track real vocal presence, at least not with this scoring method.",
    icon="🛑",
)
st.info(
    "**Honest final status:** NOT applied to the live vocal facet, and not recommended to be, at "
    "least not with this technique. The whole-clip → per-segment redesign genuinely fixed the "
    "*ordering* problem from the first attempt, and the code (`sonic_explorer/pipeline/"
    "vocal_presence.py`, `EmbeddingRepository.remove_from_index()`) is real, tested infrastructure "
    "-- but the underlying signal isn't reliable enough to trust as an automatic filter, confirmed "
    "by actual human listening, not just a broader sample size. The energy gate (already live, "
    "catching near-silent stems) remains the vocal facet's only automated quality check; both the "
    "\"instrumental stretch within a vocal song\" and \"Demucs bleed\" problems remain open, honestly "
    "unresolved limitations. A different technique -- a dedicated singing-voice-detection model, or "
    "pitch/periodicity analysis directly on the isolated stem rather than a general-purpose 527-class "
    "tagger on the mix -- might do better, but that's a new, untried investigation, not this one.",
    icon="⏳",
)

st.subheader("7b. Sound recognition as a general capability")
st.write(
    "Separate from the vocal-gate application above: the same pretrained AST/AudioSet model is a "
    "real, standalone capability -- given any clip, it tags what it hears against 527 general audio "
    "classes, no training required. Worth judging on its own: are the tags actually descriptive, or "
    "generic noise?"
)
for ex in AST_CAPABILITY_EXAMPLES:
    tag_text = " · ".join(f"{label} ({score:.0%})" for label, score in ex["tags"])
    st.markdown(f"**{ex['title']}** ({ex['genre']}) — {tag_text}")
st.caption(
    "Genuinely specific, not generic: \"3rd Chair\" resolves to actual instrument names "
    "(Cello, Bowed string instrument, Violin) with no model fine-tuning on this library at all. "
    "That specificity is what makes instrument/texture tagging a plausible future capability (e.g. "
    "tag-based search in Ask the DJ), not yet built. **Worth being precise about scope, though:** "
    "7a's human spot-check found the singing/speech keyword-threshold specifically unreliable -- "
    "that's a narrower claim than \"AST tagging doesn't work.\" The broad instrument/texture tags "
    "shown above weren't the part that failed."
)

st.subheader("7c. Harmony whitening: fixing the score geometry vs. fixing the task")
st.write(
    "§5a found harmony's random-pair baseline sitting at 0.85-0.95 cosine similarity -- the raw "
    "24-dim chroma-derived space (12 pitch classes × mean + std) has very little natural spread, so "
    "real differences barely register once L2-normalized. **Hypothesis:** whitening each dimension "
    "to zero mean / unit variance across the corpus before re-normalizing should spread the space "
    "out along directions that actually vary -- a pure post-hoc transform on vectors already "
    "computed, no re-extraction needed."
)
hw = HARMONY_WHITENING_RESULTS
whiten_cols = st.columns(2)
with whiten_cols[0]:
    st.markdown("**Before**")
    st.metric("Top-1 vs. random gap", f"{hw['before']['top1_mean'] - hw['before']['random_mean']:.3f}")
    st.metric("Top-1 vs. top-2 margin", f"{hw['before']['margin_mean']:.4f}")
    st.metric("Genre-cohesion@10", f"{hw['before']['cohesion_pct']:.1f}%")
with whiten_cols[1]:
    st.markdown("**After**")
    st.metric("Top-1 vs. random gap", f"{hw['after']['top1_mean'] - hw['after']['random_mean']:.3f}",
               delta=f"{(hw['after']['top1_mean'] - hw['after']['random_mean']) - (hw['before']['top1_mean'] - hw['before']['random_mean']):+.3f}")
    st.metric("Top-1 vs. top-2 margin", f"{hw['after']['margin_mean']:.4f}",
               delta=f"{hw['after']['margin_mean'] - hw['before']['margin_mean']:+.4f}")
    st.metric("Genre-cohesion@10", f"{hw['after']['cohesion_pct']:.1f}%",
               delta=f"{hw['after']['cohesion_pct'] - hw['before']['cohesion_pct']:+.1f}pp")
st.caption(
    "A real, honest split result. The score geometry improved dramatically -- random pairs went "
    "from a misleadingly-high 0.85 average down to essentially 0, and individual rankings got ~7x "
    "more decisive (margin 0.0027 → 0.0187). But genre-cohesion, the actual task metric, stayed "
    "flat (20.7% → 20.1%, within sampling noise). **Conclusion:** whitening fixed the symptom "
    "(a compressed, misleading score range) but not the underlying limitation -- a 24-dim chroma "
    "mean+std summary is a coarse representation of harmony, and rescaling it can't inject "
    "discriminative information that was never captured in the first place. Score-geometry health "
    "and task performance are genuinely different things; fixing one doesn't guarantee the other. "
    "Kept live regardless -- a sharper single top match is a real usability win in Moment Matcher "
    "and Ask the DJ, even without a genre-cohesion lift."
)

st.subheader("7d. Song-level aggregation: pooling segments before ranking")
st.write(
    "§5a's other finding: every facet's top-1-vs-top-2 margin is small (typically <0.01) -- with "
    "~14,600 segments and often only a few hundred per genre, there's usually a long plateau of "
    "near-tied single-segment candidates. **Hypothesis:** mean-pooling a song's segments into one "
    "vector before ranking (the same aggregation Taste Map/Explore already use for visualization) "
    "should smooth that segment-level noise into a sharper song-level signal."
)
song_level_df = pd.DataFrame(SONG_LEVEL_COMPARISON)
margin_fig = go.Figure(data=[
    go.Bar(name="Segment-level", x=[r["facet"].capitalize() for r in SONG_LEVEL_COMPARISON],
           y=[r["seg_margin"] for r in SONG_LEVEL_COMPARISON]),
    go.Bar(name="Song-level", x=[r["facet"].capitalize() for r in SONG_LEVEL_COMPARISON],
           y=[r["song_margin"] for r in SONG_LEVEL_COMPARISON]),
])
margin_fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), barmode="group",
                          yaxis_title="top1-vs-top2 margin", title="Ranking margin: segment vs. song level")
st.plotly_chart(margin_fig, width="stretch", key="song_level_margin_chart")

cohesion_fig = go.Figure(data=[
    go.Bar(name="Segment-level", x=[r["facet"].capitalize() for r in SONG_LEVEL_COMPARISON],
           y=[r["seg_cohesion"] for r in SONG_LEVEL_COMPARISON]),
    go.Bar(name="Song-level", x=[r["facet"].capitalize() for r in SONG_LEVEL_COMPARISON],
           y=[r["song_cohesion"] for r in SONG_LEVEL_COMPARISON]),
])
cohesion_fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), barmode="group",
                            yaxis_title="genre-cohesion@10 (%)", title="Task performance: segment vs. song level")
st.plotly_chart(cohesion_fig, width="stretch", key="song_level_cohesion_chart")

st.caption(
    "Margin improved for **every** facet (1.3x-2.3x sharper). Genre-cohesion improved for **5 of "
    "6** facets -- Instrumental +5.6pp, Bass +4.1pp, Vocal +3.9pp, Drums +3.0pp, Harmony +1.7pp. "
    "Sound is the one exception, slightly worse (55.4% → 52.5%) -- plausibly because Sound's "
    "per-segment specificity was already strong (the highest baseline of any facet), and averaging "
    "a song's segments blurs together genuinely different sonic moments (a quiet intro vs. a loud "
    "chorus) precisely where that segment-level precision was doing real work."
)
st.info(
    "**Status:** implemented and validated (`sonic_explorer/retrieval/song_level_index.py`) as a "
    "real, working alternative retrieval mode -- not yet wired into Moment Matcher's UI as a "
    "selectable option. A natural, low-risk follow-up given the validated improvement.",
    icon="✅",
)

st.subheader("7e. Does segment misalignment explain the vocal-gate errors? A structural cross-check")
st.write(
    "7a's segments are cut at fixed clock intervals (every ~2.5s, 5s windows), regardless of what's "
    "actually happening musically -- a boundary can fall mid-vocal-line or anywhere arbitrary. "
    "**Hypothesis:** that misalignment could explain some of 7a's confusing results. Checked directly "
    "against the Structure facet's already-computed novelty detection for the same 10 blind-listened "
    "segments -- no new audio processing, a pure correlation check against data that already existed."
)
hit = STRUCTURE_ALIGNMENT_HIT
st.markdown(f"**Confirmed hit: \"{hit['title']}\"**")
st.write(
    f"Human note: vocals only in the last 2 seconds of the sampled window (transition around "
    f"~{hit['human_transition_sec']:.0f}s). The Structure facet's novelty curve shows a real peak at "
    f"**{hit['novelty_peak_sec']:.2f}s** (strength {hit['novelty_peak_strength']:.2f}), with a "
    f"segment boundary at {hit['segment_boundary_sec']:.1f}s -- both landing right where the ear "
    f"placed the transition. Real, specific evidence that structurally-aware segmentation would have "
    f"caught this exact case."
)
st.markdown("**But it doesn't generalize to the other errors**")
straddle_df = pd.DataFrame(STRUCTURE_ALIGNMENT_STRADDLE_TABLE, columns=["Song", "Straddles a structural boundary?", "Outcome"])
st.dataframe(straddle_df, hide_index=True, width="stretch")
st.caption(
    "Straddling a structural boundary is common (7 of 10 windows) and doesn't reliably predict which "
    "cases were confusing. Both persistent errors -- \"Requiem for a Small Town\" (the false negative "
    "that survived even the 15s-context fix) and \"Thursday & Snow\" (the false positive) -- sit "
    "entirely *within* one structural segment, no boundary nearby to blame."
)
st.markdown("**Quick, cheap follow-up: do the two unexplained errors share anything?**")
dna_df = pd.DataFrame(UNEXPLAINED_ERROR_DNA_COMPARISON)
st.dataframe(dna_df, hide_index=True, width="stretch")
st.caption(
    f"Reusing already-computed song DNA (zero new processing): both unexplained errors rank #1 and "
    f"#2 lowest on structural confidence (rest of the sample: "
    f"{REST_OF_SAMPLE_STRUCTURAL_CONFIDENCE_RANGE[0]:.4f}-{REST_OF_SAMPLE_STRUCTURAL_CONFIDENCE_RANGE[1]:.4f}) "
    f"*and* #1 and #2 highest on rhythmic density (rest of the sample: "
    f"{REST_OF_SAMPLE_RHYTHMIC_DENSITY_RANGE[0]:.2f}-{REST_OF_SAMPLE_RHYTHMIC_DENSITY_RANGE[1]:.2f}). "
    "A plausible hypothesis, **not confirmed at n=2**: dense, rhythmically busy tracks with low "
    "structural contrast throughout give AST's keyword-based scoring less of a textural handle to "
    "separate voice from background, independent of window size or placement."
)
st.info(
    "**Status:** a confirmed, real mechanism for one class of error, not a general explanation -- "
    "structurally-aware segmentation would plausibly fix cases like \"Facing the Sea\" without "
    "touching cases like \"Requiem\" or \"Thursday & Snow.\" Given that scope, not worth a full "
    "segmentation redesign right now -- documented honestly as a real, bounded finding rather than "
    "either oversold or dismissed.",
    icon="🔍",
)

st.divider()

# ---------------------------------------------------------------------------
# 8. Next: see it in the app
# ---------------------------------------------------------------------------
st.header("8. Next: see it in the app")
st.write(
    "That's the methodology. The **App Walkthrough** picks up from here -- a guided pass through "
    "the live interactive pages themselves, explaining what you're looking at as you go (what the "
    "point cloud's shape means, what an edge between two nodes represents, how to read the "
    "clusters) before you drive it yourself."
)
st.page_link("pages/1_App_Walkthrough.py", label="**Continue to the App Walkthrough →**", icon="\U0001F9ED")
