"""Approach -- a bridge page between Overview and Methodology answering "why
does this pipeline look the way it does" visually, before Methodology dives
into technical depth and evidence. Built from real audio/data already in the
library wherever possible (a real waveform, real playable window clips, a
real embedding vector, real song DNA, real detected tags) rather than
abstract UI-box illustrations, so the mechanic is shown, not just described.

The demo song threading through steps 1-5 is picked dynamically (the same
`real_pair`-derived song `comparison_data.get_demo_pairs()` already computes
for Overview/Results), not a hardcoded title -- a fixed title could easily
not exist in a smaller deployed subset (this project has hit that exact bug
before; see git history), so continuity comes from reusing the same
already-safe selection, not a new hardcoded pick.

Step 2 ("seven ways of listening") needs two things this page can't fabricate:
real isolated stem audio (vocal/drums/bass/instrumental -- never persisted
anywhere in this project, only stem *embeddings* were kept) and the
sound_tags facet's real per-segment index (notebooks/11_sound_tags_facet.ipynb,
running as of this page's last edit). Both degrade to an explicit, honest
placeholder rather than a fabricated example when missing -- see
STEM_EXAMPLE_DIR and the sound_tags caption below."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from comparison_data import build_metadata_vs_real_graphs, get_demo_pairs
from components.plotting import close_by_illustration, embedding_strip_figure, song_dna_bars, waveform_figure
from resources import (
    build_dna_normalizer,
    build_normalized_dna_by_song,
    get_explanation_client,
    get_repositories,
    nav_button,
    show_data_source_banner,
    show_logo,
)
from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS
from sonic_explorer.analysis.waveform_preview import waveform_envelope
from sonic_explorer.config import WINDOW_SEC, audio_path_for
from sonic_explorer.facets.tags import GENERIC_TAG_LABELS
from sonic_explorer.pipeline.sound_tagging import deserialize_tags

# Real stem audio isn't produced by any pipeline step in this repo -- see
# this page's module docstring. Once extracted (see the notebook 03 smoke-
# test cell + a short save/zip/download cell), drop the five files here:
# mix.wav, vocal.wav, drums.wav, bass.wav, instrumental.wav, meta.json
# ({"title", "artist", "genre_top"} for whichever song was separated).
STEM_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "static" / "stem_example"

st.set_page_config(page_title="Sonic Explorer", layout="wide")
show_logo()
show_data_source_banner()

st.title("Approach")
st.write(
    "Overview made the case for analyzing audio directly instead of trusting tags. Here's how "
    "that actually works, one step at a time -- real audio and real data at every step, not "
    "abstract diagrams. **Methodology** walks through each of these in full technical depth "
    "afterward, with evidence."
)

song_repo, embedding_repo, _ = get_repositories()
all_songs = song_repo.list_songs()
songs_by_id = {s.id: s for s in all_songs}

if not all_songs:
    st.info("No songs available yet to build this walkthrough.")
    st.stop()

metadata_nodes, metadata_edges, real_nodes, real_edges, vectors, genre_by_song = build_metadata_vs_real_graphs(
    song_repo, embedding_repo, len(all_songs)
)
metadata_pair, real_pair = get_demo_pairs(song_repo, embedding_repo, len(all_songs))

# Picked dynamically, not a hardcoded title -- see module docstring.
demo_song = songs_by_id.get(real_pair.song_id_a) if real_pair is not None else all_songs[0]


@st.cache_data(show_spinner=False)
def _window_clip_bytes(song_id: int, start_sec: float, duration_sec: float) -> bytes:
    from sonic_explorer.analysis.waveform_preview import extract_window_clip

    return extract_window_clip(audio_path_for(songs_by_id[song_id]), start_sec, duration_sec)


st.divider()

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
st.header("Baseline")
st.write(
    "Before testing whether audio-based similarity works, it's worth being clear about what "
    "it's being compared against."
)
st.write(
    "Of the two existing paradigms just described, this project tests against metadata-based "
    "similarity."
)
st.caption(
    "Because our dataset has no listen counts, ratings, or user-level interaction data to build "
    "it from, collaborative filtering isn't attempted here. See more in Methodology > Data."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 1: slicing the track into windows
# ---------------------------------------------------------------------------
st.header("1. Slicing the track into windows")
st.write("Songs don't sound the same throughout their entirety.")

st.plotly_chart(
    waveform_figure(
        waveform_envelope(audio_path_for(demo_song)), title=demo_song.title,
        duration_sec=demo_song.duration_sec, color="rgb(99,110,250)",
    ),
    width="stretch", key="step1_full_waveform",
)
st.caption(f"\"{demo_song.title}\" -- real amplitude shape across the whole clip, not a flat signal.")

st.write(
    "Looking at a whole song at once risks losing or averaging out what makes any single moment "
    "distinct. So we slice each song into overlapping "
    f"{WINDOW_SEC:.0f}-second windows -- {song_repo.count_segments()} of them across the whole "
    "library right now. Here are a few real windows from the song above, each one separately "
    "playable:"
)

demo_segments = song_repo.get_segments(demo_song.id)
sample_segments = demo_segments[:: max(1, len(demo_segments) // 4)][:4] if demo_segments else []
if sample_segments:
    window_cols = st.columns(len(sample_segments))
    for col, seg in zip(window_cols, sample_segments, strict=False):
        with col:
            st.caption(f"{seg.start_sec:.1f}-{seg.end_sec:.1f}s")
            st.audio(_window_clip_bytes(demo_song.id, seg.start_sec, seg.end_sec - seg.start_sec), format="audio/wav")
else:
    st.info("No segments available yet for this song.")

st.divider()

# ---------------------------------------------------------------------------
# Step 2: seven ways of listening
# ---------------------------------------------------------------------------
st.header("2. Seven ways of listening")
st.write(
    "A listener might be responding to one specific facet of a song -- the bass, the vocals, "
    "the overall vibe, background texture, structure -- without realizing it."
)
st.write(
    "Measuring several independent qualities, rather than one blended score, makes sense given "
    "that."
)

stem_meta_path = STEM_EXAMPLE_DIR / "meta.json"
has_real_stems = stem_meta_path.exists() and (STEM_EXAMPLE_DIR / "mix.wav").exists()

if has_real_stems:
    stem_meta = json.loads(stem_meta_path.read_text(encoding="utf-8"))
    st.caption(
        f"Sound and Harmony use \"{demo_song.title}\"'s real audio, same as elsewhere on this "
        f"page. The four isolated facets below use real separated stems from a different real "
        f"song -- \"{stem_meta['title']}\" by {stem_meta['artist']} ({stem_meta['genre_top']}) -- "
        "separation is a one-time, offline step (Colab notebook 03), not something this page runs live."
    )
    mix_path_for_stems = STEM_EXAMPLE_DIR / "mix.wav"
else:
    st.info(
        "**Placeholder for four of seven.** Isolated stem audio (vocal/drums/bass/instrumental) "
        "hasn't been extracted into the app yet -- Sound and Harmony below already use real audio "
        "regardless (neither needs isolated stems). See this page's module docstring for exactly "
        "how to extract and drop in the real stem files."
    )
    mix_path_for_stems = None

demo_mix_path = audio_path_for(demo_song)
facet_defs = [
    ("Sound", demo_mix_path, "Overall timbre, instrumentation, production character.", True, "rgb(99,110,250)"),
    ("Harmony", demo_mix_path, "Key, chords, tonal color.", True, "rgb(0,204,150)"),
    ("Vocal", STEM_EXAMPLE_DIR / "vocal.wav", "Isolated voice timbre and delivery.", has_real_stems, "rgb(239,85,59)"),
    ("Drums", STEM_EXAMPLE_DIR / "drums.wav", "Isolated drum/percussion pattern and timbre.", has_real_stems, "rgb(171,99,250)"),
    ("Bass", STEM_EXAMPLE_DIR / "bass.wav", "Isolated bassline tone and pattern.", has_real_stems, "rgb(255,161,90)"),
    ("Instrumental", STEM_EXAMPLE_DIR / "instrumental.wav", "Backing instrumentation with vocals removed.", has_real_stems, "rgb(25,211,243)"),
]

facet_cols = st.columns(3)
for i, (name, path, desc, ready, color) in enumerate(facet_defs):
    with facet_cols[i % 3]:
        if ready:
            st.plotly_chart(
                waveform_figure(waveform_envelope(path), title=name, color=color, height=100),
                width="stretch", key=f"step2_facet_{name}",
            )
            st.audio(str(path))
        else:
            st.info(f"**{name}** -- placeholder, waiting on real isolated stem audio.")
        st.caption(desc)

demo_tags = [
    (label, score) for label, score in deserialize_tags(demo_song.sound_tags) if label not in GENERIC_TAG_LABELS
]
with facet_cols[len(facet_defs) % 3]:
    st.plotly_chart(
        waveform_figure(waveform_envelope(demo_mix_path), title="Sound Tags", color="rgb(255,105,180)", height=100),
        width="stretch", key="step2_facet_sound_tags",
    )
    st.audio(str(demo_mix_path))
    if demo_tags:
        st.caption(f"Detected: {', '.join(label for label, _ in demo_tags)}.")
    else:
        st.caption("Detected sounds/instruments in the mix.")
st.warning(
    "**Sound Tags is the 7th facet, added this session.** The tags shown above are real, "
    "already-computed per-song data (Methodology 7b) -- what's still pending is the per-*segment* "
    "searchable index that makes this a real similarity facet like the other six, not just LLM "
    "grounding (see Step 5). That indexing run is in progress now "
    "(`notebooks/11_sound_tags_facet.ipynb`, ~10.5-11.5h)."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 3: turning sound into an embedding
# ---------------------------------------------------------------------------
st.header("3. Turning sound into an embedding")
st.write(
    "Each measurement becomes an embedding -- a vector representation that can be directly "
    "compared. Once a song is represented this way, similarity is easy to compute."
)
if demo_song.id in vectors:
    vec = vectors[demo_song.id]
    preview_dims = min(48, len(vec))
    st.plotly_chart(
        embedding_strip_figure(vec, n_dims=preview_dims, title=f"\"{demo_song.title}\" -- first {preview_dims} of {len(vec)} dimensions"),
        width="stretch", key="step3_embedding_strip",
    )
    st.caption(
        "Each cell is one number, one coordinate in a 512-dimensional space -- not a signal over "
        "time, so no connected line between them the way a waveform has."
    )
else:
    st.info("No embedding available yet for this song.")

st.write("Once a song is a point in space, \"similar\" just means \"nearby\":")
st.plotly_chart(close_by_illustration(), width="stretch", key="step3_close_by")
st.caption(
    "Illustrative, not real data -- the real version of this, built from every song in the "
    "library at once, is in Results and Explore."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 4: a second kind of similarity -- Song DNA
# ---------------------------------------------------------------------------
st.header("4. A second kind of similarity: Song DNA")
st.write("Alongside the seven embeddings, each song also gets five plainly engineered features:")

dna_normalizer = build_dna_normalizer(song_repo, len(all_songs))
normalized_dna_by_song = build_normalized_dna_by_song(song_repo, dna_normalizer, len(all_songs))
if demo_song.id in normalized_dna_by_song:
    norm = normalized_dna_by_song[demo_song.id]
    st.plotly_chart(
        song_dna_bars([AXIS_LABELS[a] for a in AXES], [norm[a] for a in AXES], title=demo_song.title),
        width="stretch", key="step4_dna_bars",
    )
else:
    st.info("No song DNA computed yet for this song.")

st.write(
    "These support a second, simpler kind of similarity search -- distance between these numbers "
    "directly -- trading richness for direct controllability: you can sculpt a target by hand "
    "instead of only searching by example."
)
st.write(
    "**Why isn't this embedded too?** It could be, in principle. But two things kept it as plain, "
    "engineered numbers instead: first, these features (especially energy and brightness) likely "
    "already overlap with what Sound's embedding captures, since both come from the same audio, "
    "so an embedded version might add less genuinely new signal than it first appears to. Second, "
    "keeping these as plain, interpretable numbers is what makes direct manual control possible -- "
    "dragging a slider to mean \"slower\" only stays meaningful because these are engineered "
    "values, not compressed into an opaque vector."
)
st.warning(
    "**One honest limitation, separate from the embedding question:** each of these five numbers "
    "is computed once per whole song, not per moment within it. A song with a quiet intro and a "
    "loud chorus can average out to a value that doesn't really represent either part. This is a "
    "real, confirmed effect, not just a hypothetical, and it means the hand-drawn profile search "
    "currently always matches against whole songs, even when what you're really picturing is one "
    "specific moment."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 5: explaining it in plain language
# ---------------------------------------------------------------------------
st.header("5. Explaining it in plain language")
st.write(
    "A similarity score alone doesn't say why two things match. So an LLM looks at the actual "
    "signals behind the match -- the relevant facet values, the song's DNA, and what sounds are "
    "detected in the audio -- and turns that into a real explanation."
)
st.write(
    "In addition to being embedded as their own searchable similarity signal, these tags are "
    "given directly to the LLM as grounding."
)

with st.expander(f"Detected tags for \"{demo_song.title}\""):
    if demo_tags:
        for label, score in demo_tags:
            st.write(f"- {label} ({score:.0%})")
    else:
        st.write("No tags detected above the confidence floor for this song.")

st.write("A match is a number. Ask the DJ and Explore both turn that number into a sentence:")


@st.cache_data(show_spinner=False)
def _demo_explanation(song_a_id: int, song_b_id: int) -> str | None:
    client = get_explanation_client()
    if client is None:
        return None
    a, b = songs_by_id[song_a_id], songs_by_id[song_b_id]
    duration_a = min(30.0, a.duration_sec or 30.0)
    duration_b = min(30.0, b.duration_sec or 30.0)
    return client.generate_explanation(
        a.title, a.artist, a.genre_top, 0.0, duration_a,
        b.title, b.artist, b.genre_top, 0.0, duration_b,
        facet_name="sound", score=real_pair.audio_similarity if real_pair else 0.0,
    )


if real_pair is not None:
    if st.button("Reveal the explanation", key="step5_reveal"):
        explanation = _demo_explanation(real_pair.song_id_a, real_pair.song_id_b)
        if explanation is None:
            st.info(
                "No API key configured in this environment, so live generation isn't available "
                "here -- this is exactly the graceful fallback the app uses everywhere an LLM "
                "feature isn't load-bearing. In a configured environment, this reveals a real "
                "generated sentence, one character at a time."
            )
        else:
            placeholder = st.empty()
            shown = ""
            for ch in explanation:
                shown += ch
                placeholder.markdown(f"*{shown}*")
                time.sleep(0.015)
else:
    st.info("Not enough data yet for a live example.")

st.divider()

st.write("Next: **Methodology** walks through each of these steps in depth, with real evidence at every stage.")
nav_button("Continue to Methodology →", "pages/1_Methodology.py", key="nav_approach_to_methodology")
