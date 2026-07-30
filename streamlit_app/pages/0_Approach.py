"""Approach -- a bridge page between Overview and Methodology answering "why
does this pipeline look the way it does" visually, before Methodology dives
into technical depth and evidence. Built from real audio/data already in the
library wherever possible (a real waveform, real playable window clips, a
real embedding vector, real song DNA, real detected tags) rather than
abstract UI-box illustrations, so the mechanic is shown, not just described.

The demo song threading through steps 1-5 is pinned to "Freak of Nature
(Time Out Dubb)" by C-Doc -- also the real Step 2 stem-audio example
(notebooks/03), chosen specifically because its vocal/drums/bass stems are
all clearly audible and distinct (real measured RMS: vocal 0.06, drums 0.19,
bass 0.29, vs. e.g. a sparse/ambient track where some stems come out nearly
silent), which matters here since Step 2 plays each stem separately. A fixed
title is normally risky here (could easily not exist in a smaller deployed
subset -- this project has hit that exact bug before; see git history), but
this one is safe: it's force-included in `scripts/build_deploy_subset.py`'s
REQUIRED_EXAMPLE_TITLES, guaranteed present in the deployed subset
regardless of the random stratified sample. Falls back to the old dynamic
`real_pair`-derived pick if it's ever missing (e.g. synthetic dev data),
rather than crashing.

"flekkefjord" by Blear Moon (the earlier pick) was reassigned rather than
dropped -- it's a real, strong crow-sound detection example (Methodology 7b),
which fits Results' Ask the DJ Gallery's sound-recognition query ("Any songs
with crow sounds?") much better than this page's general walkthrough; that
demo is a live search_by_sound_content query, not a hardcoded reference, so
no code change was needed there to preserve it.

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
from components.plotting import (
    chord_strip_figure,
    chromagram_figure,
    close_by_illustration,
    embedding_strip_figure,
    mel_spectrogram_figure,
    song_dna_bars,
    waveform_figure,
)
from resources import (
    build_dna_normalizer,
    build_normalized_dna_by_song,
    get_explanation_client,
    get_repositories,
    hero_banner,
    nav_button,
    render_toc,
    show_data_source_banner,
    show_logo,
    sticky_header,
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

TOC_SECTIONS = [
    ("0. First, get baseline", "step-0-baseline"),
    ("1. Slicing the track into windows", "step-1-slicing"),
    ("2. Seven ways of listening", "step-2-listening"),
    ("3. Turning sound into an embedding", "step-3-embedding"),
    ("4. Song DNA", "step-4-song-dna"),
    ("5. Explaining in plain language", "step-5-explaining"),
    ("6. From audio to image", "step-6-album-art"),
]

st.set_page_config(page_title="Sonic Explorer", layout="wide")
hero_banner("approach")
render_toc(TOC_SECTIONS)
# show_logo()
show_data_source_banner()
st.title("Approach")
st.write(
    "This page lays out the framework used to test whether songs can be matched by "
    "how they actually sound: what it's measured against, how a song gets broken "
    "down, and how similarity gets computed from there."
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

# Pinned to "Freak of Nature (Time Out Dubb)" -- see module docstring for
# why this one's safe to hardcode and why it was chosen over flekkefjord.
# Falls back to the old dynamic pick if it's ever missing.
demo_song = next((s for s in all_songs if s.title == "Freak of Nature (Time Out Dubb)"), None) or (
    songs_by_id.get(real_pair.song_id_a) if real_pair is not None else all_songs[0]
)


@st.cache_data(show_spinner=False)
def _window_clip_bytes(song_id: int, start_sec: float, duration_sec: float) -> bytes:
    from sonic_explorer.analysis.waveform_preview import extract_window_clip

    return extract_window_clip(audio_path_for(songs_by_id[song_id]), start_sec, duration_sec)


st.divider()

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
st.header("0. First, get baseline", anchor="step-0-baseline")
st.write(
    "Before testing whether audio-based similarity works, it's worth being clear about what "
    "it's being compared against."
)
st.write(
    "Of the two existing paradigms just described, this project tests against metadata-based "
    "similarity."
)
st.info(
    "ⓘ Because our dataset has no listen counts, ratings, or user-level interaction data to build "
    "it from, collaborative filtering isn't attempted here. See more in Methodology > Data."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 1: slicing the track into windows
# ---------------------------------------------------------------------------
st.header("1. Slicing the track into windows", anchor="step-1-slicing")
st.write("Songs don't sound the same throughout their entirety.")
st.write(
    "Looking at a whole song at once risks losing or averaging out what makes any single moment "
    "distinct. So we slice each song into overlapping "
    f"{WINDOW_SEC:.0f}-second windows -- {song_repo.count_segments()} of them across the whole "
    "library right now. Here are a few real windows from the song above, each one separately "
    "playable:"
)
demo_segments = song_repo.get_segments(demo_song.id)
sample_segments = demo_segments[:: max(1, len(demo_segments) // 4)][:4] if demo_segments else []

st.plotly_chart(
    waveform_figure(
        waveform_envelope(audio_path_for(demo_song)), title=demo_song.title,
        duration_sec=demo_song.duration_sec, color="rgb(99,110,250)",
        highlight_ranges=[(seg.start_sec, seg.end_sec) for seg in sample_segments] or None,
        highlight_labels=[f"{i + 1}" for i in range(len(sample_segments))] or None,
    ),
    width="stretch", key="step1_full_waveform",
)
if sample_segments:
    st.caption(
        "The four shaded, numbered regions above are exactly the four windows below -- each one a "
        "real, separately playable clip."
    )
    window_cols = st.columns(len(sample_segments))
    for i, (col, seg) in enumerate(zip(window_cols, sample_segments, strict=False)):
        with col:
            st.caption(f"**{i + 1}** · {seg.start_sec:.1f}-{seg.end_sec:.1f}s")
            st.audio(_window_clip_bytes(demo_song.id, seg.start_sec, seg.end_sec - seg.start_sec), format="audio/wav")
else:
    st.info("No segments available yet for this song.")

# st.caption(f"\"{demo_song.title}\" -- real amplitude shape across the whole clip, not a flat signal.")


st.divider()

# ---------------------------------------------------------------------------
# Step 2: seven ways of listening
# ---------------------------------------------------------------------------
st.header("2. Seven ways of listening", anchor="step-2-listening")
st.write(
    "A listener might be responding to one specific facet of a song -- the bass, the vocals, "
    "the overall vibe, background texture, structure -- without realizing it."
)
st.write(
    "Measuring several independent qualities, rather than one blended score, makes sense given "
    "that. But these seven aren't all the same *kind* of measurement -- they split into three "
    "genuinely different categories, by what audio they actually run on and what computes them:"
)

stem_meta_path = STEM_EXAMPLE_DIR / "meta.json"
has_real_stems = stem_meta_path.exists() and (STEM_EXAMPLE_DIR / "mix.wav").exists()
demo_mix_path = audio_path_for(demo_song)

if has_real_stems:
    stem_meta = json.loads(stem_meta_path.read_text(encoding="utf-8"))
    if stem_meta["title"] == demo_song.title:
        st.caption(
            f"All seven below use real audio from the same song -- \"{demo_song.title}\" by "
            f"{demo_song.artist} ({demo_song.genre_top}). Stem separation is a one-time, offline "
            "step (Demucs, notebooks/03), not something this page runs live."
        )
    else:
        st.caption(
            f"Whole-mix and Detected-content facets below use \"{demo_song.title}\"'s real audio, "
            f"same as elsewhere on this page. The Stems section uses real separated stems from a "
            f"different real song -- \"{stem_meta['title']}\" by {stem_meta['artist']} "
            f"({stem_meta['genre_top']}) -- separation is a one-time, offline step (Colab notebook "
            "03), not something this page runs live."
        )
else:
    st.info(
        "**Stems placeholder.** Isolated stem audio (vocal/drums/bass/instrumental) hasn't been "
        "extracted into the app yet -- the other two categories below already use real audio "
        "regardless (neither needs isolated stems). See this page's module docstring for exactly "
        "how to extract and drop in the real stem files."
    )

#NOTE FOR CLAUDE CODE: maybe we'll put each fact in a card
# -- Category 1: whole-mix measures ------------------------------------------
st.subheader("Whole-mix measures")
st.write(
    "Sound and Harmony each measure something about the mix as a whole, the way an actual "
    "listener hears it -- overall timbre and production character, or the song's harmonic "
    "backbone. Neither is about isolating one part; separating out a single instrument first "
    "would work against what they're trying to capture, not help it. That's what sets this "
    "category apart from Stems below, where isolating one part first is the entire point."
)


@st.cache_data(show_spinner=False)
def _chroma_for_display(song_id: int):
    from sonic_explorer.analysis.waveform_preview import chroma_for_display

    return chroma_for_display(audio_path_for(songs_by_id[song_id]))


@st.cache_data(show_spinner=False)
def _key_and_chords_for_display(song_id: int):
    from sonic_explorer.analysis.key_chord import estimate_chords, estimate_key

    chroma, times = _chroma_for_display(song_id)
    return estimate_key(chroma), estimate_chords(chroma, times)


@st.cache_data(show_spinner=False)
def _mel_spectrogram_for_display(song_id: int):
    from sonic_explorer.analysis.waveform_preview import mel_spectrogram_for_display

    return mel_spectrogram_for_display(audio_path_for(songs_by_id[song_id]))


whole_mix_cols = st.columns(2)
with whole_mix_cols[0]:
    mel_db, mel_times = _mel_spectrogram_for_display(demo_song.id)
    st.markdown("**Sound**")
    st.plotly_chart(
        mel_spectrogram_figure(mel_db, mel_times, height=100),
        width="stretch", key="step2_facet_Sound",
    )
    st.audio(str(demo_mix_path))
    st.caption("Overall timbre, instrumentation, production character.")
    st.caption(
        "This is a mel-spectrogram: the raw acoustic texture (frequency energy over time) Sound is "
        "computed *from*, shown for context -- not a view of what CLAP actually \"sees.\" CLAP's "
        "similarity is a learned, black-box representation with no raw visual proxy the way chroma "
        "is directly Harmony's input. The real way to judge Sound similarity is a paired comparison "
        "-- two songs side by side (see §6c's curated examples and Overview's two-pair demo), not "
        "reading this spectrogram alone."
    )
    st.caption("Runs on: Full mix. Computed by: CLAP.")
with whole_mix_cols[1]:
    chroma, chroma_times = _chroma_for_display(demo_song.id)
    key_estimate, chord_segments = _key_and_chords_for_display(demo_song.id)

    st.markdown(f"**Harmony** -- estimated key: **{key_estimate.tonic} {key_estimate.mode}**")
    st.caption("Key, chords, tonal color.")
    st.caption(
        f"Krumhansl-Schmuckler correlation: {key_estimate.correlation:.2f} (1.0 = perfect fit to the "
        "profile; a rough confidence, not a certainty -- real songs modulate and rarely fit one "
        "textbook key profile exactly)."
    )
    st.plotly_chart(
        chromagram_figure(chroma, chroma_times, height=100),
        width="stretch", key="step2_facet_Harmony",
    )
    chord_event = st.plotly_chart(
        chord_strip_figure(chord_segments), width="stretch", key="step2_harmony_chords", on_select="rerun",
    )
    harmony_start_time = 0.0
    if chord_event and chord_event.selection and chord_event.selection.points:
        harmony_start_time = float(chord_event.selection.points[0]["customdata"][0])
    st.audio(str(demo_mix_path), start_time=harmony_start_time)
    st.caption(
        "Click a chord segment below to listen from that moment (the chromagram above is display-"
        "only -- Plotly heatmaps don't support click-to-select the way the bar strip below does). "
        "Chord labels are estimated by matching each moment's chroma against major/minor triad "
        "templates, smoothed to avoid flickering between near-tied frames -- both real closed-form "
        "techniques on top of the same chroma data, not new audio processing."
    )
    st.caption("Runs on: Full mix. Computed by: Chroma features.")

# -- Category 2: stems --------------------------------------------------------
st.subheader("Stems")
st.write(
    "Genuinely isolated audio via Demucs source separation, then measured the same way Sound is "
    "(CLAP) -- a real, physically separate signal, not a whole-mix measurement pretending to "
    "isolate one part."
)
stem_defs = [
    ("Vocal", STEM_EXAMPLE_DIR / "vocal.wav", "Isolated voice timbre and delivery.", "rgb(239,85,59)"),
    ("Drums", STEM_EXAMPLE_DIR / "drums.wav", "Isolated drum/percussion pattern and timbre.", "rgb(171,99,250)"),
    ("Bass", STEM_EXAMPLE_DIR / "bass.wav", "Isolated bassline tone and pattern.", "rgb(255,161,90)"),
    ("Instrumental", STEM_EXAMPLE_DIR / "instrumental.wav", "Backing instrumentation with vocals removed.", "rgb(25,211,243)"),
]
stem_cols = st.columns(4)
for col, (name, path, desc, color) in zip(stem_cols, stem_defs, strict=False):
    with col:
        if has_real_stems:
            st.plotly_chart(
                waveform_figure(waveform_envelope(path), title=name, color=color, height=100),
                width="stretch", key=f"step2_facet_{name}",
            )
            st.audio(str(path))
        else:
            st.info(f"**{name}** -- placeholder, waiting on real isolated stem audio.")
        st.caption(desc)
        st.caption(f"Runs on: isolated {name.lower()} stem (Demucs). Computed by: CLAP.")

# -- Category 3: detected content ---------------------------------------------
st.subheader("Detected content")
st.write(
    "The six facets above all answer \"how similar do these sound?\" -- a continuous score. "
    "Sound Tags answers a different question: \"what's actually in this mix?\" A pretrained "
    "audio-tagging model (AST) listens to each ~5s window and names what it hears -- an "
    "instrument, a sound effect, a texture -- and those labels get embedded (via CLAP's text "
    "encoder) into the same searchable space every other facet uses. That's what makes literal, "
    "content-based search possible -- \"find songs with crow sounds\" isn't a query any of the "
    "other six facets could answer, since none of them name *what's* in a mix, only how the "
    "whole thing compares to something else."
)
demo_tags = [
    (label, score) for label, score in deserialize_tags(demo_song.sound_tags) if label not in GENERIC_TAG_LABELS
]
st.markdown("**Sound Tags**")
st.audio(str(demo_mix_path))
if demo_tags:
    st.markdown(" &nbsp; ".join(f"`{label}` {score:.0%}" for label, score in demo_tags))
else:
    st.caption("No tags detected above the confidence floor for this song.")
st.caption("Runs on: full mix. Computed by: AST (tagging), then CLAP's text encoder (embedding).")

if not demo_tags:
    st.caption(
        "Note: this pulls from a separate, older per-*song* column (`songs.sound_tags`, "
        "Methodology 7b's song-level 10s-clip tagging) that hasn't been backfilled for this "
        "specific song yet (`scripts/tag_songs.py` -- free, local AST tagging, no API key needed -- "
        "exists but hasn't been run against this song) -- unrelated to the per-segment facet index "
        "used for search, which is real and live regardless. That's why this demo song shows an "
        "empty list instead of its real detected tags."
    )

st.divider()

# ---------------------------------------------------------------------------
# Step 3: turning sound into an embedding
# ---------------------------------------------------------------------------
st.header("3. Turning sound into an embedding", anchor="step-3-embedding")
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
st.header("4. A second kind of similarity: Song DNA", anchor="step-4-song-dna")
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

st.write(
    "**Five more views, added since the original five above -- none of them a sixth radar-chart "
    "axis.** The radar's five numbers are corpus-normalized scalars: comparable 0-1 numbers "
    "specifically because every song's raw value is measured against the same library-wide range "
    "(`analysis.song_dna.DNANormalizer`, see Methodology for why it also clips). These five aren't "
    "that -- they're per-song shapes, events, or single derived readings, not values comparable "
    "across songs on a shared scale -- so they live as separate views inside the same Song DNA "
    "panel rather than being forced into a sixth spoke that wouldn't mean the same thing as the "
    "other five:"
)
st.write(
    "- **Chord progression** -- the exact same per-frame chord-template matching described above "
    "(`analysis.key_chord.estimate_chords`), rendered as a labeled strip of contiguous chord "
    "segments, computed live.\n"
    "- **Novelty curve** -- read directly from the already-computed, already-persisted structure "
    "artifact Song X-Ray's structural timeline also uses (`EmbeddingRepository."
    "get_structure_timeline`), not recomputed here. See Methodology for the exact kernel and "
    "window size.\n"
    "- **Loudness contour** -- a real RMS-energy-over-TIME curve, distinct from the single "
    "whole-song `energy` scalar above (which is that same curve collapsed to one mean value) -- "
    "computed live at a cheap, display-only sample rate.\n"
    "- **Repetition rate** -- one scalar derived from the self-similarity matrix Song X-Ray "
    "already computes: the mean similarity between every pair of non-identical moments in the "
    "song. No new audio analysis, purely a reduction over data this app already had.\n"
    "- **Beats detected** -- a real `librosa.beat.beat_track` count plus the average interval "
    "between beats, computed live. Deliberately labeled \"beats,\" not \"downbeats\": this app has "
    "no bar/measure-level meter-tracking model, and the label says so honestly rather than "
    "implying a beat count gives you bar lines for free."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 5: explaining it in plain language
# ---------------------------------------------------------------------------
st.header("5. Explaining it in plain language", anchor="step-5-explaining")
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

# ---------------------------------------------------------------------------
# Step 6: an experimental album-art pipeline (feature extraction + prompt
# generation done; image generation pending a manual Colab run)
# ---------------------------------------------------------------------------
st.header("6. From audio to image: an experimental album-art pipeline", anchor="step-6-album-art")
st.write(
    "This library has no album art -- FMA's metadata never included any, and nothing in the "
    "ingestion pipeline adds it (Song X-Ray's structure/sound/harmony fingerprint composite is "
    "the honest fallback used everywhere else in this app). A separate, experimental pipeline "
    "generates AI album art FROM the same real, already-computed audio features this page "
    "describes elsewhere -- not from a generic prompt, and not from the song's title/artist text."
)
st.write(
    "**Feature extraction (done):** brightness (spectral centroid), intensity (RMS energy), mood "
    "(major/minor key, the exact same live estimate step 2 uses), pace (tempo), and detected AST "
    "sound tags -- five signals this page already describes computing elsewhere, reused for this "
    "purpose rather than recomputed."
)
st.write(
    "**Deterministic prompt generation (done), no LLM involved:** each descriptor value maps to "
    "one of several pre-written evocative phrasings -- low brightness, for instance, might render "
    "as \"shrouded in shadow,\" \"dim, low-lit tones,\" or \"a dark, brooding palette\" -- picked at "
    "random but SEEDED by the song's own ID, so the same song gets the identical prompt on every "
    "re-run. Every phrase in a generated prompt traces back to one specific real detected feature "
    "(`sonic_explorer.analysis.album_art_prompt`), kept inspectable by design rather than a black "
    "box. A fixed style constraint -- abstract/textural only, no faces, no text, no literal "
    "band-photo imagery -- is appended to every prompt."
)
st.write(
    "**Image generation (pending a manual run):** the deterministic prompts are exported "
    "(`scripts/export_album_art_prompts.py`) and consumed by a separate Colab notebook "
    "(`notebooks/12_album_art_generation.ipynb`) running SD-Turbo via `diffusers` on a free Colab "
    "GPU -- deliberately never run inside this deployed app itself (no paid API, no GPU dependency "
    "for anything a live visitor triggers)."
)
st.info(
    "**Status:** feature extraction and deterministic prompt generation are built, tested, and "
    "have already run for real against the full deployed song set (233/233 songs, real prompts, "
    "real traceable phrases). Image generation is drafted but not yet run -- once that Colab batch "
    "job actually happens, this section gets updated with real generated art, not a mockup."
)

st.divider()

st.write("Next: **Methodology** walks through each of these steps in depth, with real evidence at every stage.")
nav_button("Continue to Methodology →", "pages/1_Methodology.py", key="nav_approach_to_methodology")
