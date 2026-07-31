import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sonic_explorer.analysis.key_chord import estimate_chords
from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from sonic_explorer.analysis.waveform_preview import (
    beat_times_for_song,
    chord_tone_audio,
    chroma_for_display,
    click_track_audio,
)
from sonic_explorer.config import album_art_path_for, audio_path_for
from sonic_explorer.facets.fingerprint import composite_fingerprint, structure_fingerprint
from components.plotting import chord_strip_figure, composite_fingerprint_thumbnail, fingerprint_thumbnail
from resources import get_repositories, hero_banner, nav_button, show_data_source_banner, show_logo

st.set_page_config(page_title="Song X-Ray")
hero_banner("song_xray")
st.title("Song X-Ray")
st.caption("A song's structural anatomy -- matching colors below mean similar-sounding sections.")
nav_button("← Back to Explore", "pages/7_Explore.py", key="nav_xray_to_explore")

show_logo()
show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()

songs = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
if not songs:
    st.info("No songs in the library yet.")
    st.stop()

# Reached from Explore's "Open full Song X-Ray" button, which stashes the
# clicked song's id here before switching pages -- popped so it only applies
# once; a plain sidebar/URL visit with no context still defaults to index 0.
context_song_id = st.session_state.pop("xray_context_song_id", None)
default_index = 0
if context_song_id is not None:
    for i, s in enumerate(songs):
        if s.id == context_song_id:
            default_index = i
            break

labels = [f"{s.title} — {s.artist} ({s.genre_top})" for s in songs]
choice = st.selectbox("Pick a song", options=range(len(songs)), index=default_index, format_func=lambda i: labels[i])
song = songs[choice]

st.subheader(f"{song.title} — {song.artist}")
st.caption(f"Genre: {song.genre_top}")
if song.description:
    st.caption(f"*{song.description}*")
st.audio(str(audio_path_for(song)))

# Real AI-generated album art (Approach step 6: real per-song audio
# features -> deterministic prompt -> SD-Turbo) -- returns None (not a
# guessed path) for any song it wasn't generated for, so this section
# simply doesn't render rather than showing a broken image. A plain
# st.image() with a real file path is enough here (unlike Explore's raw
# <img> tag + base64 data URI, which exists specifically for a hover
# tooltip) -- Song X-Ray's own documentational style already uses
# st.image()'s built-in caption for this kind of explanatory text.
album_art_path = album_art_path_for(song)
if album_art_path is not None:
    st.image(
        str(album_art_path), width=220,
        caption="AI-generated album art -- built from this song's own real detected audio features "
                "(brightness, energy, key, tempo, sound tags), not a generic prompt. See Approach "
                "step 6 for exactly how.",
    )

# Fetched once, used by both the fingerprint row and the timeline chart below.
try:
    timeline = embedding_repo.get_structure_timeline(song.id)
except FileNotFoundError:
    timeline = None

try:
    matrix = embedding_repo.get_structure_matrix(song.id)
except FileNotFoundError:
    matrix = None

structure_fp = structure_fingerprint(matrix) if matrix is not None else None
sound_fp = timeline.sound_fingerprint if timeline is not None else None
harmony_fp = timeline.harmony_fingerprint if timeline is not None else None

if structure_fp is not None or sound_fp is not None or harmony_fp is not None:
    st.markdown("#### Fingerprint")
    st.caption(
        "A visual identity for the song, derived from its structure, sound, and harmony -- also usable "
        "as an album-art fallback. The composite overlays all three as color channels: where they agree "
        "the image reads bright, where they diverge, distinct color casts appear."
    )
    cols = st.columns(4)
    if structure_fp is not None:
        with cols[0]:
            st.plotly_chart(fingerprint_thumbnail(structure_fp, "Structure"), width=180, height=180, key="fp_structure")
    if sound_fp is not None:
        with cols[1]:
            st.plotly_chart(fingerprint_thumbnail(sound_fp, "Sound"), width=180, height=180, key="fp_sound")
    if harmony_fp is not None:
        with cols[2]:
            st.plotly_chart(fingerprint_thumbnail(harmony_fp, "Harmony"), width=180, height=180, key="fp_harmony")
    if structure_fp is not None and sound_fp is not None and harmony_fp is not None:
        with cols[3]:
            composite = composite_fingerprint(structure_fp, sound_fp, harmony_fp)
            st.plotly_chart(composite_fingerprint_thumbnail(composite), width=180, height=180, key="fp_composite")


@st.cache_data(show_spinner=False)
def _cached_click_track(song_id: int, audio_path: str):
    beats = beat_times_for_song(audio_path)
    return click_track_audio(audio_path, beats), beats


@st.cache_data(show_spinner=False)
def _cached_chords_for_xray(song_id: int, audio_path: str, tempo_bpm: float | None):
    chroma, times = chroma_for_display(audio_path)
    return estimate_chords(chroma, times, tempo_bpm=tempo_bpm)


@st.cache_data(show_spinner=False)
def _cached_chord_tone_track(song_id: int, audio_path: str, chords: list):
    return chord_tone_audio(audio_path, chords)


with st.expander("Verify beat & chord detection by ear"):
    st.caption(
        "Two click-track-style overlays -- the song's own audio with a synthesized marker mixed in -- so "
        "beat and chord detection can each be judged directly by ear rather than trusting a number or a "
        "label alone."
    )
    # Gated behind an explicit opt-in, not generated automatically on every
    # page load -- a REAL, measured cost, not a theoretical one: st.expander
    # has no lazy-content mechanism in Streamlit (the block inside always
    # executes regardless of collapsed/expanded state), and this section's
    # own generation -- two full-song librosa.load() + synthesis passes,
    # beat_times_for_song + click_track_audio + chroma_for_display +
    # estimate_chords + chord_tone_audio -- measured at ~87s for a cold
    # (uncached) song before this gate existed. st.cache_data still makes a
    # repeat visit to the SAME song instant either way; this gate only
    # avoids paying that cost for every song a visitor merely browses past
    # without ever caring about this QA tool.
    show_verification_audio = st.checkbox(
        "Generate verification audio for this song",
        key="xray_show_verification_audio",
        help="Real audio decoding + synthesis, a few real seconds per song the first time -- off by "
             "default so browsing songs stays fast; stays on for the rest of this session once enabled.",
    )
    if not show_verification_audio:
        st.caption("Enable the checkbox above to generate the click track and chord tone track for this song.")
    else:
        xray_audio_path = str(audio_path_for(song))

        st.markdown("###### Beat click track")
        st.caption("A short tick mixed in at every detected beat -- listen for a tick that lands off the beat.")
        with st.spinner("Generating click track…", show_time=False):
            click_audio_bytes, xray_beat_times = _cached_click_track(song.id, xray_audio_path)
        if not click_audio_bytes:
            st.warning("Could not generate a click track for this song (no audio decoded).")
        else:
            st.audio(click_audio_bytes)
            st.caption(f"{len(xray_beat_times)} beats detected.")

        st.markdown("###### Chord tone track")
        xray_chords = _cached_chords_for_xray(song.id, xray_audio_path, song.tempo_bpm)
        if not xray_chords:
            st.caption("No chords detected (near-silent or unpitched audio).")
        else:
            st.caption(
                "A soft synthesized triad (root/third/fifth, same intervals the detector itself matched "
                "against) mixed in under the song for each detected chord -- listen for a tone that clashes "
                "with the real harmony, not just a label that looks plausible."
            )
            with st.spinner("Generating chord tone track…", show_time=False):
                chord_tone_bytes = _cached_chord_tone_track(song.id, xray_audio_path, xray_chords)
            if not chord_tone_bytes:
                st.warning("Could not generate a chord tone track for this song (no audio decoded).")
            else:
                st.audio(chord_tone_bytes)
            st.plotly_chart(
                chord_strip_figure(xray_chords), width="stretch", key="xray_chord_verify_strip",
            )

st.markdown("#### Structure timeline")

selected_segment_idx = None
if timeline is None:
    st.warning("No structure timeline computed for this song yet.")
elif not timeline.has_clear_structure:
    # Not every song has clean verse/chorus repetition -- forcing a segmented
    # timeline onto an ambient/through-composed track would show either one
    # meaningless block or noisy fake boundaries. The novelty curve says so
    # honestly instead of hiding the limitation.
    st.caption(
        "This song evolves gradually rather than repeating in clear sections -- shown below as a "
        "continuous curve instead of colored blocks. Peaks are moments that sound most different "
        "from what came just before; none were sharp enough to count as a clear section change."
    )
    if timeline.novelty_curve is not None:
        curve_fig = go.Figure(go.Scatter(
            x=timeline.novelty_times, y=timeline.novelty_curve, mode="lines", fill="tozeroy",
            line=dict(color="rgb(99,110,250)"),
        ))
        curve_fig.update_layout(
            height=180,
            xaxis_title="Time (s)",
            yaxis=dict(title="novelty", showticklabels=False, range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(curve_fig, width="stretch", key="novelty_curve_chart")
else:
    st.caption(
        "Each colored block is a stretch of the song. Same color = similar-sounding sections "
        "(e.g. a verse repeating later) -- discovered automatically from the audio, not labeled by us. "
        "Click a block to loop just that section."
    )
    palette = px.colors.qualitative.Set2
    unique_labels = sorted(set(timeline.segment_labels.tolist()))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    durations = timeline.segment_ends - timeline.segment_starts
    hover_text = [f"{s:.1f}s – {e:.1f}s" for s, e in zip(timeline.segment_starts, timeline.segment_ends, strict=False)]

    timeline_fig = go.Figure(go.Bar(
        x=durations,
        y=["Structure"] * len(durations),
        base=timeline.segment_starts,
        orientation="h",
        marker_color=[color_map[lab] for lab in timeline.segment_labels.tolist()],
        marker_line_width=0,
        customdata=[[i] for i in range(len(durations))],
        hovertext=hover_text,
        hoverinfo="text",
    ))
    timeline_fig.update_layout(
        height=140,
        showlegend=False,
        xaxis_title="Time (s)",
        yaxis=dict(showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=40),
        bargap=0,
    )
    event = st.plotly_chart(timeline_fig, width="stretch", on_select="rerun", key="structure_timeline_chart")

    if event and event.selection and event.selection.points:
        selected_segment_idx = event.selection.points[0]["customdata"][0]

    if selected_segment_idx is not None:
        seg_start = float(timeline.segment_starts[selected_segment_idx])
        seg_end = float(timeline.segment_ends[selected_segment_idx])
        st.markdown(f"**Looping section:** {seg_start:.1f}s – {seg_end:.1f}s")
        st.audio(str(audio_path_for(song)), start_time=seg_start, end_time=seg_end, loop=True)

        retrieval_segments = song_repo.get_segments(song.id)
        if retrieval_segments and st.button("Find similar moments →", key="find_similar_moments_btn"):
            # Structure blocks (variable-length, self-similarity-derived) are a
            # different segmentation from the fixed ~5s/2.5s-hop segments
            # Moment Matcher searches over -- map by closest start time rather
            # than assuming they line up.
            nearest = min(retrieval_segments, key=lambda seg: abs(seg.start_sec - seg_start))
            st.session_state["mm_context"] = {"song_id": song.id, "segment_id": nearest.id}
            st.switch_page("pages/5_Moment_Matcher.py")
    else:
        st.caption("Click a colored block above to loop just that section.")

st.markdown("#### Position in the Taste Map")


@st.cache_data
def build_taste_map_df(_song_repo, _embedding_repo, cache_key):
    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    result = compute_taste_map(song_vectors)
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    return pd.DataFrame([
        {
            "song_id": p.song_id, "x": p.x, "y": p.y, "cluster": str(p.cluster),
            "title": songs_by_id[p.song_id].title, "genre": songs_by_id[p.song_id].genre_top,
        }
        for p in result.points
        if p.song_id in songs_by_id
    ])


taste_df = build_taste_map_df(song_repo, embedding_repo, embedding_repo.index_size("sound"))
this_song = taste_df[taste_df["song_id"] == song.id]

if not this_song.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=taste_df["x"], y=taste_df["y"], mode="markers",
        marker=dict(size=7, color="lightgray"), name="library", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=this_song["x"], y=this_song["y"], mode="markers",
        marker=dict(size=16, color="crimson", symbol="star"), name=song.title,
    ))
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("Not enough embedded segments yet to place this song on the Taste Map.")

with st.expander("Technical detail: raw self-similarity matrix"):
    st.caption(
        "The matrix the timeline above is derived from. The main diagonal is deliberately left blank -- "
        "bright parallel stripes off the diagonal are what mark repeated sections."
    )
    if matrix is not None:
        heatmap = px.imshow(
            matrix, color_continuous_scale="Magma", origin="lower",
            labels=dict(x="beat", y="beat", color="similarity"),
        )
        heatmap.update_layout(height=420)
        st.plotly_chart(heatmap, width="stretch")
    else:
        st.warning("No structure matrix computed for this song yet.")
