import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from sonic_explorer.analysis.network_graph import build_similarity_graph
from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS
from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
from sonic_explorer.config import audio_path_for
from sonic_explorer.facets.fingerprint import composite_fingerprint, structure_fingerprint
from components.plotting import (
    composite_fingerprint_thumbnail,
    fingerprint_thumbnail,
    network_graph_figure,
    song_dna_radar_overlay,
)
from resources import build_dna_normalizer, get_repositories, show_data_source_banner

MOMENT_MATCHER_EXAMPLE_TITLE = "Cipralex (c/ Pulso)"
XRAY_EXAMPLE_TITLE = "Cipralex (c/ Pulso)"

st.set_page_config(page_title="Sonic Explorer", page_icon="\U0001F9ED", layout="wide")

song_repo, embedding_repo, retrieval_service = get_repositories()
all_songs = song_repo.list_songs()
songs_by_title = {s.title: s for s in all_songs}


def _find_song(title: str):
    if title in songs_by_title:
        return songs_by_title[title]
    for s in all_songs:
        if s.title.startswith(title[:20]):
            return s
    return None


st.title("App Walkthrough")
st.write(
    "The **Methodology** page covered how the library was analyzed and preprocessed. This page is "
    "different: it's a guided tour of the live app itself -- what you're actually looking at on each "
    "screen, how to read it, and what the shapes/colors mean -- using the real, currently-running "
    "components, not screenshots."
)
st.page_link("pages/0_Methodology.py", label="← Back to Methodology", icon="\U0001F52C")
show_data_source_banner()

if not all_songs:
    st.info("No songs in the library yet.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# 1. Explore -- the network graph
# ---------------------------------------------------------------------------
st.header("1. Explore -- the network graph")
st.write(
    "When you land on **Explore**, you see every song as a dot, connected by lines. Here's the same "
    "live graph (sound facet, whole library):"
)


@st.cache_data
def _walkthrough_explore_graph(_song_repo, _embedding_repo, facet_name, index_size):
    all_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo, facet_name=facet_name)
    result = build_similarity_graph(all_vectors)
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    nodes_df = pd.DataFrame([
        {
            "song_id": n.song_id, "x": n.x, "y": n.y, "cluster": n.cluster,
            "title": songs_by_id[n.song_id].title, "artist": songs_by_id[n.song_id].artist,
            "genre": songs_by_id[n.song_id].genre_top,
        }
        for n in result.nodes if n.song_id in songs_by_id
    ])
    return nodes_df, result.edges


explore_nodes, explore_edges = _walkthrough_explore_graph(
    song_repo, embedding_repo, "sound", embedding_repo.index_size("sound")
)

if not explore_nodes.empty:
    st.plotly_chart(
        network_graph_figure(explore_nodes, explore_edges), width="stretch", key="wt_explore_graph",
    )

st.markdown("""
**How to read it:**
- **Position** comes from a force-directed layout of a k-nearest-neighbor graph, not a direct
  projection -- each song is pulled toward its most similar neighbors and pushed apart from
  everything else. This is *not* the same kind of "position" as the Taste Map below.
- **A line between two nodes** means one song is genuinely among the other's top few nearest
  neighbors on the selected facet -- not every pair gets a line, only real close matches.
- **Color** is a K-means cluster computed on the same vectors, purely for visual grouping.
- **A tightly-packed region** is a pocket of songs that all sound alike on this facet. **A node
  sitting off on its own with few or no visible connections** is a genuine outlier -- nothing else
  in the library sounds much like it, on this facet specifically (switching facets can change this
  completely, since "similar" means something different per facet).
- Clicking a node on the live page opens its player and lets you queue up a random / looping /
  closest-match "next" track.
""")
st.page_link("pages/6_Explore.py", label="Open Explore →", icon="\U0001F310")

st.divider()

# ---------------------------------------------------------------------------
# 2. Taste Map
# ---------------------------------------------------------------------------
st.header("2. Taste Map -- the whole library, projected")
st.write(
    "**Taste Map** looks superficially similar to Explore -- another scatter of dots -- but the "
    "position means something different here:"
)


@st.cache_data
def _walkthrough_taste_map(_song_repo, _embedding_repo, cache_key, method):
    song_vectors = mean_pool_song_vectors(_song_repo, _embedding_repo)
    result = compute_taste_map(song_vectors, method=method)
    songs_by_id = {s.id: s for s in _song_repo.list_songs()}
    return pd.DataFrame([
        {
            "song_id": p.song_id, "x": p.x, "y": p.y, "cluster": str(p.cluster),
            "title": songs_by_id[p.song_id].title, "artist": songs_by_id[p.song_id].artist,
            "genre": songs_by_id[p.song_id].genre_top,
        }
        for p in result.points if p.song_id in songs_by_id
    ])


taste_df = _walkthrough_taste_map(song_repo, embedding_repo, embedding_repo.index_size("sound"), "pca")

if not taste_df.empty:
    fig = px.scatter(
        taste_df, x="x", y="y", color="cluster",
        hover_data={"title": True, "artist": True, "genre": True, "x": False, "y": False, "cluster": False},
        title=f"{len(taste_df)} songs, PCA projection, colored by cluster",
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch", key="wt_taste_map_live")

st.markdown("""
**How to read it:**
- **Position** here comes from PCA -- the two directions of maximum spread (variance) across the
  whole library's sound embeddings. Unlike Explore's graph layout, these axes are, in principle,
  *nameable*: the live page's "Inspect these axes" expander shows you the actual songs sitting at
  each axis's extremes so you can judge for yourself whether an axis resolves into something you'd
  call e.g. "quiet/ambient vs. loud/aggressive," or doesn't resolve into anything nameable at all --
  that's a real, open question, not a guaranteed property of PCA.
- **Point density** reflects the real data: a crowded region means the library genuinely has many
  similar-sounding songs there; sparse regions mean that sonic territory is thinly covered.
- **Switching the "Color by" toggle** between cluster and genre on the live page is the actual test
  of whether sound and genre agree -- where cluster boundaries track genre boundaries, they do;
  where a cluster spans several genre colors, the audio is telling you something the label doesn't.
- Click any point on the live page to hear it immediately.
""")
st.page_link("pages/2_Taste_Map.py", label="Open Taste Map →", icon="\U0001F5FA️")

st.divider()

# ---------------------------------------------------------------------------
# 3. Song X-Ray
# ---------------------------------------------------------------------------
st.header("3. Song X-Ray -- one song's anatomy")
st.write(
    "Pick any song on the live page and it shows you that song's fingerprints, its segmented "
    "structure, and where it sits on the Taste Map. Here's a live example:"
)

xray_song = _find_song(XRAY_EXAMPLE_TITLE)
if xray_song is not None:
    st.markdown(f"**{xray_song.title}** — {xray_song.artist} ({xray_song.genre_top})")
    st.audio(str(audio_path_for(xray_song)))

    try:
        matrix = embedding_repo.get_structure_matrix(xray_song.id)
    except FileNotFoundError:
        matrix = None
    try:
        timeline = embedding_repo.get_structure_timeline(xray_song.id)
    except FileNotFoundError:
        timeline = None

    structure_fp = structure_fingerprint(matrix) if matrix is not None else None
    sound_fp = timeline.sound_fingerprint if timeline is not None else None
    harmony_fp = timeline.harmony_fingerprint if timeline is not None else None

    fp_cols = st.columns(4)
    if structure_fp is not None:
        with fp_cols[0]:
            st.plotly_chart(fingerprint_thumbnail(structure_fp, "Structure"), width="stretch", key="wt2_fp_structure")
    if sound_fp is not None:
        with fp_cols[1]:
            st.plotly_chart(fingerprint_thumbnail(sound_fp, "Sound"), width="stretch", key="wt2_fp_sound")
    if harmony_fp is not None:
        with fp_cols[2]:
            st.plotly_chart(fingerprint_thumbnail(harmony_fp, "Harmony"), width="stretch", key="wt2_fp_harmony")
    if structure_fp is not None and sound_fp is not None and harmony_fp is not None:
        with fp_cols[3]:
            composite = composite_fingerprint(structure_fp, sound_fp, harmony_fp)
            st.plotly_chart(composite_fingerprint_thumbnail(composite), width="stretch", key="wt2_fp_composite")

st.markdown("""
**How to read it:**
- Each fingerprint is a small heatmap; **brighter = higher value** in that facet's own space
  (structure = self-similarity, sound = mel-spectrogram energy, harmony = chroma strength).
- The **composite** overlays all three as red/green/blue channels. Two songs with similar-looking
  composites are worth comparing directly -- it's a fast visual pre-filter before you go listen.
- On the live page, the **structure timeline** below the fingerprints shows colored blocks (or, for
  songs without clear repetition, a novelty curve instead) -- **same color = similar-sounding
  section**, and clicking a block loops just that stretch of audio.
- The **raw self-similarity matrix**, tucked in an expander, is the actual data everything above is
  derived from -- bright diagonal-parallel stripes mark repeated sections; a mostly-flat matrix
  means the song doesn't repeat within the clip.
""")
st.page_link("pages/3_Song_XRay.py", label="Open Song X-Ray →", icon="\U0001F50D")

st.divider()

# ---------------------------------------------------------------------------
# 4. Moment Matcher
# ---------------------------------------------------------------------------
st.header("4. Moment Matcher -- finding a match, one moment at a time")
st.write(
    "Moment Matcher has two modes: pick an existing moment in a song and find sonically similar "
    "moments elsewhere (by any facet), or hand-draw a target DNA profile and find whole songs "
    "closest to that shape. Here's what a real \"existing moment\" match looks like:"
)

mm_song = _find_song(MOMENT_MATCHER_EXAMPLE_TITLE)
mm_shown = False
if mm_song is not None:
    segments = song_repo.get_segments(mm_song.id)
    if segments:
        query_seg = segments[len(segments) // 2]
        if embedding_repo.status(query_seg.id, "sound") == "done":
            matches = retrieval_service.query_by_segment(query_seg.id, facet_name="sound", k=1)
            if matches:
                match = matches[0]
                mm_shown = True
                pct = max(0.0, match.score) * 100
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.caption(f"Query moment: \"{mm_song.title}\" at {query_seg.start_sec:.1f}s–{query_seg.end_sec:.1f}s")
                    st.audio(str(audio_path_for(mm_song)), start_time=query_seg.start_sec)
                with mcol2:
                    st.caption(f"**{pct:.0f}% sound match:** \"{match.song.title}\" — {match.song.artist}")
                    st.audio(str(audio_path_for(match.song)), start_time=match.segment.start_sec)

                query_raw = {axis: getattr(mm_song, axis) for axis in AXES}
                match_raw = {axis: getattr(match.song, axis) for axis in AXES}
                if all(v is not None for v in query_raw.values()) and all(v is not None for v in match_raw.values()):
                    dna_normalizer = build_dna_normalizer(song_repo, len(all_songs))
                    query_norm = dna_normalizer.normalize(query_raw)
                    match_norm = dna_normalizer.normalize(match_raw)
                    st.plotly_chart(
                        song_dna_radar_overlay(
                            [AXIS_LABELS[a] for a in AXES],
                            [query_norm[a] for a in AXES], mm_song.title,
                            [match_norm[a] for a in AXES], match.song.title,
                        ),
                        width="stretch", key="wt_mm_radar",
                    )

if not mm_shown:
    st.info("Live example unavailable -- open Moment Matcher directly to try it.")

st.markdown("""
**How to read it:**
- The **match percentage** is cosine similarity between the two moments' embeddings on the selected
  facet, scaled to 0-100% -- it's relative to *this facet's* space, not a universal "how similar"
  score, which is exactly why switching the facet radio button can reorder the results entirely.
- The **"Compare song DNA" radar chart** overlays the query and match's normalized profiles: where
  the two shapes overlap, the songs agree on that quality (tempo, energy, brightness, harmonic
  complexity, rhythmic density); where one bulges past the other, they diverge on that one axis
  specifically -- a high sound-match score doesn't guarantee the DNA shapes agree too, and it's
  worth checking both.
- **Hand-drawn profile mode** runs the same nearest-neighbor logic, just against a target you sculpt
  with sliders instead of an existing song's real values.
- If a facet-appropriate explanation client is configured, each match also gets a one-sentence,
  plain-language reason generated by the same LLM layer used throughout the app.
""")
st.page_link("pages/4_Moment_Matcher.py", label="Open Moment Matcher →", icon="\U0001F3AF")

st.divider()

# ---------------------------------------------------------------------------
# 5. Ask the DJ
# ---------------------------------------------------------------------------
st.header("5. Ask the DJ -- a conversational front-end")
st.write(
    "Ask the DJ doesn't run its own analysis -- it's a conversational layer *over* Moment Matcher "
    "and the Taste Map. Not run live on this page (each turn is a real, billed LLM call, so it's not "
    "triggered just by loading a walkthrough page), here's an illustrative example of what happens "
    "under the hood for a real request:"
)

with st.container(border=True):
    st.markdown("**You:** *\"Find me something moodier and more stripped-back than Midnight Drive\"*")
    st.caption(
        "1. The model calls `get_song_profile` to fetch \"Midnight Drive\"'s real tempo/energy/"
        "brightness/harmonic-complexity/rhythmic-density values.\n"
        "2. It reasons about which axes \"moodier and more stripped-back\" implies (lower energy, "
        "lower rhythmic density) and nudges those specific numbers down.\n"
        "3. It calls `search_by_mood_profile` with the adjusted target -- the exact same "
        "nearest-neighbor search Moment Matcher's hand-drawn mode uses.\n"
        "4. It translates the raw results into plain language, deliberately never mentioning "
        "\"embeddings,\" \"cosine similarity,\" or facet names."
    )
    st.markdown(
        "**DJ:** *\"Try 'Low Tide' by Nettle Grove -- it keeps a similar late-night feel but pulls "
        "back the percussion and sits at a noticeably calmer energy.\"*"
    )

st.caption(
    "Every tool result is real data pulled from the same repositories every other page uses -- the "
    "model is instructed to only report what a tool call actually returned, never to invent titles, "
    "artists, or match results."
)
st.page_link("pages/5_Ask_The_DJ.py", label="Open Ask the DJ →", icon="\U0001F399️")

st.divider()

# ---------------------------------------------------------------------------
# 6. All live pages
# ---------------------------------------------------------------------------
st.header("6. All live pages")

cta_cols = st.columns(5)
with cta_cols[0]:
    st.page_link("pages/6_Explore.py", label="**Explore**", icon="\U0001F310")
    st.caption("Every song as a node in a network graph. Click one to open the player.")
with cta_cols[1]:
    st.page_link("pages/2_Taste_Map.py", label="**Taste Map**", icon="\U0001F5FA️")
    st.caption("A 2D map of the library clustered by sound. Click a point to hear it.")
with cta_cols[2]:
    st.page_link("pages/3_Song_XRay.py", label="**Song X-Ray**", icon="\U0001F50D")
    st.caption("A song's structural anatomy, fingerprints, and DNA in one place.")
with cta_cols[3]:
    st.page_link("pages/4_Moment_Matcher.py", label="**Moment Matcher**", icon="\U0001F3AF")
    st.caption("Pick a moment, get ranked matches on any facet, with explanations.")
with cta_cols[4]:
    st.page_link("pages/5_Ask_The_DJ.py", label="**Ask the DJ**", icon="\U0001F399️")
    st.caption("Describe what you want in plain language -- a conversational front-end over it all.")
