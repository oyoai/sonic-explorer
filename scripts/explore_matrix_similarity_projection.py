"""Exploratory only -- NOT wired into the app or retrieval, and not a new
facet. Quick sanity check: if two songs' Structure/Sound/Harmony self-
similarity-matrix fingerprints LOOK alike to a human eye, do they actually
land near each other once flattened and projected to 2D? A real signal here
would be evidence a matrix-similarity-based facet is worth developing
further; no signal (or a signal that's actually just "PCA rediscovered
overall image brightness") means it isn't, at least not via this naive an
approach.

Method, deliberately the simplest version, per instruction: flatten each
song's fingerprint directly (no summarization/pooling), PCA to 2D (not
UMAP -- scikit-learn is already a dependency here, umap-learn isn't, and
PCA is the simpler, more interpretable first look this task asked for),
plot as a real image scatter -- each song's own fingerprint thumbnail
rendered at its projected coordinate, not a plain dot, via Plotly's
fig.add_layout_image() (one real image per point, positioned in data
coordinates) -- Plotly rather than matplotlib to match every other chart
in this codebase and avoid adding a new dependency for a one-off script;
this script also deliberately does NOT import anything from streamlit_app/
(see CLAUDE.md's environment-dependent-import discipline -- scripts/ never
imports from streamlit_app/ at runtime), so the fingerprint->image
conversion is done locally here rather than reusing components/plotting.py's
version.

Run: .venv/Scripts/python.exe scripts/explore_matrix_similarity_projection.py
Outputs three interactive HTML files (one per facet, real hover-zoomable
Plotly figures) plus a printed nearest-neighbor sanity check against a real
demo song."""

import sys
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH  # noqa: E402
from sonic_explorer.facets.fingerprint import structure_fingerprint  # noqa: E402
from sonic_explorer.repository.db import init_db  # noqa: E402
from sonic_explorer.repository.embedding_repository import EmbeddingRepository  # noqa: E402
from sonic_explorer.repository.song_repository import SongRepository  # noqa: E402

OUTPUT_DIR = Path(
    r"C:\Users\Study\AppData\Local\Temp\claude\c--Users-Study-Documents-AI-ENG-Course-Project"
    r"\25ddf040-691b-4af0-ba49-22d0bd3866c9\scratchpad"
)
SANITY_CHECK_SONG_TITLE = "5am, Wabi Sabi"
N_NEIGHBORS_TO_REPORT = 8
THUMB_PX = 48  # each plotted fingerprint's on-screen size, in data-coordinate-relative units below


def _fingerprint_to_data_uri(fingerprint: np.ndarray) -> str:
    """Same colorscale-sampling + flip technique components/plotting.py's
    fingerprint_thumbnail_image uses (Magma, row-0-at-bottom -> row-0-at-
    top for image display), reimplemented locally so this script has zero
    streamlit_app/ imports."""
    import base64
    import io

    from PIL import Image

    flipped = np.flipud(fingerprint)
    norm = (flipped - flipped.min()) / (flipped.max() - flipped.min() + 1e-9)
    sampled = pc.sample_colorscale("Magma", norm.ravel().tolist())
    rgb = np.array([[int(c) for c in color[4:-1].split(", ")] for color in sampled], dtype=np.uint8)
    rgb = rgb.reshape(norm.shape[0], norm.shape[1], 3)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _collect_fingerprints(song_repo, embedding_repo):
    """One (song, structure_fp, sound_fp, harmony_fp) tuple per song that
    has all three -- same three fingerprints Explore's own Song DNA
    "Self-similarity matrices" section already shows, just gathered across
    the whole library here instead of one song at a time."""
    rows = []
    for song in song_repo.list_songs():
        try:
            matrix = embedding_repo.get_structure_matrix(song.id)
        except FileNotFoundError:
            continue
        try:
            timeline = embedding_repo.get_structure_timeline(song.id)
        except FileNotFoundError:
            continue
        if timeline.sound_fingerprint is None or timeline.harmony_fingerprint is None:
            continue
        rows.append((song, structure_fingerprint(matrix), timeline.sound_fingerprint, timeline.harmony_fingerprint))
    return rows


def _image_scatter(coords: np.ndarray, images: list[np.ndarray], titles: list[str], facet_label: str, out_path: Path):
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    thumb_size_x = x_range / THUMB_PX
    thumb_size_y = y_range / THUMB_PX

    fig = go.Figure(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers",
        marker=dict(size=1, opacity=0),  # invisible -- real hover target underneath the images
        hovertext=titles, hoverinfo="text",
    ))
    for (x, y), img in zip(coords, images, strict=True):
        fig.add_layout_image(
            # xref/yref default to "paper" (fixed relative to the whole
            # canvas) if not set -- ties each image to the actual data axes
            # instead, so it pans/zooms together with the grid underneath
            # it rather than staying glued in place.
            source=_fingerprint_to_data_uri(img), x=x, y=y, xref="x", yref="y",
            sizex=thumb_size_x, sizey=thumb_size_y,
            xanchor="center", yanchor="middle", layer="above",
        )
    fig.update_layout(
        title=f"{facet_label} fingerprints, flattened + PCA'd to 2D (image = each song's real fingerprint)",
        xaxis_title="PC1", yaxis_title="PC2", width=1400, height=1400,
        plot_bgcolor="rgba(20,20,20,1)",
    )
    fig.write_html(str(out_path))
    print(f"Wrote {out_path}")


def _nearest_neighbors_report(coords: np.ndarray, songs: list, target_title: str, facet_label: str):
    target_idx = next((i for i, s in enumerate(songs) if s.title == target_title), None)
    if target_idx is None:
        print(f"[{facet_label}] '{target_title}' not found among songs with all three fingerprints available.")
        return
    target_xy = coords[target_idx]
    dists = np.linalg.norm(coords - target_xy, axis=1)
    order = np.argsort(dists)
    order = [i for i in order if i != target_idx][:N_NEIGHBORS_TO_REPORT]
    print(f"\n[{facet_label}] 2D-projection nearest neighbors of '{target_title}':")
    for i in order:
        print(f"    {dists[i]:.3f}  {songs[i].title} -- {songs[i].artist} ({songs[i].genre_top})")


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)

    rows = _collect_fingerprints(song_repo, embedding_repo)
    print(f"{len(rows)} songs have all three fingerprints available.")
    songs = [r[0] for r in rows]

    for facet_label, fp_index in [("Structure", 1), ("Sound", 2), ("Harmony", 3)]:
        images = [r[fp_index] for r in rows]
        flattened = np.stack([img.ravel().astype(np.float64) for img in images])
        coords = PCA(n_components=2, random_state=42).fit_transform(flattened)

        out_path = OUTPUT_DIR / f"matrix_similarity_projection_{facet_label.lower()}.html"
        _image_scatter(coords, images, [s.title for s in songs], facet_label, out_path)
        _nearest_neighbors_report(coords, songs, SANITY_CHECK_SONG_TITLE, facet_label)


if __name__ == "__main__":
    main()
