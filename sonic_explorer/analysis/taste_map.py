"""PCA/ICA + K-means over per-song mean-pooled sound embeddings. Plain Python,
no Streamlit import anywhere in this file -- see spec 8.3's core/interface
separation.

PCA finds the directions of maximum variance; ICA finds statistically
independent directions instead -- a real, open question (spec section 6) is
whether ICA's independent components land on more individually-nameable
qualities ("sound axes") than PCA's variance-maximizing ones, or are just as
opaque in practice. compute_taste_map() supports both via `method` so the UI
can show them side by side rather than picking one -- see
streamlit_app/pages/5_Explore.py's "2D map" view and its axis-inspection
expander (backed by correlate_axes_with_features() below), the actual
evaluation surface for that question."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA

from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


def mean_pool_song_vectors(
    song_repo: SongRepository, embedding_repo: EmbeddingRepository, facet_name: str = "sound"
) -> dict[int, np.ndarray]:
    """One vector per song (not per segment) -- segment-level would put ~11
    near-duplicate points per song on the map and clutter a library-level view."""
    song_vectors: dict[int, np.ndarray] = {}
    for song in song_repo.list_songs():
        segments = song_repo.get_segments(song.id)
        vectors = [
            embedding_repo.get_vector(facet_name, seg.id)
            for seg in segments
            if embedding_repo.status(seg.id, facet_name) == "done"
        ]
        if vectors:
            song_vectors[song.id] = np.mean(vectors, axis=0)
    return song_vectors


def similarity_weighted_pool_song_vectors(
    song_repo: SongRepository, embedding_repo: EmbeddingRepository, facet_name: str = "sound"
) -> dict[int, np.ndarray]:
    """Alternative to mean_pool_song_vectors() -- same signature/output shape
    (one vector per song), but each segment is weighted by how representative
    it is of the song's *other* segments (its mean cosine similarity to them),
    rather than every segment counting equally. Deliberately favors typical/
    central segments over outliers: this function's whole purpose (like
    mean_pool_song_vectors', which it's a drop-in alternative to) is a
    coarse, denoised, library-level summary for views that need one point per
    song (Explore's graph, Taste Map) -- not a "most distinctive moment"
    detector, which is what Moment Matcher's segment-level search already is,
    deliberately kept separate. Favoring outliers here would also conflate
    "genuinely distinctive" with "just a noisy transition/silence" -- cosine
    distance from a song's other segments can't tell those apart, so
    amplifying outliers risks polluting the summary with noise as often as
    with something meaningful. Not currently used by anything by default --
    a candidate to evaluate against mean_pool_song_vectors on genre-cohesion
    before adopting it anywhere (see evaluation/genre_cohesion.py's
    song_level_genre_cohesion_at_k, which accepts precomputed vectors for
    exactly this comparison)."""
    song_vectors: dict[int, np.ndarray] = {}
    for song in song_repo.list_songs():
        segments = song_repo.get_segments(song.id)
        vectors = [
            embedding_repo.get_vector(facet_name, seg.id)
            for seg in segments
            if embedding_repo.status(seg.id, facet_name) == "done"
        ]
        if not vectors:
            continue
        if len(vectors) == 1:
            song_vectors[song.id] = vectors[0]
            continue

        matrix = np.stack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        sims = normalized @ normalized.T
        np.fill_diagonal(sims, np.nan)
        # Each segment's weight = its mean similarity to the song's *other*
        # segments -- central/typical segments (similar to the rest) score
        # higher, outliers score lower. Clipped at 0 and renormalized so a
        # degenerate all-negative-similarity song still falls back to a
        # plain (uniform-weight) mean rather than dividing by ~0 or flipping
        # sign.
        weights = np.clip(np.nanmean(sims, axis=1), 0.0, None)
        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(vectors))
            total = float(len(vectors))
        weights = weights / total

        song_vectors[song.id] = np.average(matrix, axis=0, weights=weights)
    return song_vectors


@dataclass
class TasteMapPoint:
    song_id: int
    x: float
    y: float
    cluster: int


@dataclass
class TasteMapResult:
    points: list[TasteMapPoint]
    # (PC1, PC2) explained-variance ratios -- only meaningful for PCA, whose
    # components are ordered by variance explained. FastICA's components
    # aren't ordered/scaled that way at all (independence, not variance, is
    # what it optimizes for), so this is always None for method="ica" rather
    # than reporting a number that would misleadingly imply the same
    # diagnostic applies.
    explained_variance_ratio: tuple[float, float] | None = None


def compute_taste_map(
    song_vectors: dict[int, np.ndarray], n_clusters: int = 8, random_state: int = 42, method: str = "pca"
) -> TasteMapResult:
    """method: "pca" (variance-maximizing, the Core default) or "ica"
    (statistically independent components, Strong tier -- see module
    docstring). Clustering always runs on the full embedding, not the 2D
    projection, regardless of method -- projection is for display only:
    retrieval/similarity search (retrieval/service.py, FAISS) never touches
    this reduced 2D space either, only the full, un-reduced facet vectors."""
    if method not in ("pca", "ica"):
        raise ValueError(f"Unknown method {method!r}, expected 'pca' or 'ica'")

    song_ids = list(song_vectors.keys())
    if not song_ids:
        return TasteMapResult(points=[])

    matrix = np.stack([song_vectors[sid] for sid in song_ids])

    explained_variance_ratio = None
    if matrix.shape[0] >= 2:
        n_components = min(2, matrix.shape[0], matrix.shape[1])
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
        else:
            reducer = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
        coords = reducer.fit_transform(matrix)
        if n_components == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(len(song_ids))])
        if method == "pca":
            ratios = list(reducer.explained_variance_ratio_)
            ratios += [0.0] * (2 - len(ratios))  # n_components can be 1 when there are only 2 songs
            explained_variance_ratio = (ratios[0], ratios[1])
    else:
        coords = np.zeros((len(song_ids), 2))

    k = max(1, min(n_clusters, len(song_ids)))
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(matrix)

    points = [
        TasteMapPoint(song_id=sid, x=float(coords[i, 0]), y=float(coords[i, 1]), cluster=int(labels[i]))
        for i, sid in enumerate(song_ids)
    ]
    return TasteMapResult(points=points, explained_variance_ratio=explained_variance_ratio)


@dataclass
class AxisCorrelation:
    axis: str  # "x" or "y"
    feature: str
    r: float
    p_value: float


def correlate_axes_with_features(
    x: np.ndarray, y: np.ndarray, features: dict[str, np.ndarray]
) -> list[AxisCorrelation]:
    """The rigorous, checkable half of axis interpretability (the qualitative
    "listen to the songs at the extremes" check is a UI concern, not this
    module's job): does a projection axis actually correlate with an
    already-computed, independently-meaningful feature (tempo, energy,
    brightness, ...)? A clean |r| lets an axis be *named* with real evidence.
    No correlation above noise for either axis is itself a valid, reportable
    finding -- not every PCA/ICA axis has to resolve into something nameable,
    and claiming otherwise without a checkable result would be worse than
    saying so honestly.

    x, y, and every array in features must be the same length and index-
    aligned (same song order) -- callers restrict to songs that have both a
    projected point and a fully-computed DNA profile before calling this."""
    from scipy.stats import pearsonr

    results = []
    for axis_label, values in [("x", x), ("y", y)]:
        for feature_name, feature_values in features.items():
            r, p = pearsonr(values, feature_values)
            results.append(AxisCorrelation(axis=axis_label, feature=feature_name, r=float(r), p_value=float(p)))
    return results
