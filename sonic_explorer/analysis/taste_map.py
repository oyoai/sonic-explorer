"""PCA + K-means over per-song mean-pooled sound embeddings. Plain Python, no
Streamlit import anywhere in this file -- see spec 8.3's core/interface separation."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


@dataclass
class TasteMapPoint:
    song_id: int
    x: float
    y: float
    cluster: int


@dataclass
class TasteMapResult:
    points: list[TasteMapPoint]


def compute_taste_map(
    song_vectors: dict[int, np.ndarray], n_clusters: int = 8, random_state: int = 42
) -> TasteMapResult:
    song_ids = list(song_vectors.keys())
    if not song_ids:
        return TasteMapResult(points=[])

    matrix = np.stack([song_vectors[sid] for sid in song_ids])

    if matrix.shape[0] >= 2:
        n_components = min(2, matrix.shape[0], matrix.shape[1])
        coords = PCA(n_components=n_components, random_state=random_state).fit_transform(matrix)
        if n_components == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(len(song_ids))])
    else:
        coords = np.zeros((len(song_ids), 2))

    k = max(1, min(n_clusters, len(song_ids)))
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(matrix)

    points = [
        TasteMapPoint(song_id=sid, x=float(coords[i, 0]), y=float(coords[i, 1]), cluster=int(labels[i]))
        for i, sid in enumerate(song_ids)
    ]
    return TasteMapResult(points=points)
