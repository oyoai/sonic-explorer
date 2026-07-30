"""Genre-free clustering: does unsupervised structure in the audio embedding
space actually align with genre labels, or does audio-based similarity
carve up the library differently than genre would? Real, quantitative
supporting evidence for this project's core thesis (genre_top is a
metadata proxy, not a measurement of what a song actually sounds like) --
if clusters built PURELY from audio embeddings (genre_top never touches
the clustering step itself) don't line up cleanly with genre_top, that's
a real signal that audio similarity and genre are measuring genuinely
different things, not just noisy versions of the same signal.

Adjusted Rand Index (ARI) is the standard label-agreement metric for
exactly this "do two independent partitions of the same items agree"
question: 1.0 = identical partitions, ~0.0 = no better than random
agreement, negative = worse than random. ARI (unlike raw accuracy)
doesn't require the two partitions to share a label vocabulary or even a
label count -- KMeans' cluster indices are arbitrary integers, genre_top
is 8 real genre names -- it only scores whether PAIRS of songs are
grouped together consistently across both partitions.

KMeans, not HDBSCAN: this reuses the exact clustering already used
elsewhere in this codebase (analysis/network_graph.py's node-coloring
clusters) rather than adding a new dependency -- hdbscan isn't installed
in this project's environment, checked directly before writing this.
n_clusters defaults to 8, matching this library's own real genre count
(see build_deploy_subset.py's SONGS_PER_GENRE stratification), so a
perfectly genre-aligned clustering is at least numerically reachable, not
structurally prevented by a mismatched k."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

DEFAULT_N_CLUSTERS = 8


@dataclass
class GenreFreeClusteringResult:
    facet_name: str
    n_clusters: int
    n_songs: int
    adjusted_rand_index: float
    cluster_sizes: list[int]
    genre_sizes: dict[str, int]


def cluster_and_compare_to_genre(
    song_vectors: dict[int, np.ndarray],
    genre_by_song: dict[int, str],
    facet_name: str = "sound",
    n_clusters: int = DEFAULT_N_CLUSTERS,
    seed: int = 42,
) -> GenreFreeClusteringResult:
    """song_vectors: song_id -> embedding (e.g. analysis.taste_map.
    mean_pool_song_vectors() output). genre_by_song: song_id -> genre_top
    string -- ground truth is NEVER passed into the clustering itself,
    only used afterward to SCORE the audio-only clustering against it.
    Songs missing from either dict are excluded (need both a vector and a
    genre label to be scoreable)."""
    song_ids = [sid for sid in song_vectors if sid in genre_by_song]
    matrix = np.stack([song_vectors[sid] for sid in song_ids])
    genres = [genre_by_song[sid] for sid in song_ids]

    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(matrix)
    ari = float(adjusted_rand_score(genres, labels))

    cluster_sizes = list(np.bincount(labels).tolist())
    genre_sizes: dict[str, int] = {}
    for g in genres:
        genre_sizes[g] = genre_sizes.get(g, 0) + 1

    return GenreFreeClusteringResult(
        facet_name=facet_name, n_clusters=n_clusters, n_songs=len(song_ids),
        adjusted_rand_index=ari, cluster_sizes=cluster_sizes, genre_sizes=genre_sizes,
    )
