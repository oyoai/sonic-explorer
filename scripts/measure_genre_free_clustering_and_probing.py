"""Standalone measurement, real data, real numbers: two supporting analyses
for the "audio similarity vs. genre" thesis running throughout this
project.

1. Genre-free clustering (evaluation/genre_free_clustering.py) -- KMeans
   over real per-song CLAP embeddings (genre_top never touches the
   clustering step), scored against genre_top afterward via Adjusted Rand
   Index. Real supporting evidence: if audio-only clusters agreed with
   genre almost perfectly, that would cut against the "genre is a lossy
   proxy for what a song sounds like" framing this project makes
   throughout Approach/Methodology; a low ARI is what that framing
   predicts.

2. Linear probing (evaluation/linear_probing.py) -- Ridge-regression,
   cross-validated R^2, predicting each Song DNA scalar from the SAME CLAP
   embeddings. Lighter-weight evidence: says something about what CLAP's
   embedding space geometrically encodes, not directly about genre.

Both use mean_pool_song_vectors(..., facet_name="sound") -- one 512-dim
CLAP vector per song, averaged across that song's segments, the same
pooling taste_map.py/network_graph.py already use for their own per-song
views. Run against deploy_data (the same 233-song set the album-art
prompts and sound-tag backfill were run against this session) --
DEPLOY_DB_PATH is a real, explicit choice, not sonic_explorer.config's
auto-resolved default, for the same reason export_album_art_prompts.py
targets it explicitly: this measures the actual deployed set, not
whatever happens to be on the machine running this script."""

from sonic_explorer.analysis.song_dna import AXES
from sonic_explorer.analysis.taste_map import mean_pool_song_vectors
from sonic_explorer.config import PROJECT_ROOT
from sonic_explorer.evaluation.genre_free_clustering import DEFAULT_N_CLUSTERS, cluster_and_compare_to_genre
from sonic_explorer.evaluation.linear_probing import probe_dna_from_embeddings
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

DEPLOY_DB_PATH = PROJECT_ROOT / "deploy_data" / "artifacts" / "sonic_explorer.db"
DEPLOY_ARTIFACTS_DIR = PROJECT_ROOT / "deploy_data" / "artifacts"


def main():
    conn = init_db(DEPLOY_DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=DEPLOY_ARTIFACTS_DIR)
    embedding_repo.load_index("sound")

    songs = song_repo.list_songs()
    song_vectors = mean_pool_song_vectors(song_repo, embedding_repo, facet_name="sound")
    genre_by_song = {s.id: s.genre_top for s in songs}
    dna_by_song = {s.id: {axis: getattr(s, axis) for axis in AXES} for s in songs}

    print(f"{len(songs)} songs in deployed set, {len(song_vectors)} with a real sound embedding.\n")

    print("=" * 70)
    print("1. GENRE-FREE CLUSTERING")
    print("=" * 70)
    clustering_result = cluster_and_compare_to_genre(
        song_vectors, genre_by_song, facet_name="sound", n_clusters=DEFAULT_N_CLUSTERS, seed=42,
    )
    print(f"n_songs={clustering_result.n_songs}, n_clusters={clustering_result.n_clusters}")
    print(f"Genre distribution: {clustering_result.genre_sizes}")
    print(f"Cluster sizes (KMeans, audio-only): {clustering_result.cluster_sizes}")
    print(f"\nAdjusted Rand Index (audio clusters vs. real genre_top labels): "
          f"{clustering_result.adjusted_rand_index:.4f}")
    print(
        "(1.0 = clusters perfectly reproduce genre; ~0.0 = no better than chance agreement; "
        "negative = worse than chance)"
    )

    print("\n" + "=" * 70)
    print("2. LINEAR PROBING (Ridge regression, 5-fold cross-validated R^2)")
    print("=" * 70)
    probe_results = probe_dna_from_embeddings(song_vectors, dna_by_song, axes=AXES, cv_folds=5, seed=42)
    print(f"\n{'axis':>22} | {'n_songs':>7} | {'R^2 (mean)':>10} | {'R^2 (std)':>9}")
    print("-" * 58)
    for axis in AXES:
        r = probe_results[axis]
        print(f"{axis:>22} | {r.n_songs:>7} | {r.r2_mean:>10.4f} | {r.r2_std:>9.4f}")


if __name__ == "__main__":
    main()
