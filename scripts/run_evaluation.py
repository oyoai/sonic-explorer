"""Produces the Core-tier evaluation result: genre-cohesion of nearest neighbors
vs. a random baseline. Prints a summary and saves an HTML bar chart for slides.

Reads whatever is currently synced into data/artifacts/ -- run against the
synthetic dev dataset (scripts/seed_dev_data.py) or the real synced library,
no code changes needed either way.
"""

import plotly.graph_objects as go

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH
from sonic_explorer.evaluation.genre_cohesion import genre_cohesion_at_k
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

K = 10
SAMPLE_SIZE = 500


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    embedding_repo.load_index("sound")

    result = genre_cohesion_at_k(
        song_repo, embedding_repo, facet_name="sound", k=K, sample_size=SAMPLE_SIZE
    )

    print(f"Facet: {result.facet_name}")
    print(f"k: {result.k}")
    print(f"Queries evaluated: {result.n_queries}")
    print(f"Observed genre-cohesion@{result.k}: {result.observed * 100:.1f}%")
    print(f"Random baseline:               {result.random_baseline * 100:.1f}%")

    if result.n_queries == 0:
        print("\nNo embedded segments found -- nothing to plot.")
        return

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Sound facet", "Random baseline"],
                y=[result.observed * 100, result.random_baseline * 100],
                text=[f"{result.observed * 100:.1f}%", f"{result.random_baseline * 100:.1f}%"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"Genre cohesion @ k={result.k} ({result.n_queries} queries)",
        yaxis_title="% of neighbors sharing genre",
    )

    out_path = ARTIFACTS_DIR / "genre_cohesion.html"
    fig.write_html(str(out_path))
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
