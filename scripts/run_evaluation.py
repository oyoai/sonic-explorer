"""Produces the Core-tier evaluation result: genre-cohesion of nearest neighbors
vs. a random baseline, for every registered facet (sound, harmony, and the
four stem facets). Running them side by side is also the spec's
"ablation-style finding": do facets actually diverge from each other, or is
each new one just riding sound's coattails?

Reads whatever is currently synced into data/artifacts/ -- run against the
synthetic dev dataset (scripts/seed_dev_data.py) or the real synced library,
no code changes needed either way.
"""

import json

import plotly.graph_objects as go

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH
from sonic_explorer.evaluation.genre_cohesion import genre_cohesion_at_k
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

K = 10
SAMPLE_SIZE = 500
FACETS = default_registry().names()


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)

    results = []
    for facet_name in FACETS:
        embedding_repo.load_index(facet_name)
        result = genre_cohesion_at_k(
            song_repo, embedding_repo, facet_name=facet_name, k=K, sample_size=SAMPLE_SIZE
        )
        results.append(result)

        print(f"Facet: {result.facet_name}")
        print(f"k: {result.k}")
        print(f"Queries evaluated: {result.n_queries}")
        print(f"Observed genre-cohesion@{result.k}: {result.observed * 100:.1f}%")
        print(f"Random baseline:               {result.random_baseline * 100:.1f}%")
        print()

    # Structured export for anything downstream (e.g. a presentation/walkthrough
    # page) that wants the raw numbers rather than re-parsing the chart --
    # written alongside the HTML chart, both inside the gitignored artifacts
    # dir since they're derived outputs, not source.
    results_path = ARTIFACTS_DIR / "genre_cohesion_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "k": K,
                "sample_size": SAMPLE_SIZE,
                "facets": [
                    {
                        "facet_name": r.facet_name,
                        "n_queries": r.n_queries,
                        "observed_pct": round(r.observed * 100, 1),
                        "random_baseline_pct": round(r.random_baseline * 100, 1),
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"Results saved to {results_path}")

    plottable = [r for r in results if r.n_queries > 0]
    if not plottable:
        print("No embedded segments found for any facet -- nothing to plot.")
        return

    fig = go.Figure(
        data=[
            go.Bar(name="Observed", x=[r.facet_name for r in plottable], y=[r.observed * 100 for r in plottable],
                   text=[f"{r.observed * 100:.1f}%" for r in plottable], textposition="auto"),
            go.Bar(name="Random baseline", x=[r.facet_name for r in plottable],
                   y=[r.random_baseline * 100 for r in plottable],
                   text=[f"{r.random_baseline * 100:.1f}%" for r in plottable], textposition="auto"),
        ]
    )
    fig.update_layout(
        title=f"Genre cohesion @ k={K} by facet",
        yaxis_title="% of neighbors sharing genre",
        barmode="group",
    )

    out_path = ARTIFACTS_DIR / "genre_cohesion.html"
    fig.write_html(str(out_path))
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    main()
