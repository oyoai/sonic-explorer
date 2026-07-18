# Sonic Explorer

Audio-signal-based music similarity, exploration, and remix engine. Full product
spec: [sonic_explorer_spec.md](sonic_explorer_spec.md).

## Layout

- `sonic_explorer/` — core package (facets, repositories, retrieval, pipeline). Plain
  Python, no UI dependency — see spec section 8.3.
- `streamlit_app/` — interface layer. Calls into `sonic_explorer`, contains no logic.
- `notebooks/` — exploratory + Colab pipeline notebooks (throwaway/heavy-compute side).
- `scripts/` — one-off local scripts (e.g. evaluation reporting).
- `tests/` — unit tests for the core package, run locally.
- `data/` — gitignored; synced down from Colab/Drive (curated audio, SQLite DB, FAISS
  indexes).

## Local dev setup

```
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest tests/
```

## Compute split

Heavy embedding compute (CLAP inference over the curated library) runs on Google
Colab (`notebooks/01_fma_acquire_and_curate.ipynb`, `notebooks/02_batch_embed_pipeline.ipynb`),
which installs the same `sonic_explorer` package (`pip install -e ".[colab]"`) so
there's exactly one implementation of every class — never a notebook-local
reimplementation. Resulting artifacts (`data/artifacts/*.db`, `*.index`, curated
`data/audio/`) get downloaded from Colab into this repo's `data/` folder for local
Streamlit development.
