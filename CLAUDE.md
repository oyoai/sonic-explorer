# CLAUDE.md

Guidance for Claude Code (or any AI assistant) working in this repository. Reflects conventions that actually emerged over the course of building this project, not aspirational rules.

## What this project is

Sonic Explorer: audio-signal-based music similarity, exploration, and remix engine. Full details in `sonic_explorer_spec.md`. A Streamlit app, deployed on Streamlit Community Cloud directly from this GitHub repo (not the `Dockerfile`, which exists for local/portability verification only — see its own header comment).

## Layout

```
sonic_explorer/     core package -- plain Python, NO Streamlit/UI imports anywhere in this tree
streamlit_app/       UI layer -- pages/ (numbered files) + shared modules (resources.py, comparison_data.py, engineering_data.py, components/)
scripts/             one-off / data-prep scripts -- NOT part of the installed package (no __init__.py); don't import from here into streamlit_app or sonic_explorer at runtime, or it'll work locally and break on Streamlit Cloud (see "Environment-dependent imports" below)
notebooks/           exploratory EDA -- excluded from ruff, held to a looser bar by design (see pyproject.toml's comment)
tests/                pytest; Streamlit AppTest for page-level smoke tests
data/ / deploy_data/  data/ (gitignored, full real library) wins if present; deploy_data/ (committed, small stratified subset) is the fallback Streamlit Cloud actually sees -- resolved once at import time in sonic_explorer/config.py
```

**Core/interface separation is load-bearing, not decorative.** `sonic_explorer/` never imports Streamlit. Every Streamlit page calls into `sonic_explorer` for logic and only handles orchestration + rendering itself. Keep it that way — it's what makes the core package testable without a running app and importable from notebooks/scripts unchanged.

## Environment-dependent imports (a real bug class here — don't repeat it)

Two past incidents, same root cause: code that imports fine locally (because the repo root happens to be on `sys.path`, e.g. via `cwd`) but breaks on Streamlit Community Cloud, which only has `sonic_explorer` actually pip-installed (`pip install -e .`, per `requirements.txt`).

- `librosa`/`soundfile` had to be promoted from a colab-only extra to a base dependency once the deployed app started loading real audio at runtime (Approach's waveform preview). `torch` was promoted the same way for the Engineering page's live CNN inference.
- `scripts/` is never imported from `streamlit_app/` or `sonic_explorer/` at runtime for this reason — curated numbers/prompts that originate from a `scripts/*.py` run are copied into a small data module instead (e.g. `streamlit_app/engineering_data.py`, `streamlit_app/pages/2_Results.py`'s `GENRE_COHESION_RESULTS`), with a comment pointing at the real script that produced them.

Before adding a new runtime import: check whether it's already in `pyproject.toml`'s base `dependencies`, and check whether the module you're importing is actually part of the installed package (`sonic_explorer*` per `[tool.setuptools.packages.find]`) — `streamlit_app` and `scripts` are not.

## Streamlit conventions

- **Navigation**: `Overview.py` registers every page explicitly via `st.navigation([...])` — sidebar order comes from that list's order, **not** from filename numeric prefixes. It's fine for a page's filename number to be out of sequence with where it sits in the sidebar (see `pages/8_Engineering.py`, which sits between Methodology and Results) rather than renumbering every other file just to keep prefixes contiguous.
- **Cross-page navigation**: always use `resources.py`'s `nav_button()`, never a raw `st.page_link()` — `st.page_link()` renders as an underlined link, not a button, and (for Overview specifically) string-path linking is actually broken (see `overview_page.py`'s docstring for why `OVERVIEW_PAGE` has to be imported as a `StreamlitPage` object instead).
- **Shared resources**: everything expensive/shared goes through `resources.py`'s `@st.cache_resource`/`@st.cache_data` functions (`get_repositories()`, `get_agent()`, etc.) — pages don't construct their own repos/clients.
- **LLM features degrade gracefully**: every LLM-backed feature (explanations, rerank, the DJ agent) must work with `ANTHROPIC_API_KEY` unset — check for `None` and show an `st.info()`, never crash. LLM features are a value-add, never load-bearing for the rest of the app.

## Testing

- `pytest tests/ -q` — full suite, currently ~400 tests, a few minutes.
- `ruff check .` — narrow rule set on purpose (`E9`, `F`, `B` only — see `pyproject.toml`'s comment on why this repo doesn't lint to a stricter default).
- Page-level smoke tests use `streamlit.testing.v1.AppTest`. Any test that needs `nav_button()`/`st.switch_page()` to work **must** go through the real multipage registry: `AppTest.from_file("streamlit_app/Overview.py")` then `at.switch_page("pages/N_Name.py")`, not `AppTest.from_file()` on the page directly.
- `st.tabs()` content is **not** lazy-loaded in AppTest — every tab's content executes and is queryable via `at.tabs[i]`, regardless of which tab is visually selected.
- When renaming a public identifier (function/variable/widget key), grep for it repo-wide afterward and confirm zero remaining hits before considering the rename done — this codebase has been bitten by partial renames more than once (see git history around the "naive" → "metadata baseline" rename).

## Writing style in this codebase

This repo's comments and Streamlit copy are unusually rationale-heavy by deliberate choice — long docstrings explaining *why*, cross-references between files, explicit citations. This is different from (and overrides, for this repo) a general instinct toward terse comments. When adding to `sonic_explorer/` or `streamlit_app/`, match the existing density rather than trimming it down — a future reader here benefits from the "why," not just the "what."

Two related conventions worth preserving:
- **Honest gaps over silence.** "In progress — no results yet," explicit placeholders with a path to fill in, explicit disclosure of what a comparison *doesn't* cover — this project treats an honestly-reported gap as more valuable than an omitted one. Don't paper over an unfinished or partial result with a plausible-sounding number.
- **Real citations only.** Design choices are backed by real, checkable papers where a real precedent exists (e.g. Vohra & Akama 2026 for the source-separation-plus-ABX-calibration design, Tovstogan/Serra/Bogdanov 2022 and VidTune/CHI 2026 for the 2D embedding map) — verify a citation actually exists (e.g. via a real search) before adding a new one; never invent a plausible-sounding reference.

## Key architectural facts worth knowing before touching retrieval/similarity code

- All nearest-neighbor search (`sonic_explorer/retrieval/service.py`, `embedding_repository.py`) runs FAISS `IndexFlatIP` over full, un-reduced, L2-normalized facet embeddings. PCA/ICA (`analysis/taste_map.py`) is for the 2D map's *display* only — never fed into retrieval or clustering (clustering itself always runs on the full embedding too).
- Moment Matcher's match ranking is a two-stage bi-encoder-retrieve + LLM-rerank pipeline (`llm/rerank.py`) — a real, over-fetch-then-listwise-reorder pattern, not a trained cross-encoder.
- Metadata-baseline similarity and the (in-progress) facet blend-weight regression are both instances of hybrid search / score fusion — weighted combination of independent similarity signals, not an invented rule.
