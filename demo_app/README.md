# Sonic Explorer -- AI Demo

A separate, deliberately small Streamlit app whose only job is proving one
claim: **the audio-similarity engine finds genuinely similar sound using
nothing but the raw waveform** -- no genre, artist, or tag comparison
involved.

`streamlit_app/` (the main app, untouched by this) is the full exploratory
product -- Taste Map, Song X-Ray, Moment Matcher, Ask the DJ, Mixtape
Builder. This app exists next to it, not inside it, specifically so a
live presentation doesn't have to fight against a UI built for open-ended
exploration.

## Running locally

```
streamlit run demo_app/Demo.py
```

Uses the repo's existing `.venv` / `pip install -e .` -- no separate
install step (see `requirements.txt`'s own comment). Set `ANTHROPIC_API_KEY`
(env var or `.streamlit/secrets.toml`) to also see Ask the DJ's live chat;
everything else runs and shows real matches without it, per this project's
"LLM features are a value-add, never load-bearing" rule.

## Pages (real `st.navigation`, see `Demo.py`'s own docstring for the full
list-order rationale)

- **Audio Space** (`pages/2_Visual_Exploration.py`) -- the global
  similarity space: a real, clickable network graph over the whole
  library, one facet at a time.
- **Local Similarity** (`pages/1_Moment_Matcher.py`) -- pick a song and a
  moment, see what each facet independently retrieves for it. Fully live:
  every match is a real `RetrievalService` call made on the spot.
- **Local Similarity -- Curated** (`pages/4_Moment_Matcher_Curated.py`) --
  a static sibling of the page above: six fixed query/match pairs (one per
  facet, real precomputed match percentages, no live retrieval call at
  all), for a presentation context where retrieval-quality variance would
  be a risk. Doesn't touch the live page or its underlying code.
- **Ask the DJ** (`pages/3_Ask_The_DJ.py`) -- a conversational front end
  over the same retrieval system (`sonic_explorer/llm/agent.py`'s
  `MusicAgent`, untouched here). Two static, pre-verified example
  exchanges up top (no API call), then a genuinely live free-form chat
  below, seeded to continue that same conversation -- see the page's own
  module docstring for the full mechanics.

## What's real vs. what's fixed in advance

Varies by page -- each page's own module docstring is the source of truth
for exactly what's live versus fixed, and why. In short: Audio Space and
Local Similarity are fully live (real retrieval on every interaction);
Local Similarity -- Curated and Ask the DJ's opening examples are fixed
text with real, independently-verified data underneath (real audio, real
match percentages, real prior LLM runs) rather than re-run live, chosen
for presentation reliability; Ask the DJ's free-form chat section is live.

## Files

- `Demo.py` -- page-wide config + the `st.navigation` router; holds no
  page content of its own (see its own docstring for a real, verified
  gotcha about content rendered here never reaching the DOM).
- `pages/` -- the four pages above.
- `curated_examples.py` -- Local Similarity's default song/moment seed.
- `resources.py` / `plotting.py` -- minimal, standalone copies of just the
  repository/chart helpers this app needs (not imported from `streamlit_app/`,
  which isn't an installed package -- see those files' own docstrings).
- `local_similarity_state.json` -- on-disk persistence for Local
  Similarity's per-facet song/moment picks, so a full app restart mid-talk
  doesn't lose the presenter's setup (see `resources.py`'s
  `persistent_song_and_moment()` docstring).
