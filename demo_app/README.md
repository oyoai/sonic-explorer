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
- **Moment Matcher** (`pages/4_Moment_Matcher_Curated.py`) -- six fixed
  query/match pairs, one per facet, with real precomputed match
  percentages and no live retrieval call at all -- presentation-reliable
  by design. A live, pick-your-own-song sibling of this page also exists
  (`pages/1_Moment_Matcher.py`) but isn't wired into this app's navigation
  -- see `Demo.py`'s own docstring for why.
- **Ask the DJ** (`pages/3_Ask_The_DJ.py`) -- a conversational front end
  over the same retrieval system (`sonic_explorer/llm/agent.py`'s
  `MusicAgent`, untouched here). Two static, pre-verified example
  exchanges up top (no API call), then a genuinely live free-form chat
  below, seeded to continue that same conversation -- see the page's own
  module docstring for the full mechanics.

## What's real vs. what's fixed in advance

Varies by page -- each page's own module docstring is the source of truth
for exactly what's live versus fixed, and why. In short: Audio Space is
fully live (real retrieval on every interaction); Moment Matcher and Ask
the DJ's opening examples are fixed text with real, independently-verified
data underneath (real audio, real match percentages, real prior LLM runs)
rather than re-run live, chosen for presentation reliability; Ask the DJ's
free-form chat section is live.

## Files

- `Demo.py` -- page-wide config + the `st.navigation` router; holds no
  page content of its own (see its own docstring for a real, verified
  gotcha about content rendered here never reaching the DOM).
- `pages/` -- the three pages above, plus the live Moment Matcher
  (`1_Moment_Matcher.py`) kept in the repo but not routed to.
- `curated_examples.py` -- the live Moment Matcher's default song/moment
  seed (only relevant if that page is ever wired back into the nav).
- `resources.py` / `plotting.py` -- minimal, standalone copies of just the
  repository/chart helpers this app needs (not imported from `streamlit_app/`,
  which isn't an installed package -- see those files' own docstrings).
- `local_similarity_state.json` -- on-disk persistence for the live Moment
  Matcher's per-facet song/moment picks (see `resources.py`'s
  `persistent_song_and_moment()` docstring); unused while that page isn't
  in the nav.
- `requirements.txt` -- NOT documentation-only despite appearances: `-e ..`
  really does get installed if Streamlit Community Cloud picks this file
  over the repo-root one (it prefers whichever requirements.txt sits in
  the same directory as the app's main file) -- see this file's own
  comment for a real incident where an earlier, comments-only version of
  this file broke a fresh deployment.
