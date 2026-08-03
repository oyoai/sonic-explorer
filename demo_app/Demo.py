"""Sonic Explorer -- AI Similarity Demo: a deliberately small, separate
Streamlit app whose only job is proving the retrieval system works. Nothing
in streamlit_app/ or sonic_explorer/ was touched to build this.

Three exploration modes, each its own page (real st.navigation, same
mechanism streamlit_app/Overview.py uses -- chosen over Streamlit's classic
auto-discovered pages/ directory specifically so each page keeps an
explicit, human-written sidebar title rather than one derived from its
filename, and so this router stays the one place page order/labels are
controlled). List order here is sidebar order (per CLAUDE.md's own
established convention) -- it does NOT need to match each file's own
numeric prefix, which is why "Audio Space" (file 2) is listed first:

1. Audio Space (pages/2_Visual_Exploration.py) -- the global similarity
   space: the same network graph + PCA map App Walkthrough introduces in
   the main app, reused here as a real, clickable exploration tool rather
   than static illustration. Listed first: it's the broadest, most
   immediately visual entry point into "does this actually work."
2. Moment Matcher (pages/4_Moment_Matcher_Curated.py) -- local, facet-level
   similarity: six fixed query/match pairs, one per facet, with real
   precomputed match percentages and no live retrieval_service call at
   all -- presentation-reliable by design. The LIVE, pick-your-own-song
   version (pages/1_Moment_Matcher.py) used to also be listed here for
   comparison, but was pulled from this app's navigation (not deleted --
   still a real, working page in the repo, just not wired into this
   router) once the curated version became the one actually used on
   stage; a live, latency-and-variance-exposed second entry next to it
   added risk with no presentation benefit. See pages/
   4_Moment_Matcher_Curated.py's own module docstring for the full "why"
   and exactly how its numbers were computed.
3. Ask the DJ (pages/3_Ask_The_DJ.py) -- a conversational front end over
   the same retrieval system, adapted from streamlit_app/pages/
   6_Ask_The_DJ.py's already-built, already-tested MusicAgent integration
   (see docs/ASK_THE_DJ_HANDOFF.md) -- the agent itself was untouched here,
   this is integration into demo_app, not new agent work.

This file only sets up page-wide config (st.set_page_config, once, here,
not per-page -- calling it from more than one place in the same script run
raises) and the navigation itself; it holds no page content of its own.
Real, verified gotcha (not a hypothetical): anything actually RENDERED here
before pg.run() -- e.g. an earlier revision's inject_match_pill_style()
call -- never reaches the DOM at all, confirmed by inspecting the live
page's HTML and finding zero trace of injected CSS that should have been
here. st.navigation's page-switch machinery appears to reset/own the main
render area once pg.run() takes over, discarding whatever the router
script drew into it beforehand; st.set_page_config is unaffected only
because it's metadata, not a rendered element. So: st.set_page_config stays
here (the one config call that's genuinely page-wide), but anything that
actually renders -- CSS injection included -- has to live inside the page
that needs it instead (pages/1_Moment_Matcher.py calls
inject_match_pill_style() itself now, guarded by the same session_state
idempotency flag, so it's still cheap even though only one page uses it)."""

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))
# Makes `import sonic_explorer` resolve as a plain source-tree import,
# independent of whether pip's editable install of it actually succeeded --
# real, repeated deployment failures (ModuleNotFoundError for plotly, then
# sonic_explorer itself) traced back to demo_app/ ever having its own
# requirements.txt at all: Streamlit Community Cloud prefers a
# requirements.txt in the same directory as the main file over the
# repo-root one, and every version of that local file (relative -e paths,
# even an explicit PyPI package list) hit some install-time issue on
# Cloud's actual environment that never reproduced in local testing. Fixed
# by deleting demo_app/requirements.txt entirely, so Cloud has no local
# file to prefer and falls back to the repo-root requirements.txt (`-e .`)
# -- the exact same one the OTHER, already-successfully-deployed app in
# this repo (streamlit_app/) has used all along, so it's proven to work in
# Cloud's real environment, not just locally. This sys.path insert is the
# extra belt on top: even if some future change reintroduces a local
# requirements.txt that breaks sonic_explorer's install again, the import
# still succeeds via this explicit path. Harmless if sonic_explorer IS
# pip-installed (this path just becomes a redundant, lower-priority entry
# on sys.path; `pip install -e .` already puts the same source directory
# first via a .pth file, so nothing here overrides it either way).
sys.path.insert(0, str(_APP_DIR.parent))

import streamlit as st

st.set_page_config(page_title="Sonic Explorer -- AI Demo", layout="wide")

pg = st.navigation([
    st.Page("pages/2_Visual_Exploration.py", title="Audio Space"),
    st.Page("pages/4_Moment_Matcher_Curated.py", title="Moment Matcher"),
    st.Page("pages/3_Ask_The_DJ.py", title="Ask the DJ"),
])
pg.run()
