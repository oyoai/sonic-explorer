"""Sonic Explorer -- AI Similarity Demo: a deliberately small, separate
Streamlit app whose only job is proving the retrieval system works. Nothing
in streamlit_app/ or sonic_explorer/ was touched to build this.

Four exploration modes, each its own page (real st.navigation, same
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
2. Local Similarity (pages/1_Moment_Matcher.py) -- local, facet-level
   similarity: pick a song and a moment, see what each facet independently
   retrieves for it.
3. Local Similarity -- Curated (pages/4_Moment_Matcher_Curated.py) -- a
   static, presentation-safe sibling of #2: six fixed query/match pairs
   (one per facet, real precomputed match percentages, no live
   retrieval_service call at all), for a live talk where picking a
   song/moment on stage and waiting on retrieval would be a reliability
   risk. Doesn't touch #2 or its underlying code -- see that page's own
   module docstring for the full "why" and exactly how its numbers were
   computed. Listed right after #2 since they cover the same ground, one
   live and one fixed.
4. Ask the DJ (pages/3_Ask_The_DJ.py) -- a conversational front end over
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="Sonic Explorer -- AI Demo", layout="wide")

pg = st.navigation([
    st.Page("pages/2_Visual_Exploration.py", title="Audio Space"),
    st.Page("pages/1_Moment_Matcher.py", title="Local Similarity"),
    st.Page("pages/4_Moment_Matcher_Curated.py", title="Local Similarity — Curated"),
    st.Page("pages/3_Ask_The_DJ.py", title="Ask the DJ"),
])
pg.run()
