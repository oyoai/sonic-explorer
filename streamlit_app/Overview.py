import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from overview_page import OVERVIEW_PAGE

# ---------------------------------------------------------------------------
# Navigation: Song X-Ray, Moment Matcher, and Ask the DJ are drill-down /
# companion states reached FROM Explore (select a song -> Song X-Ray; select
# a moment within it -> Moment Matcher; a persistent companion link -> Ask
# the DJ) -- not independent top-level destinations. visibility="hidden"
# keeps them fully reachable via st.switch_page/st.page_link (which Explore
# and App Walkthrough both use) while excluding them from the sidebar, so
# nobody lands on a page with no context about which song it's showing.
#
# Overview itself is registered via a callable (OVERVIEW_PAGE, defined in
# overview_page.py) rather than this file's own path -- see that module's
# docstring for why any page that wants to link back to it must import and
# pass that same StreamlitPage object, not a "Overview.py" string.
#
# pages/8_Engineering.py sits here, between Methodology and Results, even
# though its filename number doesn't match -- st.navigation()'s list order
# controls sidebar order, not the filename prefix, so giving Engineering the
# next free number (8) avoided renumbering every file from 2_Results.py
# onward (and every nav_button()/switch_page() string that references them)
# just to keep the numeric prefixes contiguous.
# ---------------------------------------------------------------------------
pg = st.navigation([
    OVERVIEW_PAGE,
    st.Page("pages/0_Approach.py", title="Approach"),
    st.Page("pages/1_Methodology.py", title="Methodology"),
    st.Page("pages/8_Engineering.py", title="Engineering"),
    st.Page("pages/2_Results.py", title="Results"),
    st.Page("pages/3_App_Walkthrough.py", title="App Walkthrough"),
    st.Page("pages/4_Song_XRay.py", title="Song X-Ray", visibility="hidden"),
    st.Page("pages/5_Moment_Matcher.py", title="Moment Matcher", visibility="hidden"),
    st.Page("pages/6_Ask_The_DJ.py", title="Ask the DJ", visibility="hidden"),
    st.Page("pages/7_Explore.py", title="Explore"),
])
pg.run()
