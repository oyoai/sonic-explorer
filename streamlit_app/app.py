import streamlit as st

# Explore is the landing experience -- no standalone splash page. The other
# four pages (Taste Map, Song X-Ray, Moment Matcher, Ask the DJ) stay fully
# built and reachable via Streamlit's sidebar nav regardless, just not
# featured on the root page anymore.
st.switch_page("pages/5_Explore.py")
