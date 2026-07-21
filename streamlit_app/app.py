import streamlit as st

# Methodology is the landing experience -- no standalone splash page. It
# narrates the actual analysis and preprocessing (data -> facets ->
# DNA/fingerprints -> retrieval -> evaluation) with real evidence at each
# step. From there, App Walkthrough interprets the live interactive pages
# themselves, which stay fully built and reachable via Streamlit's sidebar
# nav regardless.
st.switch_page("pages/0_Methodology.py")
