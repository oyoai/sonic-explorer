import streamlit as st

# The walkthrough is the landing experience -- no standalone splash page. It
# narrates the methodology (data -> facets -> DNA/fingerprints -> retrieval ->
# evaluation) with real evidence at each step, then opens into the five live
# interactive pages, which stay fully built and reachable via Streamlit's
# sidebar nav regardless.
st.switch_page("pages/0_Walkthrough.py")
