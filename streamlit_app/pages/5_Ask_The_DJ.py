import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from resources import get_agent, get_repositories, show_data_source_banner, show_logo

MAX_MESSAGES_PER_SESSION = 30  # simple abuse/cost guardrail for the public deployment (spec section 11)

st.set_page_config(page_title="Ask the DJ", page_icon="\U0001F399️")
st.title("Ask the DJ")
st.caption(
    "A conversational companion to **Explore** -- the same library, the same underlying search "
    "(Moment Matcher's matching, the Taste Map's mood profiles), just reached by describing what "
    "you want in plain language instead of clicking through the controls yourself."
)
st.page_link("pages/6_Explore.py", label="← Back to Explore", icon="\U0001F310")

show_logo()
show_data_source_banner()

song_repo, embedding_repo, retrieval_service = get_repositories()
songs = song_repo.list_songs()
if not songs:
    st.info("No songs in the library yet.")
    st.stop()

agent = get_agent()

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # raw message list the agent itself needs (SDK-shaped)
if "agent_display_log" not in st.session_state:
    st.session_state.agent_display_log = []  # [(role, text)] -- what actually gets rendered
if "agent_message_count" not in st.session_state:
    st.session_state.agent_message_count = 0

if agent is None:
    st.info(
        "Set ANTHROPIC_API_KEY to chat with the DJ. In the meantime, use Moment Matcher and the Taste "
        "Map directly."
    )
    st.stop()

with st.expander("Try asking..."):
    st.markdown(
        "- \"Find me something similar to *[a song title]* in harmony\"\n"
        "- \"I want something moodier and more stripped-back than *[a song title]*\"\n"
        "- \"What's a high-energy, bright-sounding song in the library?\"\n"
    )

for role, text in st.session_state.agent_display_log:
    with st.chat_message(role):
        st.markdown(text)

if st.session_state.agent_message_count >= MAX_MESSAGES_PER_SESSION:
    st.info("This session has reached its message limit -- refresh the page to start a new conversation.")
else:
    user_message = st.chat_input("Ask about songs in the library...")
    if user_message:
        st.session_state.agent_message_count += 1
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply, new_history = agent.send_message(st.session_state.agent_history, user_message)
                except Exception:
                    reply = "Something went wrong on my end -- try rephrasing that, or ask something else."
                    new_history = st.session_state.agent_history
            st.markdown(reply)

        st.session_state.agent_history = new_history
        st.session_state.agent_display_log.append(("user", user_message))
        st.session_state.agent_display_log.append(("assistant", reply))
