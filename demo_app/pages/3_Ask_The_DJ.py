"""Ask the DJ -- a conversational front end over the same retrieval system
Audio Space and Local Similarity already use. Adapted from the reference
implementation, streamlit_app/pages/6_Ask_The_DJ.py (already built, tested,
and demo-polished -- see docs/ASK_THE_DJ_HANDOFF.md, the source of truth
for this integration), not redesigned at the agent level: same session-state
key shapes, same inline-audio-driven-by-tool-results pattern. The agent
itself (sonic_explorer/llm/agent.py's MusicAgent, sonic_explorer/llm/
agent_tools.py's 7 tools) is untouched -- this file is integration and
presentation, not agent work.

Presentation model -- from "live walkthrough" to "static examples" (this
revision): two earlier versions of this page (a hand-typed "Try asking..."
expander, then a scripted-but-still-live guided walkthrough advanced one
click at a time) both actually called agent.send_message() when the
presenter advanced a step -- a real, live Anthropic API call, mid-talk.
Explicit direction for this revision: "2-3 messages that show the
capabilities that aren't called live, but static." STATIC_EXAMPLES is the
result -- fixed, pre-verified request/tool-calls/reply/song data, rendered
directly with no API call at all when the page loads. This is a real
tradeoff, not a strict improvement: nothing here is generated live in front
of the audience anymore, so it can't be argued to "prove" the agent is live
the way watching a real call resolve did. What it buys back is total
reliability under a fixed presentation timeslot -- no API latency, no
availability risk, no chance of the ~1-in-6 empty-reply flakiness an
earlier revision's live follow-up step carried (see git history on this
file for that investigation) -- while every example below was still
verified against a REAL, live agent.send_message() call at authoring time
(each entry's own comment says how many times), just not re-executed on
every page load. A separate, genuinely live st.chat_input for free-form
Q&A stays below the static examples for anyone who wants to see a real call
happen -- static and live coexist on this page, they're not a replacement
for each other.

STATIC_EXAMPLES (2 entries, each a list of "turns" -- usually one, but the
second is a genuine two-turn exchange, see below):
  A. "do you have any songs with crow sounds?" -- search_by_sound_content,
     a literal AI-detected-tag keyword match. Resolves to "flekkefjord" by
     Blear Moon (Experimental), verified live.
  B. Two turns, demonstrating conversational follow-up refinement, not just
     a single request/response: turn 1, "Find me something calm and
     stripped-back." -- search_by_mood_profile alone (the 5-axis DNA
     nearest-neighbor path, no keyword search involved), resolves to "Do
     Easy" by Tasseomancy, verified live twice with an identical ranked
     list of 5 both times. Turn 2, "I like that, find something similar
     but more upbeat." -- verified live twice; both runs land on the same
     top match ("Harmony To My Heartbeat" by Sally Seltmann), though the
     exact tool sequence varied slightly run to run (the model sometimes
     opened with get_song_profile/update_taste_profile before searching,
     sometimes went straight to search_by_mood_profile) -- the RICHER of
     the two verified runs is the one scripted here: get_song_profile
     (resolving "that" = Do Easy) -> update_taste_profile (logging "I
     like that" as liked=["calm", "stripped-back"]) -> search_by_mood_profile
     (energy/tempo nudged up from Do Easy's real profile). Three tools in
     one follow-up turn -- reference resolution, taste tracking, AND a new
     search -- the fullest single-turn capability demonstration on this
     page. Both turns are shown together specifically because turn 2 only
     makes sense with turn 1's real context in view.
Reply text and tool-call arguments below are copied verbatim from those
live runs (deploy_data, the same DB this app always uses) -- not written by
hand to sound plausible. Matched songs are resolved back to real Song rows
at render time via find_song_by_title (sonic_explorer/llm/agent_tools.py,
the exact same lookup the live agent itself uses) so the audio players
underneath play real, correct audio, not a hardcoded path that could drift
if deploy_data is ever rebuilt.

Reliability note worth keeping if this set grows: the exact "'TITLE' by
ARTIST" phrasing measurably hurts find_song_by_title's exact-or-
unambiguous-substring matching (can't resolve a query that's LONGER than
the real title) -- irrelevant to STATIC_EXAMPLES' own hardcoded titles
(looked up plain, never phrased that way), but still a real trap for any
NEW live-typed or scripted request added later.

Known, pre-existing limitation worth knowing (not fixed, not this
integration's to fix): search_similar_songs (agent_tools.py) often returns
the SAME song 5 times in one result set -- different segments of one song,
not deduplicated by song_id, since retrieval_service.query_by_segment()
works at the segment level.

One audio player visible at a time in the LIVE chat section (this
revision): an earlier version rendered a player for every song any tool
call mentioned, for every turn in the conversation -- by the end of even a
short exchange, the page had a half-dozen stacked <audio> elements, most
from turns nobody was looking at anymore. _render_inline_players only ever
renders the FIRST song a turn mentions (song_ids[:1], matching what the
DJ's own reply text already foregrounds as its top pick), and the
historical-replay loop below only calls it for the single most recent
assistant turn. This does NOT apply to STATIC_EXAMPLES, which render their
own audio directly -- one fixed player per turn, by design, so example B's
two turns each get their own player (Do Easy, then River) rather than
either being dropped.

Taste profile panel is an st.expander directly under the "Try it yourself"
caption (not the sidebar -- an earlier revision put it there; moved
in-line so it reads as part of this section instead of a separate area of
the page a presenter has to remember to glance at), and shows whenever
there's something to show -- no gating: an earlier "wait until the guided
walkthrough finishes" gate made sense when there was a live scripted flow
to finish; that gate is gone now and the expander just renders the moment
liked/disliked has content, full stop. That CAN now happen immediately on
page load, not only from live chat: see the next paragraph -- example B's
static card shows update_taste_profile(liked=["calm", "stripped-back"])
firing during its turn 2, and "Try it yourself" replays that same two-turn
exchange through a real API call to seed its starting context. Whether
that replay ALSO calls update_taste_profile isn't guaranteed, though --
verified directly: re-running example B's exact two prompts live, which
tool gets called alongside the mood search varies run to run (sometimes
update_taste_profile fires, sometimes the model goes straight to
search_by_mood_profile without it), even though the retrieved song was
consistently "Harmony To My Heartbeat" across every run tested. So the
expander may or may not already have content the moment "Try it yourself"
appears -- an honest, accepted consequence of seeding via a real call
rather than replaying fixed text, not a bug to chase.

"Try it yourself" continues STATIC_EXAMPLES[1]'s actual conversation, not a
fresh one -- explicit direction for this revision ("try it yourself should
be the continuation of that conversation"). _reset_conversation() achieves
this by actually replaying example B's two turns through a real
agent.send_message() call (see its own docstring for the full mechanics),
producing genuine Anthropic-API-shaped history rather than a hand-built
approximation -- the one deliberate, bounded exception to this page's
"static examples, not called live" rule: it happens once per session (or
on explicit Restart), before any chat is shown, purely to seed invisible
context for the section that's genuinely live anyway. STATIC_EXAMPLES'
own card above is completely unaffected by this -- it stays 100% fixed
text regardless of what the seeding call happens to return, so the two
never visibly disagree even though they're generated differently.

Session-state keys (agent_history, agent_display_log, agent_message_count,
taste_profile) are plain st.session_state, not st.query_params -- same
precedent Audio Space's own selection state already follows on this app,
and the one place persistence would be actively wrong: the taste-profile
expander's own copy explicitly promises "reset on refresh," which a
persisted-across-refresh session would silently break. "Restart demo"
resets all of these together -- it only affects the live chat section,
since STATIC_EXAMPLES has no state to reset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from resources import audio_path_for, get_agent, get_repositories
from sonic_explorer.llm.agent import extract_mentioned_song_ids, extract_tool_calls
from sonic_explorer.llm.agent_tools import find_song_by_title

MAX_MESSAGES_PER_SESSION = 30  # simple abuse/cost guardrail -- see docs/ASK_THE_DJ_HANDOFF.md §3

# See this file's module docstring -- each turn verified against a real,
# live agent.send_message() call at authoring time (deploy_data), then
# copied here verbatim rather than re-executed on every page load. Each
# example is a LIST of turns (usually one) because the second example is a
# genuine two-turn exchange: a follow-up refinement only makes sense with
# the first turn's real reply as conversation context, so both turns are
# shown together, not just the final one in isolation.
STATIC_EXAMPLES: list[dict] = [
    {
        "label": "Content-based search -- search_by_sound_content",
        "turns": [
            {
                "request": "do you have any songs with crow sounds?",
                "tool_calls": [{"name": "search_by_sound_content", "input": {"query": "crow"}}],
                "reply": (
                    "Yes! There's \"flekkefjord\" by Blear Moon — an experimental track with sparse "
                    "piano and distant crow sounds. If that's the kind of atmospheric, nature-infused "
                    "vibe you're after, I can dig up something similar for you."
                ),
                "song_titles": ["flekkefjord"],
            },
        ],
    },
    {
        "label": "Mood-profile search, then a follow-up refinement -- search_by_mood_profile (twice)",
        "turns": [
            {
                "request": "Find me something calm and stripped-back.",
                "tool_calls": [{
                    "name": "search_by_mood_profile",
                    "input": {
                        "tempo_bpm": 0.3, "energy": 0.2, "brightness": 0.5,
                        "harmonic_complexity": 0.4, "rhythmic_density": 0.2, "k": 5,
                    },
                }],
                "reply": (
                    "Here are some calm, minimal options:\n\n"
                    "**\"Do Easy\"** by Tasseomancy (Pop) is the closest match — low energy, sparse, "
                    "and laid-back.\n\n"
                    "If you want something even more understated, **\"Psaltery Piece\"** by Howie "
                    "Mitchell (Folk) leans more acoustic and intimate, or **\"Rörelse av Ljus\"** by "
                    "Bisamrätta (Experimental) takes a more abstract, minimalist approach.\n\n"
                    "Any of these striking your mood?"
                ),
                "song_titles": ["Do Easy"],
            },
            {
                "request": "I like that, find something similar but more upbeat.",
                "tool_calls": [
                    {"name": "get_song_profile", "input": {"song_title": "Do Easy"}},
                    {"name": "update_taste_profile", "input": {"liked": ["calm", "stripped-back"]}},
                    {
                        "name": "search_by_mood_profile",
                        "input": {
                            "tempo_bpm": 0.55, "energy": 0.45, "brightness": 0.249,
                            "harmonic_complexity": 0.594, "rhythmic_density": 0.27, "k": 5,
                        },
                    },
                ],
                "reply": (
                    "**\"Harmony To My Heartbeat\"** by Sally Seltmann is your closest match—it keeps "
                    "that restrained, minimal character but picks up the tempo and energy. Still "
                    "spare and thoughtful, just with more forward momentum.\n\n"
                    "**\"Moon Pearls\"** by Mário Gajarský is another solid option if you want a "
                    "rock-leaning take on the same vibe—moving a little faster while staying spacious."
                ),
                "song_titles": ["Harmony To My Heartbeat"],
            },
        ],
    },
]

title_col, restart_col = st.columns([5, 1])
with title_col:
    st.title("Ask the DJ")
with restart_col:
    st.write("")
    restart_clicked = st.button("Restart demo", key="restart_demo", width="stretch")
st.caption(
    "A conversational companion to Audio Space and Local Similarity -- the same library, the same "
    "underlying search (facet-based matching, mood-profile nearest-neighbor search), just reached "
    "by describing what you want in plain language instead of picking a song and moment yourself."
)

song_repo, embedding_repo, retrieval_service = get_repositories()
songs = song_repo.list_songs()
songs_by_id = {s.id: s for s in songs}
if not songs:
    st.info("No songs in the library yet.")
    st.stop()


def _render_tool_calls(tool_calls: list[dict]) -> None:
    """Which real tool(s) produced the reply above it, in call order --
    surfaced directly rather than left opaque, since for a technical
    audience seeing e.g. `search_by_mood_profile(energy=0.8, ...)` fire is
    itself part of the proof this is a real tool-calling loop over the
    actual library, not a canned response. Small/muted (st.caption) so it
    reads as a debug/trace footnote under the reply, not competing with the
    DJ's own conversational text for attention. Shared by both
    STATIC_EXAMPLES (fixed data) and the live chat section below (real
    extract_tool_calls() output) -- same rendering either way, since the
    goal is that a viewer can't tell the difference from this trace alone."""
    if not tool_calls:
        return
    lines = []
    for call in tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call["input"].items())
        lines.append(f"{call['name']}({args})")
    st.caption("🔧 " + "  →  ".join(lines))


def _render_static_example(example: dict) -> None:
    """One fixed STATIC_EXAMPLES entry, rendered as one or more chat_message
    pairs -- same visual shape as a live turn (see this file's module
    docstring for why: static and live coexist on this page, not two
    different-looking UI patterns). Renders every turn in example["turns"]
    in order, so a genuine multi-turn exchange (the mood-search-then-
    refinement example) reads as the real short conversation it is, not
    just its final answer with the setup silently dropped. song_titles are
    looked up fresh via find_song_by_title (the same lookup the real agent
    uses) rather than a hardcoded song_id, so the audio players stay
    correct even if deploy_data is ever rebuilt and ids shift -- only the
    titles need to still exist."""
    st.caption(example["label"])
    for turn in example["turns"]:
        with st.chat_message("user"):
            st.markdown(turn["request"])
        with st.chat_message("assistant"):
            _render_tool_calls(turn["tool_calls"])
            st.markdown(turn["reply"])
            for title in turn["song_titles"]:
                song = find_song_by_title(song_repo, title)
                if song is not None:
                    st.caption(f"\"{song.title}\" — {song.artist}")
                    st.audio(str(audio_path_for(song)))


st.markdown("### A few things it can do")
for example in STATIC_EXAMPLES:
    _render_static_example(example)
    st.divider()

st.caption("Picks up right where the conversation above left off — ask a follow-up to Do Easy / \"Harmony To My Heartbeat\" directly, no need to re-set the scene.")

agent = get_agent()

if agent is None:
    st.info("Set ANTHROPIC_API_KEY to chat live with the DJ. The examples above still work without one.")
    st.stop()


def _reset_conversation() -> None:
    """Seeds agent_history/taste_profile by actually replaying STATIC_EXAMPLES'
    second example's two turns through a real agent.send_message() -- so
    "Try it yourself" below is a genuine continuation of that conversation
    (same tool-use ids, same real history shape the API expects), not a
    hand-built approximation of one. agent_display_log stays empty, not
    seeded, since both turns are already visible in the static card above
    this section -- repeating them here would just duplicate what's already
    on screen. This is the one place on this page that DOES make live API
    calls before the presenter types anything -- see this file's module
    docstring for why that's an acceptable, bounded exception to "static,
    not called live": it only runs once per session (or on explicit
    Restart), it's invisible scene-setting for the live section specifically
    (never shown as if it were itself a static example), and it degrades to
    a plain, unseeded conversation on any failure rather than breaking the
    page -- the live chat still works either way, it just won't already
    know about Do Easy/Harmony To My Heartbeat if seeding failed."""
    st.session_state.agent_display_log = []  # [(role, text, song_ids, tool_calls)] -- what actually gets rendered
    st.session_state.agent_message_count = 0
    try:
        with st.spinner("Loading conversation..."):
            seed_turns = STATIC_EXAMPLES[1]["turns"]
            reply1, history1, taste1 = agent.send_message([], seed_turns[0]["request"])
            reply2, history2, taste2 = agent.send_message(history1, seed_turns[1]["request"], taste1)
        st.session_state.agent_history = history2
        st.session_state.taste_profile = taste2
    except Exception:
        st.session_state.agent_history = []
        # Session-level only -- no DB table backs this, see sonic_explorer/llm/agent.py's module
        # docstring. Threaded through agent.send_message() the same way agent_history is: passed in,
        # returned, never stored on the agent.
        st.session_state.taste_profile = {"liked": [], "disliked": []}


if "agent_history" not in st.session_state:
    _reset_conversation()
elif restart_clicked:
    _reset_conversation()
    st.rerun()

_profile = st.session_state.taste_profile
if _profile["liked"] or _profile["disliked"]:
    with st.expander("DJ preferences learned"):
        if _profile["liked"]:
            st.markdown("**Leaning toward:** " + ", ".join(_profile["liked"]))
        if _profile["disliked"]:
            st.markdown("**Steering away from:** " + ", ".join(_profile["disliked"]))
        st.caption("Picked up from what you've said in this chat only -- not saved anywhere. Resets on refresh.")


def _render_inline_players(song_ids: list[int]) -> None:
    """Real audio for the single song this turn's reply actually foregrounds
    -- see this file's module docstring on why this is sliced to just the
    first id (song_ids[:1]) rather than one player per song a tool call
    happened to return. Songs no longer in the current library are silently
    skipped rather than erroring (not expected against deploy_data's fixed
    set, but the reference page guards this too, so this mirrors it)."""
    for song_id in song_ids[:1]:
        song = songs_by_id.get(song_id)
        if song is not None:
            st.caption(f"\"{song.title}\" — {song.artist}")
            st.audio(str(audio_path_for(song)))


def _run_turn(user_message: str) -> None:
    """The one live send path -- everything typed into the free-form
    chat_input below hits the real agent, a genuine API call, unlike
    STATIC_EXAMPLES above."""
    st.session_state.agent_message_count += 1
    with st.chat_message("user"):
        st.markdown(user_message)

    turn_start_index = len(st.session_state.agent_history)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, new_history, new_taste_profile = agent.send_message(
                    st.session_state.agent_history, user_message, st.session_state.taste_profile
                )
            except Exception:
                reply = "Something went wrong on my end -- try rephrasing that, or ask something else."
                new_history = st.session_state.agent_history
                new_taste_profile = st.session_state.taste_profile
        st.markdown(reply)
        mentioned_song_ids = extract_mentioned_song_ids(new_history, turn_start_index)
        tool_calls = extract_tool_calls(new_history, turn_start_index)
        _render_tool_calls(tool_calls)
        _render_inline_players(mentioned_song_ids)

    st.session_state.agent_history = new_history
    st.session_state.taste_profile = new_taste_profile
    st.session_state.agent_display_log.append(("user", user_message, [], []))
    st.session_state.agent_display_log.append(("assistant", reply, mentioned_song_ids, tool_calls))


display_log = st.session_state.agent_display_log
last_assistant_index = max((i for i, (role, _, _, _) in enumerate(display_log) if role == "assistant"), default=-1)
for i, (role, text, song_ids, tool_calls) in enumerate(display_log):
    with st.chat_message(role):
        st.markdown(text)
        if role == "assistant":
            # Tool-call trace stays visible for every past turn (cheap text,
            # useful as a running record of what the agent actually did) --
            # unlike the audio player below, which _render_inline_players'
            # own docstring explains is deliberately kept to just the most
            # recent turn so the page doesn't accumulate a stack of players.
            _render_tool_calls(tool_calls)
            if i == last_assistant_index:
                _render_inline_players(song_ids)

if st.session_state.agent_message_count >= MAX_MESSAGES_PER_SESSION:
    st.info("This session has reached its message limit -- restart the demo above to continue.")
else:
    user_message = st.chat_input("Ask about songs in the library...")
    if user_message:
        _run_turn(user_message)
