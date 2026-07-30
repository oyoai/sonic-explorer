"""One-shot natural-language search for Explore's search bar -- deliberately
separate from llm/agent.py's MusicAgent, not a thin wrapper around it.
MusicAgent is a multi-turn conversational assistant: its system prompt
explicitly asks it to write a plain-language reply, decide when to ask a
clarifying question, etc. -- all correct behavior for Ask the DJ's
back-and-forth page, all wrong for Explore's search bar, which is one-shot
by design (query in, ranked results out, spec's own "no separate Ask-the-DJ
link because search already falls through to it" framing was itself the
symptom of these two paths never having been properly separated). Reusing
MusicAgent here directly leaked a full conversational reply into Explore's
UI ("The DJ says: ...") -- not a display bug, a real architecture bug: this
module fixes the actual coupling rather than hiding the leaked text.

Shares only the inert, stateless pieces with agent.py: agent_tools.py's tool
*definitions* and *executors* (already plain functions with no conversational
framing of their own -- see that module's docstring). Nothing here imports
MusicAgent or its SYSTEM_PROMPT.

tool_choice forces exactly one tool call (disable_parallel_tool_use=True) and
this returns that tool's structured result directly -- no second Claude
call for the model to write a reply about what it found, since Explore never
displays that reply. This is also strictly cheaper than the DJ path (one
Claude call here vs. MusicAgent.send_message()'s two: decide-tool, then
write-reply)."""

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS
from sonic_explorer.llm.agent_tools import AGENT_TOOLS, execute_tool
from sonic_explorer.llm.explain import sanitize_untrusted_text

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Only the two tools that make sense for a single free-text query with no
# reference song and no prior conversation. search_similar_songs needs an
# existing song title to compare against -- that's Explore's literal-match
# path, already resolved before natural-language search ever runs.
# get_song_profile is a lookup, not a search.
_SEARCH_TOOLS = [t for t in AGENT_TOOLS if t["name"] in ("search_by_mood_profile", "search_by_sound_content")]

SYSTEM_PROMPT = """Translate the user's natural-language music search into exactly one tool \
call -- never reply in plain text, never ask a clarifying question, always pick your own best \
interpretation and call a tool.

For mood/vibe language (e.g. "moodier", "more stripped-back", "energetic"), reason about which \
of the five profile axes (tempo, energy, brightness, harmonic complexity, rhythmic density) it \
implies and call search_by_mood_profile with numeric values.

For a request naming a specific sound, instrument, or sound event (e.g. "crow sounds", \
"saxophone"), call search_by_sound_content with that keyword.

A genre, cultural, or style descriptor (e.g. "Spanish," "trip-hop") is neither a mood word nor a \
literal sound -- approximate it as a mood-profile nudge via search_by_mood_profile rather than \
treating it as unsearchable."""


def _sanitize_tool_result(obj):
    if isinstance(obj, str):
        return sanitize_untrusted_text(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_tool_result(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_tool_result(v) for v in obj]
    return obj


def nl_search(
    client, query: str, song_repo, embedding_repo, retrieval_service, dna_normalizer,
    normalized_dna_by_song, model: str = DEFAULT_MODEL, max_tokens: int = 512,
) -> dict:
    """Returns the raw {"matches": [...], "tool_name": ..., "tool_input": ...}
    (or {"error": ...}) dict straight from whichever single tool the model
    chose -- see this module's docstring for why no conversational reply is
    generated. tool_name/tool_input are echoed back so the caller can build
    a grounded per-result explanation (see explanation_for_search_match)
    without a second LLM call."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        tools=_SEARCH_TOOLS,
        tool_choice={"type": "any", "disable_parallel_tool_use": True},
        messages=[{"role": "user", "content": query}],
    )
    tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
    if not tool_use_blocks:
        return {"error": "The model didn't choose a search tool."}

    block = tool_use_blocks[0]
    result = _sanitize_tool_result(execute_tool(
        block.name, block.input, song_repo, embedding_repo, retrieval_service,
        dna_normalizer, normalized_dna_by_song,
    ))
    result["tool_name"] = block.name
    result["tool_input"] = block.input
    return result


def explanation_for_search_match(tool_name: str, tool_input: dict, match: dict, normalized_dna_by_song: dict) -> str | None:
    """A short, grounded "why this matched" line for one search result card
    -- built directly from the data the search tool already returned, no
    extra LLM call (search_by_sound_content's matched_tags and
    search_by_mood_profile's target values are both real, structured
    outputs, not something to re-explain via a model that might embellish
    past what the data actually shows). None means "nothing grounded to
    say" -- callers should just omit the line, not fabricate one."""
    if tool_name == "search_by_sound_content":
        matched_tags = match.get("matched_tags") or []
        if matched_tags:
            return f"Matched on tag: {', '.join(matched_tags[:3])}"
        if match.get("description"):
            return "Matched via its synthesized description."
        return None

    if tool_name == "search_by_mood_profile":
        norm = normalized_dna_by_song.get(match.get("song_id"))
        if not norm:
            return None
        deltas = sorted(
            (abs(norm[axis] - tool_input.get(axis, 0.5)), axis) for axis in AXES if axis in norm
        )
        closest = [AXIS_LABELS[axis] for _, axis in deltas[:2]]
        if not closest:
            return None
        return f"Closest match on {' and '.join(closest)}."

    return None
