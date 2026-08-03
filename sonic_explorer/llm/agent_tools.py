"""Tool definitions + executors for the conversational agent layer (spec 2.5,
Strong tier -- "over Moment Matcher + Taste Map"). Four tools, each a thin
wrapper around infrastructure that already exists and is already tested
elsewhere -- no new retrieval/search logic, just exposing it to tool-calling:

- get_song_profile: song-DNA lookup (analysis/song_dna.py's normalized space)
- search_similar_songs: facet-based retrieval (retrieval/service.py)
- search_by_mood_profile: nearest-neighbor over DNA space (analysis/song_dna.py),
  the same mechanism radar-chart-as-query uses -- the spec's explicit hook for
  "make it moodier" style requests: the LLM reasons about which axes a mood
  word implies and picks numeric values itself, no hardcoded word->axis map.
- search_by_sound_content: substring match against each song's raw AST tags
  (songs.sound_tags -- see pipeline/sound_tagging.py) and synthesized
  description (songs.description) -- the DJ's only tool that can answer
  "anything with a saxophone" / "songs with crow sounds" style requests,
  which the other three tools have no way to satisfy (they need a reference
  song, a facet, or numeric mood values, not a described sound/instrument).

Plain Python, no Streamlit/Anthropic SDK import -- these functions take
already-constructed repos/services as arguments and return JSON-serializable
dicts, so they're callable identically from a real agent loop or a test.
deserialize_tags is pure json (see pipeline/sound_tagging.py's own docstring
on why it's safe to import at module level -- no torch/transformers pulled in).

Two more tools, added for the "Ask the DJ" demo pass (research-presentation
scope, not general chatbot scope -- see the improvement-priority discussion
that motivated these):

- compare_songs: direct two-song DNA comparison, so "why are these two
  similar" gets a real per-axis answer grounded in normalized DNA distance
  (the same [0,1]^5 space nearest_songs_by_dna already uses) instead of the
  model narrating from two separate get_song_profile calls it has to
  reconcile itself.
- get_random_song: the library's only answer to "surprise me" -- a thin
  random.choice over song_repo.list_songs(), deliberately not a new
  retrieval mechanism.

A seventh tool, update_taste_profile, is a different kind of thing: it
doesn't touch song_repo/embedding_repo/retrieval_service/dna_normalizer at
all, it mutates a small caller-owned dict (the session's taste_profile) the
same way the other six read from caller-owned repos. Session-level taste
adaptation, deliberately not persistent memory -- see llm/agent.py's
docstring on how the profile is threaded through send_message() the same
way conversation history already is (passed in, returned, never stored on
MusicAgent itself)."""

import math
import random

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS, nearest_songs_by_dna
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.llm.explain import FACET_DESCRIPTIONS
from sonic_explorer.pipeline.sound_tagging import deserialize_tags

# Pulled from the real registry rather than hardcoded, so a newly-registered
# facet (e.g. the stem-separated ones) becomes usable by the agent
# automatically -- same reasoning as Explore's facet multiselect and Moment
# Matcher's facet radio, both driven by this same registry.
_FACET_NAMES = default_registry().names()
_FACET_LIST_TEXT = "; ".join(f"'{name}' ({FACET_DESCRIPTIONS[name]})" for name in _FACET_NAMES)

# Keeps a long session's taste_profile -- and the per-turn system-prompt cost
# of including it -- bounded, dropping the oldest tag in a category first
# once it fills up rather than growing without limit for the length of a chat.
MAX_TASTE_ITEMS_PER_CATEGORY = 8

AGENT_TOOLS = [
    {
        "name": "get_song_profile",
        "description": (
            "Look up a song's normalized musical DNA (tempo, energy, brightness, harmonic "
            "complexity, rhythmic density -- each 0.0 to 1.0) plus its genre. Use this to find a "
            "reference song's current profile before nudging it, e.g. for 'moodier than this' "
            "style requests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_title": {"type": "string", "description": "The song's title (or a close match)."},
            },
            "required": ["song_title"],
        },
    },
    {
        "name": "search_similar_songs",
        "description": f"Find songs sonically similar to a named song on one specific facet: {_FACET_LIST_TEXT}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "song_title": {"type": "string"},
                "facet": {"type": "string", "enum": _FACET_NAMES},
                "k": {"type": "integer", "description": "How many matches to return (default 5)."},
            },
            "required": ["song_title", "facet"],
        },
    },
    {
        "name": "search_by_mood_profile",
        "description": (
            "Find songs closest to a target mood/production profile you specify directly, each "
            "axis 0.0 to 1.0: tempo_bpm (0=slow, 1=fast), energy (0=calm, 1=intense), brightness "
            "(0=dark/warm, 1=bright/crisp), harmonic_complexity (0=simple, 1=complex chords), "
            "rhythmic_density (0=sparse, 1=busy). Use this for mood-language requests ('moodier', "
            "'more stripped-back', 'more energetic', 'brighter') by reasoning about which axes that "
            "implies and picking numeric target values yourself -- if the user references an "
            "existing song, call get_song_profile first and nudge its values rather than guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tempo_bpm": {"type": "number"},
                "energy": {"type": "number"},
                "brightness": {"type": "number"},
                "harmonic_complexity": {"type": "number"},
                "rhythmic_density": {"type": "number"},
                "k": {"type": "integer", "description": "How many matches to return (default 5)."},
            },
            "required": ["tempo_bpm", "energy", "brightness", "harmonic_complexity", "rhythmic_density"],
        },
    },
    {
        "name": "search_by_sound_content",
        "description": (
            "Find songs whose audio was actually detected (via an AudioSet-trained tagger) as "
            "containing a specific named sound, instrument, or sound event -- e.g. 'crow', "
            "'saxophone', 'applause', 'sirens' -- or whose short synthesized description matches. "
            "Use this for requests naming a concrete sound/instrument/sound-event rather than a "
            "mood or an existing reference song. One or two keywords work better than a full "
            "sentence -- pick the specific noun the user cares about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A sound/instrument/event keyword, e.g. 'crow' or 'saxophone'."},
                "k": {"type": "integer", "description": "How many matches to return (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_songs",
        "description": (
            "Directly compare two named songs on their musical DNA (tempo, energy, brightness, "
            "harmonic complexity, rhythmic density) -- returns both songs' profiles, the per-axis "
            "difference between them, and an overall DNA distance. Use this whenever the user asks "
            "why two specific songs are similar or different, or asks you to compare two named "
            "tracks -- this gives a grounded, per-axis answer instead of you having to reconcile two "
            "separate get_song_profile calls yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_title_a": {"type": "string"},
                "song_title_b": {"type": "string"},
            },
            "required": ["song_title_a", "song_title_b"],
        },
    },
    {
        "name": "get_random_song",
        "description": (
            "Pick a random song from the library, optionally restricted to one genre. Use this for "
            "open-ended discovery requests with no reference song, mood, or sound in mind -- e.g. "
            "'surprise me,' 'play me something,' 'pick anything for me.'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "description": "Optional -- restrict the random pick to this genre."},
            },
            "required": [],
        },
    },
    {
        "name": "update_taste_profile",
        "description": (
            "Record explicit like/dislike feedback the user just gave about a song or direction -- "
            "e.g. 'I like this one,' 'too energetic,' 'make it darker,' 'less electronic,' 'not a "
            "fan of that.' Call this ONLY when the user actually stated a preference, in addition "
            "to (not instead of) acting on their request normally. Use short, normalized tags for "
            "what they liked/disliked (e.g. 'dark', 'low energy', 'heavy drums'), not a verbatim "
            "quote. Never call this speculatively for a request with no evaluative language in it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "liked": {
                    "type": "array", "items": {"type": "string"},
                    "description": "New short tags to remember as liked, if the user expressed any.",
                },
                "disliked": {
                    "type": "array", "items": {"type": "string"},
                    "description": "New short tags to remember as disliked, if the user expressed any.",
                },
            },
            "required": [],
        },
    },
]


def find_song_by_title(song_repo, query_title: str):
    """Exact match wins; a single unambiguous substring match is accepted;
    anything else (no match, multiple candidates) returns None so the caller
    can report a clear error back to the model rather than silently guessing
    which song was meant."""
    query_lower = (query_title or "").strip().lower()
    if not query_lower:
        return None
    songs = song_repo.list_songs()
    exact = [s for s in songs if s.title.lower() == query_lower]
    if exact:
        return exact[0]
    partial = [s for s in songs if query_lower in s.title.lower()]
    return partial[0] if len(partial) == 1 else None


def tool_get_song_profile(song_repo, dna_normalizer, song_title: str) -> dict:
    song = find_song_by_title(song_repo, song_title)
    if song is None:
        return {"error": f"No unambiguous song found matching {song_title!r}."}
    raw = {axis: getattr(song, axis) for axis in AXES}
    if any(v is None for v in raw.values()):
        return {"error": f"{song.title} has no computed DNA yet."}
    norm = dna_normalizer.normalize(raw)
    return {
        "song_id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre_top,
        "profile": {AXIS_LABELS[a]: round(norm[a], 3) for a in AXES},
    }


def tool_search_similar_songs(song_repo, embedding_repo, retrieval_service, song_title: str, facet: str, k: int = 5) -> dict:
    if facet not in _FACET_NAMES:
        return {"error": f"Unknown facet {facet!r} -- must be one of {_FACET_NAMES}."}
    song = find_song_by_title(song_repo, song_title)
    if song is None:
        return {"error": f"No unambiguous song found matching {song_title!r}."}
    segments = song_repo.get_segments(song.id)
    if not segments:
        return {"error": f"{song.title} has no segments."}
    query_seg = segments[len(segments) // 2]
    if embedding_repo.status(query_seg.id, facet) != "done":
        return {"error": f"{song.title} has no {facet} embedding yet."}
    matches = retrieval_service.query_by_segment(query_seg.id, facet_name=facet, k=k)
    return {
        "matches": [
            {
                "song_id": m.song.id,
                "title": m.song.title,
                "artist": m.song.artist,
                "genre": m.song.genre_top,
                "similarity": round(max(0.0, m.score), 3),
            }
            for m in matches
        ]
    }


def tool_search_by_mood_profile(
    song_repo,
    normalized_dna_by_song: dict[int, dict[str, float]],
    tempo_bpm: float,
    energy: float,
    brightness: float,
    harmonic_complexity: float,
    rhythmic_density: float,
    k: int = 5,
) -> dict:
    target = {
        "tempo_bpm": tempo_bpm, "energy": energy, "brightness": brightness,
        "harmonic_complexity": harmonic_complexity, "rhythmic_density": rhythmic_density,
    }
    for axis, value in target.items():
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
            return {"error": f"{axis} must be a number between 0.0 and 1.0, got {value!r}."}

    matches = nearest_songs_by_dna(target, normalized_dna_by_song, k=k)
    songs_by_id = {s.id: s for s in song_repo.list_songs()}
    return {
        "matches": [
            {
                "song_id": m.song_id,
                "title": songs_by_id[m.song_id].title,
                "artist": songs_by_id[m.song_id].artist,
                "genre": songs_by_id[m.song_id].genre_top,
                "distance": round(m.distance, 3),
            }
            for m in matches
            if m.song_id in songs_by_id
        ]
    }


def tool_search_by_sound_content(song_repo, query: str, k: int = 5) -> dict:
    """Case-insensitive substring match against each song's raw AST tags
    (the more reliable signal for a specific named sound/instrument) and its
    synthesized description (catches vibe-language that never became a
    discrete tag). Ranked by number of matching tags first, best matching
    tag's confidence second -- a description-only hit sorts after any
    tag-matched song, since a tag is a direct model detection and the
    description is a lossy LLM-compressed summary of a handful of tags."""
    query_clean = (query or "").strip().lower()
    if not query_clean:
        return {"error": "query must not be empty."}

    scored = []
    for song in song_repo.list_songs():
        tags = deserialize_tags(song.sound_tags)
        tag_hits = [(label, score) for label, score in tags if query_clean in label.lower()]
        description_hit = bool(song.description and query_clean in song.description.lower())
        if not tag_hits and not description_hit:
            continue
        best_tag_score = max((score for _, score in tag_hits), default=0.0)
        scored.append((len(tag_hits), best_tag_score, song, tag_hits, description_hit))

    scored.sort(key=lambda t: (-t[0], -t[1]))
    return {
        "matches": [
            {
                "song_id": song.id,
                "title": song.title,
                "artist": song.artist,
                "genre": song.genre_top,
                "matched_tags": [label for label, _ in tag_hits],
                "description": song.description,
            }
            for _, _, song, tag_hits, description_hit in scored[:k]
        ]
    }


def tool_compare_songs(song_repo, dna_normalizer, song_title_a: str, song_title_b: str) -> dict:
    """Euclidean distance in the same normalized [0,1]^5 DNA space
    nearest_songs_by_dna uses, computed directly between two specific named
    songs rather than one song against the whole corpus -- the pairwise case
    that mechanism doesn't cover."""
    song_a = find_song_by_title(song_repo, song_title_a)
    if song_a is None:
        return {"error": f"No unambiguous song found matching {song_title_a!r}."}
    song_b = find_song_by_title(song_repo, song_title_b)
    if song_b is None:
        return {"error": f"No unambiguous song found matching {song_title_b!r}."}
    if song_a.id == song_b.id:
        return {"error": "Both titles resolved to the same song -- pick two different songs to compare."}

    raw_a = {axis: getattr(song_a, axis) for axis in AXES}
    raw_b = {axis: getattr(song_b, axis) for axis in AXES}
    if any(v is None for v in raw_a.values()):
        return {"error": f"{song_a.title} has no computed DNA yet."}
    if any(v is None for v in raw_b.values()):
        return {"error": f"{song_b.title} has no computed DNA yet."}

    norm_a = dna_normalizer.normalize(raw_a)
    norm_b = dna_normalizer.normalize(raw_b)

    def _summary(song, norm):
        return {
            "song_id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre_top,
            "profile": {AXIS_LABELS[a]: round(norm[a], 3) for a in AXES},
        }

    return {
        "song_a": _summary(song_a, norm_a),
        "song_b": _summary(song_b, norm_b),
        "axis_differences": {AXIS_LABELS[a]: round(abs(norm_a[a] - norm_b[a]), 3) for a in AXES},
        "same_genre": song_a.genre_top == song_b.genre_top,
        "overall_dna_distance": round(math.sqrt(sum((norm_a[a] - norm_b[a]) ** 2 for a in AXES)), 3),
    }


def tool_get_random_song(song_repo, dna_normalizer, genre: str | None = None) -> dict:
    candidates = song_repo.list_songs(genre=genre) if genre else song_repo.list_songs()
    if not candidates:
        return {"error": f"No songs found for genre {genre!r}." if genre else "No songs in the library."}

    song = random.choice(candidates)
    result = {
        "song_id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre_top,
    }
    raw = {axis: getattr(song, axis) for axis in AXES}
    if all(v is not None for v in raw.values()):
        norm = dna_normalizer.normalize(raw)
        result["profile"] = {AXIS_LABELS[a]: round(norm[a], 3) for a in AXES}
    return result


def tool_update_taste_profile(
    taste_profile: dict, liked: list[str] | None = None, disliked: list[str] | None = None
) -> dict:
    """Merges new tags into taste_profile IN PLACE (mutates the dict the
    caller handed in) -- unlike every other tool_* function here, this one's
    whole job is updating shared per-turn session state, not looking
    something up, so mutating what it was given (rather than returning a
    fresh, unrelated dict) is the point: agent.py's send_message() threads
    one taste_profile dict through every tool call in a turn, so a mid-turn
    update is visible to a later tool call in that same turn.

    Deduplicated case-insensitively, and each category capped at
    MAX_TASTE_ITEMS_PER_CATEGORY, oldest dropped first -- see that
    constant's own comment on why."""
    for category, new_tags in (("liked", liked), ("disliked", disliked)):
        if not new_tags:
            continue
        existing = taste_profile.setdefault(category, [])
        seen_lower = {tag.lower() for tag in existing}
        for tag in new_tags:
            tag_clean = (tag or "").strip()
            if not tag_clean or tag_clean.lower() in seen_lower:
                continue
            existing.append(tag_clean)
            seen_lower.add(tag_clean.lower())
        overflow = len(existing) - MAX_TASTE_ITEMS_PER_CATEGORY
        if overflow > 0:
            del existing[:overflow]

    return {"liked": list(taste_profile.get("liked", [])), "disliked": list(taste_profile.get("disliked", []))}


def execute_tool(
    tool_name: str,
    tool_input: dict,
    song_repo,
    embedding_repo,
    retrieval_service,
    dna_normalizer,
    normalized_dna_by_song: dict[int, dict[str, float]],
    taste_profile: dict,
) -> dict:
    """Single dispatch point the agent loop calls -- unknown tool names or a
    tool raising internally both become a structured {"error": ...} result
    rather than propagating, so a single bad tool call can't crash the whole
    conversation turn."""
    try:
        if tool_name == "get_song_profile":
            return tool_get_song_profile(song_repo, dna_normalizer, **tool_input)
        if tool_name == "search_similar_songs":
            return tool_search_similar_songs(song_repo, embedding_repo, retrieval_service, **tool_input)
        if tool_name == "search_by_mood_profile":
            return tool_search_by_mood_profile(song_repo, normalized_dna_by_song, **tool_input)
        if tool_name == "search_by_sound_content":
            return tool_search_by_sound_content(song_repo, **tool_input)
        if tool_name == "compare_songs":
            return tool_compare_songs(song_repo, dna_normalizer, **tool_input)
        if tool_name == "get_random_song":
            return tool_get_random_song(song_repo, dna_normalizer, **tool_input)
        if tool_name == "update_taste_profile":
            return tool_update_taste_profile(taste_profile, **tool_input)
        return {"error": f"Unknown tool {tool_name!r}."}
    except TypeError as exc:
        return {"error": f"Invalid arguments for {tool_name}: {exc}"}
