"""Tool definitions + executors for the conversational agent layer (spec 2.5,
Strong tier -- "over Moment Matcher + Taste Map"). Three tools, each a thin
wrapper around infrastructure that already exists and is already tested
elsewhere -- no new retrieval/search logic, just exposing it to tool-calling:

- get_song_profile: song-DNA lookup (analysis/song_dna.py's normalized space)
- search_similar_songs: facet-based retrieval (retrieval/service.py)
- search_by_mood_profile: nearest-neighbor over DNA space (analysis/song_dna.py),
  the same mechanism radar-chart-as-query uses -- the spec's explicit hook for
  "make it moodier" style requests: the LLM reasons about which axes a mood
  word implies and picks numeric values itself, no hardcoded word->axis map.

Plain Python, no Streamlit/Anthropic SDK import -- these functions take
already-constructed repos/services as arguments and return JSON-serializable
dicts, so they're callable identically from a real agent loop or a test."""

from sonic_explorer.analysis.song_dna import AXES, AXIS_LABELS, nearest_songs_by_dna
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.llm.explain import FACET_DESCRIPTIONS

# Pulled from the real registry rather than hardcoded, so a newly-registered
# facet (e.g. the stem-separated ones) becomes usable by the agent
# automatically -- same reasoning as Explore's facet multiselect and Moment
# Matcher's facet radio, both driven by this same registry.
_FACET_NAMES = default_registry().names()
_FACET_LIST_TEXT = "; ".join(f"'{name}' ({FACET_DESCRIPTIONS[name]})" for name in _FACET_NAMES)

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
                "title": songs_by_id[m.song_id].title,
                "artist": songs_by_id[m.song_id].artist,
                "genre": songs_by_id[m.song_id].genre_top,
                "distance": round(m.distance, 3),
            }
            for m in matches
            if m.song_id in songs_by_id
        ]
    }


def execute_tool(
    tool_name: str,
    tool_input: dict,
    song_repo,
    embedding_repo,
    retrieval_service,
    dna_normalizer,
    normalized_dna_by_song: dict[int, dict[str, float]],
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
        return {"error": f"Unknown tool {tool_name!r}."}
    except TypeError as exc:
        return {"error": f"Invalid arguments for {tool_name}: {exc}"}
