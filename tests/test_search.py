"""sonic_explorer/llm/search.py -- Explore's dedicated one-shot search path,
deliberately separate from llm/agent.py's MusicAgent (see search.py's own
docstring for the real, previously-existing coupling bug this fixes: an
earlier version of Explore called MusicAgent.send_message() directly, which
is why a full conversational DJ reply used to leak into search results)."""

import json

import pytest

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.llm.agent import SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT
from sonic_explorer.llm.search import SYSTEM_PROMPT as SEARCH_SYSTEM_PROMPT
from sonic_explorer.llm.search import _SEARCH_TOOLS, explanation_for_search_match, nl_search
from sonic_explorer.models import Song
from sonic_explorer.pipeline.sound_tagging import serialize_tags
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.service import RetrievalService


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def search_deps(conn):
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn)
    retrieval_service = RetrievalService(song_repo, embedding_repo)

    crow_song = Song(filepath="/a.mp3", fma_track_id=1, title="Crow Song", artist="Artist A",
                      genre_top="Ambient", duration_sec=30.0)
    crow_id = song_repo.add_song(crow_song)
    song_repo.update_sound_tags(crow_id, serialize_tags([("Crow", 0.9), ("Bird vocalization", 0.6)]))

    mellow_song = Song(filepath="/b.mp3", fma_track_id=2, title="Mellow Song", artist="Artist B",
                        genre_top="Jazz", duration_sec=40.0)
    mellow_id = song_repo.add_song(mellow_song)
    song_repo.update_song_dna(
        mellow_id, tempo_bpm=90.0, energy=0.1, brightness=0.2, harmonic_complexity=0.5, rhythmic_density=0.5,
    )

    songs = song_repo.list_songs()
    normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in songs])
    normalized_by_song = {
        s.id: normalizer.normalize({axis: getattr(s, axis) for axis in AXES})
        for s in songs
        if all(getattr(s, axis) is not None for axis in AXES)
    }
    return song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song, crow_id, mellow_id


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, id, name, input):
        self.id = id
        self.name = name
        self.input = input


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def test_nl_search_forces_exactly_one_tool_call_no_free_text_allowed(search_deps):
    """The whole point of this module vs. MusicAgent: the model must be
    forced into calling a tool, never allowed to just reply in prose."""
    song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song, crow_id, _ = search_deps
    response = FakeResponse([FakeToolUseBlock("t1", "search_by_sound_content", {"query": "crow"})])
    client = FakeAnthropicClient(response)

    nl_search(client, "anything with crows", song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song)

    call_kwargs = client.messages.calls[0]
    assert call_kwargs["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}


def test_nl_search_uses_its_own_system_prompt_not_the_conversational_agents(search_deps):
    """Regression guard for the real coupling bug: this module must NOT
    share MusicAgent's conversational system prompt (which explicitly asks
    for plain-language chat replies) -- confirms the two are genuinely
    separate, not just visually hidden from each other."""
    song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song, crow_id, _ = search_deps
    response = FakeResponse([FakeToolUseBlock("t1", "search_by_sound_content", {"query": "crow"})])
    client = FakeAnthropicClient(response)

    nl_search(client, "crows", song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song)

    call_kwargs = client.messages.calls[0]
    assert call_kwargs["system"] == SEARCH_SYSTEM_PROMPT
    assert call_kwargs["system"] != AGENT_SYSTEM_PROMPT
    assert "friendly" not in SEARCH_SYSTEM_PROMPT.lower()
    assert "conversational" not in SEARCH_SYSTEM_PROMPT.lower()


def test_nl_search_only_exposes_query_only_tools_not_reference_song_tools():
    """search_similar_songs and get_song_profile both need an existing
    reference song title -- meaningless for a bare free-text query with no
    prior turn to reference, and not what Explore's search bar collects."""
    tool_names = {t["name"] for t in _SEARCH_TOOLS}
    assert tool_names == {"search_by_mood_profile", "search_by_sound_content"}


def test_nl_search_returns_matches_directly_with_no_conversational_reply_field(search_deps):
    """The whole fix in one assertion: the return value must carry ranked
    matches and nothing resembling free-form reply text for the UI to
    accidentally render as a chat bubble."""
    song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song, crow_id, _ = search_deps
    response = FakeResponse([FakeToolUseBlock("t1", "search_by_sound_content", {"query": "crow"})])
    client = FakeAnthropicClient(response)

    result = nl_search(client, "crow sounds", song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song)

    assert "matches" in result
    assert result["matches"][0]["song_id"] == crow_id
    assert result["tool_name"] == "search_by_sound_content"
    assert "reply" not in result
    assert "text" not in result


def test_nl_search_returns_error_when_model_somehow_returns_no_tool_use(search_deps):
    song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song, *_ = search_deps
    client = FakeAnthropicClient(FakeResponse([]))

    result = nl_search(client, "anything", song_repo, embedding_repo, retrieval_service, normalizer, normalized_by_song)

    assert "error" in result


def test_explanation_for_search_match_sound_content_names_matched_tags():
    match = {"song_id": 1, "matched_tags": ["Crow", "Bird vocalization"], "description": None}
    explanation = explanation_for_search_match("search_by_sound_content", {}, match, {})
    assert explanation == "Matched on tag: Crow, Bird vocalization"


def test_explanation_for_search_match_sound_content_falls_back_to_description():
    match = {"song_id": 1, "matched_tags": [], "description": "calm piano loop"}
    explanation = explanation_for_search_match("search_by_sound_content", {}, match, {})
    assert explanation == "Matched via its synthesized description."


def test_explanation_for_search_match_mood_profile_names_closest_axes():
    tool_input = {"tempo_bpm": 0.1, "energy": 0.1, "brightness": 0.1, "harmonic_complexity": 0.5, "rhythmic_density": 0.9}
    normalized_dna_by_song = {
        7: {"tempo_bpm": 0.15, "energy": 0.9, "brightness": 0.12, "harmonic_complexity": 0.95, "rhythmic_density": 0.15},
    }
    match = {"song_id": 7}

    explanation = explanation_for_search_match("search_by_mood_profile", tool_input, match, normalized_dna_by_song)

    assert explanation is not None
    assert "tempo" in explanation.lower() or "brightness" in explanation.lower()
    assert "energy" not in explanation.lower()  # energy has the largest delta (0.1 target vs 0.9 actual)


def test_explanation_for_search_match_returns_none_for_unknown_tool():
    assert explanation_for_search_match("some_other_tool", {}, {"song_id": 1}, {}) is None


def test_explanation_for_search_match_mood_profile_returns_none_without_dna():
    assert explanation_for_search_match("search_by_mood_profile", {}, {"song_id": 999}, {}) is None
