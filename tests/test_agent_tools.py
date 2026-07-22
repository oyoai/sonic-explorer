import numpy as np
import pytest

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.llm.agent_tools import (
    execute_tool,
    find_song_by_title,
    tool_get_song_profile,
    tool_search_by_mood_profile,
    tool_search_by_sound_content,
    tool_search_similar_songs,
)
from sonic_explorer.models import Segment, Song
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
def repos(conn):
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn)
    retrieval_service = RetrievalService(song_repo, embedding_repo)
    return song_repo, embedding_repo, retrieval_service


def add_song(song_repo, embedding_repo, title, artist, genre, dna, facet_vectors=None):
    song = Song(filepath=f"/{title}.mp3", fma_track_id=hash(title) % 100000, title=title, artist=artist,
                genre_top=genre, duration_sec=30.0)
    song_id = song_repo.add_song(song)
    if dna is not None:
        song_repo.update_song_dna(song_id, **dna)
    segments = [Segment(song_id=song_id, start_sec=i * 5.0, end_sec=(i + 1) * 5.0, segment_index=i) for i in range(3)]
    seg_ids = song_repo.add_segments(song_id, segments)
    if facet_vectors:
        for facet_name, vec in facet_vectors.items():
            for seg_id in seg_ids:
                embedding_repo.add_vector(facet_name, seg_id, np.array(vec, dtype=np.float32))
    return song_repo.get_song(song_id)


def make_dna(tempo_bpm=120.0, energy=0.5, brightness=2000.0, harmonic_complexity=0.5, rhythmic_density=2.0):
    return dict(tempo_bpm=tempo_bpm, energy=energy, brightness=brightness,
                harmonic_complexity=harmonic_complexity, rhythmic_density=rhythmic_density)


def test_find_song_by_title_exact_match(repos):
    song_repo, embedding_repo, _ = repos
    add_song(song_repo, embedding_repo, "Neon Highway", "Chrome Static", "Rock", None)
    add_song(song_repo, embedding_repo, "Neon Highway Pt. 2", "Chrome Static", "Rock", None)

    found = find_song_by_title(song_repo, "Neon Highway")
    assert found.title == "Neon Highway"


def test_find_song_by_title_unambiguous_substring(repos):
    song_repo, embedding_repo, _ = repos
    add_song(song_repo, embedding_repo, "Midnight Drive", "The Nocturnals", "Rock", None)

    found = find_song_by_title(song_repo, "midnight")
    assert found.title == "Midnight Drive"


def test_find_song_by_title_ambiguous_returns_none(repos):
    song_repo, embedding_repo, _ = repos
    add_song(song_repo, embedding_repo, "Neon Highway", "A", "Rock", None)
    add_song(song_repo, embedding_repo, "Neon Dreams", "B", "Pop", None)

    assert find_song_by_title(song_repo, "neon") is None


def test_find_song_by_title_no_match_returns_none(repos):
    song_repo, _, _ = repos
    assert find_song_by_title(song_repo, "Nonexistent Song") is None


def test_tool_get_song_profile_returns_normalized_dna(repos):
    song_repo, embedding_repo, _ = repos
    add_song(song_repo, embedding_repo, "Song A", "Artist A", "Rock", make_dna(tempo_bpm=100.0))
    add_song(song_repo, embedding_repo, "Song B", "Artist B", "Jazz", make_dna(tempo_bpm=140.0))
    normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in song_repo.list_songs()])

    result = tool_get_song_profile(song_repo, normalizer, "Song A")

    assert result["title"] == "Song A"
    assert result["genre"] == "Rock"
    assert result["profile"]["Tempo"] == pytest.approx(0.0)


def test_tool_get_song_profile_song_not_found(repos):
    song_repo, embedding_repo, _ = repos
    add_song(song_repo, embedding_repo, "Song A", "Artist A", "Rock", make_dna())
    normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in song_repo.list_songs()])

    result = tool_get_song_profile(song_repo, normalizer, "Totally Different")
    assert "error" in result


def test_tool_get_song_profile_missing_dna(repos):
    song_repo, embedding_repo, _ = repos
    add_song(song_repo, embedding_repo, "No DNA Song", "Artist", "Rock", None)
    normalizer = fit_normalizer([])

    result = tool_get_song_profile(song_repo, normalizer, "No DNA Song")
    assert "error" in result


def test_tool_search_similar_songs_returns_matches(repos):
    song_repo, embedding_repo, retrieval_service = repos
    add_song(song_repo, embedding_repo, "Query Song", "A", "Rock", None, facet_vectors={"sound": [1.0, 0.0, 0.0]})
    add_song(song_repo, embedding_repo, "Close Song", "B", "Rock", None, facet_vectors={"sound": [0.99, 0.01, 0.0]})

    result = tool_search_similar_songs(song_repo, embedding_repo, retrieval_service, "Query Song", "sound", k=5)

    assert "matches" in result
    titles = [m["title"] for m in result["matches"]]
    assert "Close Song" in titles
    assert "Query Song" not in titles  # excludes the query's own song


def test_tool_search_similar_songs_invalid_facet(repos):
    song_repo, embedding_repo, retrieval_service = repos
    add_song(song_repo, embedding_repo, "Song A", "A", "Rock", None, facet_vectors={"sound": [1.0, 0.0]})

    result = tool_search_similar_songs(song_repo, embedding_repo, retrieval_service, "Song A", "nonexistent_facet")
    assert "error" in result


def test_tool_search_similar_songs_song_not_found(repos):
    song_repo, embedding_repo, retrieval_service = repos
    result = tool_search_similar_songs(song_repo, embedding_repo, retrieval_service, "Nope", "sound")
    assert "error" in result


def test_tool_search_similar_songs_no_embedding_yet(repos):
    song_repo, embedding_repo, retrieval_service = repos
    add_song(song_repo, embedding_repo, "Unembedded Song", "A", "Rock", None)

    result = tool_search_similar_songs(song_repo, embedding_repo, retrieval_service, "Unembedded Song", "sound")
    assert "error" in result


def test_tool_search_by_mood_profile_orders_by_distance(repos):
    song_repo, embedding_repo, _ = repos
    close_song = add_song(song_repo, embedding_repo, "Close", "A", "Rock", None)
    far_song = add_song(song_repo, embedding_repo, "Far", "B", "Rock", None)
    normalized_by_song = {
        close_song.id: {axis: 0.5 for axis in AXES},
        far_song.id: {axis: 0.0 for axis in AXES},
    }

    result = tool_search_by_mood_profile(
        song_repo, normalized_by_song, tempo_bpm=0.5, energy=0.5, brightness=0.5,
        harmonic_complexity=0.5, rhythmic_density=0.5, k=2,
    )

    assert result["matches"][0]["title"] == "Close"
    assert result["matches"][1]["title"] == "Far"


def test_tool_search_by_mood_profile_rejects_out_of_range_values(repos):
    song_repo, embedding_repo, _ = repos
    result = tool_search_by_mood_profile(
        song_repo, {}, tempo_bpm=1.5, energy=0.5, brightness=0.5, harmonic_complexity=0.5, rhythmic_density=0.5,
    )
    assert "error" in result


def test_tool_search_by_sound_content_matches_raw_tags(repos):
    song_repo, embedding_repo, _ = repos
    crow_song = add_song(song_repo, embedding_repo, "Crow Song", "A", "Folk", None)
    add_song(song_repo, embedding_repo, "Sax Jam", "B", "Jazz", None)
    song_repo.update_sound_tags(crow_song.id, serialize_tags([("Crow", 0.3), ("Bird vocalization", 0.1)]))

    result = tool_search_by_sound_content(song_repo, "crow")

    titles = [m["title"] for m in result["matches"]]
    assert titles == ["Crow Song"]
    assert result["matches"][0]["matched_tags"] == ["Crow"]


def test_tool_search_by_sound_content_matches_description_when_no_tag_hit(repos):
    song_repo, embedding_repo, _ = repos
    song = add_song(song_repo, embedding_repo, "Mellow Tune", "A", "Jazz", None)
    song_repo.update_description(song.id, "Sassy hip hop with a mellow groove")

    result = tool_search_by_sound_content(song_repo, "sassy")

    assert [m["title"] for m in result["matches"]] == ["Mellow Tune"]
    assert result["matches"][0]["matched_tags"] == []


def test_tool_search_by_sound_content_ranks_tag_hits_before_description_only(repos):
    song_repo, embedding_repo, _ = repos
    tag_song = add_song(song_repo, embedding_repo, "Tag Match", "A", "Rock", None)
    song_repo.update_sound_tags(tag_song.id, serialize_tags([("Trumpet", 0.5)]))
    desc_song = add_song(song_repo, embedding_repo, "Description Match", "B", "Rock", None)
    song_repo.update_description(desc_song.id, "Bright trumpet-adjacent brass tone")

    result = tool_search_by_sound_content(song_repo, "trumpet")

    titles = [m["title"] for m in result["matches"]]
    assert titles.index("Tag Match") < titles.index("Description Match")


def test_tool_search_by_sound_content_no_match_returns_empty_list(repos):
    song_repo, embedding_repo, _ = repos
    add_song(song_repo, embedding_repo, "Song A", "A", "Rock", None)

    result = tool_search_by_sound_content(song_repo, "nonexistent_sound_xyz")
    assert result["matches"] == []


def test_tool_search_by_sound_content_empty_query_returns_error(repos):
    song_repo, embedding_repo, _ = repos
    result = tool_search_by_sound_content(song_repo, "   ")
    assert "error" in result


def test_execute_tool_dispatches_correctly(repos):
    song_repo, embedding_repo, retrieval_service = repos
    add_song(song_repo, embedding_repo, "Song A", "Artist A", "Rock", make_dna())
    normalizer = fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in song_repo.list_songs()])

    result = execute_tool(
        "get_song_profile", {"song_title": "Song A"},
        song_repo, embedding_repo, retrieval_service, normalizer, {},
    )
    assert result["title"] == "Song A"


def test_execute_tool_dispatches_search_by_sound_content(repos):
    song_repo, embedding_repo, retrieval_service = repos
    song = add_song(song_repo, embedding_repo, "Crow Song", "A", "Folk", None)
    song_repo.update_sound_tags(song.id, serialize_tags([("Crow", 0.3)]))

    result = execute_tool(
        "search_by_sound_content", {"query": "crow"},
        song_repo, embedding_repo, retrieval_service, None, {},
    )
    assert result["matches"][0]["title"] == "Crow Song"


def test_execute_tool_unknown_tool_name_returns_error(repos):
    song_repo, embedding_repo, retrieval_service = repos
    result = execute_tool("nonexistent_tool", {}, song_repo, embedding_repo, retrieval_service, None, {})
    assert "error" in result


def test_execute_tool_invalid_arguments_returns_error_not_exception(repos):
    song_repo, embedding_repo, retrieval_service = repos
    result = execute_tool(
        "get_song_profile", {"unexpected_arg": "x"},
        song_repo, embedding_repo, retrieval_service, None, {},
    )
    assert "error" in result
