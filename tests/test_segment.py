from sonic_explorer.pipeline.segment import segment_song


def test_segment_song_covers_full_duration_with_expected_count():
    segments = segment_song(song_id=1, duration_sec=30.0, window_sec=5.0, hop_sec=2.5)
    assert len(segments) == 11
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 5.0
    assert segments[-1].end_sec <= 30.0
    for seg in segments:
        assert seg.end_sec - seg.start_sec == 5.0
        assert seg.song_id == 1


def test_segment_song_indexes_are_sequential():
    segments = segment_song(song_id=1, duration_sec=30.0, window_sec=5.0, hop_sec=2.5)
    assert [s.segment_index for s in segments] == list(range(len(segments)))


def test_segment_song_shorter_than_one_window_returns_single_segment():
    segments = segment_song(song_id=1, duration_sec=3.0, window_sec=5.0, hop_sec=2.5)
    assert len(segments) == 1
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 3.0
