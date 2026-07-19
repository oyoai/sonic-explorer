from pathlib import Path

from sonic_explorer.models import Song


def make_song(fma_track_id=42, filepath="/some/other/environment/path.mp3"):
    return Song(
        filepath=filepath, fma_track_id=fma_track_id, title="T", artist="A",
        genre_top="Rock", duration_sec=30.0,
    )


def test_audio_path_for_prefers_environment_reconstruction_when_file_exists(tmp_path, monkeypatch):
    import sonic_explorer.config as config

    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path)
    real_file = tmp_path / "42.mp3"
    real_file.write_bytes(b"fake mp3 bytes")

    song = make_song(fma_track_id=42, filepath="/a/completely/different/stale/path.mp3")
    resolved = config.audio_path_for(song)

    assert resolved == real_file


def test_audio_path_for_falls_back_to_stored_filepath_when_no_local_match(tmp_path, monkeypatch):
    import sonic_explorer.config as config

    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path)  # empty dir, no {track_id}.mp3 present

    song = make_song(fma_track_id=99, filepath="/dev/audio/dev_99.wav")
    resolved = config.audio_path_for(song)

    assert resolved == Path("/dev/audio/dev_99.wav")
