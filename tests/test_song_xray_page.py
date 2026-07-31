"""AppTest smoke test for Song X-Ray. No dedicated test file existed for this
page before -- test_context_passing.py only covers the xray_context_song_id
hand-off from Explore, not the page's own content. Must go through
Overview.py + switch_page for consistency with the rest of the suite's
multipage-registry requirement."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_xray() -> AppTest:
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    at.switch_page("pages/4_Song_XRay.py")
    at.run()
    return at


def _run_xray_with_verification_enabled() -> AppTest:
    """The click-track/chord-tone verification audio is gated behind an
    explicit opt-in checkbox (see 4_Song_XRay.py's own comment: st.expander
    has no lazy-content mechanism, and generating this unconditionally on
    every page load measured at ~87s cold before that gate existed) --
    pre-seeding the checkbox's own session_state key before .run() is the
    same established pattern this suite already uses to simulate a user
    interaction without a live browser."""
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=180)
    at.switch_page("pages/4_Song_XRay.py")
    at.session_state["xray_show_verification_audio"] = True
    at.run()
    return at


@pytest.fixture(scope="module")
def xray_at() -> AppTest:
    """Shared across every read-only test using the DEFAULT (verification
    off) render -- computed once instead of per test."""
    return _run_xray()


@pytest.fixture(scope="module")
def xray_at_verified() -> AppTest:
    """Shared across every read-only test using the verification-ENABLED
    render -- a separate fixture (not a mutation of xray_at) since it's a
    genuinely different initial session_state, computed once instead of
    per test."""
    return _run_xray_with_verification_enabled()


def test_song_xray_page_runs_without_exceptions(xray_at):
    assert not xray_at.exception


def test_song_xray_verification_audio_not_generated_by_default(xray_at):
    """The real performance fix this gate exists for: with the checkbox
    left unchecked (the default), the expensive audio generation must NOT
    run -- confirmed by the honest opt-in caption showing instead of any
    click-track/chord-tone content, and the audio widget count staying at
    just the song's own player."""
    assert not xray_at.exception
    expander_labels = [e.label for e in xray_at.expander]
    assert "Verify beat & chord detection by ear" in expander_labels

    caption_texts = " ".join(c.value for c in xray_at.caption)
    assert "enable the checkbox above" in caption_texts.lower()
    assert len(xray_at.get("audio")) == 1  # only the song's own player, no click/tone tracks


def test_song_xray_shows_beat_and_chord_verification_when_enabled(xray_at_verified):
    """The click-track + chord-tone-track verification tool -- an explicit
    request to make beat AND chord detection checkable by ear, not just
    trusted as a number/label. Real audio mixing (click_track_audio,
    chord_tone_audio), so this only checks it renders without exception and
    produces real audio widgets, not that the click/tone positions are
    audible-correct (that's waveform_preview's own test coverage:
    test_click_track_audio_is_louder_at_beat_positions_than_a_silent_
    original and the chord_tone_audio equivalents)."""
    assert not xray_at_verified.exception

    audios = xray_at_verified.get("audio")
    # the song's own player + the beat click-track player -- the chord tone
    # track is a 3rd, but only when chords were actually detected, so it's
    # not asserted as an unconditional minimum here (see the dedicated
    # chord-tone-track test below for that either/or check).
    assert len(audios) >= 2


def test_song_xray_verification_expander_shows_beat_count_caption(xray_at_verified):
    assert not xray_at_verified.exception
    caption_texts = " ".join(c.value for c in xray_at_verified.caption)
    assert "beats detected" in caption_texts.lower()


def test_song_xray_verification_chord_strip_or_honest_fallback(xray_at_verified):
    """Same either/or pattern as Explore's Song DNA chord test: a chart when
    chords were detected, an honest caption when they weren't (near-silent/
    unpitched audio) -- never neither, never a crash."""
    assert not xray_at_verified.exception
    charts = xray_at_verified.get("plotly_chart")
    caption_texts = " ".join(c.value for c in xray_at_verified.caption)
    chord_chart_or_fallback = (
        any("xray_chord_verify_strip" in c.proto.id for c in charts)
        or "no chords detected" in caption_texts.lower()
    )
    assert chord_chart_or_fallback


def test_song_xray_chord_tone_track_present_when_chords_detected(xray_at_verified):
    """The harmonic equivalent of the beat click track -- a real 3rd audio
    player (song + click track + chord tone track) whenever chords were
    actually detected for this song; the honest "No chords detected"
    fallback (checked above) covers the case where there's nothing to
    sonify."""
    assert not xray_at_verified.exception
    caption_texts = " ".join(c.value for c in xray_at_verified.caption)
    if "no chords detected" in caption_texts.lower():
        return  # nothing to verify for this particular song -- not a failure
    assert len(xray_at_verified.get("audio")) >= 3
    assert "root/third/fifth" in caption_texts


def test_song_xray_no_album_art_image_when_none_generated(monkeypatch, tmp_path):
    """This dev environment's own data/ library has no album_art/
    directory at all (art was only generated for deploy_data's smaller
    set) -- album_art_path_for() must return None for every song, and the
    page must simply not render an album-art image, not crash or show a
    broken one."""
    import sonic_explorer.config as config

    monkeypatch.setattr(config, "ALBUM_ART_DIR", tmp_path)  # empty dir -- no album art for any song

    at = _run_xray()

    assert not at.exception
    image_captions = " ".join(cap for img in at.get("image") for cap in img.captions)
    assert "AI-generated album art" not in image_captions


def test_song_xray_shows_real_album_art_when_it_exists_for_the_selected_song(monkeypatch, tmp_path):
    """Simulated the same way as Explore's equivalent test -- monkeypatch
    ALBUM_ART_DIR to a real temp file for the default-selected song.
    Unlike Explore's own version of this test (which only base64-encodes
    raw bytes, never decoding them), st.image() actually opens the file
    via PIL to validate/process it -- confirmed the hard way, fake
    non-image bytes raised a real UnidentifiedImageError here -- so this
    needs a real, minimal, valid PNG, not just arbitrary bytes."""
    import io

    from PIL import Image

    import sonic_explorer.config as config
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    default_song = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))[0]
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(120, 60, 200)).save(buf, format="PNG")
    (tmp_path / f"{default_song.id}.png").write_bytes(buf.getvalue())
    monkeypatch.setattr(config, "ALBUM_ART_DIR", tmp_path)

    at = _run_xray()

    assert not at.exception
    images = at.get("image")
    assert len(images) >= 1
    image_captions = " ".join(cap for img in images for cap in img.captions)
    assert "AI-generated album art" in image_captions
