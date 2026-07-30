import io

import numpy as np
import soundfile as sf

from sonic_explorer.analysis.key_chord import ChordSegment
from sonic_explorer.analysis.waveform_preview import (
    CHROMA_DISPLAY_SR,
    MEL_DISPLAY_BINS,
    PREVIEW_SR,
    beat_times_for_song,
    chord_tone_audio,
    chroma_for_display,
    click_track_audio,
    extract_window_clip,
    mel_spectrogram_for_display,
    rms_contour,
    waveform_envelope,
    waveform_envelope_for_clip,
)


def make_sine_wav(path, duration_sec, freq=440.0, sr=PREVIEW_SR):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), audio, sr)


def test_waveform_envelope_returns_requested_length(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=5.0)

    envelope = waveform_envelope(path, n_points=100)

    assert envelope.shape == (100,)


def test_waveform_envelope_nonzero_for_real_tone(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=3.0, freq=440.0)

    envelope = waveform_envelope(path, n_points=50)

    assert envelope.max() > 0
    assert (envelope >= 0).all()


def test_waveform_envelope_silence_is_all_zero(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(PREVIEW_SR * 2), PREVIEW_SR)

    envelope = waveform_envelope(path, n_points=50)

    assert envelope.max() == 0


def test_waveform_envelope_for_clip_returns_requested_length(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=10.0, freq=440.0)

    envelope = waveform_envelope_for_clip(path, start_sec=2.0, end_sec=7.0, n_points=80)

    assert envelope.shape == (80,)
    assert envelope.max() > 0


def test_waveform_envelope_for_clip_matches_full_song_shape_at_same_offset():
    """Moment Matcher's per-moment waveform must show that moment's own
    audio, not a generic reused shape -- a clip cut from a different, silent
    stretch of the same file must look different (near-zero) from one cut
    from the tone."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tone_then_silence.wav"
        t = np.linspace(0, 5.0, int(PREVIEW_SR * 5.0), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        silence = np.zeros(int(PREVIEW_SR * 5.0))
        sf.write(str(path), np.concatenate([tone, silence]), PREVIEW_SR)

        tone_clip = waveform_envelope_for_clip(path, start_sec=0.0, end_sec=5.0, n_points=50)
        silence_clip = waveform_envelope_for_clip(path, start_sec=5.0, end_sec=10.0, n_points=50)

        assert tone_clip.max() > 0
        assert silence_clip.max() == 0


def test_extract_window_clip_returns_valid_wav_bytes_of_the_requested_duration(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=10.0, freq=440.0, sr=22050)

    clip_bytes = extract_window_clip(path, start_sec=2.0, duration_sec=5.0, sr=22050)

    assert isinstance(clip_bytes, bytes)
    audio, sr = sf.read(io.BytesIO(clip_bytes))
    assert sr == 22050
    assert abs(len(audio) / sr - 5.0) < 0.05  # real ~5s clip, not the whole 10s file


def test_extract_window_clip_offset_actually_moves_the_window(tmp_path):
    """A clip starting partway through a tone that fades in must differ from
    a clip starting at 0 -- confirms `offset` is really slicing, not just
    truncating duration from the start every time."""
    path = tmp_path / "fade.wav"
    sr = 22050
    duration_sec = 10.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    fade_in = np.clip(t / duration_sec, 0, 1)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t) * fade_in
    sf.write(str(path), audio, sr)

    early = extract_window_clip(path, start_sec=0.0, duration_sec=1.0, sr=sr)
    late = extract_window_clip(path, start_sec=8.0, duration_sec=1.0, sr=sr)

    early_audio, _ = sf.read(io.BytesIO(early))
    late_audio, _ = sf.read(io.BytesIO(late))
    assert np.abs(late_audio).mean() > np.abs(early_audio).mean() * 2  # late window is much louder (faded in)


def test_chroma_for_display_returns_twelve_rows_matching_times_length(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=5.0, freq=440.0, sr=CHROMA_DISPLAY_SR)

    chroma, times = chroma_for_display(path)

    assert chroma.shape[0] == 12
    assert chroma.shape[1] == len(times)


def test_chroma_for_display_times_are_monotonic_and_span_the_clip(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=6.0, freq=440.0, sr=CHROMA_DISPLAY_SR)

    _, times = chroma_for_display(path)

    assert (np.diff(times) > 0).all()
    assert times[0] == 0.0
    assert times[-1] < 6.0  # frame starts, not ends -- last frame starts before the clip's actual end
    assert times[-1] > 4.0  # but still real coverage of most of the clip, not just the first instant


def test_chroma_for_display_identifies_the_right_pitch_class_for_a_pure_tone(tmp_path):
    """A pure 440Hz tone is A4 -- chroma's strongest mean energy across the
    whole clip must land on the "A" bin (index 9 in librosa's default
    chroma_cqt bin order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B), a real
    correctness check on the actual pitch content, not just shape."""
    path = tmp_path / "a4.wav"
    make_sine_wav(path, duration_sec=5.0, freq=440.0, sr=CHROMA_DISPLAY_SR)

    chroma, _ = chroma_for_display(path)

    mean_energy = chroma.mean(axis=1)
    assert int(np.argmax(mean_energy)) == 9  # "A"


def test_chroma_for_display_handles_silence_without_crashing(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(CHROMA_DISPLAY_SR * 3), CHROMA_DISPLAY_SR)

    chroma, times = chroma_for_display(path)

    assert np.all(np.isfinite(chroma))
    assert len(times) > 0


def test_mel_spectrogram_for_display_returns_configured_bins_matching_times_length(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=5.0, freq=440.0, sr=CHROMA_DISPLAY_SR)

    mel_db, times = mel_spectrogram_for_display(path)

    assert mel_db.shape[0] == MEL_DISPLAY_BINS
    assert mel_db.shape[1] == len(times)


def test_mel_spectrogram_for_display_tone_has_real_spread_silence_is_flat(tmp_path):
    """A real correctness check, not just shape. librosa.power_to_db's
    ref=np.max normalizes each clip independently, so .max() is always ~0dB
    for any non-degenerate input -- not a useful cross-clip comparison. The
    real signal is spread: true silence is every bin sitting at the same
    floor value (relative to its own trivial "max"), so std is exactly 0;
    a genuine tone has real variation across frequency bins, so std is
    meaningfully positive."""
    tone_path = tmp_path / "tone.wav"
    make_sine_wav(tone_path, duration_sec=3.0, freq=440.0, sr=CHROMA_DISPLAY_SR)
    silence_path = tmp_path / "silence.wav"
    sf.write(str(silence_path), np.zeros(CHROMA_DISPLAY_SR * 3), CHROMA_DISPLAY_SR)

    tone_mel, _ = mel_spectrogram_for_display(tone_path)
    silence_mel, _ = mel_spectrogram_for_display(silence_path)

    assert silence_mel.std() == 0.0
    assert tone_mel.std() > 5.0


def test_mel_spectrogram_for_display_handles_silence_without_crashing(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(CHROMA_DISPLAY_SR * 3), CHROMA_DISPLAY_SR)

    mel_db, times = mel_spectrogram_for_display(path)

    assert np.all(np.isfinite(mel_db))
    assert len(times) > 0


def make_click_train_wav(path, duration_sec, bpm=120.0, sr=CHROMA_DISPLAY_SR):
    """A real percussive pulse train at a known tempo -- a pure sine tone (as
    make_sine_wav produces) has no rhythmic onsets for beat_track to lock
    onto, so beat detection needs its own kind of synthetic signal, the same
    way the pitch-class tests above needed a pure tone rather than noise."""
    interval_sec = 60.0 / bpm
    n_samples = int(sr * duration_sec)
    audio = np.zeros(n_samples)
    click_len = int(0.01 * sr)
    t = np.arange(click_len) / sr
    click = np.sin(2 * np.pi * 2000.0 * t) * np.exp(-t * 200.0)
    beat_sec = 0.0
    while beat_sec < duration_sec:
        start = int(beat_sec * sr)
        end = min(start + click_len, n_samples)
        audio[start:end] += click[: end - start]
        beat_sec += interval_sec
    sf.write(str(path), audio, sr)


def test_rms_contour_returns_requested_length_for_a_long_clip(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=60.0, freq=440.0)

    contour = rms_contour(path, n_points=100)

    assert contour.shape == (100,)


def test_rms_contour_is_a_real_loudness_measure_not_raw_amplitude():
    """A loud tone followed by a quiet (heavily attenuated) tone of the same
    frequency must show real RMS contrast between the two halves -- confirms
    this reads actual energy, not a constant/degenerate value."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "loud_then_quiet.wav"
        t = np.linspace(0, 5.0, int(PREVIEW_SR * 5.0), endpoint=False)
        loud = 0.9 * np.sin(2 * np.pi * 440.0 * t)
        quiet = 0.05 * np.sin(2 * np.pi * 440.0 * t)
        sf.write(str(path), np.concatenate([loud, quiet]), PREVIEW_SR)

        contour = rms_contour(path, n_points=100)

        first_half_mean = contour[:45].mean()  # away from the loud/quiet boundary
        second_half_mean = contour[55:].mean()
        assert first_half_mean > second_half_mean * 3


def test_rms_contour_silence_is_all_zero(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(PREVIEW_SR * 2), PREVIEW_SR)

    contour = rms_contour(path, n_points=50)

    assert contour.max() == 0


def test_beat_times_for_song_recovers_roughly_the_right_tempo(tmp_path):
    path = tmp_path / "clicks.wav"
    make_click_train_wav(path, duration_sec=10.0, bpm=120.0)

    beats = beat_times_for_song(path)

    assert len(beats) >= 8  # ~120bpm over 10s is ~20 beats; a loose floor, not an exact count
    intervals = np.diff(beats)
    assert abs(np.median(intervals) - 0.5) < 0.15  # 120bpm -> 0.5s/beat, loose tolerance for tracker jitter


def test_beat_times_for_song_silence_returns_no_beats(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(CHROMA_DISPLAY_SR * 3), CHROMA_DISPLAY_SR)

    beats = beat_times_for_song(path)

    assert len(beats) == 0


def test_click_track_audio_returns_valid_longer_or_equal_length_wav(tmp_path):
    path = tmp_path / "clicks.wav"
    make_click_train_wav(path, duration_sec=6.0, bpm=120.0)
    beats = beat_times_for_song(path)

    mixed_bytes = click_track_audio(path, beats)

    assert isinstance(mixed_bytes, bytes)
    audio, sr = sf.read(io.BytesIO(mixed_bytes))
    assert len(audio) / sr > 5.0  # real ~6s output, not empty/truncated


def test_click_track_audio_is_louder_at_beat_positions_than_a_silent_original():
    """Mixing a click into an otherwise-silent original must produce real
    audible energy exactly where beat_times says a beat is -- confirms this
    is actually inserting clicks, not silently passing the original through
    unchanged."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "silence.wav"
        sr = 22050
        sf.write(str(path), np.zeros(sr * 4), sr)

        beats = np.array([1.0, 2.0, 3.0])
        mixed_bytes = click_track_audio(path, beats, sr=sr)
        audio, _ = sf.read(io.BytesIO(mixed_bytes))

        at_beat = np.abs(audio[int(1.0 * sr):int(1.0 * sr) + int(0.02 * sr)]).mean()
        between_beats = np.abs(audio[int(1.3 * sr):int(1.3 * sr) + int(0.02 * sr)]).mean()
        assert at_beat > 0
        assert at_beat > between_beats * 10


def test_click_track_audio_empty_signal_returns_empty_bytes(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.zeros(0), 22050)

    result = click_track_audio(path, np.array([1.0, 2.0]))

    assert result == b""


def test_chord_tone_audio_adds_real_energy_only_during_chord_segments():
    """The harmonic equivalent of the beat click-track energy test above: a
    tone must appear exactly during its chord_segments window on an
    otherwise-silent original, and be silent before/after every segment --
    confirms this actually gates tones to their labeled time ranges, not a
    constant drone or a no-op."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "silence.wav"
        sr = 22050
        sf.write(str(path), np.zeros(sr * 4), sr)

        segments = [ChordSegment(start_sec=1.0, end_sec=2.0, label="C")]
        mixed_bytes = chord_tone_audio(path, segments, sr=sr)
        audio, _ = sf.read(io.BytesIO(mixed_bytes))

        during = np.abs(audio[int(1.5 * sr):int(1.5 * sr) + 200]).mean()
        before = np.abs(audio[int(0.5 * sr):int(0.5 * sr) + 200]).mean()
        after = np.abs(audio[int(3.0 * sr):int(3.0 * sr) + 200]).mean()
        assert during > 0
        assert before == 0
        assert after == 0


def test_chord_tone_audio_major_and_minor_produce_different_tones():
    """C major (root/+4/+7) and C minor (root/+3/+7) share the same root and
    fifth but differ on the third -- a real correctness check that the 'm'
    suffix actually changes the synthesized interval, not just the label
    text."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "silence.wav"
        sr = 22050
        sf.write(str(path), np.zeros(sr * 3), sr)

        major_bytes = chord_tone_audio(path, [ChordSegment(start_sec=0.5, end_sec=1.5, label="C")], sr=sr)
        minor_bytes = chord_tone_audio(path, [ChordSegment(start_sec=0.5, end_sec=1.5, label="Cm")], sr=sr)

        major_audio, _ = sf.read(io.BytesIO(major_bytes))
        minor_audio, _ = sf.read(io.BytesIO(minor_bytes))
        window = slice(int(1.0 * sr), int(1.0 * sr) + 2000)
        assert not np.allclose(major_audio[window], minor_audio[window])


def test_chord_tone_audio_skips_no_chord_segments():
    """'N/C' (no discernible chord) must stay silent -- sonifying a guess for
    a segment the detector itself flagged as having no clear chord would be
    actively misleading for a verification tool."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "silence.wav"
        sr = 22050
        sf.write(str(path), np.zeros(sr * 3), sr)

        mixed_bytes = chord_tone_audio(path, [ChordSegment(start_sec=0.5, end_sec=2.0, label="N/C")], sr=sr)
        audio, _ = sf.read(io.BytesIO(mixed_bytes))

        assert np.abs(audio).max() == 0


def test_chord_tone_audio_empty_signal_returns_empty_bytes(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.zeros(0), 22050)

    result = chord_tone_audio(path, [ChordSegment(start_sec=0.0, end_sec=1.0, label="C")])

    assert result == b""
