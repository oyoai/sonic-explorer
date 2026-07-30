"""Cheap, display-only audio prep for Approach's step-by-step visuals -- NOT
analysis-grade audio (that's CLAP/chroma/AST elsewhere at their real sample
rates, feeding real retrieval). This module never feeds anything back into a
similarity computation; everything here exists purely to render something on
screen.

Plain Python, no Streamlit import -- see spec 8.3's core/interface
separation, same discipline as network_graph.py/taste_map.py."""

import numpy as np

PREVIEW_SR = 4000  # low -- this is for a visual shape, not audio analysis
DEFAULT_N_POINTS = 300
CHROMA_DISPLAY_SR = 22050  # same rate the real harmony/structure analysis uses (config.STRUCTURE_SR) -- chroma needs real pitch resolution, unlike the waveform envelope's 4000Hz


def waveform_envelope(path, n_points: int = DEFAULT_N_POINTS) -> np.ndarray:
    """Downsampled amplitude envelope (max of abs() per block) -- a real
    waveform shape cheap enough to compute per page render, not a placeholder
    sine wave or synthetic shape."""
    import librosa

    audio, _ = librosa.load(str(path), sr=PREVIEW_SR, mono=True)
    if len(audio) == 0:
        return np.zeros(n_points)

    block = max(1, len(audio) // n_points)
    trimmed = audio[: block * n_points]
    return np.abs(trimmed).reshape(n_points, block).max(axis=1)


def rms_contour(path, n_points: int = DEFAULT_N_POINTS) -> np.ndarray:
    """Downsampled RMS-energy envelope (mean of librosa.feature.rms per
    block) -- a loudness-over-time shape, distinct from waveform_envelope's
    raw amplitude: RMS is a real perceptual-loudness proxy (root-mean-square
    of the signal), while the waveform envelope above is just abs()'s peak,
    which spikes on isolated transients even in a quiet passage. Same
    PREVIEW_SR=4000/display-only philosophy as waveform_envelope -- this
    feeds a Song DNA loudness-contour chart, not a similarity computation."""
    import librosa

    audio, _ = librosa.load(str(path), sr=PREVIEW_SR, mono=True)
    if len(audio) == 0:
        return np.zeros(n_points)

    rms = librosa.feature.rms(y=audio)[0]
    if len(rms) == 0:
        return np.zeros(n_points)

    block = max(1, len(rms) // n_points)
    trimmed = rms[: block * n_points]
    resized = trimmed.reshape(-1, block).mean(axis=1)
    return resized


def beat_times_for_song(path, sr: int = CHROMA_DISPLAY_SR) -> np.ndarray:
    """Real beat positions (librosa.beat.beat_track), not bar-level
    downbeats -- this project has no downbeat/meter-tracking anywhere (that
    would need a dedicated model, e.g. madmom's DBN downbeat tracker, a real
    new dependency), so this deliberately surfaces only what beat_track
    actually gives: beat-level positions, honestly labeled as such in the
    UI rather than implying bar/measure detection this doesn't do. Computed
    at CHROMA_DISPLAY_SR (not the cheap PREVIEW_SR waveform_envelope uses)
    since beat tracking accuracy degrades at very low sample rates, same
    reasoning chroma_for_display already applies for pitch resolution."""
    import librosa

    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if len(audio) == 0:
        return np.zeros(0)

    _, beat_frames = librosa.beat.beat_track(y=audio, sr=loaded_sr)
    return librosa.frames_to_time(beat_frames, sr=loaded_sr)


def waveform_envelope_for_clip(path, start_sec: float, end_sec: float, n_points: int = DEFAULT_N_POINTS) -> np.ndarray:
    """Same downsampled amplitude-envelope shape as waveform_envelope(), but
    for one exact time-range clip rather than the whole song -- Moment
    Matcher's per-moment waveform needs to show just the selected ~5s
    segment, not the full track's shape with the segment merely highlighted.
    Loads the whole file via plain librosa.load(path, sr=..., mono=True) and
    slices the resulting array in-memory (same reasoning as
    extract_window_clip() above: librosa's own offset/duration load params
    force a different, less reliable decode path than a plain full-file
    load)."""
    import librosa

    audio, loaded_sr = librosa.load(str(path), sr=PREVIEW_SR, mono=True)
    start_idx = int(start_sec * loaded_sr)
    end_idx = int(end_sec * loaded_sr)
    clip = audio[start_idx:end_idx]
    if len(clip) == 0:
        return np.zeros(n_points)

    block = max(1, len(clip) // n_points)
    trimmed = clip[: block * n_points]
    return np.abs(trimmed).reshape(n_points, block).max(axis=1)


def chroma_for_display(path, sr: int = CHROMA_DISPLAY_SR) -> tuple[np.ndarray, np.ndarray]:
    """Real chroma-over-time for the chromagram visualization (Approach's
    Harmony card) -- computed fresh at real pitch resolution, NOT a reuse of
    facets/fingerprint.py's harmony_fingerprint (a 12x32 grid, block-averaged
    across the whole song and normalized to grayscale intensity for a small
    thumbnail -- already-persisted, but too coarse and built for a different
    purpose to serve as a genuine chromagram). Also NOT harmony.py's
    per-segment 24-dim embedding, which discards the time axis entirely
    (chroma.mean(axis=1)/std(axis=1)) -- there is no already-stored artifact
    with real per-timestep resolution, so this recomputes the same way
    waveform_envelope() does: cheap, display-only, at the same sample rate
    the real analysis pipeline uses (unlike waveform_envelope's low 4000Hz)
    since chroma genuinely needs real pitch resolution to mean anything.

    Returns (chroma, times): chroma is (12, T) -- one row per pitch class,
    C first (librosa's default chroma_cqt bin order); times is (T,), the
    start-time in seconds of each column, for building a real time axis and
    mapping a click's x-position back to a playback start_time."""
    import librosa

    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if len(audio) == 0:
        return np.zeros((12, 1)), np.zeros(1)

    chroma = librosa.feature.chroma_cqt(y=audio, sr=loaded_sr)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=loaded_sr)
    return chroma, times


MEL_DISPLAY_BINS = 64  # finer than facets/fingerprint.py's 32x32 thumbnail grid, still cheap


def mel_spectrogram_for_display(path, sr: int = CHROMA_DISPLAY_SR, n_mels: int = MEL_DISPLAY_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Real mel-spectrogram-over-time for Sound's supporting visual (Approach
    Step 2) -- the raw acoustic texture (frequency energy over time) Sound is
    computed from, NOT a view of CLAP's actual embedding space (that's a
    learned, black-box representation with no raw visual proxy the way
    chroma directly *is* Harmony's input -- see this function's caller for
    the explicit framing). Computed fresh, same reasoning as
    chroma_for_display(): facets/fingerprint.py's sound_fingerprint() is a
    related, already-persisted artifact, but it's the same kind of coarse
    (32x32, whole-song block-averaged, normalized for a small thumbnail)
    artifact harmony_fingerprint was for chroma -- not suited for a real,
    properly-resolved display.

    Returns (mel_db, times): mel_db is (n_mels, T) in dB (librosa.power_to_db,
    same conversion sound_fingerprint() uses, ref=np.max); times is (T,),
    start-time in seconds per column."""
    import librosa

    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if len(audio) == 0:
        return np.zeros((n_mels, 1)), np.zeros(1)

    mel = librosa.feature.melspectrogram(y=audio, sr=loaded_sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=loaded_sr)
    return mel_db, times


def extract_window_clip(path, start_sec: float, duration_sec: float, sr: int = 22050) -> bytes:
    """Real, playable WAV bytes for one exact time-range clip of a song --
    Approach's windowing step needs actual separate playable clips per
    window, not just a visual highlight overlaid on the continuous
    full-song player. Unlike waveform_envelope above (a cheap shape for
    plotting, PREVIEW_SR=4000), this is real audio meant to be listened to,
    so it loads at a normal listening sample rate -- still not the
    analysis-grade CLAP_SR, since nothing here feeds a similarity
    computation.

    Deliberately loads the WHOLE file via plain librosa.load(path, sr=sr,
    mono=True) and slices the resulting array in-memory, rather than
    librosa's own offset/duration load params -- those force a different
    internal decode path (soundfile can't seek efficiently within a
    compressed MP3 stream, so librosa falls back to audioread) than the
    plain full-file load every other caller on this page already uses
    successfully. That fallback path broke on a real Streamlit Cloud
    deploy running Python 3.14 (ImportError inside librosa's own load(),
    redacted by Streamlit before reaching this repo's logs) -- Python
    3.13+ removed several stdlib modules audioread's older fallback chain
    can still reach for on some formats, and this project already carries
    standard-aifc/standard-sunau as an unpinned, transitively-resolved
    workaround for exactly that gap (see librosa's own DeprecationWarning
    in this repo's test output) -- not a guarantee that resolves identically
    on every fresh Python version. Loading the whole ~30s clip 4x for 4
    windows costs a few hundred ms, not a real performance concern here."""
    import io

    import librosa
    import soundfile as sf

    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    start_idx = int(start_sec * loaded_sr)
    end_idx = int((start_sec + duration_sec) * loaded_sr)
    clip = audio[start_idx:end_idx]

    buf = io.BytesIO()
    sf.write(buf, clip, loaded_sr, format="WAV")
    return buf.getvalue()


def click_track_audio(
    path, beat_times: np.ndarray, sr: int = 22050, click_freq: float = 1000.0,
    click_duration_sec: float = 0.05, click_gain: float = 0.6,
) -> bytes:
    """Real, playable WAV bytes: the song's own audio with a short
    synthesized click (an exponentially-decaying sine burst, the standard
    shape for an audible click-track tick -- a pure tone would ring on too
    long and blur into the next beat at any real tempo) mixed in at each
    detected beat_times position. This is a click-track OVERLAY, the
    standard audio-engineering technique for verifying a beat-tracker's
    output by ear against the actual music -- not a new beat-detection
    method; beat_times is expected to already come from
    beat_times_for_song() above. Same real-audio, real-sample-rate
    reasoning as extract_window_clip() (this is meant to be listened to,
    not just plotted) and the same whole-file-load-then-slice pattern for
    the same Streamlit Cloud compatibility reason -- see that function's
    own docstring. click_gain=0.6 (not 1.0) keeps the click audible without
    fully drowning out the music underneath it."""
    import io

    import librosa
    import soundfile as sf

    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if len(audio) == 0:
        return b""

    click_n = int(click_duration_sec * loaded_sr)
    t = np.linspace(0, click_duration_sec, click_n, endpoint=False)
    click_wave = np.sin(2 * np.pi * click_freq * t) * np.exp(-t * 40.0)

    mixed = audio.copy()
    for beat_sec in beat_times:
        start_idx = int(beat_sec * loaded_sr)
        if start_idx >= len(mixed):
            continue
        end_idx = min(start_idx + click_n, len(mixed))
        mixed[start_idx:end_idx] += click_wave[: end_idx - start_idx] * click_gain

    mixed = np.clip(mixed, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, mixed, loaded_sr, format="WAV")
    return buf.getvalue()


def chord_tone_audio(
    path, chord_segments, sr: int = 22050, gain: float = 0.12, octave: int = 3, fade_sec: float = 0.03,
) -> bytes:
    """The harmonic equivalent of click_track_audio() above -- same
    verification purpose (judge a detector's output by ear against the real
    audio), same real-song-plus-synthesized-marker mixing approach, just a
    sustained triad tone per chord_segments run (analysis.key_chord.
    estimate_chords() output) instead of a short percussive tick per beat.
    Root/third/fifth (major: root, +4, +7 semitones; minor: root, +3, +7 --
    the exact intervals _chord_templates() in key_chord.py already encodes,
    reused here as music theory, not reinvented), each a plain sine at
    12-tone-equal-temperament frequency (A4=440Hz reference), summed and
    averaged so three tones together aren't louder than the single-tone
    click. octave=3 (below the click's implicit register) keeps this in a
    low-mid range that reads as a distinct "chord pad" against typical vocal/
    lead ranges rather than fighting them. "N/C" (no discernible chord)
    segments are skipped entirely -- silence is the honest sonification of
    "no chord detected," not a guessed tone.

    fade_sec ramps each segment's tone in/out linearly -- an abrupt sine
    on/off is an audible click in its own right (a discontinuity at an
    arbitrary phase), which would be indistinguishable from the very
    boundary-timing errors this tool exists to help a listener judge."""
    import io

    import librosa
    import soundfile as sf

    from sonic_explorer.analysis.key_chord import PITCH_CLASSES

    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if len(audio) == 0:
        return b""

    mixed = audio.copy()
    fade_n = int(fade_sec * loaded_sr)

    for seg in chord_segments:
        if seg.label == "N/C":
            continue
        is_minor = seg.label.endswith("m")
        root_name = seg.label[:-1] if is_minor else seg.label
        if root_name not in PITCH_CLASSES:
            continue
        root_idx = PITCH_CLASSES.index(root_name)
        third_offset = 3 if is_minor else 4

        start_idx = int(seg.start_sec * loaded_sr)
        end_idx = min(int(seg.end_sec * loaded_sr), len(mixed))
        if start_idx >= len(mixed) or end_idx <= start_idx:
            continue

        n = end_idx - start_idx
        t = np.arange(n) / loaded_sr
        tone = np.zeros(n)
        for semitone_offset in (0, third_offset, 7):
            freq = 440.0 * 2.0 ** ((root_idx - 9 + semitone_offset) / 12.0 + (octave - 4))
            tone += np.sin(2 * np.pi * freq * t)
        tone /= 3.0

        actual_fade = min(fade_n, n // 2)
        if actual_fade > 0:
            ramp = np.linspace(0.0, 1.0, actual_fade)
            envelope = np.ones(n)
            envelope[:actual_fade] = ramp
            envelope[-actual_fade:] = ramp[::-1]
            tone *= envelope

        mixed[start_idx:end_idx] += tone * gain

    mixed = np.clip(mixed, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, mixed, loaded_sr, format="WAV")
    return buf.getvalue()
