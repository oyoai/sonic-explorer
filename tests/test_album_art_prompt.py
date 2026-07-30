from sonic_explorer.analysis.album_art_prompt import (
    SOUND_TAG_EXCLUDE,
    STYLE_SUFFIX,
    SongDescriptors,
    build_album_art_prompt,
)
from sonic_explorer.analysis.song_dna import fit_normalizer


def make_normalizer():
    """A 3-point corpus spanning low/mid/high on each bucketed axis, so
    _bucket's thirds land predictably for a test song sitting near an
    extreme."""
    stats = [
        {"tempo_bpm": 60.0, "energy": 0.02, "brightness": 500.0},
        {"tempo_bpm": 120.0, "energy": 0.1, "brightness": 2500.0},
        {"tempo_bpm": 200.0, "energy": 0.3, "brightness": 5000.0},
    ]
    return fit_normalizer(stats)


def make_descriptors(**overrides):
    defaults = dict(
        song_id=1, tempo_bpm=120.0, energy=0.1, brightness=2500.0,
        key_tonic="C", key_mode="major", sound_tags=[],
    )
    defaults.update(overrides)
    return SongDescriptors(**defaults)


def test_prompt_is_deterministic_for_the_same_song_id():
    normalizer = make_normalizer()
    descriptors = make_descriptors(song_id=42, sound_tags=["Electric guitar", "Piano"])

    first = build_album_art_prompt(descriptors, normalizer)
    second = build_album_art_prompt(descriptors, normalizer)

    assert first.prompt_text == second.prompt_text


def test_different_song_ids_can_pick_different_phrases_for_the_same_bucket():
    """Same descriptors, different song_id -- since phrase choice is seeded
    by song_id, two songs landing in the identical bucket aren't guaranteed
    (or expected) to get word-for-word identical prompts. Checked over many
    ids so this isn't flaky on an unlucky single draw."""
    normalizer = make_normalizer()
    prompts = {
        build_album_art_prompt(make_descriptors(song_id=i), normalizer).prompt_text
        for i in range(30)
    }
    assert len(prompts) > 1


def test_high_brightness_and_low_intensity_map_to_the_right_bucket_phrases():
    normalizer = make_normalizer()
    descriptors = make_descriptors(brightness=5000.0, energy=0.02, tempo_bpm=120.0)

    prompt = build_album_art_prompt(descriptors, normalizer)

    from sonic_explorer.analysis.album_art_prompt import BRIGHTNESS_PHRASES, INTENSITY_PHRASES

    assert any(p in prompt.prompt_text for p in BRIGHTNESS_PHRASES["high"])
    assert any(p in prompt.prompt_text for p in INTENSITY_PHRASES["low"])


def test_major_and_minor_mode_produce_different_mood_phrases():
    normalizer = make_normalizer()
    major = build_album_art_prompt(make_descriptors(song_id=1, key_mode="major"), normalizer)
    minor = build_album_art_prompt(make_descriptors(song_id=1, key_mode="minor"), normalizer)

    from sonic_explorer.analysis.album_art_prompt import MOOD_PHRASES

    assert any(p in major.prompt_text for p in MOOD_PHRASES["major"])
    assert any(p in minor.prompt_text for p in MOOD_PHRASES["minor"])
    assert not any(p in major.prompt_text for p in MOOD_PHRASES["minor"])


def test_every_phrase_traces_back_to_a_real_input_value():
    """The whole point of `trace` -- an inspectable mapping table, not a
    black box. Every entry's value string must reference the actual raw
    descriptor that produced it."""
    normalizer = make_normalizer()
    descriptors = make_descriptors(
        song_id=5, tempo_bpm=200.0, energy=0.3, brightness=5000.0,
        key_tonic="A", key_mode="minor", sound_tags=["Piano"],
    )

    prompt = build_album_art_prompt(descriptors, normalizer)

    assert len(prompt.trace) >= 5  # brightness, intensity, mood, pace, 1 sound tag
    for phrase, source in prompt.trace.items():
        assert phrase in prompt.prompt_text
        assert any(
            marker in source for marker in ("brightness=", "energy=", "key=", "tempo_bpm=", "sound_tag=")
        )


def test_missing_descriptors_degrade_gracefully_instead_of_crashing():
    normalizer = make_normalizer()
    descriptors = SongDescriptors(
        song_id=9, tempo_bpm=None, energy=None, brightness=None, key_tonic=None, key_mode=None, sound_tags=[],
    )

    prompt = build_album_art_prompt(descriptors, normalizer)

    assert prompt.trace == {}
    assert STYLE_SUFFIX in prompt.prompt_text


def test_near_universal_sound_tags_are_excluded_from_phrasing():
    """'Music'/'Speech'/'Animal' etc. -- every song could plausibly get
    these, so they carry no visually-distinctive information; they must
    never surface as a phrase even though they're real detected tags."""
    normalizer = make_normalizer()
    descriptors = make_descriptors(sound_tags=list(SOUND_TAG_EXCLUDE))

    prompt = build_album_art_prompt(descriptors, normalizer)

    assert not any("sound_tag=" in v and any(t in v for t in SOUND_TAG_EXCLUDE) for v in prompt.trace.values())


def test_uncurated_sound_tag_still_gets_an_honest_fallback_phrase():
    """A label with no hand-written variant in SOUND_TAG_PHRASES must still
    produce SOME grounded phrase (the generic 'hints of X' fallback), not
    silently drop the tag or crash."""
    normalizer = make_normalizer()
    descriptors = make_descriptors(sound_tags=["Whistling"])  # deliberately not in the curated table

    prompt = build_album_art_prompt(descriptors, normalizer)

    assert "whistling" in prompt.prompt_text.lower()


def test_max_sound_tags_caps_how_many_tag_phrases_are_included():
    normalizer = make_normalizer()
    descriptors = make_descriptors(sound_tags=["Piano", "Electric guitar", "Drum kit", "Saxophone"])

    prompt = build_album_art_prompt(descriptors, normalizer, max_sound_tags=2)

    tag_trace_entries = [v for v in prompt.trace.values() if v.startswith("sound_tag=")]
    assert len(tag_trace_entries) == 2


def test_prompt_always_ends_with_the_fixed_style_suffix():
    normalizer = make_normalizer()
    prompt = build_album_art_prompt(make_descriptors(), normalizer)

    assert prompt.prompt_text.endswith(STYLE_SUFFIX)
