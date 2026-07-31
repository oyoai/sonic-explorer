from sonic_explorer.analysis.album_art_prompt import (
    SOUND_TAG_EXCLUDE,
    STYLE_SUFFIX,
    PercentileBucketer,
    SongDescriptors,
    build_album_art_prompt,
    fit_percentile_bucketer,
)


def make_bucketer():
    """A 3-point corpus spanning low/mid/high on each bucketed axis, so
    bucket()'s thirds land predictably for a test song sitting near an
    extreme. Percentile-based (equal-count thirds), not the min-max range
    thirds song_dna.DNANormalizer uses -- see album_art_prompt's module
    docstring for why that distinction is the whole point of this module."""
    stats = [
        {"tempo_bpm": 60.0, "energy": 0.02, "brightness": 500.0},
        {"tempo_bpm": 120.0, "energy": 0.1, "brightness": 2500.0},
        {"tempo_bpm": 200.0, "energy": 0.3, "brightness": 5000.0},
    ]
    return fit_percentile_bucketer(stats)


def make_descriptors(**overrides):
    defaults = dict(
        song_id=1, tempo_bpm=120.0, energy=0.1, brightness=2500.0,
        key_tonic="C", key_mode="major", sound_tags=[], genre="Rock", title="Test Song", artist="Test Artist",
    )
    defaults.update(overrides)
    return SongDescriptors(**defaults)


def test_prompt_is_deterministic_for_the_same_song_id_title_and_artist():
    bucketer = make_bucketer()
    descriptors = make_descriptors(song_id=42, sound_tags=["Electric guitar", "Piano"])

    first = build_album_art_prompt(descriptors, bucketer)
    second = build_album_art_prompt(descriptors, bucketer)

    assert first.prompt_text == second.prompt_text


def test_different_song_ids_can_pick_different_phrases_for_the_same_bucket():
    """Same descriptors, different song_id -- since phrase choice is seeded
    by (song_id, title, artist), two songs landing in the identical bucket
    aren't guaranteed (or expected) to get word-for-word identical prompts.
    Checked over many ids so this isn't flaky on an unlucky single draw."""
    bucketer = make_bucketer()
    prompts = {
        build_album_art_prompt(make_descriptors(song_id=i), bucketer).prompt_text
        for i in range(30)
    }
    assert len(prompts) > 1


def test_different_titles_or_artists_can_pick_different_phrases_for_the_same_song_id():
    """The whole reason title/artist were folded into the seed alongside
    song_id: two DIFFERENT songs that happen to share a song_id in a test
    fixture (never true for real data, but the seed shouldn't secretly
    ignore title/artist either) should be able to diverge in phrasing."""
    bucketer = make_bucketer()
    prompts = {
        build_album_art_prompt(make_descriptors(song_id=1, title=f"Song {i}", artist=f"Artist {i}"), bucketer).prompt_text
        for i in range(30)
    }
    assert len(prompts) > 1


def test_prompt_is_deterministic_even_though_seed_uses_title_and_artist():
    """Guards against a real, tempting bug: seeding with Python's builtin
    hash() of a string is salted per-process (PYTHONHASHSEED) and would
    silently break determinism across separate runs -- exactly the
    property a re-runnable batch export depends on. This can't directly
    prove cross-process stability from within one test process, but it
    does confirm the seed derivation is a pure function of its inputs
    (hashlib-based), not proc-local hash()."""
    from sonic_explorer.analysis.album_art_prompt import _seed_for

    assert _seed_for(42, "Some Song", "Some Artist") == _seed_for(42, "Some Song", "Some Artist")
    assert _seed_for(42, "Some Song", "Some Artist") != _seed_for(42, "Other Song", "Some Artist")


def test_high_brightness_and_low_intensity_map_to_the_right_bucket_phrases():
    bucketer = make_bucketer()
    descriptors = make_descriptors(brightness=5000.0, energy=0.02, tempo_bpm=120.0)

    prompt = build_album_art_prompt(descriptors, bucketer)

    from sonic_explorer.analysis.album_art_prompt import BRIGHTNESS_PHRASES, INTENSITY_PHRASES

    assert any(p in prompt.prompt_text for p in BRIGHTNESS_PHRASES["high"])
    assert any(p in prompt.prompt_text for p in INTENSITY_PHRASES["low"])


def test_bucketing_is_by_population_rank_not_numeric_range():
    """The real bug this module's PercentileBucketer replaced: a corpus
    with one extreme outlier used to starve the "high" bucket under
    min-max-range thirds, even for a song sitting well above the bulk of
    the population. Here, a value at the 90th percentile of a skewed
    corpus must land "high" by rank, even though it's nowhere near the
    numeric midpoint of [min, max]."""
    skewed_stats = [{"tempo_bpm": None, "energy": v, "brightness": None} for v in [0.01] * 9 + [100.0]]
    bucketer = fit_percentile_bucketer(skewed_stats)

    # 0.02 sits at the 90th percentile of this population (above all nine
    # 0.01s), but numeric-range thirds of [0.01, 100.0] would still call it
    # "low" -- percentile-rank bucketing must call it "high".
    assert bucketer.bucket("energy", 0.02) == "high"


def test_minor_key_high_intensity_gets_the_energetic_minor_phrase_not_the_somber_one():
    """The real, measured bug this exists to fix: a fast/forceful minor-key
    song used to get unconditionally somber/wistful phrasing, contradicting
    its own intensity. High-intensity minor-key songs must draw from
    MOOD_PHRASES_MINOR_HIGH_ENERGY instead of MOOD_PHRASES["minor"]."""
    bucketer = make_bucketer()
    descriptors = make_descriptors(key_mode="minor", energy=0.3)  # 0.3 is this bucketer's top energy value -> "high"

    prompt = build_album_art_prompt(descriptors, bucketer)

    from sonic_explorer.analysis.album_art_prompt import MOOD_PHRASES, MOOD_PHRASES_MINOR_HIGH_ENERGY

    assert any(p in prompt.prompt_text for p in MOOD_PHRASES_MINOR_HIGH_ENERGY)
    assert not any(p in prompt.prompt_text for p in MOOD_PHRASES["minor"])


def test_major_and_minor_mode_produce_different_mood_phrases_at_non_high_intensity():
    bucketer = make_bucketer()
    major = build_album_art_prompt(make_descriptors(song_id=1, key_mode="major", energy=0.1), bucketer)
    minor = build_album_art_prompt(make_descriptors(song_id=1, key_mode="minor", energy=0.1), bucketer)

    from sonic_explorer.analysis.album_art_prompt import MOOD_PHRASES

    assert any(p in major.prompt_text for p in MOOD_PHRASES["major"])
    assert any(p in minor.prompt_text for p in MOOD_PHRASES["minor"])
    assert not any(p in major.prompt_text for p in MOOD_PHRASES["minor"])


def test_genre_contributes_a_grounded_phrase():
    bucketer = make_bucketer()
    descriptors = make_descriptors(genre="Electronic")

    prompt = build_album_art_prompt(descriptors, bucketer)

    from sonic_explorer.analysis.album_art_prompt import GENRE_PHRASES

    assert any(p in prompt.prompt_text for p in GENRE_PHRASES["Electronic"])
    assert any(v.startswith("genre=") for v in prompt.trace.values())


def test_unlisted_genre_still_gets_an_honest_fallback_phrase():
    bucketer = make_bucketer()
    descriptors = make_descriptors(genre="Blues")  # not one of this corpus's real 8 FMA genres

    prompt = build_album_art_prompt(descriptors, bucketer)

    assert "blues" in prompt.prompt_text.lower()


def test_missing_genre_contributes_no_phrase_and_no_crash():
    bucketer = make_bucketer()
    descriptors = make_descriptors(genre=None)

    prompt = build_album_art_prompt(descriptors, bucketer)

    assert not any(v.startswith("genre=") for v in prompt.trace.values())


def test_title_and_artist_never_appear_literally_in_the_prompt_text():
    """STYLE_SUFFIX explicitly rules out literal text/imagery -- title/
    artist may widen the RNG seed (see the determinism tests above) but
    must never themselves become prompt phrase content."""
    bucketer = make_bucketer()
    descriptors = make_descriptors(title="Midnight Reverie", artist="The Wandering Echoes")

    prompt = build_album_art_prompt(descriptors, bucketer)

    assert "Midnight Reverie" not in prompt.prompt_text
    assert "The Wandering Echoes" not in prompt.prompt_text


def test_every_phrase_traces_back_to_a_real_input_value():
    """The whole point of `trace` -- an inspectable mapping table, not a
    black box. Every entry's value string must reference the actual raw
    descriptor that produced it."""
    bucketer = make_bucketer()
    descriptors = make_descriptors(
        song_id=5, tempo_bpm=200.0, energy=0.3, brightness=5000.0,
        key_tonic="A", key_mode="minor", sound_tags=["Piano"], genre="Pop",
    )

    prompt = build_album_art_prompt(descriptors, bucketer)

    assert len(prompt.trace) >= 6  # brightness, intensity, mood, pace, genre, 1 sound tag
    for phrase, source in prompt.trace.items():
        assert phrase in prompt.prompt_text
        assert any(
            marker in source
            for marker in ("brightness=", "energy=", "key=", "tempo_bpm=", "genre=", "sound_tag=")
        )


def test_missing_descriptors_degrade_gracefully_instead_of_crashing():
    bucketer = make_bucketer()
    descriptors = SongDescriptors(
        song_id=9, tempo_bpm=None, energy=None, brightness=None, key_tonic=None, key_mode=None,
        sound_tags=[], genre=None,
    )

    prompt = build_album_art_prompt(descriptors, bucketer)

    assert prompt.trace == {}
    assert STYLE_SUFFIX in prompt.prompt_text


def test_near_universal_sound_tags_are_excluded_from_phrasing():
    """'Music'/'Speech'/'Animal' etc. -- every song could plausibly get
    these, so they carry no visually-distinctive information; they must
    never surface as a phrase even though they're real detected tags."""
    bucketer = make_bucketer()
    descriptors = make_descriptors(sound_tags=list(SOUND_TAG_EXCLUDE))

    prompt = build_album_art_prompt(descriptors, bucketer)

    assert not any("sound_tag=" in v and any(t in v for t in SOUND_TAG_EXCLUDE) for v in prompt.trace.values())


def test_uncurated_sound_tag_still_gets_an_honest_fallback_phrase():
    """A label with no hand-written variant in SOUND_TAG_PHRASES must still
    produce SOME grounded phrase (the generic 'hints of X' fallback), not
    silently drop the tag or crash."""
    bucketer = make_bucketer()
    descriptors = make_descriptors(sound_tags=["Whistling"])  # deliberately not in the curated table

    prompt = build_album_art_prompt(descriptors, bucketer)

    assert "whistling" in prompt.prompt_text.lower()


def test_max_sound_tags_caps_how_many_tag_phrases_are_included():
    bucketer = make_bucketer()
    descriptors = make_descriptors(sound_tags=["Piano", "Electric guitar", "Drum kit", "Saxophone"])

    prompt = build_album_art_prompt(descriptors, bucketer, max_sound_tags=2)

    tag_trace_entries = [v for v in prompt.trace.values() if v.startswith("sound_tag=")]
    assert len(tag_trace_entries) == 2


def test_prompt_always_ends_with_the_fixed_style_suffix():
    bucketer = make_bucketer()
    prompt = build_album_art_prompt(make_descriptors(), bucketer)

    assert prompt.prompt_text.endswith(STYLE_SUFFIX)


def test_fit_percentile_bucketer_handles_no_data():
    bucketer = fit_percentile_bucketer([])
    assert isinstance(bucketer, PercentileBucketer)
    assert bucketer.bucket("energy", 0.5) is None
