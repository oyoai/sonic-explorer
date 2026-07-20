import pytest

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer, nearest_songs_by_dna


def make_raw(tempo_bpm, energy, brightness, harmonic_complexity, rhythmic_density):
    return {
        "tempo_bpm": tempo_bpm,
        "energy": energy,
        "brightness": brightness,
        "harmonic_complexity": harmonic_complexity,
        "rhythmic_density": rhythmic_density,
    }


def test_fit_normalizer_computes_min_max_per_axis():
    stats = [
        make_raw(100.0, 0.1, 1000.0, 0.2, 1.0),
        make_raw(140.0, 0.3, 3000.0, 0.8, 3.0),
    ]
    normalizer = fit_normalizer(stats)

    assert normalizer.mins["tempo_bpm"] == 100.0
    assert normalizer.maxs["tempo_bpm"] == 140.0
    assert normalizer.mins["energy"] == 0.1
    assert normalizer.maxs["energy"] == 0.3


def test_normalize_maps_extremes_to_zero_and_one():
    stats = [make_raw(100.0, 0.1, 1000.0, 0.2, 1.0), make_raw(140.0, 0.3, 3000.0, 0.8, 3.0)]
    normalizer = fit_normalizer(stats)

    low = normalizer.normalize(stats[0])
    high = normalizer.normalize(stats[1])

    for axis in AXES:
        assert low[axis] == pytest.approx(0.0)
        assert high[axis] == pytest.approx(1.0)


def test_normalize_midpoint_is_half():
    stats = [make_raw(100.0, 0.1, 1000.0, 0.2, 1.0), make_raw(140.0, 0.3, 3000.0, 0.8, 3.0)]
    normalizer = fit_normalizer(stats)

    mid = normalizer.normalize(make_raw(120.0, 0.2, 2000.0, 0.5, 2.0))
    for axis in AXES:
        assert mid[axis] == pytest.approx(0.5, abs=1e-6)


def test_normalize_handles_none_values():
    stats = [make_raw(100.0, 0.1, 1000.0, 0.2, 1.0), make_raw(140.0, 0.3, 3000.0, 0.8, 3.0)]
    normalizer = fit_normalizer(stats)

    raw_with_missing = {"tempo_bpm": None, "energy": 0.2, "brightness": None,
                         "harmonic_complexity": 0.5, "rhythmic_density": 2.0}
    result = normalizer.normalize(raw_with_missing)
    assert result["tempo_bpm"] == 0.0
    assert result["brightness"] == 0.0
    assert result["energy"] == pytest.approx(0.5)


def test_fit_normalizer_handles_empty_corpus():
    normalizer = fit_normalizer([])
    result = normalizer.normalize(make_raw(120.0, 0.2, 2000.0, 0.5, 2.0))
    assert all(v == 0.0 for v in result.values())


def test_fit_normalizer_ignores_songs_with_missing_stats():
    stats = [
        make_raw(100.0, 0.1, 1000.0, 0.2, 1.0),
        {"tempo_bpm": None, "energy": None, "brightness": None, "harmonic_complexity": None, "rhythmic_density": None},
        make_raw(140.0, 0.3, 3000.0, 0.8, 3.0),
    ]
    normalizer = fit_normalizer(stats)
    assert normalizer.mins["tempo_bpm"] == 100.0
    assert normalizer.maxs["tempo_bpm"] == 140.0


def make_norm(**overrides):
    base = {axis: 0.5 for axis in AXES}
    base.update(overrides)
    return base


def test_nearest_songs_by_dna_orders_by_distance():
    normalized_by_song = {
        1: make_norm(**{axis: 0.0 for axis in AXES}),   # far from target
        2: make_norm(),                                  # exactly the target
        3: make_norm(tempo_bpm=0.6),                      # close to the target
    }
    target = make_norm()

    matches = nearest_songs_by_dna(target, normalized_by_song, k=3)

    assert [m.song_id for m in matches] == [2, 3, 1]
    assert matches[0].distance == pytest.approx(0.0, abs=1e-9)
    assert matches[1].distance < matches[2].distance


def test_nearest_songs_by_dna_respects_k():
    normalized_by_song = {i: make_norm(tempo_bpm=i / 10) for i in range(10)}
    target = make_norm(tempo_bpm=0.0)

    matches = nearest_songs_by_dna(target, normalized_by_song, k=3)

    assert len(matches) == 3
    assert [m.song_id for m in matches] == [0, 1, 2]


def test_nearest_songs_by_dna_handles_empty_corpus():
    assert nearest_songs_by_dna(make_norm(), {}, k=6) == []
