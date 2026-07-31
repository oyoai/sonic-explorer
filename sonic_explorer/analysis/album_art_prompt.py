"""Deterministic, template-with-variety prompt text for AI-generated album
art, built from real per-song audio descriptors -- NOT an LLM call. Every
phrase in a generated prompt traces back to one specific detected feature
(see PromptTrace below), by explicit design: this exists to be inspectable,
not a black box that happens to produce plausible-sounding text.

Five descriptor axes -- brightness (spectral centroid), intensity (RMS
energy), mood (major/minor key), pace (tempo), genre (song.genre_top) --
each bucketed/mapped to a real, grounded phrase. A 6th axis, sound tags
(real AST/AudioSet labels already persisted per song, see
pipeline.sound_tagging), contributes zero or more additional phrases when
present.

Bucketing (tempo/energy/brightness -> low/mid/high) is corpus-*population*-
relative via PercentileBucketer (below), not song_dna.DNANormalizer's
min-max range. That distinction mattered in practice: DNANormalizer splits
the raw [min, max] range into equal-width thirds, so a handful of outlier
tracks stretching the range can starve the "high" bucket almost entirely --
confirmed on the real deploy set, where tempo's "high" bucket held just 1%
of songs, energy's 8%, brightness's 8%, systematically under-firing the
"forceful"/"driving"/"radiant" phrasing even for genuinely energetic songs
and producing a library-wide skew toward muted, subdued album art. This
module deliberately does NOT change DNANormalizer itself -- the Song DNA
radar chart and nearest_songs_by_dna() depend on its min-max semantics for
a different, still-correct-for-them purpose -- so PercentileBucketer is a
separate, self-contained bucketer just for this module.

"Intensity" uses energy (mean RMS, already computed/persisted) rather than
crest factor (peak/RMS ratio) -- crest factor isn't computed anywhere in
this codebase, and adding a new signal-processing pass for it wasn't worth
it when RMS energy already captures the same "how forceful does this
sound" intuition. If you have a real, hearing-verified reason crest factor
would materially change the intensity phrasing, that would be a defensible
follow-up.

Mood used to map minor-key songs to somber/wistful language unconditionally
-- a real music-theory oversimplification (plenty of energetic, upbeat
music is in a minor key), which combined with the bucketing skew above to
push a high-energy minor-key song toward "wistful, introspective" phrasing
that directly contradicted its own intensity. Minor-key songs that also
land in the high-intensity bucket now draw from a separate, energetic-
minor phrase pool instead (see MOOD_PHRASES_MINOR_HIGH_ENERGY).

Phrase selection is deterministic per song: random.Random seeded from a
hash of (song_id, title, artist) -- not song_id alone -- so two songs that
happen to land in identical buckets (same tempo/energy/brightness/mood/
genre bucket combination) aren't guaranteed to draw the same phrases, while
still being fully reproducible run to run for the exact same song. Uses
hashlib rather than Python's builtin hash() specifically because builtin
hash() of a string is salted per-process (PYTHONHASHSEED) and would NOT
reproduce the same value across separate runs -- important for a batch
export whose output (scripts/export_album_art_prompts.py) feeds a
separate, possibly-rerun Colab image-generation step."""

import hashlib
import random
from bisect import bisect_left
from dataclasses import dataclass, field

BUCKET_AXES = ["tempo_bpm", "energy", "brightness"]

BRIGHTNESS_PHRASES = {
    "low": [
        "shrouded in shadow",
        "dim, low-lit tones",
        "a dark, brooding palette",
        "muted and murky hues",
    ],
    "mid": [
        "balanced, natural tones",
        "an even, neutral palette",
        "neither bright nor dark",
    ],
    "high": [
        "radiant and luminous",
        "sharp, glinting highlights",
        "a bright, sunlit palette",
        "crisp and shimmering light",
    ],
}

INTENSITY_PHRASES = {
    "low": [
        "quiet and restrained",
        "a hushed, delicate presence",
        "soft-edged and understated",
    ],
    "mid": [
        "a steady, grounded intensity",
        "measured and even in force",
    ],
    "high": [
        "forceful and dense",
        "a powerful, driving intensity",
        "bold, saturated energy",
    ],
}

MOOD_PHRASES = {
    "major": [
        "an open, uplifted mood",
        "a warm, major-key brightness",
        "an optimistic emotional cast",
    ],
    "minor": [
        "a somber, minor-key undertone",
        "a wistful, introspective mood",
        "a shadowed emotional cast",
    ],
}

# Used instead of MOOD_PHRASES["minor"] specifically when a minor-key song
# ALSO lands in the high-intensity bucket -- minor key alone doesn't mean
# sad (a lot of upbeat, driving music is written in a minor key), so a
# fast/forceful minor-key song gets phrasing that keeps the minor-key
# color without contradicting its own measured intensity.
MOOD_PHRASES_MINOR_HIGH_ENERGY = [
    "a driving, minor-key intensity",
    "an edgy, minor-key tension",
    "a restless, high-voltage minor-key pulse",
]

PACE_PHRASES = {
    "low": [
        "a slow, unhurried pace",
        "drifting, spacious motion",
        "languid, floating movement",
    ],
    "mid": [
        "a steady, walking pace",
        "measured, even movement",
    ],
    "high": [
        "a fast, propulsive pace",
        "urgent, driving motion",
        "rapid, energetic movement",
    ],
}

# Curated variants for the AST/AudioSet labels actually observed in this
# library (checked directly against the deployed DB's real songs.sound_tags
# column, not guessed against the full ~527-class AudioSet vocabulary --
# see this module's own test/the export script's own comment for the real
# coverage number, which is genuinely sparse right now). A label reaching
# this function that ISN'T in this table still gets a phrase -- see
# _phrase_for_tag's fallback -- just a more generic, lightly-templated one
# rather than a hand-written variant.
SOUND_TAG_PHRASES: dict[str, list[str]] = {
    "Drum": ["percussive textures", "the pulse of real drums"],
    "Drum kit": ["a full drum kit's presence", "kit-driven percussion"],
    "Percussion": ["scattered percussive accents", "rhythmic percussion"],
    "Bass drum": ["a deep, thudding low end", "heavy bass-drum weight"],
    "Piano": ["resonant piano tones", "the ring of struck piano strings"],
    "Guitar": ["the grain of a real guitar", "string-driven texture"],
    "Electric guitar": ["jagged, electric-guitar edges", "amplified string bite"],
    "Acoustic guitar": ["warm, wooden acoustic tone", "unamplified string warmth"],
    "Steel guitar, slide guitar": ["a sliding, metallic string whine", "steel-string glide"],
    "Plucked string instrument": ["plucked string detail", "fine, articulated string texture"],
    "Bass guitar": ["a deep, rounded low end", "low-register string weight"],
    "Saxophone": ["a breathy brass wail", "reed-driven brass texture"],
    "Brass instrument": ["bright, blaring brass", "the shine of brass"],
    "Clarinet": ["a reedy woodwind thread", "clarinet's woody tone"],
    "Synthesizer": ["synthetic, electronic texture", "cold synthesizer tones"],
    "Electronic music": ["electronic, machine-made texture", "synthetic production"],
    "Ambient music": ["a wide, ambient wash", "diffuse, atmospheric texture"],
    "New-age music": ["soft, atmospheric layers", "gentle ambient washing"],
    "Hip hop music": ["a hip-hop groove", "beat-driven, rhythmic drive"],
    "Rapping": ["rhythmic, spoken cadence", "a rapped vocal drive"],
    "Country": ["a rootsy, twanging character", "rural, string-driven warmth"],
    "Gong": ["a resonant metallic strike", "a deep gong resonance"],
    "Reverberation": ["a spacious, echoing wash", "reverb-soaked depth"],
}

# Near-universal or non-visually-distinctive labels every song could plausibly
# get tagged with (or which are likely AST false positives on real music, e.g.
# "Crow"/"Animal") -- excluded from the candidate pool rather than forced into
# a phrase, since a phrase here wouldn't actually differentiate this song's
# art from any other's. Still real, honest data -- just not visual material.
SOUND_TAG_EXCLUDE = {"Music", "Musical instrument", "Speech", "Animal", "Crow", "Caw"}

# This library's real 8 FMA top-level genres (same fixed list
# streamlit_app/components/plotting.py's _KNOWN_GENRES uses for the network
# graph's genre coloring -- duplicated rather than imported, since
# sonic_explorer/ never imports from streamlit_app/, see CLAUDE.md). A
# genre reaching this function that isn't one of these 8 (shouldn't happen
# against this corpus, but a different/future corpus might have others)
# still gets a phrase via the honest generic fallback in _phrase_for_genre.
GENRE_PHRASES: dict[str, list[str]] = {
    "Electronic": ["a pulse of synthetic color", "cool, circuit-lit geometry"],
    "Experimental": ["unpredictable, fractured forms", "textures that resist a single shape"],
    "Folk": ["warm, handmade grain", "an earthy, acoustic warmth"],
    "Hip-Hop": ["bold, street-worn texture", "a raw, beat-driven energy"],
    "Instrumental": ["a spacious, wordless atmosphere", "an open instrumental canvas"],
    "International": ["richly patterned, cross-cultural motifs", "woven, global texture"],
    "Pop": ["polished, vivid color blocking", "a bright, catchy visual hook"],
    "Rock": ["gritty, high-contrast edges", "a raw, amplified texture"],
}

STYLE_SUFFIX = (
    "Abstract, textural album art. No faces, no text, no literal band-photo imagery."
)


@dataclass
class SongDescriptors:
    """The raw, ungrounded-in-anything-but-real-computation inputs for one
    song's prompt. tempo_bpm/energy/brightness come straight off the Song
    row (already-persisted DNA scalars); key_tonic/key_mode from a live
    estimate_key() call (never persisted -- see analysis.key_chord);
    sound_tags from the persisted songs.sound_tags column via
    pipeline.sound_tagging.deserialize_tags (already gracefully [] for a
    song with none); genre/title/artist likewise straight off the Song row.
    title/artist are NEVER turned into phrase text (STYLE_SUFFIX explicitly
    rules out literal text/imagery in the art) -- they only widen the RNG
    seed, see build_album_art_prompt."""

    song_id: int
    tempo_bpm: float | None
    energy: float | None
    brightness: float | None
    key_tonic: str | None
    key_mode: str | None  # "major" or "minor", or None if never estimated
    sound_tags: list[str] = field(default_factory=list)  # label strings only, scores not needed for phrasing
    genre: str | None = None
    title: str = ""
    artist: str = ""


@dataclass
class AlbumArtPrompt:
    song_id: int
    prompt_text: str
    trace: dict[str, str]  # phrase -> the real feature/bucket it came from, for inspectability


@dataclass
class PercentileBucketer:
    """Corpus-relative low/mid/high buckets by POPULATION rank (equal-COUNT
    thirds) rather than song_dna.DNANormalizer's equal-WIDTH numeric thirds
    -- see this module's docstring for the real, measured skew that
    difference caused. sorted_values holds each axis's real corpus values,
    pre-sorted once at fit time so bucket() can binary-search a rank
    instead of rescanning the whole corpus per song."""

    sorted_values: dict[str, list[float]]

    def bucket(self, axis: str, value: float | None) -> str | None:
        values = self.sorted_values.get(axis)
        if value is None or not values:
            return None
        rank = bisect_left(values, value) / len(values)
        if rank < 1 / 3:
            return "low"
        if rank < 2 / 3:
            return "mid"
        return "high"


def fit_percentile_bucketer(all_raw_stats: list[dict[str, float | None]]) -> PercentileBucketer:
    sorted_values = {
        axis: sorted(s[axis] for s in all_raw_stats if s.get(axis) is not None) for axis in BUCKET_AXES
    }
    return PercentileBucketer(sorted_values=sorted_values)


def _phrase_for_tag(label: str, rng: random.Random) -> str:
    variants = SOUND_TAG_PHRASES.get(label)
    if variants:
        return rng.choice(variants)
    return f"hints of {label.lower()}"  # honest, still-grounded fallback for an uncurated label


def _phrase_for_genre(genre: str, rng: random.Random) -> str:
    variants = GENRE_PHRASES.get(genre)
    if variants:
        return rng.choice(variants)
    return f"a {genre.lower()} character"  # honest, still-grounded fallback for an unlisted genre


def _seed_for(song_id: int, title: str, artist: str) -> int:
    """hashlib, not builtin hash() -- str hashing is salted per-process
    (PYTHONHASHSEED) and would silently break the determinism a re-runnable
    batch export depends on (see module docstring)."""
    material = f"{song_id}|{title}|{artist}".encode()
    return int(hashlib.sha256(material).hexdigest(), 16)


def build_album_art_prompt(
    descriptors: SongDescriptors, bucketer: PercentileBucketer, max_sound_tags: int = 2,
) -> AlbumArtPrompt:
    """One deterministic prompt per song. rng is seeded from a hash of
    (song_id, title, artist) -- not song_id alone -- so re-running this
    against the same library always produces the exact same prompt for the
    exact same song (title/artist don't change between runs for a given
    song_id), while two different songs that happen to land in identical
    buckets aren't guaranteed to draw identical phrases."""
    rng = random.Random(_seed_for(descriptors.song_id, descriptors.title, descriptors.artist))

    phrases: list[str] = []
    trace: dict[str, str] = {}

    brightness_bucket = bucketer.bucket("brightness", descriptors.brightness)
    if brightness_bucket:
        phrase = rng.choice(BRIGHTNESS_PHRASES[brightness_bucket])
        phrases.append(phrase)
        trace[phrase] = f"brightness={descriptors.brightness:.1f} -> {brightness_bucket}"

    intensity_bucket = bucketer.bucket("energy", descriptors.energy)
    if intensity_bucket:
        phrase = rng.choice(INTENSITY_PHRASES[intensity_bucket])
        phrases.append(phrase)
        trace[phrase] = f"energy={descriptors.energy:.4f} -> {intensity_bucket}"

    if descriptors.key_mode == "minor" and intensity_bucket == "high":
        phrase = rng.choice(MOOD_PHRASES_MINOR_HIGH_ENERGY)
        phrases.append(phrase)
        trace[phrase] = f"key={descriptors.key_tonic} minor, energy -> high (energetic-minor variant)"
    elif descriptors.key_mode in MOOD_PHRASES:
        phrase = rng.choice(MOOD_PHRASES[descriptors.key_mode])
        phrases.append(phrase)
        trace[phrase] = f"key={descriptors.key_tonic} {descriptors.key_mode}"

    pace_bucket = bucketer.bucket("tempo_bpm", descriptors.tempo_bpm)
    if pace_bucket:
        phrase = rng.choice(PACE_PHRASES[pace_bucket])
        phrases.append(phrase)
        trace[phrase] = f"tempo_bpm={descriptors.tempo_bpm:.0f} -> {pace_bucket}"

    if descriptors.genre:
        phrase = _phrase_for_genre(descriptors.genre, rng)
        phrases.append(phrase)
        trace[phrase] = f"genre={descriptors.genre!r}"

    candidate_tags = [t for t in descriptors.sound_tags if t not in SOUND_TAG_EXCLUDE]
    for label in candidate_tags[:max_sound_tags]:
        phrase = _phrase_for_tag(label, rng)
        phrases.append(phrase)
        trace[phrase] = f"sound_tag={label!r}"

    body = ", ".join(phrases) if phrases else "an unmarked, undetermined character"
    prompt_text = f"Album art evoking {body}. {STYLE_SUFFIX}"
    return AlbumArtPrompt(song_id=descriptors.song_id, prompt_text=prompt_text, trace=trace)
