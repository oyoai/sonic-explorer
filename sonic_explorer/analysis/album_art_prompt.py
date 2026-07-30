"""Deterministic, template-with-variety prompt text for AI-generated album
art, built from real per-song audio descriptors -- NOT an LLM call. Every
phrase in a generated prompt traces back to one specific detected feature
(see PromptTrace below), by explicit design: this exists to be inspectable,
not a black box that happens to produce plausible-sounding text.

Four descriptor axes -- brightness (spectral centroid), intensity (RMS
energy), mood (major/minor key), pace (tempo) -- each bucketed into
low/mid/high (mood: major/minor) relative to THIS CORPUS's own actual
range via analysis.song_dna.DNANormalizer, the same corpus-relative
normalization Song DNA's radar chart already uses -- not arbitrary
hardcoded thresholds. A 5th axis, sound tags (real AST/AudioSet labels
already persisted per song, see pipeline.sound_tagging), contributes
zero or more additional phrases when present.

"Intensity" uses energy (mean RMS, already computed/persisted) rather than
crest factor (peak/RMS ratio) -- crest factor isn't computed anywhere in
this codebase, and adding a new signal-processing pass for it wasn't worth
it when RMS energy already captures the same "how forceful does this
sound" intuition. If you have a real, hearing-verified reason crest factor
would materially change the intensity phrasing, that would be a defensible
follow-up.

Phrase selection is deterministic per song: random.Random(song_id) seeds
each pick, so re-running this against the same library always produces the
exact same prompt for the exact same song -- important for a batch export
whose output (scripts/export_album_art_prompts.py) feeds a separate,
possibly-rerun Colab image-generation step."""

import random
from dataclasses import dataclass, field

from sonic_explorer.analysis.song_dna import DNANormalizer

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
    song with none)."""

    song_id: int
    tempo_bpm: float | None
    energy: float | None
    brightness: float | None
    key_tonic: str | None
    key_mode: str | None  # "major" or "minor", or None if never estimated
    sound_tags: list[str] = field(default_factory=list)  # label strings only, scores not needed for phrasing


@dataclass
class AlbumArtPrompt:
    song_id: int
    prompt_text: str
    trace: dict[str, str]  # phrase -> the real feature/bucket it came from, for inspectability


def _bucket(normalized_value: float | None) -> str | None:
    if normalized_value is None:
        return None
    if normalized_value < 1 / 3:
        return "low"
    if normalized_value < 2 / 3:
        return "mid"
    return "high"


def _phrase_for_tag(label: str, rng: random.Random) -> str:
    variants = SOUND_TAG_PHRASES.get(label)
    if variants:
        return rng.choice(variants)
    return f"hints of {label.lower()}"  # honest, still-grounded fallback for an uncurated label


def build_album_art_prompt(
    descriptors: SongDescriptors, normalizer: DNANormalizer, max_sound_tags: int = 2,
) -> AlbumArtPrompt:
    """One deterministic prompt per song. rng is seeded by song_id alone
    (not song_id + anything else) so the exact same descriptors always
    produce the exact same prompt, run to run -- a batch re-export after,
    say, adding more songs to the library still gives every PREVIOUSLY
    existing song_id the identical prompt it had before, not a shuffled
    one (list order/count elsewhere in the corpus never enters this
    picture)."""
    rng = random.Random(descriptors.song_id)
    normalized = normalizer.normalize({
        "tempo_bpm": descriptors.tempo_bpm, "energy": descriptors.energy, "brightness": descriptors.brightness,
    })

    phrases: list[str] = []
    trace: dict[str, str] = {}

    brightness_bucket = _bucket(normalized.get("brightness")) if descriptors.brightness is not None else None
    if brightness_bucket:
        phrase = rng.choice(BRIGHTNESS_PHRASES[brightness_bucket])
        phrases.append(phrase)
        trace[phrase] = f"brightness={descriptors.brightness:.1f} -> {brightness_bucket}"

    intensity_bucket = _bucket(normalized.get("energy")) if descriptors.energy is not None else None
    if intensity_bucket:
        phrase = rng.choice(INTENSITY_PHRASES[intensity_bucket])
        phrases.append(phrase)
        trace[phrase] = f"energy={descriptors.energy:.4f} -> {intensity_bucket}"

    if descriptors.key_mode in MOOD_PHRASES:
        phrase = rng.choice(MOOD_PHRASES[descriptors.key_mode])
        phrases.append(phrase)
        trace[phrase] = f"key={descriptors.key_tonic} {descriptors.key_mode}"

    pace_bucket = _bucket(normalized.get("tempo_bpm")) if descriptors.tempo_bpm is not None else None
    if pace_bucket:
        phrase = rng.choice(PACE_PHRASES[pace_bucket])
        phrases.append(phrase)
        trace[phrase] = f"tempo_bpm={descriptors.tempo_bpm:.0f} -> {pace_bucket}"

    candidate_tags = [t for t in descriptors.sound_tags if t not in SOUND_TAG_EXCLUDE]
    for label in candidate_tags[:max_sound_tags]:
        phrase = _phrase_for_tag(label, rng)
        phrases.append(phrase)
        trace[phrase] = f"sound_tag={label!r}"

    body = ", ".join(phrases) if phrases else "an unmarked, undetermined character"
    prompt_text = f"Album art evoking {body}. {STYLE_SUFFIX}"
    return AlbumArtPrompt(song_id=descriptors.song_id, prompt_text=prompt_text, trace=trace)
