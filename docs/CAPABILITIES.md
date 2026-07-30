# Sonic Explorer — Complete Capability Reference

This document is not a build history (see `PROJECT_HISTORY.md` for that) and not a
rebuild plan (`REBUILD_PLAN.md`). It's a snapshot of **everything the system can
actually do today**, laid out in the order each capability needs to exist for the
next one to make sense — which also happens to roughly track the order a user moves
through the app itself: from the raw data, through the engines that turn audio into
comparable signal, through what gets derived per song, into the ways you can actually
search and match, through the layers that explain results in plain language, into the
screens that put all of it in front of a user, and finally the operational
capabilities (evaluation, deployment, security) that keep the whole thing honest and
running.

---

## Part 1: The Library Itself

The system operates on **1,400 songs** drawn from the Free Music Archive (FMA),
Creative Commons–licensed, spanning **8 genres**: Electronic, Experimental, Folk,
Hip-Hop, Instrumental, International, Pop, and Rock (unevenly represented —
International and Electronic are the largest groups, several genres sit at well under
half that count, which matters when reading any genre-based evaluation number: a
facet has an easier time on an over-represented genre). Every clip is a uniform
**30.0-second preview**, not a full track — confirmed empirically (every one of the
1,400 clips falls within 0.05 seconds of exactly 30.0s), and this fact directly shapes
what the Structure facet can honestly claim to show (see Part 3).

Each song is chopped into fixed-length, overlapping **segments** — 5.0-second
windows, 2.5-second hop — giving roughly 10–11 segments per song and **14,602
segments total** across the library. Segments, not whole songs, are the unit almost
every facet actually operates on: when you "match a moment," you're matching one
5-second window against every other 5-second window in the library on whichever facet
you've selected.

---

## Part 2: The Six Similarity Facets — What "Sounds Similar" Actually Means, Six Different Ways

The system's central idea is that "this sounds like that" isn't one measurement. Six
independent facets each capture a genuinely different aspect of a song, computed
completely separately, searchable completely separately, with no facet's score ever
folded into another's by default.

**Sound** — the general timbre/instrumentation/production-character facet, computed
via **CLAP** (`laion/clap-htsat-unfused`, a pretrained joint audio-text embedding
model, 512-dimensional output, run at 48kHz). This is the facet that captures the
broadest, most holistic "what does this sound like" signal — production style,
overall texture, instrumentation blend — and empirically the strongest one: its
nearest neighbors share genre 54.4% of the time versus an 11.9% random baseline
(genre-cohesion@10, see Part 11), by far the largest margin of any facet.

**Harmony** — key, chord color, and tonal similarity, computed via chroma-CQT
features (12 pitch classes, mean + standard deviation = 24 dimensions) rather than
any pretrained model — a hand-crafted feature specifically because harmony is a
narrower, more precisely definable property than "general sound." The raw chroma
space has very little natural spread (random pairs of segments score 0.85–0.95
cosine similarity before any correction), so the live index applies a corpus-wide
**whitening transform** (each dimension rescaled to zero mean / unit variance before
re-normalizing) purely to spread the score geometry out — this sharpens individual
match rankings substantially (top-1-vs-top-2 margin roughly 7× tighter after
whitening) without changing what the facet is fundamentally capable of separating;
its genre-cohesion (21.4% vs. 11.7% baseline) is the weakest of the six, honestly
reflecting that a 24-dimensional mean+std summary is a coarse representation of
harmony, not a failure of the whitening fix itself.

**Vocal, Drums, Bass, Instrumental** — four facets built by first running **Demucs**
(a pretrained source-separation model, `htdemucs`) on each song to split it into
isolated vocal/drums/bass/other stems, then running the *same* CLAP model used for
the Sound facet on each isolated stream instead of the full mix. This means "vocal
similarity" is architecturally nothing more than "Sound facet, but pointed at an
isolated stem" — no separate embedding technique, no separate architecture, just a
different audio input. Every one of these four facets is protected by an **energy
gate**: before a segment's isolated-stem audio ever gets embedded, its RMS energy is
compared against the full mix's RMS for that same segment, and if the stem is
near-silent relative to the mix (an instrumental track's "vocal" stem, an a cappella
track's "drum" stem), that segment is marked **skipped** rather than embedded — a
near-silent stem embedded anyway would make unrelated quiet tracks look falsely
similar to each other purely because they're both silent, not because they share any
real signal. Current library coverage after the gate: Vocal 8,674 segments actually
embedded (~59% of segments have real vocal content), Drums 10,395 (~71%), Bass 10,484
(~72%), Instrumental 13,858 (~95%, since nearly every song has *some* backing
instrumentation). Genre-cohesion: Instrumental 41.5%, Vocal 38.1%, Drums 37.4%, Bass
26.7% — all clearly above their respective random baselines (11.8–13.9%).

Every facet implements the same `Facet` interface (`embed`, `similarity`) and is
listed in a central `FacetRegistry` — nothing in the UI, the retrieval logic, or the
agent's tools ever hardcodes which facets exist; every facet-picker, every dropdown,
every tool description pulls the current facet list live from this registry.

---

## Part 3: Structure and Abstractivity — the One Facet That's Visualized, Not Searched

Structure is architecturally different from the six facets above: it's a **song-level
visualization**, not a per-segment embedding you can run nearest-neighbor search
against. Given a song, a **self-similarity matrix** (beat-synchronized chroma,
compared against itself across time) reveals which moments in the song sound like
which other moments — bright off-diagonal stripes mean a section repeats later (a
verse coming back, a chorus recurring). The main diagonal is deliberately left
blank/zeroed (a song trivially matches itself at every instant; the interesting signal
is *other* matches), so the visualization is designed around bright *off-diagonal*
stripes as the landmark, not the diagonal itself.

That raw matrix is hard to read for a non-specialist, so it's turned into a
**segmented timeline**: a single horizontal color-coded bar, where identical colors
mean similar-sounding stretches of the song. The segmentation algorithm is
agglomerative clustering constrained so that a frame can only ever merge with its
immediate temporal neighbor — guaranteeing every resulting segment is a genuinely
contiguous stretch of time, never a scattered "these unrelated moments happen to
share a color" result. Clicking a segment on the timeline loops just that stretch of
audio via the player's native start/end controls, no audio regeneration needed.

Not every song has clean, repeating sections — an ambient or through-composed track
genuinely doesn't, and forcing a segmented timeline onto one would show either one
meaningless block or noisy fake boundaries. A **novelty curve** (a checkerboard
detector slid along a dense self-similarity signal, the Foote 2000 technique) measures
how sharply the song's character changes moment to moment; its overall flatness
becomes a **structural-confidence score**. When a song's structure is confident
(1,394 of 1,399 songs with computed structure — 99.6% — clear the bar), the segmented
timeline shows. When it isn't, the app shows the continuous novelty curve itself
instead, framed honestly ("this song evolves gradually rather than repeating in clear
sections") rather than pretending a confident segmentation exists. Because every clip
is only 30 seconds long, the honest framing throughout is that Structure shows *how a
song's texture evolves across its first 30 seconds*, not full-song verse/chorus form.

---

## Part 4: What Gets Computed and Stored Per Song

Once a song's audio has been analyzed, several compact, reusable artifacts get
computed and persisted so nothing needs recomputing at query time.

**Song DNA** — five scalar numbers per song, computed from cheap, well-established
signal-processing techniques (no model, no training): **tempo** (BPM, via beat
tracking), **energy** (mean RMS loudness), **brightness** (mean spectral centroid —
how much high-frequency content is present), **harmonic complexity** (Shannon entropy
of the mean chroma vector — how "spread out" across many chords/pitches a song is,
normalized against the maximum possible entropy), and **rhythmic density** (onset
rate — how busy the rhythm is). Every song's raw values get normalized against the
whole library's actual range (a `DNANormalizer` fit once on the corpus), so "high
energy" always means "high relative to what's actually in this library," never an
arbitrary absolute scale. These five numbers are what every radar chart in the app —
the static comparison overlay, the hand-drawn query target, the agent's mood-profile
search — actually operates on.

**Fingerprints** — small, precomputed visual thumbnails, one per facet: a
**structure fingerprint** (a downsampled tile of the self-similarity matrix), a
**sound fingerprint** (a small mel-spectrogram thumbnail, capturing acoustic texture
visually), and a **harmony fingerprint** (a chroma-gram strip — 12 rows, since chroma
only ever has 12 meaningful pitch classes). A **composite fingerprint** overlays all
three as color channels of one image (structure→red, harmony→green, sound→blue):
where all three agree the image reads bright and neutral; where they diverge, the
image casts a distinct color — a real technique (comparable to false-color scientific
imaging) that encodes three separate similarity dimensions into one small image,
usable both as a genuine visual-identity element and as an album-art fallback for
songs FMA doesn't have artwork for.

**Description** — a short, natural-language phrase ("calm piano," "sassy hip hop,"
"chaotic ambient with voices and sirens") synthesized by an LLM from a song's raw
sound tags (below) plus its normalized DNA. This is a deliberately vivid, compressed
2–5-word summary meant for quick human scanning on Song X-Ray and Explore — 1,399 of
1,400 songs currently have one (the one gap has no computed DNA at all, an unrelated
pre-existing data hole).

**Sound tags** — the *raw* output of a pretrained audio tagger (**AST**,
`MIT/ast-finetuned-audioset-10-10-0.4593`, trained against the 527-class AudioSet
taxonomy) run on a representative 10-second slice from the middle of each track,
stored as the top 6 detected labels with their confidence scores (e.g. `[("Crow",
0.3), ("Bird vocalization", 0.1)]`). This is what lets the system answer "does any
song actually contain a specific named sound or instrument" — a much more precise
signal than the compressed Description phrase above, which necessarily drops
most detected tags in favor of a punchy summary. All 1,399 described songs currently
carry their raw tags too.

---

## Part 5: Ways to Actually Search and Match

With the facets and per-song artifacts in place, the system offers four genuinely
different ways to find "similar" — each answering a different kind of question.

**Moment-level matching** — the foundational retrieval mechanism. Pick a specific
~5-second moment in a song and one facet, and get back the nearest neighbors from
across the *entire* library on that facet alone, ranked by cosine similarity (via a
FAISS `IndexIDMap2` index per facet). Switching which facet is active reorders the
results entirely, since "similar" means something structurally different per facet —
a song can score a strong sound match and a weak vocal match to the same query moment
simultaneously, and that divergence is itself real, presentable signal, not noise.

**Song-level (whole-song) matching** — the same mechanism, but with every candidate
song's segments mean-pooled into a single vector before ranking, searched against a
separate song-level FAISS index. This smooths out segment-level noise: with roughly
14,600 individual segments and often only a few hundred per genre, single-segment
rankings tend to have a long plateau of near-tied candidates. Song-level pooling
sharpens the ranking margin for every facet (1.3×–2.3× tighter) and actually improves
genre-cohesion for five of the six facets — Sound is the one exception, since its
per-segment specificity is already strong and pooling blurs together genuinely
different moments within one song (a quiet intro versus a loud chorus). Available as
a selectable "Match against: Whole songs" mode.

**Mood-profile search** — instead of picking an existing song as the query, sculpt a
target directly: five sliders (tempo, energy, brightness, harmonic complexity,
rhythmic density), each 0–1 in the same normalized DNA space every song already lives
in. The system finds the nearest songs to that hand-drawn point by distance — no
embedding, no facet selection, pure DNA-space nearest-neighbor. This is also exactly
the mechanism the conversational agent uses under the hood for "make it moodier"
style requests: the agent reasons about which of the five axes a mood word implies,
picks numeric target values itself, and calls the identical search.

**Sound-content search** — find songs whose audio was actually *detected* as
containing a specific named sound, instrument, or event, e.g. "crow," "saxophone,"
"sirens." This is a case-insensitive substring match against each song's raw sound
tags first (the reliable signal, since it's a direct model detection), falling back to
the synthesized Description text second (catching vibe-language that never became a
discrete tag, like "sassy"). Tag-matched songs always rank ahead of description-only
matches. This is the only search mechanism that can answer a request naming a
concrete sound or instrument with no reference song and no facet selection involved.

---

## Part 6: Explaining Results in Plain Language

Raw similarity scores or facet names never appear on screen; three LLM-powered layers
turn retrieval output into something a non-technical listener can actually use.

**Match explanations** — every ranked match in Moment Matcher gets a single
plain-language sentence explaining *why* it matched on the active facet (e.g. "both
tracks feature a sparse, intimate vocal delivery layered with subtle synth
textures"), generated by an LLM given the two songs' titles/artists/genres, the
matched facet, and the similarity score — explicitly instructed never to mention
"cosine similarity," "embeddings," or raw numbers, and never to invent sensory detail
that isn't traceable to something the retrieval actually returned.

**Re-ranking** — a two-stage retrieve-then-rerank pipeline: the system first
over-fetches 15 candidates by plain cosine similarity, then hands the whole candidate
list to an LLM in one call to reason jointly over the query and every candidate at
once, re-sorting them down to a final top 6. This gets the benefit of a model
reasoning about the *whole* candidate set together (closer to what a real
cross-encoder reranker would do) without training one — and if the model's response
is ever malformed or out of range, the system silently falls back to the original
cosine ordering rather than crashing or dropping results.

**Descriptive-tag synthesis** — the same underlying mechanism that produces the
"calm piano" / "sassy hip hop" style Description field (Part 4): an LLM given a
song's raw AST tags plus its normalized DNA, instructed to write one vivid, natural
short phrase rather than just concatenating tag names.

All three of these LLM touchpoints share the exact same security discipline:
untrusted text (song titles, artist names — anything that ultimately came from FMA's
metadata, not from the system itself) is run through a sanitizer that strips the
delimiter characters that would otherwise let a crafted title escape its data block
and inject fake instructions, and every system prompt explicitly frames that data
block as inert text the model must never treat as a command. This defense was
specifically red-teamed against the live API (not a mock) with delimiter-escape and
plain-English "reveal your instructions" attempts — all handled correctly.

---

## Part 7: Ask the DJ — the Conversational Layer

A chat interface that turns every mechanism above into something you can just
describe in plain English rather than operate through toggles and sliders. Under the
hood it's a tool-calling loop against the Anthropic API: the model is given four
tools and decides for itself which to call, potentially chaining several in one turn.

- **`get_song_profile`** — look up a named song's real DNA values, used to anchor a
  "moodier than *this*" style request in the song's actual numbers rather than a
  guess.
- **`search_similar_songs`** — facet-based retrieval given a reference song and a
  named facet.
- **`search_by_mood_profile`** — the DNA-space nearest-neighbor search from Part 5;
  the model reasons about which axes a mood word implies and picks the numeric
  target itself.
- **`search_by_sound_content`** — the tag/description search from Part 5, for
  requests naming a specific sound or instrument rather than a mood or reference song.

A typical multi-step request ("find me something moodier and more stripped-back than
*Midnight Drive*") chains `get_song_profile` (fetch the real values) →
reasoning about which axes "moodier and stripped-back" implies → `search_by_mood_profile`
with the adjusted target → a plain-language reply built only from what the tools
actually returned. The system prompt explicitly instructs the model to commit to its
own best interpretation of an unusual or ambiguous request rather than handing the
user a menu of options to choose from, and to never claim a song has a
quality/instrument/sound unless a tool call actually confirmed it. Conversation
history is owned by the caller (Streamlit's session state), not the agent object
itself, and every tool result is sanitized the same way explanation/rerank text is.
A per-session message cap (30) guards against runaway API cost on the public
deployment. This whole layer was red-teamed directly, including a tool-result
injection attempt via a song with a deliberately malicious title — the agent never
echoed the injected content or deviated from its role.

---

## Part 8: Visual Exploration — Explore and Song X-Ray

**Explore** is the app's hub — the screen every other interactive capability is
reached from. It offers two ways to see the whole library at once, toggled at the
top:

- **Network graph** — a force-directed layout of a real k-nearest-neighbor graph
  (not a projection): each song is a node, pulled toward its genuinely closest
  neighbors and pushed apart from everything else, with an actual edge drawn only
  between real top-neighbor pairs. Following an edge from node to node is a literal
  "who's actually similar to this" path through the library.
- **2D map** — a PCA or ICA projection of the whole library instead, colorable by
  either the K-means cluster discovered purely from audio or the known genre label —
  a direct visual test of whether sonic similarity and genre agree, or diverge. An
  "Inspect these axes" panel runs a rigorous two-step interpretability check: first
  quantitative (does this axis actually correlate with one of the five DNA features —
  a clean correlation lets you *name* the axis with real evidence), then, only for
  whatever the correlation doesn't explain, a qualitative fallback of listening to
  the songs sitting at each axis's extremes.

Both views share filtering (genre multiselect, tempo range, "My Library only" scope)
and an "Up Next" queue with three modes — random, loop, or closest-match on a chosen
facet — so exploring never has to stop at one song.

Clicking any song opens its detail panel in place: title, artist, genre, its
synthesized Description, an inline player, save/unsave, and (from here) a button that
opens the *full* **Song X-Ray** view for that exact song — passing the song's
identity along so X-Ray never lands on an arbitrary default.

**Song X-Ray** is the deep, single-song view: all four fingerprints, the segmented
structure timeline (or novelty curve, whichever is honest for that song) with
click-to-loop, and the song's position highlighted on the library-wide 2D map. Once
a structure segment is selected, a "Find similar moments" button maps that
structural block to the nearest matching fixed-window retrieval segment (the two are
genuinely different segmentations, so the mapping is by closest start time) and
opens **Moment Matcher** pre-loaded with that exact song and moment — the second
half of Explore's drill-down chain.

---

## Part 9: Moment Matcher — the Core Matching Interface

Moment Matcher is where the four search mechanisms from Part 5 actually get operated
by hand. Two top-level modes:

**Existing-song moment mode** — pick a song, a specific ~5-second moment (or the
whole song, via the granularity toggle), and a facet; get back ranked matches with
plain-language explanations and LLM reranking (Part 6), each shown with a
"Compare song DNA" radar overlay so you can see not just *that* two moments matched,
but *how* they agree or diverge across all five DNA axes even when the matched facet
says they're close.

**Hand-drawn profile mode** — the mood-profile search from Part 5, operated directly
via the radar sliders, re-running automatically the moment you release a slider.

When reached from Song X-Ray with a specific song and moment already selected (Part
8), both selectors default to that exact context instead of an arbitrary first
alphabetical song — Moment Matcher is designed to be a destination you arrive at with
intent, not one you land on cold.

---

## Part 10: Personal Library

A lightweight bookmarking capability with no real authentication system behind it (a
single implicit library is enough to demonstrate the feature) — any song can be
saved or unsaved from its detail panel in Explore, and the "My Library" scope filter
restricts Explore's graph/map to just the saved subset.

---

## Part 11: Evaluation and Diagnostic Capabilities

These aren't user-facing features in the usual sense, but they're real, load-bearing
capabilities the rest of the system depends on for credibility — every claim of "this
facet works" traces back to one of these.

**Genre-cohesion@k** — the system's primary quantitative metric: for a sample of
real queries per facet, what fraction of the top-k nearest neighbors share the query's
genre label, compared against a random-pair baseline? Genre is treated explicitly as
a *proxy*, not ground truth for "sounds similar" — but a facet showing no lift over
random would be a real red flag. Current numbers (k=10, 500 queries/facet): Sound
54.4% vs. 11.9% baseline; Instrumental 41.5% vs. 11.8%; Vocal 38.1% vs. 13.9%; Drums
37.4% vs. 13.1%; Bass 26.7% vs. 12.5%; Harmony 21.4% vs. 11.7% — every facet clears
its baseline by a wide margin, and the facets visibly diverge from each other rather
than one riding another's coattails.

**Score-distribution diagnostics** — a deeper check than the pass/fail cohesion
number: for each facet, how does a real top-1 match's score compare to a random
pair's score, across the whole distribution? This is what revealed harmony's
collapsed embedding space (random pairs already scoring 0.85–0.95 before whitening) —
a mechanistic explanation for a weak facet that the cohesion number alone never would
have shown.

**Axis-correlation checking** — for the 2D map's PCA/ICA projections, a rigorous
test of whether an axis is actually *nameable*: correlate it against the five DNA
features first (a clean |r| ≥ 0.4 lets you say what the axis represents with real
evidence), falling back to a qualitative listen only for whatever doesn't resolve
that way.

**CNN genre-classifier baseline** — a small, genuinely *trained* neural network
(three convolutional blocks over log-mel spectrograms, no pretrained model of any
kind involved) used as an independent comparison point: does a model that has *never*
seen CLAP, chroma, or AST — trained from scratch, only on this library's genre labels
— separate genre better or worse than the facets built on frozen pretrained
embeddings? Result: 47.2% test accuracy against a 12.5% random baseline (8 balanced
classes) — a real, non-trivial signal from raw spectrograms alone, confirming the
library's genre labels correlate with something genuinely audible, independent of any
of the pretrained models the rest of the app leans on.

**Calibration-rating collection** — a standalone, blind pairwise-rating tool
(no title, artist, or algorithmic score shown, to avoid biasing a judgment with
recognition rather than actual sound) that draws pairs stratified across
high/medium/random similarity bands off the Sound facet's real retrieval results,
rather than naive random sampling (which would skew almost entirely to "obviously
dissimilar" and give little usable variance). This is the prerequisite data source
for a planned blend-weight regression and a conditional CLAP fine-tuning decision —
both still pending real rating volume.

---

## Part 12: Holding It All Together — App Structure and Navigation

The app is organized as a guided narrative followed by open exploration, not a flat
list of unrelated screens. **Overview** is the real landing page (project pitch, plus
two explicitly-marked-as-unwritten placeholder sections for a naive-baseline
comparison and related work). **Methodology** walks through the analysis and
preprocessing story with real embedded evidence at every step, including a full
section of honest hypothesis→test→result case studies. **Results** holds the outcome
numbers on their own page — genre-cohesion, the CNN baseline, and an honest
"no results yet" status for the calibration regression rather than a fabricated
number. **App Walkthrough** is a guided tour of the live interactive components
themselves. **Explore** is the actual hub for hands-on use, with **Song X-Ray**,
**Moment Matcher**, and **Ask the DJ** reachable only as contextual drill-downs from
it (hidden from the top-level sidebar entirely, but still directly reachable via a
button/link once you're already mid-interaction) — nobody lands on Moment Matcher
cold with a meaningless default selection.

A data-source banner warns whenever the app is pointed at synthetic placeholder data
rather than the real library, so a demo or screenshot never gets silently confused
for real results. A wordmark logo renders consistently across every page's
sidebar/header via Streamlit's native logo API. The whole interface commits to a dark
theme.

---

## Part 13: Operational Capabilities — Deployment, Portability, and Security

**Deployment portability** — the same codebase runs identically against the full
1,400-song local library or a small 200-song stratified subset (`deploy_data/`,
genre-balanced, committed to git) with zero code changes: `config.py` automatically
prefers the full local dataset if present and falls back to the subset otherwise,
which is exactly what happens once deployed to Streamlit Community Cloud, since that
platform only ever sees what's actually in the repository.

**Containerization and CI** — a Docker image (base dependencies only — the deployed
app never runs audio-processing/model-inference itself, only reads precomputed
artifacts) that's been actually built and run end-to-end, not just written on faith.
A GitHub Actions workflow runs linting (ruff) and the full test suite (currently 307
tests, covering core logic, the agent/tool layer, and every page via Streamlit's
headless testing harness) plus a Docker build check on every push.

**Security posture** — beyond the prompt-injection defenses already described in
Part 6, the conversational agent and explanation layers have been directly
red-teamed against the live API across 14 adversarial prompts spanning instruction
override, hallucination bait, tool misuse, data-framing injection, fabrication bait,
scope overreach, and a direct extraction attempt — all 14 handled correctly. Secrets
(the Anthropic API key) are read through the deployment platform's own secrets
manager, never committed to the repository. Every public-facing LLM feature carries a
simple per-session call cap as a cost/abuse guardrail.
