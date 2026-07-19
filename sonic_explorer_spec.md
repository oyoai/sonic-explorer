# Sonic Explorer — Product Specification

## 1. Vision

An AI-driven music exploration platform: explore your library by the actual **audio
signal** — sound, structure, and harmony — not metadata, tags, or genre labels. The
exploration and the underlying models (facets, embeddings, clustering, fingerprints,
the agent layer) are the point of the product. Playback exists to support exploring:
whatever you click on, hover over, or match to should just play, immediately, so
exploring never feels disconnected from listening. It's in service of the
exploration, not a separate "player app" identity competing for attention.

**One-line pitch:** *"Explore your music library by how it actually sounds, see why
things are similar, and let the system remix them for you — while you listen."*

**Playback support** (present wherever exploration happens, not a separate feature
set): play/pause, seek, and a sense of "what's playing now," available inline in the
Taste Map, Song X-Ray, Moment Matcher, and Mixtape Builder alike, so clicking a point
or a match immediately plays it. Loop control for a selected section (from structure
segmentation) supports close listening while exploring structure. Pitch/speed
adjustment (regenerate-on-release, reuses the Mixtape Builder's tempo-compatibility
machinery) supports exploring how a match holds up under small changes.

---

## 2. Product Modules

### 2.1 Explore Tab (Taste Map)
A 2D visual map of the **entire library**, clustered by sonic character rather than
genre. Click a point → hear the song. Nearby points sound similar. This is the primary
non-technical entry point: no metrics, no vectors on screen, just spatial closeness
as an intuitive proxy for similarity.

**Add song**: a user can add a new song not currently in the library. This runs the
same segment → embed (all facets) → store pipeline on-demand for one song, rather than
only as a batch job — a direct validation that the facet-registry/repository design
(section 8.2) is genuinely reusable, not just a one-time script.

**Save song**: bookmark an existing library song into "my collection" — a lightweight
DB record, no processing needed.

**My Taste Map** (personal view): a Taste Map scoped to the user's saved songs only.
- *Core*: reuse the global PCA/ICA projection, filter/highlight just the saved songs.
- *Strong*: recompute PCA/ICA on the saved subset alone — can surface clusters that
  only emerge within a smaller personal library (two songs that look distant on the
  global map, dominated by whole-library variance, may be near-neighbors within a
  user's own world). Comes with its own small "does this actually look different/
  better" evaluation.

No real auth system needed for this — single implicit "my library" (session state or
one default user record) is enough to demonstrate the feature; full accounts would add
scope with no course-session mapping to justify it.

### 2.2 Song X-Ray
Pick any song → see its structural anatomy and other per-song visual aspects, plus
its position in the Taste Map.

**Structure view (Core)**: a segmented timeline as the primary view — a single
horizontal color-coded bar representing the song's duration, where identical colors
mean repeated/similar sections (e.g. "green = verse, blue = chorus, green repeats").
The raw self-similarity matrix is kept as a secondary "technical detail" view
underneath, not the main thing a viewer sees first. Note: the matrix's diagonal is
intentionally zeroed by `librosa.segment.recurrence_matrix` (excluding trivial
self/near-self similarity so it doesn't drown out real repeated-section stripes) —
this is expected behavior, not a bug, and the visualizations above are designed not
to rely on a visible diagonal as a landmark.

**Multi-strip view (Strong, sequenced after the radar chart overlay below)**: stacked,
aligned horizontal strips over the same timeline — waveform + beat grid, energy/
loudness curve (RMS), spectral brightness (spectral centroid), chroma/key (once the
harmony facet exists), and the structure segments strip. Richer than more informative
on its own — sequenced after the radar chart since that has a stronger
explainability payoff for the same rough effort.

**Radar chart "song DNA" overlay (Strong, build first)**: aggregate each song into a
handful of per-facet numbers (e.g. average tempo, energy, brightness, harmonic
complexity, rhythmic density) plotted as a radar/spider chart. Overlaying two songs'
radar charts (semi-transparent, distinct colors) makes agreement/divergence visually
obvious at a glance — where shapes overlap, they agree; where one bulges past the
other, they diverge. Reuses facet computations already being done, so it's cheap
once per-song aggregate stats exist. Ties directly into Moment Matcher's "why did
these match" explanation — shown side-by-side with the LLM's text explanation.

**Loop a section (Core)**: trim playback to a selected start/end — most naturally a
segment from the structure analysis (e.g. loop just the chorus) — and loop it via the
player, no audio regeneration needed. Part of baseline listening functionality, not
an add-on.

**Pitch/speed adjustment (Strong)**: not live/continuous — Streamlit's player doesn't
support manipulating audio while it plays. The realistic version: adjust a
pitch/speed control, release it, regenerate the clip (`librosa.effects.pitch_shift` /
`time_stretch`, or `pyrubberband` for higher quality) and reload the player with the
result. Reuses the same time-stretching machinery the Mixtape Builder already needs
for tempo-compatible transitions — this is surfacing existing machinery as a
user-facing control, not new modeling work.

**Radar chart as query (Strong, same tier as the static overlay, additive — not a
replacement for it)**: let the user drag each radar axis by hand to sculpt a target
profile (e.g. high energy, low brightness, high rhythmic density) rather than only
picking an existing song as reference. That target becomes a point in the same
aggregate-stat space every song lives in, and the system does nearest-neighbor search
against it — "find songs closest to the shape I just drew." Reuses the exact same
per-song aggregate stats as the static overlay, so it's a small addition, not new
infrastructure. After matching, overlay the user's drawn target with the actual
matched song's radar, so the user can see how close the match really is. This also
gives the conversational agent layer a natural hook later — a request like "make it
moodier" can just nudge the same radar values programmatically, sharing one mechanism
between manual and agent-driven interaction.

On slider release (not continuous drag), automatically re-run the nearest-neighbor
search and switch playback to the new top match, with a short crossfade (a second or
two of overlap) rather than an abrupt cut. This is the buildable version of "the
matched song reacts as I adjust the chart" — Streamlit reruns on each interaction
rather than maintaining a continuous audio stream, so this fits the framework's grain
naturally. *(Future work, not planned: continuous real-time audio morphing while
actively dragging — would need a persistent low-latency audio stream, realistically a
JS-based web audio app rather than Streamlit; a genuinely different, larger build than
anything else in this spec.)*

*(Future work, not planned: a richer D3.js/Sigma.js-style interactive network
visualization for the Taste Map or a "how songs connect" secondary view — genuinely
more powerful and visually distinctive than what Plotly/PyVis can do within Streamlit,
but it's a separate JS skillset and embedding approach, a bigger lift than anything
else in this spec. Plotly (already in use) or PyVis remain the realistic in-scope
options for interactive graph/network visuals.)*

### 2.3 Moment Matcher
Pick a specific ~4–8 second moment in a song → get ranked matches from elsewhere in
the library, each with a plain-language explanation of *why* it matched. This is the
core proof-of-concept: does audio-signal similarity actually work.

**Facets** — similarity isn't one thing, so matching happens along several separate,
independently-computed aspects of a song:

*Whole-mix facets:*
- **Sound/timbre** — CLAP embeddings (or MFCC fallback) — overall texture,
  instrumentation, production character
- **Harmony** — chroma features — key, chord color, tonal similarity
- **Structure** — self-similarity descriptors — the shape/arrangement of a song, with
  a structural-confidence score (see Song X-Ray) distinguishing clearly-sectioned
  songs from abstract/through-composed ones
- **Abstractivity** — the structural-confidence/novelty-curve-flatness measure,
  exposed as its own facet: how strongly a song repeats in clear sections vs. evolves
  gradually
- **Surprise/predictability** — a research-grounded proxy (drawing on music-cognition
  work on expectation violation, e.g. Huron) measuring how unexpected a moment is
  given what preceded it — harmonic surprise (prediction error over the chroma time
  series) and dynamic surprise (sudden energy/RMS jumps after a low-variance stretch)
  combined into a surprise-over-time curve. This is a proxy correlated with
  frisson-inducing moments in the research literature, not a direct physiological
  measurement — framed that way explicitly, not oversold. Distinct from the other
  facets in that it's about a moment's relationship to what came before it in the
  same song, not a cross-song comparison.

*Stem-separated facets* (via a pretrained source-separation model, e.g. Demucs —
splits a track into vocals/drums/bass/other before computing similarity, rather than
only ever comparing the full mixed-down song):
- **Vocal** — isolated vocal stem, voice/delivery similarity independent of the
  backing
- **Drums** — isolated drum stem
- **Bass** — isolated bass stem
- **Instrumental/backing** — the "other" stem (or full mix minus vocals) — general
  backing-track similarity

*Flagged as harder, real research-level future work, not now:*
- **Guitar** specifically — mainstream separation models don't isolate guitar as its
  own stem (it's lumped into "other" along with keys/synths); genuine guitar-only
  separation is a less mature research area
- **Effects/accents** (reverb, distortion, delay) — not a separable stem, would need
  dedicated feature engineering (e.g. reverb decay estimation, distortion/clipping
  measures) rather than a separation model
- **Lyrics/meaning** — semantic similarity of what a song's about, via transcription
  (Whisper or similar) + text embedding, once vocals are isolated

*Set aside, not measurable from audio alone:* frisson itself (a subjective
physiological response) — the surprise/predictability facet above is the measurable,
research-grounded proxy, kept explicitly distinct from a frisson claim.

Each stem-separated facet reuses the exact same `Facet` interface and registry as the
whole-mix ones — a "vocal similarity" facet is just `SoundFacet` run on an isolated
audio stream rather than the full mix, no architecture change needed.

**Selecting facets — allow any subset, including all, shown side by side rather than
blended by default:** pick one facet → ranked matches for just that aspect. Pick
several (or all) → each facet's own top matches shown in its own panel, not averaged
into one score — a forced average can hide the interesting case where a song matches
strongly on one facet (e.g. vocal timbre) and not at all on another (e.g.
instrumental) — which is itself a real, presentable finding, not noise to smooth
over. The weighted-blend slider (see the radar-chart-as-query feature) still has a
place *within* this — once a subset is chosen, optionally blend those into one
combined ranking — but "show each aspect separately" is the default view.

UI framing stays plain-language regardless of how many facets are active: labeled
toggles/panels per aspect (e.g. "Vocal", "Drums", "Harmony") — never "cosine
similarity" or "embedding dimension" on screen.

### 2.4 Mixtape Builder
Given a starting song (and optionally a vibe/duration request), the system chains
songs together using harmonic compatibility (key/mode matching, Camelot-wheel style),
tempo compatibility, and structure-aware transition points (matching one song's
outro/breakdown to the next song's intro) — then renders an actual playable
crossfade/transition, not just a claim.

### 2.5 Conversational Agent Layer
A natural-language front-end over all of the above: *"find me something moodier and
more stripped-back than this"* or *"build me a 15-minute chill set starting here."*
The agent decides which facet(s) to weight, calls the retrieval/compatibility tools,
and returns results with explanations — turning the module toggles into a
conversation for non-technical users.

### 2.6 Song Fingerprints (visual identity, Core + Strong)
Each song gets a small visual "fingerprint" per facet, used both as an album-art
fallback (FMA's art coverage is inconsistent — worth checking coverage on the curated
set early) and as a genuine visual identity throughout the app:

- **Structure fingerprint (Core)**: a small downsampled tile of the self-similarity
  matrix — reuses the existing structure computation, just rendered small.
- **Sound fingerprint (Core)**: a small mel-spectrogram/MFCC-derived thumbnail —
  captures acoustic texture visually; doesn't require CLAP, cheap once audio is
  loaded.
- **Harmony fingerprint (Strong)**: a small chroma-gram strip, built alongside the
  harmony facet once it exists.
- **Composite fingerprint (Strong)**: the three facet fingerprints combined into one
  image via RGB-channel overlay — structure → red, harmony → green, sound → blue,
  each normalized to grayscale intensity first. Where all three agree the composite
  reads bright/white; where they diverge, distinct color casts appear. This is a real
  technique (comparable to false-color scientific imaging combining separate
  wavelength channels), not just a visual flourish, and gives a legitimate
  methodology point: three similarity dimensions encoded as the three channels of one
  image. Two songs similar in structure but not harmony would show similarly-patterned
  but differently-colored composites — genuinely informative, not decorative.
- **Rendering choice**: use perceptually-uniform colormaps (viridis, magma, plasma —
  matplotlib built-ins) for the individual fingerprints rather than flat two-tone
  heatmaps — these are an actual data-visualization standard (color intensity maps
  linearly to value, avoiding misread intensity), and they produce genuinely rich,
  gradient-like color without being decorative for its own sake. The composite's
  blended coloring emerges naturally from stacking three continuous channels, giving
  an organic "beautiful gradient" effect that's actually load-bearing data, not
  applied polish — this is the intended alternative to decorative gradients/glow
  effects, which read as generic rather than considered (see UI notes elsewhere in
  this spec).

### 2.7 Visualization Layer

**Live sound visualization (during playback)**: visual response to the audio as it
actually plays, not just static pre-computed analysis images — a waveform that
scrubs/highlights with playback position, a live frequency/spectrum display, or a
particle/bar visualizer reacting to the audio's amplitude and frequency content in
real time. Makes listening itself part of the exploration experience, not just a
background activity while looking at static charts. Can draw on the same underlying
signal data as the fingerprints (spectral/energy features) — a live view of the same
information that's shown statically elsewhere.

**Relationship/network graph view**: an alternative way to see how songs relate,
alongside the Taste Map's spatial view — songs as nodes, edges connecting genuinely
similar songs (per facet or blended), so relationships are visible as an explicit
graph rather than only inferred from proximity on a 2D map. Useful as its own
exploration mode: start at one song, see its direct neighbors, follow an edge to a
neighbor's neighbors, discover paths through the library rather than only viewing it
from above. Built with Plotly or PyVis (NetworkX for the underlying graph
structure/layout) to stay within the existing Python/Streamlit stack — see the
future-work note earlier in this section on more advanced (D3.js-style) network
visualization tooling if there's ever room to go further.

---

| Capability | Technique | Course session |
|---|---|---|
| Sound similarity | CLAP embeddings (audio+text joint space) | Bi-encoder architecture (24) |
| Harmony similarity | Chroma / pitch-class features | Feature engineering (6-7) |
| Structure similarity | Self-similarity matrices | Unsupervised/clustering (11-12) |
| Retrieval | FAISS / ChromaDB vector store | Vector stores (29) |
| Re-ranking | Two-stage retrieve-then-rerank | Cross-encoder/re-ranking (24) |
| Explanation | LLM natural-language reasoning | LLM as instrument (28-29) |
| Taste Map | PCA and/or ICA + K-means | Unsupervised learning (11-12) |
| Interpretable axes | ICA (independent components as inspectable "sound axes") | Unsupervised learning (11-12) |
| Conversational layer | Agent with tool-calling | AI Agents & Multi-Agents (30) |
| Tool architecture | MCP client-server | Building MCP architecture (31-32) |
| Evaluation | Genre-cohesion baseline + personal test set | Model evaluation (10) |

**On ICA specifically**: used two ways — (1) as an alternative/complement to PCA for
the Taste Map, where independent components are more likely to correspond to
individually-interpretable qualities than PCA's variance-maximizing components; (2) as
a stretch-tier source-separation tool if time allows.

**On "several kinds of similarity"**: general-purpose embeddings like CLAP capture a
blended, holistic notion of "what this sounds like" — they do not cleanly separate
sound from harmony from structure. That separation has to come from using distinct,
targeted features per facet. This is a real, presentable finding, not just an
assumption — worth stating explicitly in results/limitations.

---

## 4. Tiered Scope

### Core (must work — safety net; alone, a complete presentable project)
- Data: FMA subset, ~1,000–2,000 tracks, segmented into windows
- Sound facet: CLAP (or MFCC fallback) embeddings → FAISS retrieval
- Structure facet (basic): self-similarity matrix per song → feeds Song X-Ray
- Taste Map: PCA + K-means, 2D, clickable, click-to-play
- Moment Matcher: sound-only matching
- Evaluation: genre-cohesion vs. random baseline

### Strong (should work — real value-add)
- Harmony facet: chroma features as second matching axis
- Facet toggle in UI (plain-language framing)
- LLM explanation layer ("why these matched")
- ICA for Taste Map (compare against PCA, inspect interpretable axes)
- Song X-Ray (full, own screen)
- Re-ranking step
- Conversational agent front-end over Moment Matcher + Taste Map

### Stretch (only if Core + Strong solid with days to spare)
- Mixtape Builder (harmonic + tempo compatibility, structure-aware transitions, rendered audio)
- Agent-orchestrated Mixtape Builder (multi-step planning loop)
- Planner + Critic multi-agent pattern
- MCP client-server wrapping of tools
- Multi-moment query (combine 2+ selected moments)
- Sequence-aware matching (DTW over embedding sequences — real research territory)
- Explainability follow-up agent ("why not this other song instead?")

**Cut rule (applies at every tier boundary):** each new facet/module gets an early
checkpoint. If it doesn't show a meaningful signal (e.g., harmony facet isn't
separating anything, ICA components aren't interpretable), that becomes a documented,
honest limitation — not a blocker, and not something to debug into the time budgeted
for the next tier. Core never depends on Strong or Stretch working.

---

## 5. Risk Points

- Does CLAP's embedding space separate "similar-feeling" audio, or mostly cluster by
  genre/instrumentation? — resolved early via direct inspection, before building
  downstream features on top of it.
- Do harmony/structure facets actually add distinguishable signal beyond the sound
  facet, or do they collapse into the same clusters?
- Are ICA components interpretable in practice, or just as opaque as PCA's?
- Does the personal "does this actually hit" test set show any learnable pattern?
- Mixtape transitions: does structure-aware transition-finding actually sound better
  than a naive crossfade, or is the difference inaudible?
- Agent layer: does tool-calling reliably pick the right facet/tool, or does it need
  significant prompt iteration to behave predictably?

---

## 6. Day-by-Day Plan (~10 days)

| Days | Focus | Checkpoint |
|---|---|---|
| 1–3 | Core: data → embeddings → structure → Taste Map → basic Moment Matcher | **Go/no-go**: if not solid by day 3, stop adding scope, polish Core only |
| 4–6 | Strong, one piece at a time: harmony facet, facet toggle, LLM explanations, ICA, Song X-Ray, re-ranking, agent front-end | Each piece gets its own go/no-go before starting the next |
| 7–8 | Stretch, only if ahead of schedule: Mixtape Builder, agent orchestration, MCP wrapping | Attempt in listed order; stop at any point and still have a complete product |
| 9–10 | Buffer + demo polish + slides | Presentation ready, live demo rehearsed |

---

## 7. Evaluation Plan

- **Quantitative**: genre-cohesion of neighbors vs. random baseline, per facet
- **Qualitative**: personal "tickle" test set — do matches actually feel right
- **Structural**: visual inspection of Taste Map clustering (tight vs. scattered)
- **Comparative**: PCA vs. ICA axis interpretability
- **Ablation-style finding**: do facets actually diverge (sound match ≠ harmony match
  ≠ structure match for the same query), or do they collapse together?

---

## 8. Software Architecture

### 8.1 Storage — two tools, different jobs
- **Vector store** (FAISS/Chroma): the actual embeddings, built for fast similarity search.
- **Relational DB** (SQLite — no need for anything heavier): song metadata, segment
  boundaries, which facets have been computed for which segment (so nothing gets
  recomputed on every run), calibration test ratings, and later, user interaction logs.
  Minimal schema: `songs`, `segments`, `embedding_status`, `calibration_ratings`.
- **Compute-once, not recompute**: check `embedding_status` before embedding a segment;
  if already done, retrieve from the vector store instead of recomputing. Protects the
  live demo from slow recomputation too.

### 8.2 Class structure / design patterns
- `Song` — id, filepath, metadata, list of `Segment`s
- `Segment` — id, parent song, start/end time, holds computed embeddings per facet
- `Facet` (abstract interface: `embed(segment)`, `similarity(vec_a, vec_b)`) —
  `SoundFacet`, `HarmonyFacet`, `StructureFacet` all implement this (**Strategy pattern**)
- `FacetRegistry` — holds all registered facets; used by both the manual UI and the
  agent's tools, so adding a facet later = one new registry entry, not scattered edits
- `SongRepository` / `EmbeddingRepository` — wrap all DB/vector-store access so no raw
  SQL or FAISS calls happen elsewhere in the codebase (**Repository pattern**) — this is
  the seam between exploratory analysis and production code
- `RetrievalService` — query segment + facet(s) → repository → ranked matches; the one
  source of truth both the Streamlit UI and the agent's `search_by_facet` tool call into

**Workflow narrative**: exploratory work (EDA, testing whether CLAP separates
meaningfully, PCA vs ICA comparisons) happens in notebooks — fast, throwaway. Once
proven, logic gets promoted into the class structure above — clean, typed, reusable.
Worth stating explicitly in the presentation: this is literally how real ML teams work.

### 8.3 Core/interface separation
Core logic (facet registry, retrieval, compatibility checks) is plain Python with no UI
dependency. The interface layer (Streamlit UI, agent) calls into core logic but contains
none of it. This means Streamlit is today's front-end, not a structural commitment — a
future real web app (FastAPI + React, say) would mean rewriting only the interface
layer, not the ML/retrieval work underneath.

---

## 9. Deep Learning Additions

- **Fine-tuned embeddings (Strong tier)**: use the calibration test set (human
  similarity ratings on song pairs) to fine-tune CLAP via contrastive/triplet loss —
  pull "similar"-rated pairs closer, push "dissimilar" pairs apart. Real training, reuses
  data already being collected for evaluation. Report correlation-with-human-ratings
  before vs. after fine-tuning.
- **CNN genre classifier (Core/Strong)**: small CNN on mel-spectrograms, used as a
  trained-model baseline for the genre-cohesion evaluation — parallel story to the
  hand-crafted-features-vs-CLAP comparison.
- **Autoencoder for structure encoding (Stretch)**: learn a compressed structure
  embedding as an alternative/addition to the self-similarity-matrix approach; slots
  into the facet registry as just another facet.
- **LSTM over segment sequences (Stretch)**: temporal model of a song's shape, directly
  feeds the "sequence-aware matching" stretch item. Real added debugging surface — only
  attempt with Strong tier solid and time to spare.
- **Calibration → trained blend weights**: fit a small regression (facet scores →
  predicted human rating) on the calibration data, producing learned blend weights
  instead of an arbitrary slider default.
- **Note**: CLAP itself stays pretrained-only in Core — no training risk there. The BCI
  connection some of this superficially resembles (subjective response you can't
  directly observe) is a loose conceptual link only, not a technical one — worth not
  overclaiming this if it comes up.

---

## 10. Deployment Plan

- **Target**: Streamlit Community Cloud or Hugging Face Spaces — both free, both deploy
  near-directly from a GitHub repo, both give genuine "it runs somewhere a stranger can
  use it" experience without infra complexity.
- **Sequencing**: deploy the Core version early (right after Core is solid, before
  starting Strong-tier work), then push updates as Strong-tier features land — avoids a
  scramble to deploy everything at once at the end.
- **Real gotchas to expect** (genuinely useful interview material, not just risk):
  dependency differences between local and deployed (`requirements.txt` drift, audio
  libraries behaving differently), secrets management for the LLM API key (platform
  secrets manager, not a local `.env`), dataset size limits for hosting (may need a
  smaller curated dataset for the deployed version vs. the full local one — a
  legitimate, presentable tradeoff), API cost/rate-limiting for the public LLM calls,
  and cold-start latency on free-tier hosting.

---

## 11. Working Style for the Build Phase

When building this in Claude Code: explain each step briefly before doing it (what
we're building and why), flag it clearly on reaching a tiered checkpoint or cut-rule
decision point, and surface gotchas at the moment they're actually relevant rather than
all upfront. Keep default explanations short; give a longer, deeper explanation only
when explicitly asked for one — default to moving efficiently, not narrating at length.

---

## 12. Presentation Arc (12 min, no code)

1. Motivation — why signal-based, explainable music similarity is interesting
2. Problem definition — what "done" looks like (demo definition)
3. Data & EDA
4. Methodology — facets, embeddings, Taste Map, agent layer (as far as time built it)
5. Live demo — Taste Map exploration → Moment Matcher → (if built) Mixtape Builder
6. Results — evaluation findings, including honest facet-divergence findings
7. Limitations & future work — anything cut, anything that didn't separate cleanly