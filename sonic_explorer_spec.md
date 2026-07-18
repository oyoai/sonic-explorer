# Sonic Explorer — Product Specification

## 1. Vision

A music player and exploration platform where similarity is based on the actual
**audio signal** — sound, structure, and harmony — not metadata, tags, or genre
labels. The system is explainable (every match comes with a plain-language reason),
explorable (a visual map you wander, not a search box you query), and generative
(it can combine songs into new transitions and mixtapes based on what it's learned
about them).

**One-line pitch:** *"Explore your music library by how it actually sounds, see why
things are similar, and let the system remix them for you."*

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
Pick any song → see its structural anatomy (self-similarity matrix rendered as a
visual "shape" — verse/chorus/bridge segmentation) and its position in the Taste Map.
Makes the structure-analysis work visible as its own feature, not a buried step.

### 2.3 Moment Matcher
Pick a specific ~4–8 second moment in a song → get ranked matches from elsewhere in
the library, each with a plain-language explanation of *why* it matched. This is the
core proof-of-concept: does audio-signal similarity actually work.

**Facets** — similarity isn't one thing, so matching happens along three separate,
independently-computed axes:
- **Sound/timbre** — CLAP embeddings (or MFCC fallback) — texture, instrumentation
- **Harmony** — chroma features — key, chord color, tonal similarity
- **Structure** — self-similarity descriptors — the shape/arrangement of a song

UI framing stays plain-language: a toggle or blend slider — *"Match by: Sound /
Structure / Harmony"** — never "cosine similarity" or "embedding dimension" on screen.

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

---

## 3. Underlying Techniques

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
