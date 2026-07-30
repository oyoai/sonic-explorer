# Sonic Explorer — Complete Project History

This document walks through everything that was actually built, in the order it was
built, with the real reasoning behind each decision, every bug that was found and how
(or whether) it got fixed, and where things stand today. It's written so that someone
with zero prior context can read start to finish and understand the whole project —
not a bullet-point summary, the real story.

Sonic Explorer is a Streamlit app for exploring a personal music library by how songs
actually *sound* — timbre, harmony, structure, isolated vocal/drum/bass/instrumental
character — instead of by genre tags or metadata. The pitch: click around a map of
your library built entirely from audio signal, find moments that sound alike across
completely different genres, get plain-language explanations of *why* they matched,
and talk to a conversational DJ that can search the same underlying data.

Everything is built on one architectural spine that never changed across the whole
project: a `Facet` is a strategy-pattern interface (`embed(segment)`,
`similarity(vec_a, vec_b)`); a `FacetRegistry` holds every registered facet so nothing
downstream (UI, evaluation scripts, the agent's tools) ever hardcodes a facet list;
`SongRepository`/`EmbeddingRepository` wrap every bit of SQLite/FAISS access so no raw
SQL or vector-store call happens anywhere else; `RetrievalService` is the single
choke-point every query — manual UI or agent tool — goes through. Core logic (facets,
retrieval, repositories) is plain Python with zero Streamlit dependency; the interface
layer calls into it but contains none of the actual logic itself. That separation is
why the app could be restructured twice (once early, once very late) without touching
any of the underlying model or retrieval code.

---

## Part 1: Foundation — Data, Core Package, and the First Pipeline

### The dataset

The library is a curated subset of the **Free Music Archive (FMA)** — Creative
Commons-licensed tracks. The curated set that ended up being used is **1,400 songs
across 8 genres** (Electronic, Experimental, Folk, Hip-Hop, Instrumental,
International, Pop, Rock), acquired via a Colab notebook
(`notebooks/01_fma_acquire_and_curate.ipynb`) that mounts Google Drive and writes a
`curated_tracks.csv` manifest so the download/curation step only ever has to run once.
A local-equivalent script (`scripts/acquire_fma.py`) exists for anything that doesn't
need GPU.

One fact about this dataset shaped a lot of later decisions and is worth stating
explicitly up front: **every clip is a 30.0-second preview, not a full track.**
(Confirmed empirically later via a duration histogram in the app's own EDA — every one
of the 1,400 clips falls within 0.05s of exactly 30.0 seconds.) This is *why* the
Structure facet was later reframed as "how a song's texture evolves across its first
30 seconds" rather than full verse/chorus form — 30 seconds usually isn't long enough
for a section to repeat.

### Core package skeleton (Day 1)

The very first commit laid down the shape everything else would sit inside:

- `Song` / `Segment` dataclasses.
- The `Facet` strategy interface, with `SoundFacet` as the first real implementation —
  **CLAP** (`laion/clap-htsat-unfused`, lazy-imported so importing the module never
  requires torch unless the facet is actually used), with an MFCC fallback path.
- `FacetRegistry`.
- SQLite schema (`songs`, `segments`, `embedding_status`) plus `SongRepository` /
  `EmbeddingRepository`, the latter wrapping FAISS `IndexIDMap2` indexes.
- 17 unit tests, none requiring GPU or real audio — the discipline of testing core
  logic against fakes/synthetic data rather than needing real audio or a GPU to run
  the suite was established here and never broken.

### Batch embedding pipeline (Day 2)

`sonic_explorer/pipeline/segment.py` implements fixed-window/hop segmentation
(promoted out of a single-song exploratory notebook, `audio_deep_dive.ipynb`).
`pipeline/embed_library.py` is the library-scale batch job, built with
**compute-once resumability** from the very start: it checks `embedding_status`
before re-embedding anything, and — critically — skips reloading audio entirely for
songs that are *already fully embedded*, not just the embedding step itself.

A real bug was caught immediately: `EmbeddingRepository`'s FAISS index path was
hardcoded to the package's local `data/artifacts/`, which inside a Colab session
resolves to the *ephemeral* `/content` clone rather than the Drive-mounted artifacts
folder — meaning every index would get silently wiped the moment the Colab session
disconnected. Fixed by making `artifacts_dir` an explicit constructor argument instead
of an assumption.

`notebooks/02_batch_embed_pipeline.ipynb` is the notebook that actually clones the
repo into Colab, mounts Drive, installs the package via `pip install -e ".[colab]"`
(the same `sonic_explorer` package everything else uses — never a
notebook-local reimplementation of any logic), runs the batch pipeline, and
sanity-checks retrieval before syncing anything down. A follow-up commit fixed the
notebook's clone/install cell, which had been using shell magic that silently
*continued* — and printed a false "installed" success message — even when `git clone`
actually failed; a real run against a private repo hit exactly this (an auth error
that got masked). Replaced with `subprocess` calls that check return codes and raise
immediately.

### Interface layer, built in parallel against synthetic data

Rather than block UI work on the real Colab embedding job, the interface layer was
built in parallel against a small **synthetic genre-clustered dataset**
(`scripts/seed_dev_data.py`) — fake audio, fake embeddings clustered by a fake genre
label — designed so the real synced artifacts could swap in later with **zero code
changes**. This pattern (build against synthetic data with the exact same shape as
real data, verify the swap-in works, never write UI code that "knows" it's talking to
fake data) recurs constantly through this project and is one of its more important
disciplines.

- `sonic_explorer/retrieval/service.py` — `RetrievalService`, the single entry point
  everything queries through from here on. `query_by_segment` snaps to a
  *precomputed* vector rather than re-embedding live.
- `sonic_explorer/analysis/taste_map.py` — PCA + K-means over per-song mean-pooled
  sound embeddings, plain Python, no Streamlit dependency.
- The first Streamlit pages: a landing page and a clickable Taste Map (Plotly
  scatter, click-to-play wired through a `custom_data`-carried `song_id`).

A real bug surfaced immediately on actually *running* the app (not just importing it):
SQLite connections default to single-thread-only, but Streamlit reruns script code
across a thread pool — the cached DB connection blew up cross-thread the moment two
reruns overlapped. Fixed with `check_same_thread=False`. This is the first of many
places in the project's history where a bug only showed up once something was
*actually run end-to-end*, not just imported or unit-tested — verification via
Streamlit's `AppTest` harness (headless script execution that catches real exceptions
a plain import or `curl` would miss) became a standing discipline from here forward.

### Structure facet, Song X-Ray, Moment Matcher (completing Core tier)

`sonic_explorer/facets/structure.py`'s `compute_self_similarity_matrix` was promoted
out of the exploratory notebook's beat-synced chroma recurrence-matrix cell. This was
kept as a **song-level artifact**, not routed through the `Facet`/registry retrieval
path the way `SoundFacet` is — Moment Matcher stays sound-only for Core tier per the
spec, and Structure is a visualization, not something you search by.

`Song X-Ray` (self-similarity heatmap + Taste Map position) and `Moment Matcher`
(pick a song + moment, get ranked sound matches, phrased as "92% sonic match" — never
"cosine similarity" on screen) shipped together. All four pages verified via `AppTest`
plus a real local dev server; Moment Matcher's actual output confirmed retrieval
clusters correctly by genre against the synthetic data.

### Genre-cohesion evaluation

`sonic_explorer/evaluation/genre_cohesion.py`'s `genre_cohesion_at_k` — observed vs.
random-baseline neighbor genre agreement — is the project's core quantitative metric
from here through the very end. It deliberately excludes same-song neighbors from
both the observed and random arms. `scripts/run_evaluation.py` prints the comparison
and saves an HTML bar chart. First run, against the synthetic dev data: 100% observed
vs. 22.1% random baseline (k=10, 224 queries, 4 balanced genres) — confirmed the whole
metric pipeline end-to-end before it ever touched real data.

`sonic_explorer/pipeline/build_structure_library.py` — the library-scale, CPU-only
structure-matrix computation (no CLAP/GPU needed) — was built and ready to run the
moment real curated audio synced down, using the same compute-once discipline
(checking whether each song's `.npy` artifact already exists) as everything else.

A small but real bug caught by the new tests: `EmbeddingRepository.get_vector` passed
`segment_id` straight to FAISS's `reconstruct()`, which rejects numpy integer types
(e.g. from `rng.choice` or a pandas row) — fixed by casting to plain `int`.

---

## Part 2: Plugging In Real Data — Where Things Actually Broke

This is the part of the project where synthetic data got swapped for the real
1,400-song, 14,580-segment library, and where a run of genuinely important bugs
surfaced — every one of them only visible once real data and real scale were
involved.

### Filepath portability (Colab → local)

Songs embedded via the Colab notebook have `filepath` pointing at the Drive-mounted
path inside that Colab session — meaningless once synced to a local machine.
`scripts/repath_audio_paths.py` (safe to re-run) repoints every song to
`data/audio/{fma_track_id}.mp3` once real curated audio lands locally. This is the
first instance of a portability bug class that recurs later for local→deployed paths
too (see "Deployment prep" below).

### The FAISS checkpoint data-loss bug

This is the single most important bug in the project's early history. Plugging the
real 1,400-song / 14,580-segment library into the local app for the first time, Taste
Map and Song X-Ray crashed with a FAISS "key not found" error.

**Root cause:** `mark_done()` used to fire immediately per-vector, inside
`add_vector()`, completely decoupled from whenever `save_index()` next actually
persisted the FAISS index to disk. A Colab disconnect between checkpoints left 22
segments (2 songs) marked `'done'` in the SQLite DB with **no corresponding vector
ever actually saved** — silent data loss that only became visible once something tried
to read those vectors back.

**Fix:** split the concerns cleanly. `EmbeddingRepository.add_to_index()` now does
FAISS-add only, with zero DB status change; `add_vector()` (used by dev seeding,
tests, anything that doesn't checkpoint) keeps its old immediate-mark-done behavior.
`pipeline/embed_library.py`'s `run_batch_embedding` now owns checkpointing internally:
vectors go into the index via `add_to_index()` as they're computed, and `mark_done()`
only fires for a batch of segments *right after* `save_index()` has actually persisted
them to disk — so `'done'` can never outlive an unsaved vector, no matter when the
process dies. A new `reset_status()` method reverts a segment back to `'pending'`, and
`scripts/repair_orphaned_embeddings.py` was written as a one-off repair for data
already affected: it found and fixed exactly the 22/14,580 orphaned segments
(0.14% of the real library). A regression test simulates a crash mid-batch
(`checkpoint_every=2`, crash on the 3rd song) and asserts every `'done'` segment has an
actually-reconstructable vector.

This exact "never mark complete before the artifact is durably saved" discipline
becomes the template every later pipeline (structure, DNA, novelty, stems) follows.

### Structure computation crashes on real audio

Running the structure pipeline against the real library crashed 150 songs in with a
librosa `ParameterError` about `width=1`. Some real tracks — near-silent intros,
sparse/ambient sections, non-rhythmic clips — produce a beat-synced chroma with only
1–2 frames, too few for `recurrence_matrix`'s default width. Fixed by checking the
synced frame count and falling back to framewise (unsynced) chroma, which has orders
of magnitude more frames and is effectively always safe.

Immediately after, a *different* track crashed the same fix: its decoded audio was
short enough that even the framewise fallback had too few frames. Rather than chase
every possible real-world audio anomaly (corrupt downloads, near-empty decodes — a
real risk at 1,400-file scale), the batch pipeline was reworked to **isolate failures
per-song**: catch, report via an `on_error` callback, move on. No `.npy` gets written
for a failed song, so the existing compute-once check naturally retries it on the next
run. This per-song error-isolation pattern is reused for every later batch job
(stems, descriptive tagging).

### The diagonal saga (a real reversal, worth understanding fully)

Reviewing real Song X-Ray output, the self-similarity heatmap showed no bright
diagonal. Verified with real data before touching anything: `diag(matrix)` was
*exactly* 0.0 for every song, ruling out a colorscale/display issue.

**Root cause:** `librosa.segment.recurrence_matrix` defaults to `self=False`, which
explicitly zeroes the main diagonal — correct behavior for *recurrence/repeat-finding*
(the trivial self-match is noise when hunting for *other* matching regions), but wrong
for something being called a *self-similarity* matrix, where every moment matching
itself perfectly is the whole point. Fixed with `self=True`, pinned by a new test.

Then, the very next day, the spec itself was updated to **explicitly document the
zeroed-diagonal behavior as intentional** — a real conflict with the fix just made.
The `self=True` change was **reverted back to `self=False`**, with the test flipped to
pin the zeroed-diagonal behavior instead, and all real structure/fingerprint data for
all 1,400 songs was regenerated to match. The lesson here (expanded on in the
companion rebuild-plan document) is to settle a convention like this against the
actual downstream use case *before* treating a surprising result as automatically a
bug.

### The structure-timeline granularity saga

Reading a raw self-similarity heatmap is hard for a non-technical viewer even with
good contrast. The fix: a single horizontal color-coded timeline bar as the *primary*
Song X-Ray view (matching colors = similar-sounding sections), with the raw matrix
demoted to a "technical detail" expander. `analyze_structure()` computes chroma/
beat-tracking once and derives both views from the same underlying frames, so they
describe the exact same data. K-Means (not the exploratory notebook's DBSCAN) was
chosen for predictable behavior across 1,400 songs without per-song `eps` tuning.

A real bug was caught before this went further: raw per-beat K-Means labels flip far
too often to read as clean blocks — confirmed on real songs (e.g. 22 segments across
27 seconds, most under 1.5s). A minimum-segment-duration merge pass plus an
adjacent-same-label collapse pass were added so every emitted segment is phrase-scale
(≥3s).

That patch, though, was treating a symptom. The real diagnosis came a day later: raw
per-frame K-Means labels flip *constantly* even within one perceptual section
(confirmed: 18–46 label changes per ~30s clip on real songs). Merging away all those
short spurious runs just propagated *one* label across most of the song instead —
confirmed against the **full real library: 850/1,399 songs (61%) collapsed to exactly
one segment**, useless for a "matching colors = repeated sections" view.

**The real fix:** agglomerative clustering constrained to a **tri-diagonal
(temporal-adjacency-only) connectivity graph**. Frame *i* can only ever merge with
frame *i−1* or *i+1*, which guarantees each of the *k* final clusters is one
contiguous time interval *by construction* — no possibility of the confetti K-Means
produced. Verified on the same real songs used to diagnose the bug: exactly *k*=6
clean segments every time, versus 18–46 before. The minimum-duration merge pass stays
in as a safety net for genuinely short edge segments, but is no longer the primary
mechanism.

---

## Part 3: Deployment Prep, Fingerprints, Song DNA

### Deployment prep — done early, per the spec's own advice

The spec explicitly recommends deploying the Core version early, before starting
Strong-tier work, rather than scrambling to deploy everything at the end — and this
was actually followed:

- `config.py`'s `DATA_DIR` resolves to `data/` (gitignored, the full local dataset) if
  present, else falls back to `deploy_data/` (a small subset *committed to git*) —
  since Streamlit Community Cloud only ever sees what's actually in the repo, no
  Drive/local-disk access the way Colab has.
- `audio_path_for(song)` resolves audio via `AUDIO_DIR` + `fma_track_id` at **read
  time**, rather than trusting `song.filepath`, which is only ever valid in the
  environment it was written in — the exact same class of bug `repath_audio_paths.py`
  fixed for Colab→local, this time for local→deployed.
- A real `.gitignore` bug was caught before it could bite: `*.db` and `*.index` were
  unscoped and would have silently excluded `deploy_data`'s own database and FAISS
  index too. Scoped to `data/**/` only.
- `requirements.txt` is just `-e .` — single source of truth with `pyproject.toml`'s
  base dependencies, no duplication/drift risk. The deployed app never runs CLAP
  inference or structure computation itself, so the `colab`/`dev` extras (torch,
  transformers, librosa, Demucs) are never installed there.
- `scripts/build_deploy_subset.py` builds a stratified sample (~25 songs/genre, seed
  42) of the real library into `deploy_data/`, copying already-computed
  vectors/audio/structure — no recompute needed. First generated as a 200-song subset,
  verified by *temporarily hiding* `data/` and confirming all pages worked cleanly
  against the subset alone before restoring it.

This subset needed rebuilding, and got rebuilt, every single time a new facet or
column landed — song DNA, harmony vectors, harmony+composite fingerprints, novelty
fields, the `is_saved` migration, stem vectors, the whitened harmony index, and
(much later) `description`/`sound_tags`. Each rebuild is its own commit in the
history. One of these rebuilds is where a real bug was caught late in the project (see
Part 8) — worth remembering as you read the rebuild-plan document's advice to
generalize this the moment a new persisted field exists.

### Song fingerprints and Song DNA

`facets/fingerprint.py` added `structure_fingerprint()` (a pure-numpy downsample of
the SSM, no audio dependency — safe to compute anywhere, including the deployed app)
and `sound_fingerprint()` (a mel-spectrogram thumbnail, needs librosa — so
precomputed in the structure batch pipeline, reusing audio already loaded there,
rather than computed lazily in the UI, keeping the deployed app free of a librosa
dependency). `harmony_fingerprint()` followed once the harmony facet existed — kept as
a *strip*, not square, since a chroma-gram only ever has 12 meaningful rows.
`composite_fingerprint()` overlays all three as RGB channels
(structure→red, harmony→green, sound→blue): where they agree the image reads
bright/white, where they diverge distinct color casts appear.

`facets/song_dna.py`'s `compute_raw_song_dna()` produces five cheap, well-established
librosa-derived scalars per song: tempo (beat tracking), energy (mean RMS), brightness
(mean spectral centroid), harmonic complexity (Shannon entropy of mean chroma,
normalized), rhythmic density (onset rate). `analysis/song_dna.py`'s `DNANormalizer`
does corpus-wide min-max scaling per axis, since e.g. "brightness in raw Hz" is
meaningless without knowing the library's actual range. `components/plotting.py`'s
`song_dna_radar_overlay()` renders two songs' normalized profiles as semi-transparent
overlaid radar traces.

A DNA-specific real bug: `build_deploy_subset.py` was silently dropping song DNA
entirely (it only ever copied base `Song` fields) — caught before pushing, by actually
checking the deployed radar chart worked rather than just "no exceptions." This is the
first instance of a bug class — "the deploy-subset builder doesn't know about a new
field" — that recurs again, much later, with `description`/`sound_tags` (Part 8).

### Structural confidence — the "Abstractivity" facet

Not every song has clean verse/chorus repetition; forcing a segmented timeline onto an
ambient/through-composed track would show either one meaningless block or noisy fake
boundaries. `facets/novelty.py`'s `compute_novelty_curve()` slides a Gaussian-tapered
checkerboard kernel (Foote, 2000) along a **dense** cosine-similarity matrix
(deliberately *not* the sparse k-NN recurrence matrix used for the heatmap — checkerboard
novelty needs a dense signal). The kernel size adapts to real seconds (~4s each side),
not frame count, since frame duration differs ~20× between beat-synced and
framewise-fallback chroma.

Before writing formal tests, this was validated against two synthetic ground-truth
cases: a pure tone correctly shows 0 peaks; two audibly distinct halves (a tritone
apart) show exactly 1 peak, landing within 0.01s of the true midpoint. Only then was
it wired into `analyze_structure()`, which now also returns `novelty_curve`,
`has_clear_structure`, and `structural_confidence` (the curve's own standard
deviation, as a continuous flatness measure). This was deliberately kept as a
**confidence gate** on the existing (already-validated, already-tuned) agglomerative
segmentation, rather than rebuilt around novelty-peak boundaries directly — the more
"textbook" approach, but far riskier to already-working code for the same underlying
goal.

Verified on the real library: **1,394/1,399 songs (99.6%) show clear structure**; the
5 that don't are qualitatively sensible — an Experimental-genre track, an a cappella
track, a couple of ambient-leaning Electronic tracks — checked by name, not just by
exception-free rendering.

### Harmony facet

`HarmonyFacet` — chroma-CQT mean/std, 24-dim — registered alongside `SoundFacet`. The
batch embedding function was generalized to embed *any number* of facets from one
shared audio load per song, with fully independent per-facet compute-once tracking, so
harmony's addition needed no second audio-loading pass. First genre-cohesion
comparison, the spec's explicit "ablation-style finding": sound showed strong cohesion
(54.4% vs. 11.9% random baseline) while harmony was weaker but still clearly above
baseline (21.2% vs. 11.7%) — confirming the two facets capture genuinely different
signal rather than one riding the other's coattails.

A real bug, caught via `AppTest`: `resources.py` only ever called
`embedding_repo.load_index("sound")`, so switching Moment Matcher to harmony raised a
`KeyError` even though the DB correctly reported segments as embedded — `status()`
reads from SQLite, but `get_vector()`/`search()` need the FAISS index actually loaded
into memory. Fixed to load every registered facet's index.

### LLM explanation layer — the first LLM integration, and the security pattern it set

Each Moment Matcher match gets a one-sentence plain-language explanation via the
Anthropic API (`claude-haiku-4-5`). `sonic_explorer/llm/explain.py` keeps prompt
construction and the API call in core (plain Python, no Streamlit dependency).

This is where the project's prompt-injection defense pattern was established, and it
never changed afterward: song titles/artists are untrusted input flowing into the
prompt. Two *independent* layers — `sanitize_untrusted_text()` strips the delimiter
characters themselves so untrusted text literally can't close a `<song_data>` block
and open a fake one, **and** the system prompt explicitly frames that block as inert
data the model must never treat as instructions. A real red-team pass was run against
the *live* API (not a mocked client) at this point already: a normal call, a
delimiter-escape + fake `SYSTEM` override, and a plain-English "reveal your system
prompt" attempt — all three behaved correctly.

Secrets go through `st.secrets` (the platform secrets manager once deployed) with an
env-var fallback for local dev; a missing key degrades gracefully (matches still
render, just without the explanation line) rather than crashing. A simple per-session
call cap (60) is the abuse/cost guardrail for the public deployment.

---

## Part 4: Strong-Tier Build-Out

### ICA as a Taste Map alternative

`compute_taste_map()` gained a `method="pca"|"ica"` parameter — FastICA as an
alternative 2D projection, with clustering always running on the full embedding
regardless of which projection is used for display. This directly answers one of the
spec's own stated risk questions: are ICA's independent components more individually
nameable than PCA's variance-maximizing ones? Spot-checked against the real
1,400-song library: both landed on legible axes (International/acoustic vs.
Electronic/Hip-Hop; Rock vs. ambient/Electronic), but ICA mostly *rediscovered* the
same two axes rather than surfacing new independent structure — reported as a real,
presentable finding for this corpus, not a failure of the feature.

### Radar chart as query

A new "Query by" mode in Moment Matcher: sculpt a target profile across the five
song-DNA axes and get nearest songs by distance in the same normalized [0,1]⁵ space
every song already lives in for the static radar overlay — genuinely just distance +
ranking over existing infrastructure, no new pipeline. Search re-runs automatically on
slider release (Streamlit's natural rerun behavior gives this for free). Playback
switches via a plain audio-source swap rather than a rendered crossfade — a
deliberate, discussed scope cut to avoid adding librosa/soundfile as base runtime
dependencies for the deployed app.

### LLM-based re-ranking

A two-stage retrieve-then-rerank pipeline: stage 1 over-fetches 15 candidates by
cosine similarity (the existing FAISS search); stage 2 has Claude reason jointly over
the query and the *whole* candidate list in one listwise call, resorting it down to
the final top 6. This gets the "reason about query and candidates together" property
real cross-encoder reranking wants, without training a cross-encoder — much cheaper
and faster than one pointwise call per candidate. Reuses `explain.py`'s exact
sanitization/framing defenses. `parse_rerank_response()` falls back to the original
cosine order on any malformed/out-of-range/hijacked response — reranking degrades to
"no reranking," never a crash or dropped candidates. A live red-team pass (delimiter
escape + fake override, "reveal your system prompt") both failed to hijack it.

### Ask the DJ — the conversational agent

A chat interface backed by an Anthropic tool-calling loop (`llm/agent.py`'s
`MusicAgent`), with three tools at launch (`llm/agent_tools.py`), each a thin wrapper
over already-tested infrastructure:

- `get_song_profile` — DNA lookup, so the agent can reason about a reference song's
  actual values rather than guessing.
- `search_similar_songs` — facet-based retrieval, reuses `RetrievalService` directly.
- `search_by_mood_profile` — nearest-neighbor over DNA space, the exact mechanism
  radar-chart-as-query already uses. This is the spec's explicit hook for "make it
  moodier" style requests: the LLM reasons about which axes a mood word implies and
  picks numeric target values *itself*, no hardcoded word→axis mapping.

Conversation history is caller-owned (a plain list passed in and back out, never
stored inside `MusicAgent`), keeping the agent a stateless, `cache_resource`-safe
object while Streamlit owns per-session chat state — the same separation every other
LLM client in the package uses. Tool results flow through the Anthropic API's
structured `tool_result` content-block boundary — a real, meaningful difference from
`explain.py`/`rerank.py`'s hand-rolled delimited prompts, since the API itself keeps
tool output structurally separate from instructions — plus `sanitize_untrusted_text()`
on every tool-result string as defense-in-depth.

A live red-team pass in a throwaway in-memory library (never touching real data)
tried direct prompt injection in the user message and tool-result injection via a
song with a malicious title embedding a fake closing tag plus a fake `SYSTEM`
override asking the model to leak its prompt and say "PWNED" — both failed. Normal
requests were verified too, including a "moodier and more stripped-back than X"
request correctly chaining `get_song_profile → search_by_mood_profile`.

### Explore page, My Library

A new page: a song-as-node similarity graph, with "Explore (global)" and "My Library"
built as **one shared code path** filtered by which songs' vectors go in — a filter on
the same view, not two implementations. `analysis/network_graph.py` builds a
k-nearest-neighbor graph (k=4 default) over the same mean-pooled sound vectors Taste
Map already uses, laid out with `networkx.spring_layout` (force-directed) rather than
a projection, since "follow an edge to a neighbor's neighbors" only makes sense with
actual graph edges. Clicking a node opens a side-panel identity card in place —
title/artist, real tempo, key/year explicitly marked "not yet computed" rather than
faked, cluster ID, which facets are embedded, all four fingerprints, an inline player.

A new "save song" feature: an `is_saved` migration column, `save_song()`/
`unsave_song()`, `list_songs(saved_only=True)`.

### Stem-separated facets (Demucs)

Four new facets — vocal, drums, bass, instrumental (`facets/stems.py`) — each is
literally just `SoundFacet` (CLAP) run on an isolated audio stream instead of the
full mix. No new architecture, only a different registered name — the payoff of the
`Facet` strategy pattern being real from day one. Separation itself
(`pipeline/separation.py`, Demucs `htdemucs`) is heavy, GPU-preferred inference, so it
runs on Colab (`notebooks/03_stem_separation_and_embed.ipynb`), following the exact
same "heavy compute lives on Colab" pattern the sound facet established.

The Demucs integration was initially written from documented API knowledge alone —
this local dev machine has no GPU and doesn't have torch/Demucs installed — explicitly
flagged as an unvalidated caveat in the code until a real Colab smoke test (separate
one song, listen to all four stems inline) confirmed it actually worked: an
instrumental/experimental song correctly separated into a near-empty vocal stem plus
audibly distinct drums/bass/other.

`pipeline/embed_stems.py` mirrors `embed_library.py`'s compute-once/checkpoint
discipline exactly (a segment only ever marked done after its vector is durably
saved), but also isolates per-song failures the way the structure pipeline does — a
bad file is more likely across a 1,400-song Demucs run than during cheap chroma
extraction.

Two real notebook bugs surfaced running this for real: the smoke-test cell picked a
random song without checking it actually existed in this Colab session's synced
folder (hit a real `FileNotFoundError`); and the manifest was reconstructing audio
paths as `"{track_id}.mp3"` when curated audio actually sits in a subfolder — fixed to
read `curated_tracks.csv` directly, the same source notebook 02 already uses, instead
of guessing the layout.

**The near-silent-stem problem.** A real question came up mid-build: should the
pipeline check for vocals before trying to extract them? Since Demucs always produces
all four stems in one pass regardless, there's no separation-time cost to save — but
blindly embedding a *near-silent* stem (an instrumental track's vocal stem, or any
facet on a track that genuinely lacks it) is worse than wasted compute: near-silent
segments from *different* songs would all embed close together in CLAP space, making
unrelated instrumental tracks look "similar" on the vocal facet purely because
they're both quiet. Generalized beyond vocals to every stem facet (an a cappella
track's drum/bass stems have the identical problem): an energy gate compares each
segment's stem RMS against the full mix's RMS *per segment*, not per song, so a track
with a silent intro but real vocals later still gets those later parts embedded.
`EmbeddingRepository.mark_skipped()` gives "deliberately not computed" its own status,
distinct from `'pending'` — without it, resumability would re-run the full Demucs
separation on every future invocation for a result that was never going to change.

Explore's facet selector became a multiselect pulling live from the registry, and
multi-facet selection blends via `build_blended_similarity_graph()` — averaging each
facet's own cosine-similarity *matrix* rather than raw vectors, since different facets
live in different, often differently-sized embedding spaces with no shared vector
space to average within.

Two hardcoded-facet-list bugs were caught and fixed around this time: Moment Matcher's
"Match by" radio and Ask the DJ's `search_similar_songs` tool were both still hardcoded
to `["sound", "harmony"]` from before the stem facets existed — generalized to pull
live from `default_registry()`. Generalizing surfaced a *further* latent bug:
`explain.py`'s `FACET_DESCRIPTIONS` dict only had entries for sound/harmony and
deliberately raises `KeyError` on an unknown facet — selecting any stem facet in
Moment Matcher would have crashed the explanation/rerank calls the moment real stem
data landed. Caught and fixed before it ever fired in practice.

`scripts/merge_colab_db.py` was written here for a real, not hypothetical, need:
local-only work (harmony/structure/DNA, computed after the last Colab sync) had
diverged from the DB snapshot notebook 03 started from, so downloading its output and
overwriting the local DB would have silently discarded all of that local work. The
script merges in just the requested facets' `embedding_status` rows by `segment_id`,
refuses to run if song/segment counts don't match between the two DBs, and backs up
the local DB before writing anything. (This exact script gets reused again, much
later, for the final stem-facet reprocessing sync — see Part 8.)

First six-facet genre-cohesion comparison, confirming every facet carries real signal
above the ~12% random baseline: sound 54.4%, instrumental 40.6%, drums 36.6%, vocal
36.1%, bass 27.2%, harmony 21.2%.

### A real production crash

Reported live: clicking a node in a real browser crashed with `KeyError: 0` on
`event.selection.points[0]["customdata"][0]` — never caught by `AppTest`, which
doesn't replicate a real browser's actual Plotly selection-event payload. Root cause:
the edges trace (densely crisscrossing the graph, no `customdata` configured) can
catch a click landing close to a line rather than precisely on a node marker,
producing a selection point with unusable `customdata`. A new
`extract_selected_song_id()` helper handles every failure mode defensively (missing
key, empty list, non-indexable value, non-dict-like point) by resolving to `None`
instead of crashing. A new `tests/test_plotting.py` exercises these defensive branches
directly against the pure function — this is exactly the class of bug `AppTest`
missed, so a plain unit test on the extracted logic is the real regression guard here,
not another simulated Streamlit event.

---

## Part 5: The Narrative Pages — Methodology, App Walkthrough, and Real EDA

Genre-cohesion results were exported to JSON alongside the existing HTML chart, for
pulling real numbers into a presentation page without re-parsing a chart.

A methodology walkthrough page became the app's new landing point: a step-by-step
narrative (data → facets → DNA/fingerprints → retrieval → evaluation → live demo)
showing real evidence at every step instead of just asserting it. Curated evidence
(genre-cohesion numbers, 12 real nearest-neighbor examples, DNA/structure validation)
is embedded directly in the page source rather than loaded from `data/artifacts/` at
runtime, since that directory is gitignored and wouldn't exist post-deploy.

It was then expanded significantly based on feedback that it needed to actually *show*
the work, not just assert it: every comparison got real side-by-side audio players so
differences could be heard, not just read; the DNA section got a real radar-chart
overlay instead of a bare number table; the fingerprints section became a live song
picker (5 curated songs spanning the structural-confidence range) reusing Song
X-Ray's exact rendering logic instead of one fixed screenshot; a Taste Map section
showed the real PCA+K-means projection twice (cluster-colored vs. genre-colored) as a
direct visual test of whether sonic clusters track genre.

The landing page was then split — two distinct concerns had been living in one page:
how the library was analyzed/preprocessed, versus how to actually use and interpret
the live interactive app. **Methodology** kept the analysis narrative; a new **App
Walkthrough** page became a guided pass through Explore, Taste Map, Song X-Ray, Moment
Matcher, and Ask the DJ using the real, currently-running components (not
screenshots), each section explaining what the shapes/colors/positions actually mean.

Shortly after, the standalone Taste Map page was retired entirely — Taste Map and
Explore had become redundant (both "the whole library as a scatter/graph"). Explore
gained a View toggle (network graph / 2D map) covering both. Axis interpretability
(both in Methodology and in Explore's own "Inspect these axes" expander) became a
genuinely rigorous two-step check instead of a qualitative-only guess:
`correlate_axes_with_features()` correlates each PCA/ICA axis against the five
already-computed DNA features first — PCA's y-axis turned out to be well-explained by
energy/brightness/harmonic-complexity moving together (r > 0.4), while PCA's x-axis
and most ICA axes don't resolve to any single feature, reported honestly rather than
papered over. The "songs at the extremes" listen became the explicit *secondary*,
qualitative fallback only for whatever the correlation check doesn't explain.

Ask the DJ's system prompt was fixed for two real observed failures at this point:
it was handing ambiguous/unusual requests back to the user as a menu of options
instead of committing to an interpretation and searching, and its match explanations
were inventing plausible-sounding sensory detail (specific instruments, "vibe") that
no tool result had actually returned. Both fixes are pinned by regression-guard tests
directly on the prompt text, since live model behavior isn't something a fake-client
test can assert.

Real EDA was added to Methodology, not just curated highlight examples: a
track-duration histogram (which empirically *confirmed* the 30.0-second-clip finding
rather than just asserting it) and an artist-distribution check (unique count,
top-10, median tracks/artist — checking whether the library is dominated by a
handful of prolific artists, which would let a facet "cheat" by learning artist
production style rather than genre); full-library histograms for all five DNA axes;
Taste Map projection-value histograms; and — most consequentially — a per-facet
top-1-vs-random-pair score-distribution check
(`evaluation/retrieval_diagnostics.py`), sampled live from the real FAISS indexes.
This directly revealed harmony's collapsed embedding space (random pairs already
scoring 0.85–0.95) as the *mechanistic* reason it underperforms on genre-cohesion —
something genre-cohesion alone never would have surfaced.

---

## Part 6: The Case-Study Day — Four Hypothesis→Test→Result Investigations

This was the single most intensive day of investigative work in the project, and it's
worth walking through in full because each investigation follows the same honest
discipline: state a hypothesis, test it against the real library, report the real
result — including when the result is a failure, a mixed outcome, or a fix that isn't
actually wired into the live app yet.

### Investigation 1: The vocal-facet AST cross-check (the one that ultimately failed)

Methodology's own honest limitation noted that Demucs' "vocal" stem can carry real
energy from non-vocal content — a confirmed case, a track called "3rd Chair"
(cello/violin), scored a 0.58 stem-to-mix energy ratio, well above the energy gate's
0.05 threshold, despite having no real vocals. **Hypothesis:** a pretrained AudioSet
tagger (AST) could independently check whether a song actually contains singing/speech,
catching what the energy gate can't.

**First attempt (failed): score the whole 30-second clip at once.**
`pipeline/vocal_presence.py`'s first design ran AST over the whole clip. Result: there
is no threshold that sorts this correctly — "3rd Chair" (the exact bleed case this was
supposed to catch) scored *higher* on the "Speech" label than two genuinely real-vocal
songs it must not exclude. Diagnosis: AST's output over a full 30s clip is a
continuous distribution across all 527 AudioSet classes, not a sparse detector —
dominant instrumental/percussive content in the mix swamps genuinely-present-but-quieter
vocals into the same tiny-probability noise floor that residual background "vocal"
mass sits at in truly instrumental tracks.

**Redesign: score each ~5-second segment individually, take the max.** A shorter
window has less competing instrumental content, so a real vocal moment doesn't get
drowned out. Validated against 9 confirmed cases: every "keep" song scored ≥0.020,
every "exclude" song scored ≤0.016 — a threshold around 0.018 sorted all 9 correctly,
"3rd Chair" included.

**Then it was reworked again, from song-level to segment-level gating specifically**
(a real design correction, worth calling out): a genuinely vocal song still has
purely-instrumental stretches (intro, bridge, break) that correctly score low on
vocal presence — gating whole *songs* in/out was the wrong unit, since Moment Matcher
queries a specific *moment* and expects similar moments back, not "this song generally
has vocals." `filter_vocal_facet_by_ast.py` was rewritten to score and gate each
segment individually via `remove_from_index` + `mark_skipped` per segment, so an
instrumental bridge in an otherwise-vocal song gets excluded from vocal retrieval
without touching that song's real vocal segments.

**The reality check that killed the technique.** The 9-song validation had been
checked against *assumed* labels (genre + curated-example status), not real listening.
Before trusting it at library scale, a 400-segment random sample across the whole
library (not restricted to any genre) found **56.2% of segments scoring below
threshold** — far too high to explain as normal instrumental intros/bridges alone.
That prompted an actual **blind human-listening spot-check**: 10 segments, judged with
no model score or label shown, via a temporary standalone tool
(`scripts/vocal_spotcheck_app.py`), compared to the model's verdict only afterward.

**Result: only 6/10 agreed.** Three false negatives (real vocals the model missed,
scoring 0.017–0.0179) and one false positive (confidently scored 0.0228 as vocal, no
real vocals present) — the false positive's score *exceeds every false negative's*, so
no single threshold can fix both simultaneously. This isn't a calibration problem; the
underlying keyword-max score simply doesn't reliably track real vocal presence with
this technique. **Documented honestly: NOT applied to the live vocal facet, and not
recommended with this technique** — the energy gate (already live) remains the vocal
facet's only automated quality check, and both the "instrumental stretch within a
vocal song" and "Demucs bleed" problems remain open, honestly unresolved limitations.

A follow-up correlation check (Investigation 4, below) asked whether fixed-window
segmentation itself explained any of these errors.

### Investigation 2: Harmony whitening

Score distributions had already shown harmony's random-pair baseline sitting at
0.85–0.95 — the raw 24-dim chroma-derived space has very little natural spread, so
real differences barely register once L2-normalized. **Hypothesis:** whitening each
dimension to zero mean/unit variance across the corpus before re-normalizing should
spread the space out along directions that actually vary — a pure post-hoc transform
on already-computed vectors, no re-extraction needed
(`analysis/embedding_whitening.py`, `scripts/whiten_harmony_index.py`).

**Result: a real, honest split outcome.** The score geometry improved dramatically —
random-pair scores dropped from a misleadingly-high ~0.85 average to essentially 0,
and individual rankings got roughly 7× more decisive (top1-vs-top2 margin
0.0027 → 0.0187). But genre-cohesion, the actual task metric, stayed flat
(20.7% → 20.1%, within sampling noise). Conclusion: whitening fixed the *symptom* (a
compressed, misleading score range) but not the underlying limitation — a 24-dim
chroma mean+std summary is a coarse representation of harmony, and rescaling it can't
inject discriminative information that was never captured in the first place. Kept
live regardless, since a sharper single top match is a real usability win in Moment
Matcher and Ask the DJ, even without a genre-cohesion lift.

### Investigation 3: Song-level aggregation

Score distributions revealed every facet's top-1-vs-top-2 margin is small
(typically <0.01) — with ~14,600 segments and often only a few hundred per genre,
there's usually a long plateau of near-tied single-segment candidates.
**Hypothesis:** mean-pooling a song's segments into one vector before ranking (the
same aggregation Taste Map/Explore already use for visualization) should smooth that
segment-level noise into a sharper song-level signal
(`retrieval/song_level_index.py`).

**Result: validated, and mostly positive.** Ranking margin improved for *every*
facet (1.3×–2.3× sharper). Genre-cohesion improved for 5 of 6 facets — Instrumental
+5.6pp, Bass +4.1pp, Vocal +3.9pp, Drums +3.0pp, Harmony +1.7pp. Sound was the lone
exception, slightly worse (55.4% → 52.5%) — plausibly because Sound's per-segment
specificity was already the strongest of any facet, and averaging a song's segments
blurs together genuinely different sonic moments (a quiet intro vs. a loud chorus)
precisely where that segment-level precision was doing real work. **Status when
documented:** implemented and validated, but not yet wired into Moment Matcher's UI as
a selectable option. (This was wired in the very next day — see Part 7.)

### Investigation 4: Does segment misalignment explain the vocal-gate errors?

For the same 10 blind-listened vocal-gate segments, checked whether fixed-clock-interval
segmentation explains the confusing errors, by correlating against the Structure
facet's *already-computed* novelty detection for those same segments — no new audio
processing, a pure correlation check against data that already existed. A temporary
visualization tool plotted each segment's novelty curve with the sampled window and
structural boundaries overlaid.

**A confirmed hit:** "Facing the Sea" — the human note was "vocals only in the last 2
seconds of the sampled window" (~8s transition); the real novelty curve shows a peak
at 8.96s with a segment boundary at 9.0s, right where the ear placed it. **But it
doesn't generalize:** straddling a structural boundary is common (7/10 windows) and
doesn't predict which cases were actually confusing — the two persistent errors
("Requiem for a Small Town," "Thursday & Snow") both sit entirely *within* one
structural segment, no boundary nearby to blame. A quick, cheap follow-up (reusing
already-computed song DNA, zero new processing) found those same two unexplained
errors rank #1 and #2 lowest on structural confidence *and* #1 and #2 highest on
rhythmic density in the sample — suggestive at n=2, not confirmed, that dense,
structurally-uniform tracks give AST's keyword scoring less textural handle to work
with, independent of window placement. **Conclusion:** a confirmed, real mechanism for
*one class* of error, not a general explanation — documented as a bounded finding, not
oversold into a segmentation redesign the timeline didn't justify.

### Wiring in what was validated

The day after these investigations, song-level aggregation (Investigation 3) was
actually wired into Moment Matcher's UI as a selectable "Match against" mode: "A
specific moment" (unchanged segment-level behavior) versus "Whole songs" (mean-pools
the query song's segments, searches the already-tested song-level FAISS index).
Whole-song mode deliberately skips LLM reranking/explanations (there's no one specific
matched moment to explain) but keeps the DNA-comparison radar chart, which already
operates at song granularity. `deploy_data` was refreshed with the whitened harmony
index at the same time.

---

## Part 7: Calibration Infrastructure, Docker/CI, and the Deep-Learning Additions

This section covers the batch of work done specifically to round out the spec's
Section 9 (Deep Learning Additions) and Section 12 (Resume-Gap Tooling) items,
sequenced deliberately: the human-time bottleneck (calibration rating) was built
*first* so a human could start rating in parallel with everything else, ahead of
the zero-risk tooling and the CNN baseline.

### Calibration-rating infrastructure

The `calibration_ratings` table had existed in the schema since early in the project
but had zero rows and no read/write code anywhere. `CalibrationRepository` wraps it
properly. `generate_calibration_pairs()` draws a 350-pair set across three similarity
bands — high, medium, and random — pulled off the *sound facet's real retrieval
results*, rather than uniform random sampling: a naive random sample over ~14,600
segments skews almost entirely to "obviously dissimilar," giving a rating set with
little real variance for a regression or fine-tuning objective to learn from.

`scripts/calibration_rating_app.py` is a standalone, multi-session rating tool —
progress persists to the real database, not session state, so closing the tab and
returning later resumes exactly where it left off. Deliberately **blind**: no
title, artist, or algorithm score shown, to avoid biasing similarity judgments with
recognition rather than actual sound.

This is explicitly the bottleneck-by-human-time prerequisite for the blend-weight
regression and (conditionally) CLAP fine-tuning from spec Section 9 — built first, so
rating could start immediately in parallel with the rest of the day's work.

### Docker, CI, linting, red-teaming

- **Docker**: a `Dockerfile` (base deps only — the deployed app never runs audio
  processing itself, matching `requirements.txt`'s existing reasoning) plus
  `.dockerignore`. Actually built and run locally to verify — not written on faith:
  the image builds clean, the container serves HTTP 200, and `deploy_data`'s 200
  songs are queryable inside it end-to-end.
- **CI**: `.github/workflows/ci.yml` runs ruff + pytest on every push/PR, plus a
  second job building the Docker image — catches dependency drift, syntax errors, or
  an image that stops building before it ever reaches deployment.
- **ruff**: found and fixed 20 real (if minor) issues across the codebase — mostly
  `zip()` calls without an explicit `strict=` parameter, one dead variable
  assignment, a couple of unused loop variables, and one `assert False` inside a
  `try` block rewritten to a proper `raise`. `.pre-commit-config.yaml` wires in
  ruff's lint-and-fix hook (deliberately *not* `ruff-format`, which would have
  touched ~57 files for pure style churn against this codebase's existing deliberate
  compact style, with zero bug-catching value).
- **Red-teaming**: 14 adversarial prompts run against the *real* live agent and the
  real Anthropic API, not a mock — instruction-override attempts, hallucination
  bait, tool misuse via adversarial inputs, injection via untrusted-data framing,
  fabrication bait, scope overreach, and a direct extraction attempt. **All 14
  handled correctly.** Full transcripts in `scripts/red_team_findings.md`.

### CNN genre classifier baseline

The one Section 9 item that doesn't depend on the calibration dataset: a genuinely
*trained* neural network — not another pretrained-embedding-and-retrieval story like
everything else in this project (CLAP, chroma, Demucs, and AST are all
pretrained-and-frozen). Deliberately small — three convolutional blocks — a baseline
comparison point, not a research architecture.

Split into two modules specifically for CI: `analysis/mel_features.py` (pure
numpy/librosa, no torch, fully testable in the main CI environment) and
`analysis/genre_cnn.py` (the actual `nn.Module`, with a module-level torch import —
never imported by the deployed app or the main test suite; its tests use
`pytest.importorskip("torch")` so CI skips them cleanly rather than failing on a
dependency deliberately not installed there).

`scripts/train_genre_cnn.py` trains on the real local library with a stratified
train/val/test split, caching extracted mel-spectrograms (the expensive part) to
disk. **Result: 47.2% test accuracy against a 12.5% random baseline** (8 balanced
classes) — a real, non-trivial signal from spectrograms alone, with no pretrained
model and no facet engineering.

### LLM-on-top-of-AST descriptive tagging

AST already tags instruments/sound events per song (the same model
`vocal_presence.py` uses, reused here unfiltered via `sound_tagging.get_descriptive_tags`).
A new synthesis step feeds those raw tags plus normalized song DNA into the same LLM
layer (and the exact same injection defenses) Ask the DJ's explanations already use,
producing a short natural-language description — "calm piano," "sassy hip hop" — via
new `build_description_messages`/`generate_description` functions. This is a
batch-precompute step (`scripts/generate_song_descriptions.py`, checkpointed every 25
songs), since AST needs torch/transformers, which the deployed app deliberately never
installs.

A real bug in the first version of this script: every *other* AST call site in the
codebase (the vocal-presence check, the prevalence sampler) always slices a short
window before calling the classifier — this script passed the *entire* loaded clip
instead. AST ("...-10-10-...") is trained on 10-second windows; fixed to take a
representative 10-second slice from the middle of the track (avoiding a cold-open
silence/fade-in). Run against the real library: **1,399/1,400 songs got a
description** (the one skip has no song DNA computed at all — a pre-existing gap,
unrelated to this fix). Spot-checked descriptions read naturally and match genre
("Heavy metal with spoken word", "Dreamy synth pop", "Hypnotic dub reggae").

---

## Part 8: The Final Stretch — App Restructure, the DJ's Sound-Content Search, and Syncing the Reprocessing Pass

This is the most recent block of work, done in rapid succession.

### Real landing page, Methodology/Results split

`app.py` had been, for a while, a bare `st.switch_page("pages/0_Methodology.py")`
passthrough with no content of its own — a real, if minor, bug. It became a genuine
introduction page: what the project is and why, plus two deliberately honest
placeholders (a naive-baseline demo, a related-work section) left as explicit stubs
rather than guessed at, since writing them without real supporting content would have
been worse than an honest "not written yet" note.

Methodology was split along a process-vs-outcome line: **Methodology** keeps the
analysis/preprocessing narrative and the case-study section (an iterative *process*);
a new **Results** page holds the outcome numbers — the genre-cohesion evaluation
(moved verbatim), a new section finally surfacing the CNN classifier's actual training
run (which had been computed but never shown anywhere in the app), and an honest
"in progress, no results yet" status for the calibration/blend-weight regression
rather than fabricated placeholder numbers. New flow: intro → Methodology → Results →
App Walkthrough → the individual interactive pages → Explore last (unrestricted
free-form exploration, positioned after the guided narrative has already built
context).

The app's own sidebar was showing a literal `"app"` label — Streamlit derives a
multipage app's sidebar text directly from the filename (underscores → spaces, no
title-casing; there's no separate "display name" setting for the main script).
Renamed `app.py` → `Overview.py`, and swept every reference: six `AppTest` suites'
entry-point paths, Methodology's back-link, and the Dockerfile's `ENTRYPOINT`.

### Explore becomes the real hub

Song X-Ray, Moment Matcher, and Ask the DJ had been sitting as independent top-level
sidebar entries with no inherent context — visiting any of them directly never
crashed, but each defaulted to an arbitrary first-alphabetical song/moment with zero
connection to whatever a user might have just been looking at, undermining the
intended "drill down from what you're exploring" flow. Fixed by migrating page
registration from Streamlit's legacy `pages/` folder auto-discovery to
`st.navigation()`, using `visibility="hidden"` for the three drill-down pages —
confirmed, via Streamlit's own documentation and an isolated experiment, that hidden
pages stay fully reachable through `switch_page`/`page_link` while disappearing from
the sidebar. (A full merge of the roughly 500 lines of UI code across those three
pages directly into Explore was considered and explicitly rejected as the
higher-risk option.)

Context now genuinely carries through: Explore's "Open full Song X-Ray" button
stashes the selected song's ID in session state before switching pages; Song X-Ray
consumes it (popped immediately, so it applies exactly once — a stale/bookmarked
direct visit still degrades to a sane default, not a stuck stale selection). Once a
structure-timeline segment is selected on Song X-Ray, a "Find similar moments" button
maps that structural block to the *nearest fixed-window retrieval segment* — they're
genuinely different segmentations, and the mapping is done by nearest start-time, not
assumed alignment — and hands song+segment to Moment Matcher the same way. A
persistent "Ask the DJ" companion link sits on Explore itself, since the DJ needs no
song-specific context (it's a genuinely global chat tool).

A logo (user-provided) was wired in via `st.logo()`, appearing consistently across
every page's sidebar/header and clickable back to Overview. The source file had a
solid white background; since the app is dark-theme-only, a derived transparent
white-wordmark asset was generated via a simple distance-from-white alpha threshold
(safe for this kind of clean text-only mark), with the original kept untouched as the
source asset.

### The DJ learns to search by sound content

A real, reported gap: asked "any songs with crow sounds," the DJ said it could only
search by song title or a reference track — none of its three existing tools
(`get_song_profile`, `search_similar_songs`, `search_by_mood_profile`) can satisfy a
request naming a specific sound/instrument with no reference song. The synthesized
`description` field (Part 7) already existed but isn't enough on its own for *exact*
content search: a short 2–5-word synthesized phrase routinely drops specific/niche
tags a user might actually ask about — "crow" wouldn't survive into a phrase like
"rustic folk."

A new `songs.sound_tags` column stores the *raw* AST tags themselves (JSON-encoded
`[[label, score], ...]`), persisted going forward by `generate_song_descriptions.py`
alongside the description it already generates, and backfilled for all 1,399
already-described songs via a tags-only pass (re-tags, but skips the LLM call
entirely, since the existing description doesn't need regenerating) —
**1,399/1,399 succeeded**. A new `search_by_sound_content` tool matches a query
against both the raw tags (the reliable signal for a named sound/instrument) and the
description text (catching vibe-language that never became a discrete tag), ranking
tag-matched songs first. Verified against the real backfilled data: a live "crow"
query returns a real match with "Crow" in its matched tags; "saxophone," "guitar,"
and "siren" all returned correct real matches too.

### Syncing the completed stem-facet reprocessing pass

The Colab stem-separation notebook had a reprocessing run in flight for most of the
project's later history, aimed at the same near-silent-stem problem described in Part
4 — fixing a data-quality issue where a handful of near-silent isolated stems were
being indexed as if meaningful, on the corrected/hardened separation+embedding
pipeline. Once it completed (all 1,400 songs processed, final index sizes: vocal
8,674, drums 10,395, bass 10,484, instrumental 13,858), the result was synced down via
the exact same `merge_colab_db.py` script written back in Part 4 — 1,400 songs /
14,602 segments matched exactly, no divergence — and the four corrected FAISS indexes
were swapped in (sound and harmony untouched, since neither depends on stem
separation).

The genre-cohesion evaluation was re-run against the corrected data. Sound and harmony
were unchanged, as expected. The four stem facets shifted a few points each, and —
worth noting as its own honest finding — the **random-pair baseline itself also
moved** for those four facets (roughly 11.7% → 12.5–13.9%), since removing near-silent
noise from the index changed what a typical *random* pair looks like, not just what a
*good* match looks like.

Rebuilding `deploy_data/` after this surfaced one more real bug in the same family as
the earlier song-DNA one (Part 3): `build_deploy_subset.py` predated the
`description`/`sound_tags` columns entirely and would have silently dropped them from
every future rebuild of the deployed subset — which would have quietly broken the
descriptive-tagging feature and the DJ's sound-content search on the live deployment
the next time this script ran. Fixed to copy both fields, and confirmed all 200 songs
in the rebuilt subset carry them through correctly.

---

## Where Things Stand Today

**App structure**, top to bottom: `Overview.py` (real intro page, with two explicit
placeholders still unwritten) → **Methodology** (the full analysis narrative and
case-study section) → **Results** (genre-cohesion numbers, the CNN baseline, an
honest "pending" status for the calibration regression) → **App Walkthrough** (a
guided tour of the live components) → **Explore**, the actual hub for everything
else, from which **Song X-Ray**, **Moment Matcher**, and **Ask the DJ** are reached
as contextual drill-downs rather than independent destinations.

**Data**: 1,400 real FMA songs, 8 genres, 14,602 segments, six retrieval facets
(sound, harmony, vocal, drums, bass, instrumental) plus Structure/Abstractivity as a
visualized (not retrieval-indexed) facet. Every song has song DNA, structure/sound/
harmony/composite fingerprints, a synthesized natural-language description, and raw
AST sound tags. `deploy_data/` is a 200-song stratified subset kept in sync with all
of the above for the Streamlit Community Cloud deployment.

**Genre-cohesion (k=10, current numbers, post-reprocessing):** Sound 54.4% vs. 11.9%
random; Harmony 21.4% vs. 11.7%; Vocal 38.1% vs. 13.9%; Drums 37.4% vs. 13.1%; Bass
26.7% vs. 12.5%; Instrumental 41.5% vs. 11.8%. CNN genre-classifier baseline: 47.2%
test accuracy vs. 12.5% random.

**Deep-learning additions status:** the CNN baseline is done. The calibration-rating
tool is live and collecting data, but **still at zero ratings collected** as of this
writing — which means the blend-weight regression (Section 9) hasn't run, and the
CLAP fine-tuning go/no-go decision (explicitly conditional on that regression, never a
firm commitment) hasn't been made. This is the single open item actually blocking
further deep-learning work.

**Explicitly deferred, not started:** presentation prep (spec Section 15); the
Overview page's two placeholder sections (naive-baseline demo, related work); a
multi-facet side-by-side view and a Song X-Ray multi-strip view (both flagged early on
as "worth doing if time allows," never confirmed into final scope).

**Test suite**: 307 tests, all passing, covering core logic (facets, repositories,
retrieval, evaluation), the agent/tool layer, and every page via Streamlit's `AppTest`
harness — CI runs ruff + the full suite + a Docker build on every push.
