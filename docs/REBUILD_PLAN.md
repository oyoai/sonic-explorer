# Sonic Explorer — If Rebuilding From Scratch Today

This is a companion to `PROJECT_HISTORY.md`. That document is what actually happened,
in order. This one asks a different question: **knowing everything the real build
taught us, what's the correct order to build this in, and why?**

This isn't "what's different" — it's a real, followable build sequence, phase by
phase, with the reasoning for *why that phase belongs there* baked in. Several phases
land in a different order than the real history, specifically because of lessons the
real history paid to learn. Those reorderings are called out explicitly wherever they
happen, rather than silently presented as if this was obvious from the start.

## The lessons this plan is built around

Before the phases, it's worth naming the recurring patterns from the real build that
this plan is designed to front-load, since they explain most of the reordering below:

1. **Validate before scaling, every single time, no exceptions.** CLAP's embedding
   space, the novelty-curve algorithm, Demucs separation, the vocal-gate AST scoring —
   every one of these got a small, cheap, real-data-or-synthetic-ground-truth check
   *before* a library-scale batch job was ever kicked off. This worked every time it
   was actually done. The one time a validation step got skipped past — trusting a
   9-song "assumed label" check as if it were equivalent to real human judgment for
   the vocal-gate — cost a full investigation cycle to discover the technique
   fundamentally didn't work. Lesson: any check that gates or filters data (an energy
   gate, a confidence gate, a threshold) needs a real human-judgment validation step,
   not just an internally-consistent one, before it's trusted.

2. **Never mark something "done" before its result is durably saved.** The FAISS
   data-loss bug (Part 2 of the history) happened because `mark_done()` was decoupled
   from `save_index()`. Once fixed, the corrected pattern — checkpoint saves the
   artifact, *then* marks status, atomically as one step — was reused for every later
   pipeline without incident. This should be the checkpoint pattern from the very
   first batch job, not discovered via real data loss on song #1,378.

3. **Settle representational conventions against the actual downstream use case
   before treating a surprising result as a bug.** The self-similarity diagonal
   got "fixed" (self=True) and then reverted (back to self=False) a day later, once
   it turned out the zeroed diagonal was the intended convention all along. A five-minute
   conversation about "what should the diagonal mean here, given what the timeline
   view actually needs" before writing the fix would have skipped the churn.

4. **For any timeline/segmentation task, temporal contiguity is a hard constraint,
   not a hope.** Plain K-Means on a sequence of frames does not know frames are
   ordered — it happily assigns cluster 3 to frame 10, cluster 1 to frame 11, and
   cluster 3 again to frame 12. This was discovered the hard way (confetti segments,
   a merge-pass patch that fixed the symptom, then a real fix three commits later:
   agglomerative clustering constrained to a tri-diagonal connectivity graph, so a
   cluster *is* a contiguous interval by construction). If you already know you're
   segmenting a *sequence*, use a sequence-respecting algorithm from the first attempt.

5. **Generalize a registry/facet pattern completely the moment it exists — don't let
   any call site hardcode the old, shorter list.** Multiple real bugs in the history
   are exactly this: a facet gets added, and some UI component or evaluation script
   that was written before it existed still has the old facet list hardcoded, and
   breaks (sometimes loudly, sometimes silently) the moment someone actually exercises
   the new facet through that path. A one-time grep for every place a facet list is
   written out by hand, done the moment the registry pattern is introduced, prevents
   this whole bug class.

6. **If you persist a derived value from a model, persist the raw output too, not
   just the summary you happen to need right now.** The `sound_tags` backfill (Part 8)
   exists because only the *synthesized description* (a lossy, short phrase) was
   originally saved from AST's tagging pass — when a later feature needed to search
   the raw tags, all 1,399 songs had to be re-tagged from scratch (a real, if bounded,
   re-compute cost) just to capture data that was already computed once and then
   thrown away. Persist the raw model output at the same time you persist any
   derived/summarized version of it.

7. **Any script that builds a deployable subset or export needs to be treated as
   part of the schema — the moment a new persisted field exists, that script needs
   updating in the same commit, not discovered later via a silent gap.** This bit the
   project twice (song DNA, then description/sound_tags).

8. **Decide the information architecture (what's a hub, what's a drill-down, what
   needs context to be meaningful) before building each screen as its own page.**
   Song X-Ray, Moment Matcher, and Ask the DJ were each built as independent
   sidebar-level pages, worked fine on their own, and then needed a real
   architecture pass (much later) to become properly contextual drill-downs from
   Explore. That pass wasn't hard, but it was entirely avoidable if the hub/drill-down
   decision had been made explicit before the first of those three pages existed.

9. **Start any human-time-bottlenecked work in parallel, on day one of the relevant
   phase, not after everything else is built.** Calibration ratings gate the
   blend-weight regression and the CLAP fine-tuning decision — real deep-learning
   work — but the rating tool itself was built quite late in the real timeline. A
   human can only rate so many pairs per day regardless of how fast the engineering
   moves; that clock should start ticking as early as it possibly can.

10. **Do the zero-risk tooling first, literally, because the spec already says so.**
    Section 12 of the spec explicitly names Docker+CI+ruff+pre-commit as
    "zero-risk, do these first" — and in the real build, this was mostly true (deploy
    prep happened early), but the actual Docker/CI/ruff/red-teaming batch landed only
    in the last few days. It costs nothing to do this in week one, and it protects
    every commit that follows.

With those in hand, here's the sequence.

---

## Phase 0 — Repo skeleton, tooling, and the core/interface split (Day 1, before any model touches audio)

Set up the whole zero-risk foundation before writing a single line of facet code:
`pyproject.toml` with base + `dev` + `colab` extras defined from the start (so the
"deployed app never installs heavy ML deps" boundary is a real dependency-group split
from day one, not something discovered and retrofitted later); `pytest` configured;
`ruff` configured and passing on an empty repo; `.pre-commit-config.yaml` wired to
ruff's lint-and-fix hook; a `Dockerfile` (base deps only) and a GitHub Actions
workflow that runs ruff + pytest + a Docker build on every push, even though there's
nothing to test yet. Decide and write down the core/interface separation rule (core
package is plain Python, zero Streamlit imports; the interface layer calls into core
and contains no logic of its own) as an actual sentence in the README, since this
single discipline is what makes every later restructure (and there are several)
cheap instead of a rewrite.

*Why first:* this is pure upside with zero risk to momentum, exactly per the spec's
own Section 12 guidance — and every single day after this one benefits from CI
catching a regression immediately instead of at the end.

## Phase 1 — Data acquisition and real EDA (before any modeling)

Acquire the FMA subset (Colab notebook + local equivalent script, same pattern the
real build used and which worked well: heavy/GPU-preferred steps live in Colab,
everything else can run locally). Before writing a single facet, actually look at the
data: genre distribution, track-duration distribution, artist concentration. In the
real build, the "every clip is exactly 30.0 seconds" fact — which directly shaped how
Structure and its confidence-gating had to be framed later — wasn't confirmed via a
histogram until Methodology's EDA section, quite late. Confirm it on day one instead;
it changes how you scope the Structure facet's ambitions from the very first line of
code you write for it, rather than requiring a later reframing.

*Why here, not later:* every fact you learn about your data early is a fact you don't
have to retrofit an explanation for later. The real project's EDA additions were all
genuinely valuable — pulling them to day one costs nothing and removes a later
"wait, why does Structure only cover 30 seconds" surprise.

## Phase 2 — Core package: models, repositories, and a checkpoint discipline that's correct from the start

`Song`/`Segment` dataclasses, the `Facet` strategy interface, `FacetRegistry`,
SQLite schema + `SongRepository`/`EmbeddingRepository` wrapping FAISS. This is
identical in shape to the real Day 1 work, with exactly one addition made explicit
from the start rather than discovered via a data-loss incident: **the checkpoint
contract.** Write `add_to_index()` (FAISS write only) and the batch-job's own
checkpoint loop (mark-done only fires immediately after a successful `save_index()`
for that batch) as the *only* pattern any batch pipeline in this project ever uses,
documented once, right here, in the repository layer's own docstring. Every later
pipeline (structure, DNA, novelty, stems, descriptive tagging) inherits this for free
instead of needing its own bug-then-fix cycle.

Also decide, right here, the answer to "does mark-done-but-vector-missing ever
happen, and what do we do about it" — write `reset_status()` and a generic
orphan-repair script *now*, as part of the pipeline's test suite (simulate a crash
mid-batch, assert no orphaned "done" rows), rather than writing it reactively once
real data loss is discovered.

*Why this order:* this is the one piece of infrastructure every single subsequent
phase depends on. Getting the checkpoint contract right here, once, is far cheaper
than discovering it's wrong on song 1,378 of a 1,400-song real run.

## Phase 3 — Sound facet (CLAP), validated before anything is built on top of it

Implement `SoundFacet` (CLAP, lazy-imported, MFCC fallback). Before writing any
retrieval UI: does CLAP's embedding space actually separate anything meaningful, or
does it just cluster by loudness/genre-obviously? This is explicitly one of the
spec's own named risk points ("resolved early via direct inspection, before building
downstream features on top of it") — in the real build this genuinely was checked
early, and it should stay exactly that early here. A five-minute nearest-neighbor
spot-check against a handful of known-similar and known-dissimilar songs is enough to
green-light the rest of the sound-facet work.

Build the batch embedding pipeline against this facet, using Phase 2's checkpoint
contract. Build a small synthetic genre-clustered dev dataset in parallel — the real
build's choice to develop UI against fast, fake, correctly-shaped data while the real
Colab job runs in the background was a genuinely good call; keep it exactly as-is.

## Phase 4 — RetrievalService + first UI slice, verified with AppTest from the first page

`RetrievalService` as the single choke-point for every query. First Streamlit pages:
a clickable Taste Map (PCA + K-means) and a bare Moment Matcher. Establish the
`AppTest`-plus-real-local-server verification habit *here*, on the very first page,
rather than letting it become a discipline that gets adopted a few features in. The
SQLite cross-thread bug (`check_same_thread=False`) and the FAISS-numpy-int
`reconstruct()` bug both only ever showed up by actually running the app, not by
importing it — start running it for real immediately.

## Phase 5 — Genre-cohesion evaluation, built early and re-run after every facet

`genre_cohesion_at_k()` plus `scripts/run_evaluation.py`. Verify it end-to-end
against the synthetic dev data first (a clean, predictable expected number is the
right thing to confirm the whole metric pipeline against), then treat it as the
mandatory go/no-go checkpoint the spec's own cut-rule calls for: run it after *every*
new facet, not just at big milestones. This is unchanged from the real build, which
did this well — keep it.

Also build the score-distribution diagnostic (`retrieval_diagnostics.py`'s
top-1-vs-random-pair check) *at the same time* as genre-cohesion, not much later.
In the real build this diagnostic is what eventually revealed harmony's collapsed
embedding space as the mechanistic reason it underperformed — that insight is far
more useful sitting next to the harmony facet's very first genre-cohesion number than
arriving as a retrospective EDA addition weeks later. Make "cohesion number *and*
score-distribution shape" the standard two-part checkpoint for every facet from here
on, including facets not yet built.

## Phase 6 — Deployment skeleton, early (per the spec, and it worked)

`deploy_data`'s stratified-subset builder, portable audio paths
(`audio_path_for()` resolving at read time instead of trusting a stored filepath —
apply this lesson immediately rather than waiting to discover the Colab→local and
local→deployed path-portability bugs separately, since they're the same bug class),
`requirements.txt` as `-e .` against `pyproject.toml`'s base deps only. Deploy the
Core version (Sound facet + Taste Map + Moment Matcher only) to Streamlit Community
Cloud right now, before Strong-tier work starts. This is literally what the spec
recommends and the real build actually followed it — keep doing that.

**Establish the rule right here, in the deploy-subset builder's own docstring or a
CI check**: any new column added to the `songs` table, or any new artifact type a
song can have, must be added to `build_deploy_subset.py` in the same commit that
introduces it. This single sentence, written down at the moment the script is first
created, would have prevented both the song-DNA-dropped bug and the much later
description/sound_tags-dropped bug — the same failure mode hit twice in the real
history because nothing forced anyone to remember the rule the second time.

## Phase 7 — Structure facet, built with the lessons already applied

This is the phase where the real build's iteration cost was highest, and where this
plan diverges most from simply "do it in the same order." Build it once, correctly,
using what the real project had to learn the hard way:

1. **Decide the diagonal convention up front.** Ask: what does the *timeline view*
   (the thing users will actually look at) need the diagonal to mean? A
   self-similarity heatmap's diagonal being trivially "identical to itself" is not
   the interesting signal; the interesting signal is *other* matching regions. Land
   on `self=False` (matching what the spec ultimately settled on) as the starting
   choice, and skip the fix-then-revert cycle entirely.
2. **Handle short/sparse audio from the start.** Real FMA tracks include near-silent
   intros and ambient/sparse sections that produce too few beat-synced chroma frames
   for a default-width recurrence matrix. Build the framewise-chroma fallback (and
   the "still too short even for that" per-song error-isolation wrapper) as part of
   the pipeline's first version, not as two separate emergency patches discovered
   mid-batch-run.
3. **Segment the timeline with a sequence-aware algorithm from the first attempt.**
   Skip plain K-Means over frames entirely — go straight to agglomerative clustering
   constrained to a tri-diagonal (temporal-adjacency-only) connectivity graph, which
   guarantees contiguous segments by construction. This one decision avoids the
   confetti-segments bug, the merge-pass patch that only treated the symptom, and the
   discovery (on 61% of the real library) that the patch didn't actually work.
4. **Build the novelty-curve / structural-confidence gate at the same time as
   segmentation, not as a later addition.** You already know from Phase 1's EDA that
   not every 30-second clip will have clean structure — build the "does this song
   even have clear sections" check alongside the segmentation itself, validated
   against the same synthetic ground-truth cases the real build used (a pure tone →
   0 peaks; two audibly distinct halves → exactly 1 peak, accurate to within 0.01s)
   before ever running it against real audio.

Run the structure batch pipeline against the real library using Phase 2's checkpoint
contract and per-song error isolation from the start. Add song fingerprints
(structure + sound) as a cheap addition riding the same audio load — this part of the
real build was already efficient; keep it as-is.

## Phase 8 — Calibration-rating tool (moved up, deliberately)

Build `CalibrationRepository`, `generate_calibration_pairs()` (stratified by
similarity band off real sound-facet retrieval, not naive random sampling — this
design choice was correct in the real build and should be kept exactly), and the
blind standalone rating app. Hand it to a human rater *today*, in parallel with
everything from Phase 9 onward.

*Why moved this early, against the real build's actual order:* this is the single
clearest "start the human-time bottleneck as early as possible" lesson from the whole
project. The blend-weight regression and the CLAP fine-tuning decision both depend on
having enough real ratings, and a person can only rate so many pairs a day no matter
how fast the rest of the engineering moves. In the real build this tool didn't exist
until near the very end, which is why — even after every facet, every case study,
every piece of tooling was done — the calibration dataset still sat at zero ratings.
Moving this to Phase 8 (as soon as the sound facet's real retrieval results exist to
draw stratified pairs from) buys the maximum possible number of real days for a human
to actually rate pairs against everything else that follows.

## Phase 9 — Song DNA + radar overlay

`compute_raw_song_dna()`, `DNANormalizer`, the static radar-chart overlay in Moment
Matcher. Unchanged in substance from the real build — this was efficiently sequenced
already (cheap librosa features, reused everywhere downstream). One change: update
`build_deploy_subset.py` in the *same commit* that adds the DNA columns, per the
Phase 6 rule — no separate "oops, forgot to copy DNA" fix commit needed.

## Phase 10 — Harmony facet, with the score-distribution check as a mandatory step

`HarmonyFacet` (chroma-CQT). Run genre-cohesion *and* the score-distribution
diagnostic (Phase 5's standing rule) immediately. In the real build, harmony's
collapsed random-pair-baseline problem (0.85–0.95) wasn't discovered until an EDA
pass much later — here, it's visible on day one of the harmony facet's existence,
which means the whitening fix (Phase 15 below) could be attempted right away instead
of waiting for a retrospective diagnostic pass to reveal the problem. Add the harmony
fingerprint and the composite (structure+sound+harmony) fingerprint at the same time,
since it's the same cheap reuse-existing-audio-load pattern as before. Update
`resources.py` to load *every* registered facet's index — not just the first one that
exists — as a standing rule from the moment a second facet exists, rather than
discovering the `KeyError` when someone switches to it.

## Phase 11 — LLM explanation layer, with security built in from message one

The first LLM integration point. Build `sanitize_untrusted_text()` and the
inert-data system-prompt framing *as part of* the first prompt that ever includes
untrusted song metadata — not retrofitted after the feature works. Run a live
red-team check (delimiter-escape attempt, plain-English extraction attempt) against
the real API before calling this feature done, exactly as the real build did. Every
later LLM feature (reranking, the agent, descriptive tagging) reuses this exact
sanitize-plus-frame pattern without needing its own security design pass — which is
exactly what happened in the real build, and is worth explicitly deciding to keep:
build the pattern once, well, the first time an untrusted string touches a prompt.

## Phase 12 — ICA alternative, radar-as-query, LLM reranking, Ask the DJ

These four Strong-tier items were well-sequenced in the real build and don't need
reordering relative to each other — they're additive, low-risk, and each reuses
already-tested infrastructure (ICA reuses the PCA projection code path; radar-as-query
reuses the DNA normalizer; reranking reuses the explanation layer's security pattern;
the agent reuses `RetrievalService` and the DNA search directly). Build them in this
order. For the agent specifically: decide the tool list by asking "what request
patterns can a user actually type that none of Moment Matcher's existing UI
mechanisms satisfy" rather than mechanically wrapping every existing function as a
tool — this is exactly how the real `search_by_mood_profile` tool was scoped, and
it's the right instinct to keep deliberate.

## Phase 13 — Decide the information architecture before building Explore, Song X-Ray, Moment Matcher, or Ask the DJ as separate pages

This is the second explicit reordering relative to the real build. Before writing
*any* of these four pages, write down the actual navigation model: **Explore is the
single hub.** Song X-Ray is reached by selecting a song from Explore. Moment Matcher
is reached by selecting a moment within Song X-Ray. Ask the DJ is a persistent,
context-free companion reachable from Explore at any time. None of the three
drill-down pages is ever a standalone sidebar destination.

Build Explore first, as the hub — the network-graph/2D-map view, click-to-open song
detail. Then build Song X-Ray as something *only ever reached with a song ID already
in hand* (no "pick a song" dropdown defaulting to index 0 — there is no meaningful
default, because it's never meant to be landed on cold). Same for Moment Matcher: it
only ever receives a song+moment from Song X-Ray's own interaction, never defaults to
an arbitrary first song. Use `st.navigation()` with `visibility="hidden"` for these
two pages from the moment they're created, not migrated to later — Explore is the
only page with a visible top-level sidebar entry among these four.

*Why this avoids real cost:* in the real build, all three of these pages were built
as independent, fully-functional, individually-fine pages, and only later — after
enough usage revealed that landing on Moment Matcher cold with an arbitrary default
song is meaningless — did the architecture get corrected. The correction itself
wasn't expensive, but it was *entirely avoidable* by making the hub/drill-down
decision before writing the first of these four pages instead of after.

## Phase 14 — Stem-separated facets, with the near-silent-stem gate built in from the very first embed

Demucs separation, smoke-tested on real Colab GPU (listen to all four stems from one
real song) before committing to a full 1,400-song run — this validate-before-scaling
step was done correctly in the real build; keep it exactly. But build the energy gate
(`_has_meaningful_energy`, comparing each segment's stem RMS against the full mix's
RMS *per segment*) and `mark_skipped()` as part of the *very first* stem-embedding
run, not as a fix discovered after noticing quiet unrelated tracks clustering
together. This is knowable in advance — any isolated-stem facet can produce
near-silent output for a track that doesn't feature that element — so build the gate
before the first batch run touches real data, not after a retrieval-quality
investigation finds the symptom.

At the moment these four new facets are registered, do the one-time sweep the real
build had to do reactively: grep every place a facet list is written out by hand
(Moment Matcher's radio options, the agent's `search_similar_songs` tool,
`FACET_DESCRIPTIONS` in the explanation layer, `build_deploy_subset.py`,
`run_evaluation.py`) and confirm every one of them pulls from `default_registry()`
instead. This was a real, multi-commit cleanup in the actual history, entirely
avoidable by doing the sweep once, right here, the moment the facet count first grows
past two.

## Phase 15 — The case-study investigations, run as standard checkpoints rather than a single big investigative day

With the score-distribution diagnostic already running from Phase 5/10 onward, the
harmony-whitening and song-level-aggregation investigations aren't really separate
"case study" work anymore — they're the natural next step the moment the diagnostic
shows a collapsed space (harmony) or a thin top1-vs-top2 margin (every facet). Run
them as soon as those signals appear, rather than batching four investigations into
one intensive day near the end of the project. Wire in whatever the investigation
validates (song-level aggregation as a selectable Moment Matcher mode; harmony
whitening applied to the live index) in the same work session that validates it,
rather than leaving a validated-but-unwired gap for a later commit to close.

For the vocal-facet AST cross-check specifically: **do the human blind-listening
validation step immediately after the first small-sample (9-song) validation passes
— not after a much later 400-segment prevalence check raises an alarm.** The real
build's mistake here wasn't the hypothesis or the redesign (whole-clip → per-segment
was the right fix for the right reason); it was trusting a small sample checked
against *assumed* labels as if it were equivalent to real human judgment, for long
enough that a full per-song vs. per-segment redesign cycle happened before the
technique's fundamental unreliability was discovered. The lesson from lesson #1 above
applies most sharply here: the moment a check is going to *gate or filter real data*
(remove segments from a live index), it needs a real blind human-listening validation
before that gate goes live — 10 minutes of blind listening, done early, is cheaper
than a redesign cycle followed by discovering the technique doesn't work anyway.

## Phase 16 — Descriptive tagging, with raw tags persisted from day one

Build the AST-tags-to-natural-language-description synthesis exactly as the real
build did (reuse the explanation layer's security pattern; batch-precompute since AST
needs torch/transformers the deployed app never installs; slice a representative
window before calling the classifier, learning immediately from Phase 14's stem-facet
energy-gate work that every AST call site in this codebase must window its input, so
this isn't a bug to discover independently here). The one change: **persist the raw
tags (`sound_tags`, JSON-encoded) in the very same batch pass that persists the
synthesized description**, not as a separate later backfill. This costs nothing extra
— the tags are already computed in memory as an input to the LLM call — and it means
a future "search by named sound/instrument" feature (which arrives in the very next
phase) never needs a costly re-tagging pass over the whole library just to recover
data that was already computed once and discarded.

## Phase 17 — The DJ's sound-content search tool

Because Phase 16 already persisted raw tags, this phase is now just: write
`search_by_sound_content()` (match against tags first, description text second, rank
tag-matches higher) and wire it into the agent's tool list and system prompt. No
backfill job needed — this is the concrete payoff of lesson #6.

## Phase 18 — CNN genre classifier baseline

Unchanged from the real build: split into a torch-free feature-extraction module and
a torch-dependent model module for CI's sake from the start (this split was done
correctly in the real build — keep it), train against the real library, report
against the genre-cohesion numbers already sitting in Results for direct comparison.

## Phase 19 — Blend-weight regression and the CLAP fine-tuning decision

Because Phase 8 started calibration collection on day one of the relevant era of the
project instead of near the very end, there should be a real, non-trivial number of
ratings sitting in the database by the time every other phase above is done. Run the
blend-weight regression against real data here. Make the CLAP fine-tuning go/no-go
call based on what that regression actually shows and how much time is genuinely
left — kept explicitly conditional, never a standing commitment, exactly as the real
build's own decision framing correctly insisted on.

## Phase 20 — Narrative pages, written incrementally throughout, not as a late writing exercise

This is the one phase that isn't really a phase — it's a standing practice that
should run alongside every phase above, exactly as it mostly did in the real build.
Every time a facet ships, a case study concludes, or an evaluation number changes,
update Methodology/Results with the real number and a short honest note *in the same
work session*, rather than letting narrative-writing become its own separate,
large, late task. The real project did this reasonably well (Methodology grew
commit-by-commit alongside the actual work); the one place it fell behind was the
Overview page's placeholders and a full "explain the whole project end to end"
document — which is, not coincidentally, exactly the kind of document this file and
its companion `PROJECT_HISTORY.md` are.

## Phase 21 — Buffer, presentation prep

Exactly as the spec's own day-by-day plan reserves the last stretch for this — with
the real difference that, following this plan, presentation prep starts with a
genre-cohesion story, a CNN baseline, four documented case studies, *and* real
calibration-regression results already in hand, rather than reaching the buffer days
with the regression still blocked on a rating tool that only just got built.

---

## The short version

If there's one sentence to take from this whole document: **the real build's biggest
recurring cost wasn't any individual technical mistake — CLAP, chroma, Demucs, AST,
PCA/ICA, the CNN all worked essentially as expected — it was a handful of sequencing
decisions** (checkpoint discipline discovered via data loss instead of designed in;
segmentation algorithm chosen without the sequence-awareness constraint in mind;
information architecture decided after three pages already existed independently;
raw model output discarded and later needing recomputation; the human-rating
bottleneck started last instead of first). Every one of those is fixable for free by
deciding the *policy* once, early, and then simply following it — which is exactly
what this reordered sequence tries to do.
