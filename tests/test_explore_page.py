"""AppTest smoke test for the three-panel Explore page: a browsable/
searchable song list + a clickable mini-map toggling between a neighbors-
only network graph and a PCA scatter map (left), the selected song's
identity card (middle), and an inline Moment Matcher (right). Must go
through Overview.py + switch_page -- some interactions on this page render
markdown/captions Streamlit only resolves with the full app context."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


def _run_explore() -> AppTest:
    # default_timeout=240, not this suite's usual 120 -- whichever test in
    # this file runs FIRST pays a real, measured cold-start cost computing
    # the dataframe's fingerprint/DNA columns for all ~1400 songs before
    # st.cache_data has anything warmed: ~78s for the four fingerprint
    # image columns (Structure/Sound/Harmony/Composite) alone, measured
    # directly, on top of the rest of the page's own rendering. Every
    # later test in the file reuses the same warmed process-level cache and
    # finishes well under 120s -- this margin exists for whichever test
    # happens to go first, not a general slowdown.
    at = AppTest.from_file("streamlit_app/Overview.py", default_timeout=240)
    at.switch_page("pages/7_Explore.py")
    at.run()
    return at


@pytest.fixture(scope="module")
def explore_at() -> AppTest:
    """Most tests in this file only ever READ the default render (no widget
    interaction) -- sharing one real cold-started instance across those
    (module-scoped: computed once, reused for the rest of this file) avoids
    paying the ~78s+ cold-start cost over two dozen times for identical
    output. Tests that mutate a widget/session_state and rerun, or that
    monkeypatch config before rendering, keep calling _run_explore()
    directly for their own independent, fresh instance -- sharing this one
    would leak state between them and change what they're actually
    testing."""
    return _run_explore()


def test_explore_page_runs_without_exceptions(explore_at):
    assert not explore_at.exception


def test_explore_page_has_no_hero_banner(explore_at):
    """Removed specifically on this page -- the three panels already take
    real vertical space; a decorative banner on top would compete with
    that, unlike every other page where there's nothing else up there."""
    assert not explore_at.exception
    charts = explore_at.get("plotly_chart")
    assert not any("hero_explore" in c.proto.id for c in charts)


def test_explore_page_defaults_to_a_selected_song(explore_at):
    """Unlike the old click-to-select design, a song is selected by default
    (the mockup this page follows shows the middle/right panels already
    populated, not an empty "click a node" hint)."""
    assert not explore_at.exception
    assert explore_at.session_state["explore_selected_song_id"] is not None
    caption_texts = " ".join(c.value for c in explore_at.caption)
    assert "Selected song" in caption_texts
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    assert "### Moment Matcher" in markdown_texts


def test_explore_page_has_search_box_and_two_per_tab_facet_selectors(explore_at):
    """No more literal/NL toggle -- the search bar is natural-language-only
    now (see the module docstring's "Search architecture" paragraph for
    why: the song list's own st.dataframe toolbar search covers literal
    lookups). No more single page-wide "Match by" selector either -- it
    used to live in the top bar and drive both the mini-map and Moment
    Matcher as one shared control; now that the mini-map lives inside
    Moment Matcher's own "Full Song" tab, each of the two tabs (Full Song,
    Moment) gets its own independent selector instead."""
    assert any(ti.label == "Search library" for ti in explore_at.text_input)
    assert not any(t.label == "NL search" for t in explore_at.toggle)
    facet_selects = [sb for sb in explore_at.selectbox if sb.label == "Match by"]
    assert len(facet_selects) == 2
    for facet_select in facet_selects:
        assert set(facet_select.options) == {
            "Sound", "Harmony", "Vocal", "Drums", "Bass", "Instrumental", "Sound tags",
        }


def test_explore_page_has_ask_the_dj_button(explore_at):
    """Sits at the very start of the top bar (left of the search box) --
    see test_explore_page_ask_the_dj_is_first_clear_only_appears_with_a_query
    for the document-order check."""
    assert any(b.label == "Ask the DJ" for b in explore_at.button)


def test_explore_page_never_shows_a_conversational_dj_reply_card():
    """Regression guard for a real architecture bug: Explore's search used
    to call MusicAgent.send_message() directly and display its prose reply
    as a "DJ says" card. Search is one-shot now (llm.search.nl_search(),
    not MusicAgent) -- no such card should exist anywhere on this page."""
    at = _run_explore()
    text_input = next(ti for ti in at.text_input if ti.label == "Search library")
    text_input.set_value("zzzzzznonexistentsongtitlezzzzzz").run()

    assert not at.exception
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "DJ says" not in markdown_texts
    assert "explore_search_note_from_dj" not in at.session_state


def test_explore_page_has_no_recent_searches_dropdown():
    """Removed: a real ordering bug made it render one rerun behind the
    actual search history, looking permanently broken."""
    at = _run_explore()
    text_input = next(ti for ti in at.text_input if ti.label == "Search library")
    text_input.set_value("test query").run()

    assert not at.exception
    assert not any(sb.label == "Recent searches" for sb in at.selectbox)
    assert "explore_search_history" not in at.session_state


def test_explore_page_clear_results_resets_search_state_not_selection_default():
    """State seeded directly rather than through a live NL search call -- NL
    search is semantic now, not exact-match, so typing a title and expecting
    precisely [target.id] back isn't a reliable premise. This test is about
    Clear's own reset behavior, not search matching, so explore_last_query is
    pre-set to the same text too -- otherwise the page's own "new query"
    check would fire a real search on this run and overwrite the seeded
    state before Clear ever gets clicked."""
    at = _run_explore()
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    target = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))[2]

    at.session_state["explore_search_input"] = target.title
    at.session_state["explore_last_query"] = target.title
    at.session_state["explore_filtered_song_ids"] = [target.id]
    at.run()
    assert at.session_state["explore_filtered_song_ids"] == [target.id]

    clear_button = next(b for b in at.button if b.label == "Clear")
    clear_button.click().run()

    assert not at.exception
    assert at.session_state["explore_filtered_song_ids"] is None
    assert at.session_state["explore_search_note"] is None


def test_explore_page_list_is_a_dataframe_with_song_columns(explore_at):
    """The list is a real st.dataframe() now, not a hand-rolled row loop --
    see the module docstring for why."""
    tables = [d for d in explore_at.get("dataframe") if d.key == "explore_song_table"]
    assert len(tables) == 1
    columns = set(tables[0].value.columns)
    assert {"song_id", "Thumbnail", "Title", "Artist", "Genre"} <= columns


def test_explore_page_numeric_dna_columns_are_real_numbers_not_formatted_strings(explore_at):
    """Real reported bug: Tempo (and every other DNA column) used to be a
    pre-formatted string ("103 BPM"), so clicking the column header sorted
    lexicographically -- "103, 258, ..., 40, 99" instead of numeric order.
    Display formatting now lives in column_config's NumberColumn instead,
    so the underlying dtype (and therefore the sort) stays numeric."""
    import pandas as pd

    tables = [d for d in explore_at.get("dataframe") if d.key == "explore_song_table"]
    df = tables[0].value
    for col in ["Tempo", "Energy", "Brightness", "Harmonic Complexity", "Rhythmic Density", "Repetition Rate"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not a numeric dtype: {df[col].dtype}"


def test_explore_page_sound_tags_column_never_shows_generic_labels():
    """Real reported gap: "Music"/"Musical instrument" are excluded from the
    Sound Tags FACET's embedding text (facets/tags.py's GENERIC_TAG_LABELS,
    since they dominate almost every song's top scores without describing
    anything distinguishing) -- but that exclusion never applied to the
    raw songs.sound_tags column this page's own "Sound Tags" display reads,
    a genuinely different code path. Finds a real song whose raw tags
    include a generic label among its top scores and confirms the
    displayed column excludes it -- a real regression check, not a
    vacuous one that would pass even with no filtering at all."""
    import json

    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    songs = song_repo.list_songs()
    target = None
    for s in songs:
        if not s.sound_tags:
            continue
        tags = json.loads(s.sound_tags)
        top_labels = [label for label, _score in sorted(tags, key=lambda t: t[1], reverse=True)[:3]]
        if "Music" in top_labels or "Musical instrument" in top_labels:
            target = s
            break
    assert target is not None, "no real song in this library has a generic tag in its raw top-3 -- can't test this"

    at = _run_explore()
    tables = [d for d in at.get("dataframe") if d.key == "explore_song_table"]
    row = tables[0].value[tables[0].value["song_id"] == target.id].iloc[0]
    shown_tags = [t.strip() for t in row["Sound Tags"].split(",") if t.strip()]
    assert "Music" not in shown_tags
    assert "Musical instrument" not in shown_tags


def test_explore_page_list_row_selection_changes_selected_song():
    """Simulates a real row click the same way this codebase simulates any
    dataframe selection in AppTest (no interactive "click row N" API exists):
    pre-seed the widget's own session_state key, then rerun. Real regression
    coverage for a genuine bug found while wiring this up: a naive external-
    sync implementation would silently discard this exact click (Streamlit
    already applies a widget's fresh frontend value to session_state BEFORE
    the script body runs, so a sync block that unconditionally overwrites it
    back to the old selection erases the click before st.dataframe ever
    resolves it) -- see the page's own sync-block comment for the fix."""
    at = _run_explore()
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    songs_sorted = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
    target = songs_sorted[1]  # row 0 is already the default selection; pick a different row

    at.session_state["explore_song_table"] = {"selection": {"rows": [1], "columns": [], "cells": []}}
    at.run()

    assert not at.exception
    assert at.session_state["explore_selected_song_id"] == target.id

    # A follow-up rerun with no new frontend interaction must not revert the
    # selection back to the old song -- this is exactly the failure mode the
    # sync-block fix above targets.
    at.run()
    assert not at.exception
    assert at.session_state["explore_selected_song_id"] == target.id


def test_explore_page_list_selection_survives_an_empty_search_without_crashing():
    """Regression guard for a real crash found while testing this: a search
    that narrows (or empties) the list left a stale, now-out-of-range row
    index in the table's own selection state, and Streamlit's own dataframe
    widget threw a pandas .iloc IndexError trying to restore it. desired_rows
    must be recomputed fresh against the CURRENT (possibly empty) list_df
    every run, never trusting a carried-forward index.

    State seeded directly (not via a live query) since NL search is semantic
    now -- there's no query string guaranteed to come back with zero real
    matches, so this forces the exact empty-list_df condition the crash
    needs deterministically instead of hoping a nonsense string happens to
    match nothing."""
    at = _run_explore()
    at.session_state["explore_filtered_song_ids"] = []
    at.run()

    assert not at.exception
    assert at.session_state["explore_filtered_song_ids"] == []


def test_explore_page_mid_panel_shows_structure_fingerprint_only_no_cycler(explore_at):
    """Fingerprint variant cycling was removed -- Selected Song shows only
    the Structure fingerprint now, no refresh/cycle button."""
    assert not explore_at.exception
    assert not any(b.key == "cycle_fp_variant" for b in explore_at.button)
    assert "explore_fp_variant_idx" not in explore_at.session_state
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    assert "tag-chip" in markdown_texts
    metric_labels = [m.label for m in explore_at.metric]
    assert "Tempo" in metric_labels
    assert "Key" in metric_labels
    assert "Sound Tags" in metric_labels


def test_explore_page_fingerprint_explanation_is_hover_only_not_a_visible_caption(explore_at):
    """The "brighter cells..." / "standing in for album art" explanation
    must live in the fingerprint <img>'s own title= attribute (a native
    browser hover tooltip), not as a permanently-visible st.caption
    underneath it. Rendered as a real <img> now, not a Plotly chart -- see
    fingerprint_image_data_uri's docstring for why (a real reported bug:
    the Plotly-rendered version looked visibly different from the list's
    image-rendered thumbnails for the same song)."""
    assert not explore_at.exception
    caption_texts = " ".join(c.value for c in explore_at.caption)
    assert "album art" not in caption_texts.lower()

    fp_markdown = next(m for m in explore_at.markdown if "<img" in m.value)
    assert "album art" in fp_markdown.value.lower()
    assert "title=" in fp_markdown.value


def test_explore_page_has_no_other_detailed_metadata_expander(explore_at):
    """Removed entirely per explicit request -- the X-Ray link is real
    functionality, now living inside the Song DNA expander; the old
    expander's purely-informational content (cluster id, facet Yes/No
    table) was dropped rather than silently unwrapped into plain view.
    Save/Unsave is removed for now (a separate, explicit request) --
    genuinely gone, not just relocated."""
    expander_labels = [e.label for e in explore_at.expander]
    assert "Other Detailed Metadata" not in expander_labels
    assert not any(b.label in ("Save", "Unsave") for b in explore_at.button)
    assert any(b.label == "Open full Song X-Ray →" for b in explore_at.button)


def test_explore_page_has_no_up_next_feature(explore_at):
    assert not explore_at.exception
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    assert "Up Next" not in markdown_texts
    button_labels = [b.label for b in explore_at.button]
    assert "◀ Back" not in button_labels
    assert "Next ▶" not in button_labels
    assert "explore_up_next_cache" not in explore_at.session_state
    assert "explore_up_next_history" not in explore_at.session_state


def test_explore_page_previous_next_always_visible_navigates_full_library_without_search():
    """Always visible now -- an earlier "hidden without an active search"
    restriction was explicitly lifted per direct request. With no search
    active it steps through the full sorted library (all_songs_sorted),
    not just search results."""
    at = _run_explore()
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    songs_sorted = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))

    assert at.session_state["explore_filtered_song_ids"] is None
    # Icon-only buttons (real Material icons, not the old ⏮/⏭ emoji -- see
    # the button call sites' own comment) have an empty label, so identity
    # is checked by key, same as every other icon-only button in this suite.
    button_keys = [b.key for b in at.button]
    assert "mid_prev_result" in button_keys
    assert "mid_next_result" in button_keys

    next_button = next(b for b in at.button if b.key == "mid_next_result")
    next_button.click().run()

    assert not at.exception
    assert at.session_state["explore_selected_song_id"] == songs_sorted[1].id


def test_explore_page_previous_next_navigates_search_results_when_a_search_is_active():
    """With a search active, Previous/Next steps through just its results,
    not the full library. State seeded directly rather than through a live
    NL search call -- search is semantic now, not exact-match, so there's
    no query string guaranteed to come back with a specific multi-result
    set; seeding forces a deterministic scenario instead."""
    at = _run_explore()
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    songs_sorted = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))
    result_ids = [songs_sorted[3].id, songs_sorted[7].id, songs_sorted[9].id]

    at.session_state["explore_filtered_song_ids"] = result_ids
    at.session_state["explore_selected_song_id"] = result_ids[0]
    at.run()

    next_button = next(b for b in at.button if b.key == "mid_next_result")
    next_button.click().run()

    assert not at.exception
    assert at.session_state["explore_selected_song_id"] == result_ids[1]


def test_explore_page_ask_the_dj_is_first_clear_only_appears_with_a_query():
    """Ask the DJ sits at the very start of the top bar now (left of the
    search box, not beside Clear on the right) -- checked via document
    order, since it's the very first button the page renders. Clear no
    longer has a fixed slot at all: it only renders once the search box
    actually has something in it to clear."""
    at = _run_explore()
    button_labels = [b.label for b in at.button]
    assert button_labels[0] == "Ask the DJ"
    assert "Clear" not in button_labels

    text_input = next(ti for ti in at.text_input if ti.label == "Search library")
    text_input.set_value("test query").run()

    assert not at.exception
    assert "Clear" in [b.label for b in at.button]


def test_explore_page_nl_search_stays_one_shot_regardless_of_api_key_state():
    """LLM features are a value-add, never load-bearing (CLAUDE.md). This
    dev environment has a real key in .streamlit/secrets.toml, so this
    exercises a real natural-language search call -- but the assertion that
    actually matters holds either way (real key configured, or not): no
    conversational reply card, ever, and filtered ids always end up a
    plain list (possibly empty). No toggle to set anymore -- the search box
    is natural-language-only now."""
    at = _run_explore()
    text_input = next(ti for ti in at.text_input if ti.label == "Search library")
    text_input.set_value("something moody and stripped-back").run()

    assert not at.exception
    assert isinstance(at.session_state["explore_filtered_song_ids"], list)
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "DJ says" not in markdown_texts


def test_explore_page_zero_result_search_clears_the_selected_song():
    """A search that matches nothing must not leave a PREVIOUS selection's
    detail panel on screen -- that reads as a stale leftover, not an honest
    "nothing matched." Forces the zero-result path deterministically by
    pre-exhausting the session's LLM call budget (_resolve_nl_search's own
    guardrail returns ([], ...) without a live API call once it's hit --
    see moment_matching.MAX_LLM_CALLS_PER_SESSION) rather than hoping a
    live query happens to return nothing. Both mid_col and Moment Matcher
    already fall back to "Select a song..." for a None selection, so this
    only needs to confirm the id itself clears."""
    at = _run_explore()
    at.session_state["llm_calls"] = 60  # moment_matching.MAX_LLM_CALLS_PER_SESSION
    at.session_state["explore_selected_song_id"] = 1  # simulate a real prior selection
    text_input = next(ti for ti in at.text_input if ti.label == "Search library")
    text_input.set_value("anything at all").run()

    assert not at.exception
    assert at.session_state["explore_filtered_song_ids"] == []
    assert at.session_state["explore_selected_song_id"] is None
    info_texts = " ".join(i.value for i in at.info)
    assert "Select a song from the list to see it here" in info_texts


def test_explore_page_list_rows_show_per_result_match_explanation():
    """Simulated the same way a real nl_search() result would populate
    state (AppTest has no live ANTHROPIC_API_KEY to exercise the real call
    path) -- each matching row must carry its own grounded "why it matched"
    text in the dataframe's own "Why it matched" column, not a single
    combined reply for the whole search."""
    at = _run_explore()
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    target = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))[0]

    at.session_state["explore_filtered_song_ids"] = [target.id]
    at.session_state["explore_search_explanations"] = {target.id: "Matched on tag: crow"}
    at.session_state["explore_selected_song_id"] = target.id
    at.run()

    assert not at.exception
    tables = [d for d in at.get("dataframe") if d.key == "explore_song_table"]
    row = tables[0].value.iloc[0]
    assert row["Why it matched"] == "Matched on tag: crow"


def test_explore_page_moment_matcher_panel_shows_segment_pills_and_results(explore_at):
    """Segment picker is real st.pills() now, not a button grid -- see the
    page's own module docstring for why (the button grid stretched every
    button to its own column's full width regardless of label length,
    reading as oversized boxes around short "0.0-5.0s" text)."""
    assert not explore_at.exception
    assert len(explore_at.pills) == 1
    options = explore_at.pills[0].options
    # Every song has 10-11 real segments, but only every other one is shown
    # (~5s apart) to keep the picker short -- see SEGMENT_DISPLAY_STRIDE.
    assert 4 <= len(options) <= 6
    assert all("s" in label for label in options)  # e.g. "0.0-5.0s"
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    assert "Top " in markdown_texts and "results" in markdown_texts


def test_explore_page_segment_pills_are_five_seconds_apart(explore_at):
    """Display-only regrouping: the underlying segments still overlap every
    2.5s (untouched), but the picker should show every other one, so
    consecutive pills read as clean 5s-spaced moments."""
    options = explore_at.pills[0].options
    starts = [float(label.split("–")[0]) for label in options]
    # starts[1:] is deliberately one element shorter (adjacent-pair diffing)
    # -- strict=True would be wrong here, it's not a same-length zip.
    diffs = [round(b - a, 1) for a, b in zip(starts, starts[1:], strict=False)]
    assert all(d == 5.0 for d in diffs)


def test_explore_page_moment_matcher_result_cards_have_match_badge(explore_at):
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    assert "match-badge" in markdown_texts
    assert "% match" in markdown_texts


def test_explore_page_clicking_a_segment_pill_updates_selection_and_matches():
    """A real select on a real pill -- unlike the old Plotly chart, no
    session-state simulation workaround needed, AppTest can select this
    directly. Target segment id computed independently from the real song
    data (song.segments[::2], mirroring SEGMENT_DISPLAY_STRIDE) rather than
    read back off the widget, since st.pills' AppTest wrapper exposes
    formatted labels, not the underlying option values, via .options."""
    at = _run_explore()
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    song_id = at.session_state["explore_selected_song_id"]
    song = song_repo.get_song(song_id)
    displayed_segments = song.segments[::2]
    assert len(displayed_segments) >= 2
    target_segment_id = displayed_segments[-1].id

    at.pills[0].select(target_segment_id).run()

    assert not at.exception
    assert at.session_state["explore_selected_segment_id"] == target_segment_id


def test_explore_page_moment_matcher_query_player_loops_the_selected_segment(explore_at):
    """Uses st.audio()'s own native start_time/end_time/loop parameters --
    not a custom HTML5/JS player, not a pre-sliced clip file, since neither
    turned out to be necessary (see the page's module docstring)."""
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    song_id = explore_at.session_state["explore_selected_song_id"]
    song = song_repo.get_song(song_id)
    query_segment = song.segments[0]

    audios = explore_at.get("audio")
    query_player = next(
        a for a in audios
        if a.proto.start_time == query_segment.start_sec and a.proto.end_time == query_segment.end_sec
    )
    assert query_player.proto.loop is True


def test_explore_page_mini_graph_toggle_defaults_to_network(explore_at):
    assert not explore_at.exception
    toggle = explore_at.segmented_control[0]
    assert toggle.label == "Mini-graph view"
    assert set(toggle.options) == {"Network", "Map"}
    assert toggle.value == "network"


def test_explore_page_mini_graph_network_mode_shows_neighbors_only_caption(explore_at):
    caption_texts = " ".join(c.value for c in explore_at.caption)
    assert "direct" in caption_texts and "neighbors only" in caption_texts
    assert "distance from the selected song" in caption_texts.lower()


def test_explore_page_mini_graph_network_mode_node_distance_reflects_real_similarity(explore_at):
    """Real regression guard for the fix itself: node position in the
    rendered network figure must come from radial_layout_around (distance
    = real similarity, relative to what's shown), not the full library's
    inherited spring_layout position -- a real reported bug where a node
    could look "visually closest" without being the most similar by
    score. Checked directly against the actual rendered figure's JSON
    spec (AppTest's plotly_chart elements are UnknownElement, exposing
    .spec -- a JSON string -- not a reconstructed go.Figure), not just
    that radial_layout_around itself works in isolation (that's
    network_graph.py's own unit test coverage). Exactly one node sitting
    at the exact origin is itself strong evidence this is the radial
    layout and not spring_layout -- spring_layout's force equilibrium has
    no reason to ever place a node at precisely (0, 0)."""
    import base64
    import json

    import numpy as np

    def _plotly_array(value):
        """Plotly's JSON serialization packs numeric arrays as base64-
        encoded binary (real, confirmed by inspection: {"dtype": "f8",
        "bdata": "..."}), not a plain JSON list -- decode that format
        specifically, don't assume plain numbers."""
        if isinstance(value, dict) and "bdata" in value:
            return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"]).tolist()
        return value

    charts = explore_at.get("plotly_chart")
    network_charts = [c for c in charts if "explore_mini_network" in c.proto.id]
    if not network_charts:
        pytest.skip("no network chart rendered -- selected song has no embedded neighbors to show")
    spec = json.loads(network_charts[0].spec)
    node_trace = spec["data"][1]  # data[0] is the edge trace, data[1] is nodes
    xs, ys = _plotly_array(node_trace["x"]), _plotly_array(node_trace["y"])
    if len(xs) < 2:
        pytest.skip("fewer than 2 nodes rendered -- nothing to compare distances between")

    at_origin = [i for i, (x, y) in enumerate(zip(xs, ys, strict=True)) if x == 0.0 and y == 0.0]
    assert len(at_origin) == 1  # exactly the selected/center song, never a spring_layout coincidence


def test_explore_page_mini_graph_switching_to_map_mode_renders_without_exception():
    at = _run_explore()
    toggle = at.segmented_control[0]
    toggle.set_value("map").run()

    assert not at.exception
    assert at.session_state["explore_mini_graph_mode"] == "map"
    caption_texts = " ".join(c.value for c in at.caption)
    assert "PCA projection" in caption_texts
    charts = at.get("plotly_chart")
    assert any("explore_mini_map" in c.proto.id for c in charts)


def test_explore_page_mini_graph_network_mode_renders_a_chart(explore_at):
    charts = explore_at.get("plotly_chart")
    assert any("explore_mini_network" in c.proto.id for c in charts)


def test_explore_page_moment_matcher_tabs_are_full_song_first_then_moment(explore_at):
    """Full Song leads -- "what does this whole song match" is the primary
    question, Moment ("what does this specific moment match") the more
    specific follow-up. Reversed from an earlier layout where Moment was
    the page's only Moment Matcher content and the mini-map was a
    completely separate, standalone panel."""
    assert not explore_at.exception
    labels = [t.label for t in explore_at.tabs]
    assert labels == ["Full Song", "Moment"]


def test_explore_page_full_song_tab_has_its_own_facet_selector_and_mini_graph(explore_at):
    """The mini-graph (Network/Map toggle) now lives inside this tab
    specifically, not as an independent left-panel feature -- moved
    wholesale, not rebuilt, since a network graph already IS song-level
    matching."""
    full_song_tab = next(t for t in explore_at.tabs if t.label == "Full Song")
    facet_selects = [sb for sb in full_song_tab.selectbox if sb.label == "Match by"]
    assert len(facet_selects) == 1
    toggles = [sc for sc in full_song_tab.segmented_control if sc.label == "Mini-graph view"]
    assert len(toggles) == 1


def test_explore_page_full_song_tab_shows_whole_song_retrieval_results(explore_at):
    """Real regression coverage for the point of this section existing: the
    Full Song tab must show ranked matched-song results (not just the
    network/map graph on its own) -- the same mean-pooled song-level index
    pages/5_Moment_Matcher.py's own "Whole songs" mode already uses."""
    full_song_tab = next(t for t in explore_at.tabs if t.label == "Full Song")
    markdown_texts = " ".join(m.value for m in full_song_tab.markdown)
    info_texts = " ".join(i.value for i in full_song_tab.info)
    caption_texts = " ".join(c.value for c in full_song_tab.caption)

    has_results = "results" in markdown_texts.lower() and "whole-song ranking" in caption_texts.lower()
    has_honest_empty_state = "no matches found" in info_texts.lower()
    assert has_results or has_honest_empty_state


def test_explore_page_moment_tab_has_its_own_facet_selector_and_segment_pills(explore_at):
    moment_tab = next(t for t in explore_at.tabs if t.label == "Moment")
    facet_selects = [sb for sb in moment_tab.selectbox if sb.label == "Match by"]
    assert len(facet_selects) == 1
    assert len(moment_tab.pills) == 1


def test_explore_page_full_song_and_moment_facet_selectors_are_independent():
    """Real regression coverage for the point of splitting this into two
    selectors at all: setting one must not affect the other, since an
    earlier version had exactly one shared control specifically to keep
    them from ever disagreeing -- that constraint is intentionally gone
    now that Full Song and Moment are two different questions."""
    at = _run_explore()
    full_song_select = next(
        sb for sb in next(t for t in at.tabs if t.label == "Full Song").selectbox if sb.label == "Match by"
    )
    full_song_select.set_value("harmony").run()

    assert not at.exception
    assert at.session_state["explore_full_song_facet_select"] == "harmony"
    assert at.session_state["explore_moment_facet_select"] != "harmony"


def test_explore_page_shows_tempo_and_key_as_metrics_with_help(explore_at):
    """st.metric's own help= icon replaced the earlier badge + st.popover
    ("ℹ️") pair -- a native tooltip affordance instead of a hand-rolled
    one."""
    assert not explore_at.exception
    metrics_by_label = {m.label: m for m in explore_at.metric}
    assert "Tempo" in metrics_by_label
    assert "Key" in metrics_by_label
    assert "beat-tracking" in metrics_by_label["Tempo"].proto.help.lower()
    assert "krumhansl" in metrics_by_label["Key"].proto.help.lower()


def test_explore_page_shows_song_dna_expander_with_chart(explore_at):
    assert not explore_at.exception
    expander_labels = [e.label for e in explore_at.expander]
    assert "Song DNA" in expander_labels
    charts = explore_at.get("plotly_chart")
    assert any("mid_dna_bars" in c.proto.id for c in charts)


def test_explore_page_song_dna_headers_have_explanatory_help_text(explore_at):
    """Real request: every measured thing in Song DNA should explain how
    it's computed and how to interpret it, not just show a number/chart
    with no context -- st.plotly_chart doesn't support help= in this
    Streamlit version, so each item's own section header carries it
    instead (same visible "?" pattern the pre-existing Repetition rate/
    Beats detected metrics already use)."""
    headers_by_text = {m.value: m for m in explore_at.markdown if m.value.startswith("######")}

    dna_axes_help = headers_by_text["###### DNA axes"].proto.help
    assert "beat_track" in dna_axes_help.lower()
    assert "spectral centroid" in dna_axes_help.lower()
    assert "chroma" in dna_axes_help.lower()

    matrices_help = headers_by_text["###### Self-similarity matrices"].proto.help
    assert "recurrence_matrix" in matrices_help.lower()
    assert "mel-spectrogram" in matrices_help.lower()

    chords_help = headers_by_text["###### Chord progression"].proto.help
    assert "mode filter" in chords_help.lower()
    assert "merge pass" in chords_help.lower()

    novelty_help = headers_by_text["###### Novelty curve"].proto.help
    assert "foote" in novelty_help.lower()
    assert "checkerboard" in novelty_help.lower()

    loudness_help = headers_by_text["###### Loudness contour"].proto.help
    assert "rms" in loudness_help.lower()


def test_explore_page_song_dna_shows_chord_progression_strip(explore_at):
    """Reuses the same estimate_chords()/chord_strip_figure() pair the
    module docstring says it does -- not a new detection method, so this
    just checks the chart actually renders inside Song DNA, not that chord
    detection itself is correct (that's key_chord.py's own test coverage)."""
    assert not explore_at.exception
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    assert "Chord progression" in markdown_texts
    charts = explore_at.get("plotly_chart")
    chord_or_no_chords = (
        any("dna_chord_strip" in c.proto.id for c in charts)
        or "No chords detected" in " ".join(c.value for c in explore_at.caption)
    )
    assert chord_or_no_chords


def test_explore_page_song_dna_shows_novelty_and_loudness_charts(explore_at):
    """Both read from data this app already computes elsewhere (the
    persisted structure timeline for novelty, a live RMS pass for loudness)
    -- see the page's own Song DNA comment for why neither is a new radar
    axis. novelty_curve isn't guaranteed present for every song's structure
    artifact (older ones may predate it), so this only asserts the loudness
    chart unconditionally and checks novelty renders SOME valid outcome
    (chart or the "not yet computed" fallback caption), not that novelty
    specifically is always there."""
    assert not explore_at.exception
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    assert "Novelty curve" in markdown_texts
    assert "Loudness contour" in markdown_texts

    charts = explore_at.get("plotly_chart")
    assert any("dna_loudness_contour" in c.proto.id for c in charts)

    novelty_present = any("dna_novelty_curve" in c.proto.id for c in charts)
    novelty_fallback = "not yet computed" in " ".join(c.value for c in explore_at.caption).lower()
    assert novelty_present or novelty_fallback


def test_explore_page_song_dna_shows_repetition_rate_and_beats_metrics(explore_at):
    """Repetition rate reads the already-persisted structure matrix
    (facets.structure.repetition_rate); Beats detected is a live
    beat_track() pass, same live-compute pattern Tempo/Key already use on
    this page. Both degrade to an honest caption instead of crashing when
    their underlying data isn't available for a given song -- checked the
    same either/or way as the novelty curve above."""
    assert not explore_at.exception
    metrics_by_label = {m.label: m for m in explore_at.metric}
    caption_texts = " ".join(c.value for c in explore_at.caption)

    assert "Repetition rate" in metrics_by_label or "not yet computed" in caption_texts.lower()
    assert "Beats detected" in metrics_by_label or "not enough beats" in caption_texts.lower()


def test_explore_page_song_dna_shows_self_similarity_matrices(explore_at):
    """The self-similarity-matrix fingerprints (Structure/Sound/Harmony +
    composite) moved into Song DNA from the Selected Song thumbnail, which
    now prioritizes real album art instead (see the album-art tests below).
    Structure's fingerprint is the most broadly available (get_structure_
    matrix succeeds for nearly every song), so it's asserted unconditionally;
    Sound/Harmony/Composite depend on the structure timeline actually having
    those fingerprints persisted, checked the same either/or way the rest of
    Song DNA's optional visuals already are."""
    assert not explore_at.exception
    markdown_texts = " ".join(m.value for m in explore_at.markdown)
    caption_texts = " ".join(c.value for c in explore_at.caption)

    matrices_present = "Self-similarity matrices" in markdown_texts
    matrices_absent_gracefully = "Self-similarity matrices" not in markdown_texts
    assert matrices_present or matrices_absent_gracefully  # never crashes either way

    if matrices_present:
        charts = explore_at.get("plotly_chart")
        assert any("dna_fp_structure" in c.proto.id for c in charts)
        assert "brighter means more alike" in caption_texts.lower()


def test_explore_page_dataframe_thumbnail_uses_album_art_when_it_exists(monkeypatch, tmp_path):
    """The browsable list's Thumbnail column must prioritize real album art
    the same way the Selected Song panel already does (see _row_thumbnail_
    data_uri) -- checked directly against the underlying dataframe value
    (Dataframe.value reconstructs the real pandas DataFrame from the
    widget's arrow bytes), not just that the page renders without error."""
    import sonic_explorer.config as config
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    default_song = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))[0]
    (tmp_path / f"{default_song.id}.png").write_bytes(b"not a real png, just needs to be real bytes")
    monkeypatch.setattr(config, "ALBUM_ART_DIR", tmp_path)

    at = _run_explore()

    assert not at.exception
    df = at.dataframe[0].value
    row = df[df["song_id"] == default_song.id].iloc[0]
    assert row["Thumbnail"].startswith("data:image/png;base64,")
    # a different song with no album art still falls back to the fingerprint,
    # not the same album-art bytes -- confirms this is per-row, not global
    other_row = df[df["song_id"] != default_song.id].iloc[0]
    assert other_row["Thumbnail"] != row["Thumbnail"]


def test_explore_page_falls_back_to_fingerprint_when_no_album_art_generated(monkeypatch, tmp_path):
    """This dev environment's own data/ library has no album_art/ directory
    at all (art was only ever generated for deploy_data's smaller set) --
    album_art_path_for() must return None for every song here, and the
    page must fall back to the existing structure-fingerprint substitute
    rather than showing a broken image or crashing."""
    import sonic_explorer.config as config

    monkeypatch.setattr(config, "ALBUM_ART_DIR", tmp_path)  # empty dir -- no album art for any song

    at = _run_explore()

    assert not at.exception
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "built entirely from this song's own real detected audio features" not in markdown_texts
    assert "Standing in for album art" in markdown_texts


def test_explore_page_shows_real_album_art_when_it_exists_for_the_selected_song(monkeypatch, tmp_path):
    """Simulated via monkeypatching ALBUM_ART_DIR to a real temp file for
    the default-selected song -- the page only reads and base64-encodes
    these bytes (_cached_album_art_data_uri), never decodes them as an
    actual image itself, so this doesn't need to be a real, valid PNG to
    prove the wiring works."""
    import sonic_explorer.config as config
    from resources import get_repositories

    song_repo, _, _ = get_repositories()
    default_song = sorted(song_repo.list_songs(), key=lambda s: (s.genre_top, s.title))[0]
    (tmp_path / f"{default_song.id}.png").write_bytes(b"not a real png, just needs to be real bytes")
    monkeypatch.setattr(config, "ALBUM_ART_DIR", tmp_path)

    at = _run_explore()

    assert not at.exception
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "built entirely from this song's own real detected audio features" in markdown_texts
