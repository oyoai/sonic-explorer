"""Unit tests for the adapter mechanism only -- NOT a test that loads the
real Qwen2.5-0.5B-Instruct model (a real ~1GB download + real CPU
inference, inappropriate for a fast, network-independent test suite this
project keeps to a few minutes). _generate is monkeypatched the same way
pipeline/sound_tagging.py's own tests inject a fake tagger_fn instead of
loading real AST weights. The actual model's real behavior -- including
the real, measured prompt-injection finding -- is documented directly in
scripts/red_team_local_description_findings.md, not asserted here as a
pass/fail unit test (a "the model outputs PWNED" test would read as
backwards -- that's a documented finding, not a behavior to pin)."""

from sonic_explorer.llm.explain import ExplanationClient
from sonic_explorer.llm.local_client import LocalTransformersClient


def test_messages_create_wraps_generate_output_in_content_text_shape():
    """The exact duck-typed shape ExplanationClient's own docstring
    requires: `.messages.create(...)` -> object with `.content[0].text`."""
    client = LocalTransformersClient()
    client._generate = lambda system, user_text, max_tokens: "a generated phrase"

    response = client.messages.create(model="unused", max_tokens=30, system="sys", messages=[{"content": "usr"}])

    assert response.content[0].text == "a generated phrase"


def test_generate_is_called_with_system_and_user_text_and_max_tokens():
    client = LocalTransformersClient()
    captured = {}

    def fake_generate(system, user_text, max_tokens):
        captured["system"] = system
        captured["user_text"] = user_text
        captured["max_tokens"] = max_tokens
        return "reply"

    client._generate = fake_generate
    client.messages.create(model="unused", max_tokens=42, system="the system prompt", messages=[{"content": "the user text"}])

    assert captured == {"system": "the system prompt", "user_text": "the user text", "max_tokens": 42}


def test_model_is_not_loaded_until_generate_is_actually_called():
    """Lazy-load discipline, same reasoning as pipeline/sound_tagging.py's
    _ensure_tagger_loaded -- constructing a client (or importing this
    module) must stay cheap even for a caller that never actually
    generates anything."""
    client = LocalTransformersClient()
    assert client._pipe is None


def test_explanation_client_works_end_to_end_against_a_fake_local_backend():
    """Confirms explain.py needed ZERO changes to support this backend --
    ExplanationClient(LocalTransformersClient()) round-trips through
    generate_description exactly like it does with a fake Anthropic-shaped
    client in test_explain.py."""
    client = LocalTransformersClient()
    client._generate = lambda system, user_text, max_tokens: "  calm ambient drift  "
    explanation_client = ExplanationClient(client)

    result = explanation_client.generate_description(
        title="Song", artist="Artist", genre="Ambient", tags=[],
        tempo_bpm=0.3, energy=0.2, brightness=0.4, harmonic_complexity=0.3, rhythmic_density=0.2,
    )

    assert result == "calm ambient drift"
