"""Local, free, CPU-only alternative to a real Anthropic-backed client --
loads a small instruction-tuned model directly via `transformers` (no
separate server process the way Ollama would need, and no new external
dependency beyond what pipeline/sound_tagging.py already requires for AST
tagging). Implements the exact duck-typed interface llm/explain.py's
ExplanationClient docstring already specifies (anything with
`.messages.create(model=, max_tokens=, system=, messages=)` ->
object with `.content[0].text`) -- explain.py itself needed ZERO changes
to support this, which is the real confirmation that interface really was
built swappable, not just described that way.

Ollama itself (a separate local server + its own model-pull step) isn't
installed in the environment this was built in (checked directly: neither
`ollama` nor `winget` is on PATH here), so this reuses infrastructure
already proven working in THIS environment -- HF Hub downloads, a
CPU-only torch install -- rather than adding a new kind of dependency
(an external service) on top of it. An Ollama-backed adapter would look
almost identical (swap _generate()'s pipeline call for an HTTP request to
localhost:11434/api/chat) if Ollama becomes available later -- the
duck-typed interface genuinely doesn't care which backend implements it.

Deliberately small (Qwen2.5-0.5B-Instruct, ~500M params) and CPU-only --
no GPU available in this environment (torch.cuda.is_available() confirmed
False directly), and every real caller of this (song-description
synthesis) is a short, 2-5-word generation per call, not a task that
needs a large model's reasoning depth."""


class _TextBlock:
    def __init__(self, text: str):
        self.text = text


class _Message:
    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


class _Messages:
    def __init__(self, client: "LocalTransformersClient"):
        self._client = client

    def create(self, model: str, max_tokens: int, system: str, messages: list[dict]) -> _Message:
        user_text = messages[0]["content"]
        return _Message(self._client._generate(system, user_text, max_tokens))


class LocalTransformersClient:
    """Drop-in for anthropic.Anthropic(...) -- pass an instance of this to
    ExplanationClient(client=...) instead. Lazy-loads the model on first
    real use (same discipline as pipeline/sound_tagging.py's
    _ensure_tagger_loaded), not at construction or import time -- creating
    a client (or importing this module) must stay cheap even in a caller
    that never actually generates anything with it."""

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._model_name = model_name
        self._pipe = None
        self.messages = _Messages(self)

    def _ensure_loaded(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from transformers import pipeline

        self._pipe = pipeline(
            "text-generation", model=self._model_name, dtype=torch.float32, device=-1,
        )
        return self._pipe

    def _generate(self, system: str, user_text: str, max_tokens: int) -> str:
        pipe = self._ensure_loaded()
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        # do_sample=False (greedy decoding): deterministic output for a
        # short descriptive task, same reasoning generate_description's
        # real callers already want reproducibility for (a re-run against
        # the same song shouldn't produce a different phrase at random).
        output = pipe(chat, max_new_tokens=max_tokens, do_sample=False)
        reply_message = output[0]["generated_text"][-1]
        return reply_message["content"].strip()
