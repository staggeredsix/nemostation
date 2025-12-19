from daystrom_dml.gpt_runner import _OpenAICompatibleBackend


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _build_backend():
    return _OpenAICompatibleBackend(
        base_url="https://example.invalid",
        api_key=None,
        model_name="dummy-model",
    )


def test_openai_backend_handles_null_content(monkeypatch):
    backend = _build_backend()

    def fake_post(*args, **kwargs):
        return _DummyResponse({"choices": [{"message": {"content": None}}]})

    monkeypatch.setattr("daystrom_dml.gpt_runner.requests.post", fake_post)

    text, usage = backend.generate("hello")

    assert text == ""
    assert usage is None


def test_openai_backend_handles_text_field(monkeypatch):
    backend = _build_backend()

    def fake_post(*args, **kwargs):
        return _DummyResponse({"choices": [{"text": "Hi there"}]})

    monkeypatch.setattr("daystrom_dml.gpt_runner.requests.post", fake_post)

    text, usage = backend.generate("hello")

    assert text == "Hi there"
    assert usage is None
