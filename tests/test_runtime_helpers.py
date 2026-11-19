from types import SimpleNamespace

import cloud
import hybrid
import ollama_client
import run_app
from models import create_response


def test_response_factory_fills_optional_fields():
    result = create_response(route="LOCAL", answer="answer")

    assert result == {
        "route": "LOCAL",
        "answer": "answer",
        "sources": [],
        "context": "",
        "memory_saved": None,
        "memory_score": None,
        "answer_score": None,
        "answer_route": None
    }


def test_hybrid_prompt_includes_question_and_document_context():
    prompt = hybrid.build_hybrid_prompt("what part?", "private excerpt")

    assert "what part?" in prompt
    assert "private excerpt" in prompt
    assert "prioritize information found" in prompt


def test_cloud_approved_call_returns_response(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="remote answer"))]
                )
            )
        )
    )
    monkeypatch.setattr(cloud, "get_client", lambda: fake_client)

    result = cloud.ask_cloud("prompt", approval_confirmed=True)

    assert result["answer"] == "remote answer"
    assert result["route"] == "CLOUD"


def test_ollama_client_retries_transient_errors(monkeypatch):
    attempts = []

    def call(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise ConnectionError("local model starting")
        return {"ok": True}

    monkeypatch.setattr(ollama_client, "OLLAMA_MAX_RETRIES", 1)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda _delay: None)

    assert ollama_client._call_with_retry(call, model="test") == {"ok": True}
    assert len(attempts) == 2


def test_ollama_client_does_not_retry_client_errors(monkeypatch):
    attempts = []

    class ClientError(Exception):
        status_code = 404

    def call(**_kwargs):
        attempts.append(True)
        raise ClientError("model not found")

    monkeypatch.setattr(ollama_client, "OLLAMA_MAX_RETRIES", 3)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda _delay: None)

    try:
        ollama_client._call_with_retry(call)
    except ClientError:
        pass
    else:
        raise AssertionError("Expected model error to propagate")

    assert len(attempts) == 1


def test_launcher_rejects_second_instance(monkeypatch, tmp_path):
    monkeypatch.setattr(run_app, "PROJECT_DIR", tmp_path)
    first_lock = run_app.acquire_instance_lock()
    try:
        assert first_lock is not None
        assert run_app.acquire_instance_lock() is None
    finally:
        first_lock.close()
