import pytest

import cloud
import orchestrator
from models import create_response


def local_result(answer="local", context="", sources=None):
    return create_response(
        route="LOCAL",
        answer=answer,
        context=context,
        sources=sources or []
    )


def prepare_orchestrator(monkeypatch):
    monkeypatch.setattr(orchestrator, "auto_remember", lambda _question: None)
    monkeypatch.setattr(orchestrator, "build_memory_context", lambda _question: "")
    monkeypatch.setattr(
        orchestrator,
        "inject_memory",
        lambda question, **_kwargs: question
    )


@pytest.mark.parametrize(
    "proposed,cloud_enabled,hybrid_enabled,expected",
    [
        ("LOCAL", True, True, "LOCAL"),
        ("CLOUD", True, True, "CLOUD"),
        ("HYBRID", True, True, "HYBRID"),
        ("HYBRID", True, False, "CLOUD"),
        ("CLOUD", False, True, "LOCAL"),
        ("HYBRID", False, True, "LOCAL"),
    ]
)
def test_automatic_route_switches(
    monkeypatch, proposed, cloud_enabled, hybrid_enabled, expected
):
    monkeypatch.setattr(orchestrator, "ENABLE_CLOUD_ESCALATION", cloud_enabled)
    monkeypatch.setattr(orchestrator, "ENABLE_HYBRID_ESCALATION", hybrid_enabled)

    assert orchestrator._resolve_automatic_route(proposed) == expected


def test_disabled_memory_switches_skip_extraction_and_injection(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_MEMORY_EXTRACTION", False)
    monkeypatch.setattr(orchestrator, "ENABLE_MEMORY_INJECTION", False)
    monkeypatch.setattr(
        orchestrator,
        "auto_remember",
        lambda _question: (_ for _ in ()).throw(AssertionError("extraction ran"))
    )
    monkeypatch.setattr(
        orchestrator,
        "build_memory_context",
        lambda _question: (_ for _ in ()).throw(AssertionError("injection ran"))
    )
    received = {}

    def fake_local(question, history, memory_context=""):
        received["question"] = question
        received["memory_context"] = memory_context
        return local_result()

    monkeypatch.setattr(orchestrator, "ask_local", fake_local)

    result = orchestrator.ask("a question", route_override="LOCAL")

    assert received["question"] == "a question"
    assert result["memory_saved"] is None


def test_memory_injection_is_used_when_enabled(monkeypatch):
    monkeypatch.setattr(orchestrator, "ENABLE_MEMORY_EXTRACTION", False)
    monkeypatch.setattr(orchestrator, "ENABLE_MEMORY_INJECTION", True)
    monkeypatch.setattr(
        orchestrator,
        "build_memory_context",
        lambda _question: "Known User Facts:\n- owns a Corvette"
    )
    received = {}

    def fake_local(question, history, memory_context=""):
        received["question"] = question
        received["memory_context"] = memory_context
        return local_result()

    monkeypatch.setattr(orchestrator, "ask_local", fake_local)
    orchestrator.ask("question", route_override="LOCAL")

    assert received["question"] == "question"
    assert "owns a Corvette" in received["memory_context"]


def test_disabling_answer_evaluation_keeps_local_answer(monkeypatch):
    prepare_orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator, "route_question", lambda _question: "LOCAL")
    monkeypatch.setattr(orchestrator, "ENABLE_ANSWER_EVALUATION", False)
    monkeypatch.setattr(orchestrator, "ask_local", lambda **_kwargs: local_result())
    monkeypatch.setattr(
        orchestrator,
        "evaluate_response",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("evaluation ran"))
    )

    result = orchestrator.ask("question")

    assert result["route"] == "LOCAL"
    assert result["answer_score"] is None


def test_cloud_disabled_uses_local_for_automatic_cloud_route(monkeypatch):
    prepare_orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator, "route_question", lambda _question: "CLOUD")
    monkeypatch.setattr(orchestrator, "ENABLE_CLOUD_ESCALATION", False)
    monkeypatch.setattr(orchestrator, "ENABLE_ANSWER_EVALUATION", False)
    monkeypatch.setattr(orchestrator, "ask_local", lambda **_kwargs: local_result())

    result = orchestrator.ask("question")

    assert result["route"] == "LOCAL"
    assert result.get("confirmation_required") is None


def test_hybrid_override_stages_exact_private_context_for_review(monkeypatch):
    prepare_orchestrator(monkeypatch)
    sources = [{"source_number": 1, "file": "/private/notes.txt", "chunk": 0, "distance": 0.1}]
    monkeypatch.setattr(
        orchestrator,
        "ask_local",
        lambda **_kwargs: local_result(context="private excerpt", sources=sources)
    )

    result = orchestrator.ask("question", route_override="HYBRID")

    assert result["confirmation_required"] is True
    assert result["pending_openai"]["route"] == "HYBRID"
    assert "private excerpt" in result["pending_openai"]["prompt"]
    assert "/private/notes.txt" in result["privacy_items"][-1]


def test_hybrid_escalation_disabled_falls_back_to_cloud_request(monkeypatch):
    prepare_orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator, "route_question", lambda _question: "HYBRID")
    monkeypatch.setattr(orchestrator, "ENABLE_HYBRID_ESCALATION", False)
    monkeypatch.setattr(orchestrator, "ENABLE_CLOUD_ESCALATION", True)

    result = orchestrator.ask("question")

    assert result["route"] == "CLOUD"
    assert result["confirmation_required"] is True
    assert "Retrieved private document excerpts" not in result["privacy_items"]


def test_answer_evaluator_escalation_obeys_switches(monkeypatch):
    prepare_orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator, "route_question", lambda _question: "LOCAL")
    monkeypatch.setattr(orchestrator, "ask_local", lambda **_kwargs: local_result(context="private"))
    monkeypatch.setattr(
        orchestrator,
        "evaluate_response",
        lambda **_kwargs: {"score": 50, "recommended_route": "HYBRID"}
    )
    monkeypatch.setattr(orchestrator, "ENABLE_HYBRID_ESCALATION", False)

    result = orchestrator.ask("question")

    assert result["route"] == "CLOUD"
    assert result["answer_route"] == "CLOUD"
    assert result["confirmation_required"] is True


def test_approved_openai_request_is_sent_with_explicit_approval(monkeypatch):
    calls = []

    def fake_cloud(prompt, *, approval_confirmed):
        calls.append((prompt, approval_confirmed))
        return create_response(route="CLOUD", answer="answer")

    monkeypatch.setattr(orchestrator, "ask_cloud", fake_cloud)
    pending = {
        "route": "HYBRID",
        "prompt": "exact approved prompt",
        "sources": [],
        "context": "context",
        "answer_score": 55,
        "answer_route": "HYBRID"
    }

    result = orchestrator.approve_openai_request(pending)

    assert calls == [("exact approved prompt", True)]
    assert result["route"] == "HYBRID"
    assert result["answer"] == "answer"
    assert result["answer_score"] == 55


def test_cancelled_hybrid_request_returns_staged_local_answer(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "ask_cloud",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled requests must not call OpenAI")
        )
    )
    pending = {
        "answer_score": 58,
        "pending_openai": {
            "local_fallback": local_result(
                answer="Answer already prepared locally",
                sources=[{"source_number": 1, "file": "notes.txt"}]
            )
        }
    }

    result = orchestrator.cancel_openai_request(pending)

    assert result["route"] == "LOCAL"
    assert result["answer_route"] == "LOCAL"
    assert result["answer"] == "Answer already prepared locally"
    assert result["sources"] == [{"source_number": 1, "file": "notes.txt"}]
    assert result["answer_score"] == 58
    assert result["confirmation_required"] is False


def test_cancelled_cloud_request_asks_local_documents(monkeypatch):
    prepare_orchestrator(monkeypatch)
    received = {}

    def fake_local(question, history, memory_context=""):
        received["question"] = question
        received["history"] = history
        received["memory_context"] = memory_context
        return local_result(answer="Local documents answer")

    monkeypatch.setattr(orchestrator, "ask_local", fake_local)
    monkeypatch.setattr(
        orchestrator,
        "ask_cloud",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled requests must not call OpenAI")
        )
    )
    history = [{"role": "user", "content": "Earlier question"}]

    result = orchestrator.cancel_openai_request({
        "memory_saved": None,
        "memory_score": None,
        "pending_openai": {
            "fallback_question": "Current question",
            "fallback_history": history
        }
    })

    assert received == {
        "question": "Current question",
        "history": history,
        "memory_context": ""
    }
    assert result["route"] == "LOCAL"
    assert result["answer"] == "Local documents answer"


def test_cloud_client_refuses_unapproved_requests(monkeypatch):
    monkeypatch.setattr(
        cloud,
        "get_client",
        lambda: (_ for _ in ()).throw(AssertionError("client should not be created"))
    )

    with pytest.raises(PermissionError):
        cloud.ask_cloud("sensitive prompt")
