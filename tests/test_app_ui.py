from pathlib import Path

from streamlit.testing.v1 import AppTest

import orchestrator


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


def pending_result():
    return {
        "route": "CLOUD",
        "answer": "Approval required",
        "sources": [],
        "context": "",
        "answer_score": None,
        "answer_route": None,
        "memory_saved": None,
        "memory_score": None,
        "confirmation_required": True,
        "privacy_items": ["Question and remembered facts"],
        "pending_openai": {
            "route": "CLOUD",
            "prompt": "Exact prompt to review",
            "sources": [],
            "context": "",
            "answer_score": None,
            "answer_route": None
        }
    }


def test_app_renders_and_handles_local_errors_without_crashing(monkeypatch):
    monkeypatch.setattr(orchestrator, "ask", lambda **_kwargs: {
        "route": "ERROR",
        "answer": "I couldn't complete that request.",
        "sources": [],
        "answer_score": None,
        "memory_saved": None,
        "memory_score": None
    })

    at = AppTest.from_file(str(APP_FILE)).run()
    assert not at.exception
    assert at.title[0].value == "🤖 AI Router"

    at.chat_input[0].set_value("question").run()

    assert not at.exception
    assert any("I couldn't complete that request." in item.value for item in at.markdown)


def test_app_shows_request_and_sends_only_after_approval(monkeypatch):
    monkeypatch.setattr(orchestrator, "ask", lambda **_kwargs: pending_result())
    sent = []

    def approve(pending):
        sent.append(pending["prompt"])
        return {
            "route": "CLOUD",
            "answer": "Approved answer",
            "sources": []
        }

    monkeypatch.setattr(orchestrator, "approve_openai_request", approve)
    at = AppTest.from_file(str(APP_FILE)).run()
    at.chat_input[0].set_value("/cloud question").run()

    assert not at.exception
    assert at.session_state["pending_openai"]["pending_openai"]["prompt"] == (
        "Exact prompt to review"
    )
    assert any(item.value == "Exact prompt to review" for item in at.code)
    assert sent == []

    at.button(key="approve_openai_request").click().run()

    assert not at.exception
    assert sent == ["Exact prompt to review"]
    assert at.session_state["pending_openai"] is None
    assert any("Approved answer" in item.value for item in at.markdown)


def test_app_cancel_does_not_send_pending_request(monkeypatch):
    monkeypatch.setattr(orchestrator, "ask", lambda **_kwargs: pending_result())
    sent = []
    monkeypatch.setattr(
        orchestrator,
        "approve_openai_request",
        lambda _pending: sent.append("sent")
    )
    at = AppTest.from_file(str(APP_FILE)).run()
    at.chat_input[0].set_value("/cloud question").run()

    at.button(key="cancel_openai_request").click().run()

    assert not at.exception
    assert sent == []
    assert at.session_state["pending_openai"] is None
    assert any("OpenAI request cancelled" in item.value for item in at.markdown)
