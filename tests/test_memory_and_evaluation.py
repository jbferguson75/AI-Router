import answer_evaluator
import memory
import memory_extractor
import memory_filter


def ollama_text(content):
    return {"message": {"content": content}}


def test_memory_save_search_context_and_delete(monkeypatch, memory_collection):
    monkeypatch.setattr(memory, "get_embedding", lambda _text: [1.0, 0.0])

    assert memory.save_memory("User owns a Corvette.") is True
    assert memory.search_memory("Corvette") == ["User owns a Corvette."]
    assert "User owns a Corvette." in memory.build_memory_context("Corvette")
    assert memory.save_memory("User owns a Corvette.") is False
    assert memory.delete_memory("User owns a Corvette.") is True
    assert memory.memory_count() == 0


def test_memory_rejects_blank_text():
    assert memory.save_memory("  ") is False


def test_memory_filter_accepts_score_at_threshold(monkeypatch):
    monkeypatch.setattr(memory_filter, "ollama_chat", lambda **_kwargs: ollama_text("60"))

    assert memory_filter.should_save_memory("long term preference") == (True, 60)


def test_memory_filter_rejects_invalid_score(monkeypatch):
    monkeypatch.setattr(memory_filter, "ollama_chat", lambda **_kwargs: ollama_text("unknown"))

    assert memory_filter.score_memory("some fact") == 0


def test_memory_extraction_recognizes_none_and_candidate(monkeypatch):
    responses = iter([ollama_text("NONE"), ollama_text("User manages ten engineers.")])
    monkeypatch.setattr(memory_extractor, "ollama_chat", lambda **_kwargs: next(responses))

    assert memory_extractor.extract_memory("What is the weather?") is None
    assert memory_extractor.extract_memory("I manage ten engineers") == (
        "User manages ten engineers."
    )


def test_auto_remember_scores_and_saves_candidate(monkeypatch):
    monkeypatch.setattr(
        memory_extractor,
        "extract_memory",
        lambda _text: "User manages ten engineers."
    )
    monkeypatch.setattr(
        memory_extractor,
        "should_save_memory",
        lambda _text: (True, 88)
    )
    monkeypatch.setattr(memory_extractor, "save_memory", lambda _text: True)

    assert memory_extractor.auto_remember("I manage ten engineers") == {
        "memory": "User manages ten engineers.",
        "score": 88
    }


def test_answer_evaluator_parses_score_and_recommends_route(monkeypatch):
    monkeypatch.setattr(answer_evaluator, "ollama_chat", lambda **_kwargs: ollama_text("75"))

    assert answer_evaluator.evaluate_response("q", "context", "answer") == {
        "score": 75,
        "recommended_route": "LOCAL"
    }
    assert answer_evaluator.recommend_route(50) == "HYBRID"
    assert answer_evaluator.recommend_route(39) == "CLOUD"


def test_answer_evaluator_fails_closed_on_invalid_output(monkeypatch):
    monkeypatch.setattr(answer_evaluator, "ollama_chat", lambda **_kwargs: ollama_text("75/100"))

    assert answer_evaluator.score_answer("q", "context", "answer") == 0
