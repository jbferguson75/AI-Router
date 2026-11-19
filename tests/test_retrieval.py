import chat
import router
from models import create_response


def test_router_uses_local_route_for_strong_match(monkeypatch):
    monkeypatch.setattr(router, "search_similarity", lambda _question: 0.2)
    assert router.route_question("question") == "LOCAL"


def test_router_uses_hybrid_route_for_moderate_match(monkeypatch):
    monkeypatch.setattr(router, "search_similarity", lambda _question: 1.0)
    monkeypatch.setattr(
        router,
        "llm_route",
        lambda _question: (_ for _ in ()).throw(AssertionError("unexpected LLM route"))
    )
    assert router.route_question("question") == "HYBRID"


def test_router_uses_llm_route_for_weak_match(monkeypatch):
    monkeypatch.setattr(router, "search_similarity", lambda _question: 1.5)
    monkeypatch.setattr(router, "llm_route", lambda _question: "CLOUD")
    assert router.route_question("question") == "CLOUD"


def test_empty_collection_does_not_request_embedding(monkeypatch, document_collection):
    monkeypatch.setattr(
        router,
        "get_embedding",
        lambda _text: (_ for _ in ()).throw(AssertionError("should not embed"))
    )
    assert router.search_similarity("question") == 999.0


def test_local_answer_uses_retrieved_context_and_returns_sources(
    monkeypatch, document_collection
):
    document_collection.add(
        ids=["chunk-1"],
        documents=["The recommended kit is 1477."],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"source": "/docs/car.txt", "chunk": 2}]
    )
    prompts = []
    monkeypatch.setattr(chat, "get_embedding", lambda _text: [1.0, 0.0])
    monkeypatch.setattr(
        chat,
        "ollama_chat",
        lambda **kwargs: prompts.append(kwargs["messages"][0]["content"])
        or {"message": {"content": "Use kit 1477 [SOURCE 1]."}}
    )

    result = chat.ask_local("Which kit?")

    assert result["answer"] == "Use kit 1477 [SOURCE 1]."
    assert "The recommended kit is 1477." in result["context"]
    assert result["sources"][0]["file"] == "/docs/car.txt"
    assert result["sources"][0]["chunk"] == 2
    assert "Use only retrieved context" in prompts[0]


def test_local_answer_explains_empty_index(document_collection):
    result = chat.ask_local("Question with no indexed docs")

    assert "No documents have been indexed" in result["answer"]
    assert result["sources"] == []
