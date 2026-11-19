import chat
import router
from models import create_response


def test_router_uses_local_route_for_strong_match(monkeypatch):
    monkeypatch.setattr(router, "search_similarity", lambda _question: 0.2)
    assert router.route_question("question") == "LOCAL"


def test_router_keeps_relevant_corvette_rebuild_match_local(monkeypatch):
    monkeypatch.setattr(router, "search_similarity", lambda _question: 0.813)
    monkeypatch.setattr(
        router,
        "llm_route",
        lambda _question: (_ for _ in ()).throw(
            AssertionError("strong local document matches should not escalate")
        )
    )

    assert router.route_question("What rebuild kit should I get?") == "LOCAL"


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


def test_local_history_omits_ui_notices_and_stale_abstentions():
    history = [
        {"role": "user", "content": "What year is my Corvette?"},
        {"role": "assistant", "content": "OpenAI approval is required."},
        {
            "role": "assistant",
            "content": "I don't know based on the provided documents."
        },
        {"role": "assistant", "content": "Your car has a carburetor."},
    ]

    formatted = chat.format_history(history)
    search_query = chat.build_search_query("What year is my Corvette?", history)

    assert "What year is my Corvette?" in formatted
    assert "Your car has a carburetor." in formatted
    assert "OpenAI approval is required" not in formatted
    assert "I don't know based on the provided documents" not in formatted
    assert "OpenAI approval is required" not in search_query
    assert "I don't know based on the provided documents" not in search_query


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
    assert "both Retrieved Context and Known User Facts as evidence" in prompts[0]


def test_local_prompt_treats_titles_and_memories_as_evidence(
    monkeypatch, document_collection
):
    document_collection.add(
        ids=["corvette"],
        documents=["1980 Corvette Maintenance Notes"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"source": "/docs/corvette_notes.txt", "chunk": 0}]
    )
    prompts = []
    monkeypatch.setattr(chat, "get_embedding", lambda _text: [1.0, 0.0])
    monkeypatch.setattr(
        chat,
        "ollama_chat",
        lambda **kwargs: prompts.append(kwargs["messages"][0]["content"])
        or {"message": {"content": "Your Corvette is a 1980 [SOURCE 1]."}}
    )

    result = chat.ask_local(
        "What year is my Corvette?",
        memory_context="Known User Facts:\n- User owns a 1980 Corvette."
    )

    assert result["answer"] == "Your Corvette is a 1980 [SOURCE 1]."
    assert "1980 Corvette Maintenance Notes" in prompts[0]
    assert "User owns a 1980 Corvette." in prompts[0]
    assert "Treat titles and headings in retrieved documents as factual evidence" in prompts[0]


def test_local_retrieval_deduplicates_identical_document_chunks(
    monkeypatch, document_collection
):
    duplicate = "Project Phoenix Target: Reduce processing time by 25%."
    document_collection.add(
        ids=["phoenix-1", "phoenix-2", "phoenix-3"],
        documents=[duplicate, duplicate, duplicate],
        embeddings=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        metadatas=[
            {"source": "/docs/Project Phoenix.docx", "chunk": 0},
            {"source": "/docs/Project Phoenix 2.docx", "chunk": 0},
            {"source": "/docs/Project Phoenix 3.docx", "chunk": 0},
        ]
    )
    monkeypatch.setattr(chat, "get_embedding", lambda _text: [1.0, 0.0])
    monkeypatch.setattr(
        chat,
        "ollama_chat",
        lambda **_kwargs: {"message": {"content": "Reduce processing time by 25%. [SOURCE 1]"}}
    )

    result = chat.ask_local("What target does Project Phoenix have?")

    assert result["context"].count(duplicate) == 1
    assert len(result["sources"]) == 1


def test_local_answer_explains_empty_index(document_collection):
    result = chat.ask_local("Question with no indexed docs")

    assert "No documents have been indexed" in result["answer"]
    assert result["sources"] == []
