import chromadb
from ollama_client import chat as ollama_chat, embed as ollama_embed

from models import (
    create_response
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

from config import (
    CHAT_MODEL,
    EMBED_MODEL,
    DOCUMENT_COLLECTION,
    CHROMA_DB_DIR,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    DOCUMENT_RESULTS,
    CHAT_HISTORY_MESSAGES,
    DEBUG_RETRIEVAL,
    DEBUG_PROMPTS
)

# --------------------------------------------------
# Chroma
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR
)

collection = client.get_or_create_collection(
    DOCUMENT_COLLECTION
)

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

def get_embedding(text):

    response = ollama_embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]

# --------------------------------------------------
# History Helpers
# --------------------------------------------------

def build_search_query(
    question,
    history
):

    history = history or []

    recent = history[-CHAT_HISTORY_MESSAGES:]

    parts = []

    for item in recent:

        parts.append(
            f"{item['role']}: "
            f"{item['content']}"
        )

    parts.append(
        f"user: {question}"
    )

    return "\n".join(parts)


def format_history(
    history
):

    history = history or []

    recent = history[-CHAT_HISTORY_MESSAGES:]

    parts = []

    for item in recent:

        parts.append(
            f"{item['role']}: "
            f"{item['content']}"
        )

    return "\n".join(parts)

# --------------------------------------------------
# Local RAG
# --------------------------------------------------

def ask_local(
    question,
    history=None,
    num_results=DOCUMENT_RESULTS
):

    if collection.count() == 0:
        return create_response(
            route="LOCAL",
            answer=(
                "I don't know based on the provided documents. "
                "No documents have been indexed yet."
            )
        )

    search_query = (
        build_search_query(
            question,
            history
        )
    )

    embedding = get_embedding(
        search_query
    )

    results = collection.query(
        query_embeddings=[
            embedding
        ],
        n_results=num_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    distances = (
        results["distances"][0]
    )

    context_parts = []

    for i, (
        doc,
        meta
    ) in enumerate(
        zip(
            documents,
            metadatas
        ),
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

File:
{meta['source']}

Chunk:
{meta['chunk']}

{doc}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    if DEBUG_RETRIEVAL:
        print(f"[RETRIEVAL] returned {len(documents)} chunks")
        for source, distance in zip(metadatas, distances):
            print(
                f"[RETRIEVAL] {source['source']} "
                f"chunk={source['chunk']} distance={distance:.4f}"
            )

    prompt = f"""
You are answering questions
using retrieved documents.

Conversation:

{format_history(history)}

Retrieved Context:

{context}

Question:

{question}

Rules:

1. Use only retrieved context.
2. Cite using [SOURCE X]
3. If unknown say:

I don't know based on
the provided documents.
"""

    if DEBUG_PROMPTS:
        print(f"[LOCAL PROMPT]\n{prompt}")

    response = ollama_chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_ctx": OLLAMA_NUM_CTX,
            "temperature": OLLAMA_TEMPERATURE
        }
    )

    answer = (
        response["message"]
        ["content"]
    )

    sources = []

    for i, (
        meta,
        distance
    ) in enumerate(
        zip(
            metadatas,
            distances
        ),
        start=1
    ):

        sources.append(
            {
                "source_number": i,
                "file": meta[
                    "source"
                ],
                "chunk": meta[
                    "chunk"
                ],
                "distance": distance
            }
        )

    return create_response(
        route="LOCAL",
        answer=answer,
        sources=sources,
        context=context
    )

# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    history = []

    while True:

        question = input(
            "\nQuestion: "
        )

        if (
            question.lower()
            == "quit"
        ):
            break

        result = ask_local(
            question,
            history
        )

        print(
            "\n"
            + result["answer"]
        )

        history.append(
            {
                "role": "user",
                "content": question
            }
        )

        history.append(
            {
                "role": "assistant",
                "content":
                result["answer"]
            }
        )
