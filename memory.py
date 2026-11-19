import uuid

import chromadb
from ollama_client import embed as ollama_embed
from config import MEMORY_DISTANCE_THRESHOLD

CHROMA_DB_DIR = "./chroma_db"
MEMORY_COLLECTION_NAME = "memories"

EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR
)

memory_collection = (
    client.get_or_create_collection(
        MEMORY_COLLECTION_NAME
    )
)


def get_embedding(text):

    response = ollama_embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]


def _is_reusable_memory(memory_text):
    """Reject extraction explanations and query summaries as user facts."""
    normalized = memory_text.strip().lower()
    return not (
        normalized.startswith("output:")
        or normalized.startswith("user is inquiring")
        or normalized.startswith("user is asking")
    )


def save_memory(
    memory_text,
    prevent_duplicates=True
):

    memory_text = memory_text.strip()

    if not memory_text or not _is_reusable_memory(memory_text):
        return False

    if prevent_duplicates:

        matches = search_memory(
            memory_text,
            n_results=1
        )

        if matches:

            if (
                matches[0]
                .lower()
                .strip()
                ==
                memory_text
                .lower()
                .strip()
            ):
                return False

    embedding = get_embedding(
        memory_text
    )

    memory_collection.add(
        ids=[str(uuid.uuid4())],
        documents=[memory_text],
        embeddings=[embedding]
    )

    return True


def search_memory(
    query,
    n_results=5
):

    embedding = get_embedding(
        query
    )

    results = memory_collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "distances"]
    )

    docs = results.get(
        "documents",
        []
    )

    distances = results.get("distances", [])
    if not docs or not distances:
        return []

    memories = []
    for memory_text, distance in zip(docs[0], distances[0]):
        if (
            distance <= MEMORY_DISTANCE_THRESHOLD
            and _is_reusable_memory(memory_text)
        ):
            memories.append(memory_text)

    return memories


def build_memory_context(
    query,
    n_results=5
):

    memories = search_memory(
        query,
        n_results=n_results
    )

    if not memories:
        return ""

    lines = [
        "Known User Facts:",
        ""
    ]

    for memory in memories:

        lines.append(
            f"- {memory}"
        )

    return "\n".join(lines)


def get_all_memories():

    results = memory_collection.get()

    return results.get(
        "documents",
        []
    )


def memory_count():

    return len(
        get_all_memories()
    )


def delete_memory(
    memory_text
):

    results = memory_collection.get()

    ids = results.get(
        "ids",
        []
    )

    docs = results.get(
        "documents",
        []
    )

    delete_ids = []

    for memory_id, doc in zip(
        ids,
        docs
    ):

        if (
            doc.lower().strip()
            ==
            memory_text.lower().strip()
        ):

            delete_ids.append(
                memory_id
            )

    if delete_ids:

        memory_collection.delete(
            ids=delete_ids
        )

        return True

    return False
