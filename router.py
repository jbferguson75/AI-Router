import chromadb
from ollama_client import chat as ollama_chat, embed as ollama_embed
CHROMA_DB_DIR = "./chroma_db"

DOCUMENT_COLLECTION = "documents"

EMBED_MODEL = "nomic-embed-text"

from config import (
    ROUTER_MODEL,
    LOCAL_THRESHOLD,
    HYBRID_THRESHOLD,
    EMBED_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    DEBUG_ROUTING,
    DEBUG_PROMPTS
)

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR
)

collection = client.get_or_create_collection(
    DOCUMENT_COLLECTION
)


def get_embedding(text):

    response = ollama_embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]


def search_similarity(question):

    if collection.count() == 0:
        return 999.0

    embedding = get_embedding(
        question
    )

    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["distances"]
    )

    try:

        return results[
            "distances"
        ][0][0]

    except Exception:

        return 999.0


def llm_route(question):

    prompt = f"""
Decide whether this question should be
answered using private documents or
a cloud model.

Respond with only:

LOCAL

or

CLOUD

Question:

{question}
"""

    if DEBUG_PROMPTS:
        print(f"[ROUTER PROMPT]\n{prompt}")

    response = ollama_chat(
        model=ROUTER_MODEL,
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
        .strip()
        .upper()
    )

    if "CLOUD" in answer:
        return "CLOUD"

    return "LOCAL"


def route_question(question):

    distance = search_similarity(
        question
    )

    if DEBUG_ROUTING:
        print(f"[DISTANCE] {distance}")

    if distance < LOCAL_THRESHOLD:

        return "LOCAL"

    if distance < HYBRID_THRESHOLD:

        return "HYBRID"

    return llm_route(
        question
    )
