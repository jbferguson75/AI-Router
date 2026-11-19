def create_response(
    route,
    answer,
    sources=None,
    context="",
    memory_saved=None,
    memory_score=None,
    answer_score=None,
    answer_route=None
):
    return {
        "route": route,
        "answer": answer,
        "sources": sources or [],
        "context": context,
        "memory_saved": memory_saved,
        "memory_score": memory_score,
        "answer_score": answer_score,
        "answer_route": answer_route
    }
