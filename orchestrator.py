from chat import ask_local
from hybrid import build_hybrid_prompt
from cloud import ask_cloud

from router import route_question

from memory import (
    build_memory_context
)

from memory_extractor import (
    auto_remember
)

from answer_evaluator import (
    evaluate_response
)
from config import (
    ENABLE_ANSWER_EVALUATION,
    ENABLE_MEMORY_EXTRACTION,
    ENABLE_MEMORY_INJECTION,
    ENABLE_HYBRID_ESCALATION,
    ENABLE_CLOUD_ESCALATION,
    DEBUG_ROUTING,
    DEBUG_MEMORY,
    DEBUG_EVALUATION
)

# --------------------------------------------------
# Memory Injection
# --------------------------------------------------

def inject_memory(question):

    if not ENABLE_MEMORY_INJECTION:
        if DEBUG_MEMORY:
            print("[MEMORY] injection disabled")
        return question

    memory_context = build_memory_context(
        question
    )

    if not memory_context:
        return question

    return f"""
{memory_context}

Current Question:

{question}
"""


def _resolve_automatic_route(route):
    """Apply the automatic escalation switches to a proposed route."""
    if route == "LOCAL":
        return "LOCAL"

    if not ENABLE_CLOUD_ESCALATION:
        return "LOCAL"

    if route == "HYBRID" and not ENABLE_HYBRID_ESCALATION:
        return "CLOUD"

    return route

# --------------------------------------------------
# Memory Metadata Helper
# --------------------------------------------------

def attach_memory_metadata(
    result,
    memory_result
):

    result["memory_saved"] = None
    result["memory_score"] = None

    if memory_result:

        result["memory_saved"] = (
            memory_result["memory"]
        )

        result["memory_score"] = (
            memory_result["score"]
        )

    return result


def _stage_openai_request(
    route,
    prompt,
    local_result=None,
    answer_score=None,
    answer_route=None
):
    sources = local_result["sources"] if local_result else []
    context = local_result["context"] if local_result else ""

    result = {
        "route": route,
        "answer": "OpenAI approval is required before this request is sent.",
        "sources": sources,
        "context": context,
        "answer_score": answer_score,
        "answer_route": answer_route,
        "confirmation_required": True,
        "privacy_items": [
            "The question and any remembered user facts included in the request."
        ],
        "pending_openai": {
            "route": route,
            "prompt": prompt,
            "sources": sources,
            "context": context,
            "answer_score": answer_score,
            "answer_route": answer_route
        }
    }

    if route == "HYBRID":
        result["privacy_items"].append(
            "Retrieved private document excerpts shown in the exact request below."
        )

    if sources:
        result["privacy_items"].append(
            "Retrieved files: " + ", ".join(
                sorted({source["file"] for source in sources})
            )
        )

    return result


def approve_openai_request(pending_openai):
    """Send a staged request only after the UI receives user approval."""
    result = ask_cloud(
        pending_openai["prompt"],
        approval_confirmed=True
    )
    result["route"] = pending_openai["route"]
    result["sources"] = pending_openai["sources"]
    result["context"] = pending_openai["context"]
    result["answer_score"] = pending_openai["answer_score"]
    result["answer_route"] = pending_openai["answer_route"]
    result["confirmation_required"] = False
    return result

# --------------------------------------------------
# Main Entry Point
# --------------------------------------------------

def ask(
    question,
    history=None,
    route_override=None
):

    if history is None:
        history = []

    # ---------------------------------------------
    # Automatic Memory Extraction
    # ---------------------------------------------

    memory_result = None
    if ENABLE_MEMORY_EXTRACTION:
        memory_result = auto_remember(question)
    elif DEBUG_MEMORY:
        print("[MEMORY] extraction disabled")

    # ---------------------------------------------
    # Inject Retrieved Memory
    # ---------------------------------------------

    enriched_question = inject_memory(
        question
    )

    # ==================================================
    # HARD ROUTE OVERRIDES
    # ==================================================

    if route_override:

        route = (
            route_override
            .strip()
            .upper()
        )

        #
        # LOCAL OVERRIDE
        #

        if route == "LOCAL":
            result = ask_local(
                question=enriched_question,
                history=history
            )

            result["answer_score"] = None
            result["answer_route"] = None

            return attach_memory_metadata(
                result,
                memory_result
            )

        #
        # HYBRID OVERRIDE
        #

        if route == "HYBRID":
            local_result = ask_local(
                question=enriched_question,
                history=history
            )
            result = _stage_openai_request(
                "HYBRID",
                build_hybrid_prompt(enriched_question, local_result["context"]),
                local_result=local_result
            )

            return attach_memory_metadata(
                result,
                memory_result
            )

        #
        # CLOUD OVERRIDE
        #

        if route == "CLOUD":

            result = _stage_openai_request(
                "CLOUD",
                enriched_question
            )

            return attach_memory_metadata(
                result,
                memory_result
            )

    # ==================================================
    # AUTOMATIC ROUTING
    # ==================================================

    proposed_route = route_question(
        enriched_question
    )
    route = _resolve_automatic_route(proposed_route)

    if DEBUG_ROUTING:
        print(f"[AUTO ROUTE] proposed={proposed_route} selected={route}")

    # ==================================================
    # LOCAL WITH EVALUATION
    # ==================================================

    if route == "LOCAL":

        local_result = ask_local(
            question=enriched_question,
            history=history
        )

        if not ENABLE_ANSWER_EVALUATION:
            local_result["answer_score"] = None
            local_result["answer_route"] = None
            return attach_memory_metadata(local_result, memory_result)

        evaluation = evaluate_response(
            question=enriched_question,
            context=local_result["context"],
            answer=local_result["answer"]
        )

        answer_score = evaluation["score"]
        proposed_recommendation = evaluation["recommended_route"]
        recommended_route = _resolve_automatic_route(
            proposed_recommendation
        )

        if DEBUG_EVALUATION:
            print(
                f"[ANSWER EVALUATION] score={answer_score} "
                f"proposed={proposed_recommendation} "
                f"selected={recommended_route}"
            )

        #
        # Keep LOCAL
        #

        if recommended_route == "LOCAL":

            result = local_result

        #
        # Escalate HYBRID
        #

        elif recommended_route == "HYBRID":

            if DEBUG_ROUTING:
                print("[ESCALATING TO HYBRID]")

            result = _stage_openai_request(
                "HYBRID",
                build_hybrid_prompt(enriched_question, local_result["context"]),
                local_result=local_result,
                answer_score=answer_score,
                answer_route=recommended_route
            )

        # Escalate to cloud after staging its exact request for approval.
        else:

            result = _stage_openai_request(
                "CLOUD",
                enriched_question,
                answer_score=answer_score,
                answer_route=recommended_route
            )

        if recommended_route == "LOCAL":
            result = local_result

        result["answer_score"] = (
            answer_score
        )

        result["answer_route"] = (
            recommended_route
        )

        return attach_memory_metadata(
            result,
            memory_result
        )

    # ==================================================
    # AUTO HYBRID
    # ==================================================

    if route == "HYBRID":

        local_result = ask_local(
            question=enriched_question,
            history=history
        )
        result = _stage_openai_request(
            "HYBRID",
            build_hybrid_prompt(enriched_question, local_result["context"]),
            local_result=local_result
        )

        return attach_memory_metadata(
            result,
            memory_result
        )

    # ==================================================
    # AUTO CLOUD
    # ==================================================

    result = _stage_openai_request(
        "CLOUD",
        enriched_question
    )

    return attach_memory_metadata(
        result,
        memory_result
    )

# --------------------------------------------------
# Standalone Testing
# --------------------------------------------------

if __name__ == "__main__":

    history = []

    while True:

        user_input = input(
            "\nQuestion: "
        ).strip()

        if user_input.lower() in [
            "quit",
            "exit"
        ]:
            break

        override = None
        question = user_input

        if user_input.startswith(
            "/local "
        ):

            override = "LOCAL"

            question = user_input[
                len("/local "):
            ].strip()

        elif user_input.startswith(
            "/hybrid "
        ):

            override = "HYBRID"

            question = user_input[
                len("/hybrid "):
            ].strip()

        elif user_input.startswith(
            "/cloud "
        ):

            override = "CLOUD"

            question = user_input[
                len("/cloud "):
            ].strip()

        result = ask(
            question=question,
            history=history,
            route_override=override
        )

        print(
            "\n=================================================="
        )

        print(
            f"FINAL ROUTE: "
            f"{result['route']}"
        )

        print(
            "==================================================\n"
        )

        print(
            result["answer"]
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
                "content": result["answer"]
            }
        )
