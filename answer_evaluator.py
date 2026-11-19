from ollama_client import chat as ollama_chat

# --------------------------------------------------
# Configuration
# --------------------------------------------------

from config import (
    EVALUATOR_MODEL,
    LOCAL_ACCEPT_THRESHOLD,
    HYBRID_ACCEPT_THRESHOLD,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    DEBUG_EVALUATION,
    DEBUG_PROMPTS
)
# --------------------------------------------------
# Score Answer
# --------------------------------------------------

def score_answer(
    question,
    context,
    answer
):
    """
    Returns 0-100 confidence score.

    Higher means the answer is strongly
    supported by the retrieved context.
    """

    normalized_answer = " ".join(answer.lower().split())
    abstention_phrases = (
        "i don't know",
        "i do not know",
        "cannot determine",
        "can't determine",
        "not enough information",
        "insufficient information",
    )
    if context.strip() and any(
        phrase in normalized_answer for phrase in abstention_phrases
    ):
        # An abstention may be unhelpful when evidence exists, but it does not
        # contradict that evidence. Keep it below local acceptance while
        # avoiding the evaluator's contradiction score and cloud escalation.
        score = HYBRID_ACCEPT_THRESHOLD
        if DEBUG_EVALUATION:
            print(f"[ANSWER SCORE] {score} (unsupported abstention)")
        return score

    prompt = f"""
You are evaluating answer quality.

Question:

{question}

Retrieved Context:

{context}

Answer:

{answer}

Evaluate how well the answer is supported by all supplied evidence. Titles,
headings, and Known User Facts count as evidence. An answer that says "I don't
know" is an abstention, not a contradiction; score it as weakly helpful rather
than zero. Reserve 0 for an answer that directly contradicts the evidence.
Check names, numbers, units, and technical terms exactly. A changed technical
term or number is unsupported even when it sounds similar to the source.

Scoring:

100
Answer is completely supported.

90
Answer is strongly supported.

70
Answer is partially supported.

50
Answer is an unsupported abstention or contains a mix of supported and
unsupported claims.

30
Answer is mostly unsupported.

0
Answer contradicts the context.

Return ONLY a number.
"""

    if DEBUG_PROMPTS:
        print(f"[EVALUATOR PROMPT]\n{prompt}")

    response = ollama_chat(
        model=EVALUATOR_MODEL,
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

    content = (
        response["message"]["content"]
        .strip()
    )

    try:

        score = int(content)

        score = max(
            0,
            min(score, 100)
        )

        if DEBUG_EVALUATION:
            print(f"[ANSWER SCORE] {score}")

        return score

    except Exception:

        return 0

# --------------------------------------------------
# Route Recommendation
# --------------------------------------------------

def recommend_route(score):

    if score >= LOCAL_ACCEPT_THRESHOLD:
        return "LOCAL"

    if score >= HYBRID_ACCEPT_THRESHOLD:
        return "HYBRID"

    return "CLOUD"

# --------------------------------------------------
# Evaluate Full Response
# --------------------------------------------------

def evaluate_response(
    question,
    context,
    answer
):

    score = score_answer(
        question,
        context,
        answer
    )

    recommended_route = (
        recommend_route(score)
    )

    if DEBUG_EVALUATION:
        print(f"[RECOMMENDED ROUTE] {recommended_route}")

    return {
        "score": score,
        "recommended_route": recommended_route
    }

# --------------------------------------------------
# Test Mode
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What rebuild kit is recommended?"
    )

    context = """
Recommended Parts:
- Edelbrock Rebuild Kit 1477
"""

    answer = (
        "The recommended rebuild kit "
        "is the Edelbrock 1477 rebuild kit."
    )

    result = evaluate_response(
        question,
        context,
        answer
    )

    print(result)
