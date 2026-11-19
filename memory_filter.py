from ollama_client import chat as ollama_chat

from config import (
    MEMORY_SCORER_MODEL,
    MEMORY_SAVE_THRESHOLD,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    DEBUG_MEMORY,
    DEBUG_PROMPTS
)


def score_memory(
    memory_text
):

    prompt = f"""
You are evaluating a memory.

Return ONLY a single integer
between 0 and 100.

100 = definitely save forever

0 = definitely do not save

Store:

- personal preferences
- long-term goals
- relationships
- ownership
- family information
- career information
- skills
- ongoing projects

Do NOT store:

- weather
- greetings
- temporary plans
- one-time tasks
- random observations
- simple questions

Memory:

{memory_text}
"""

    if DEBUG_PROMPTS:
        print(f"[MEMORY SCORER PROMPT]\n{prompt}")

    response = ollama_chat(
        model=MEMORY_SCORER_MODEL,
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

        return score

    except Exception:

        return 0


def should_save_memory(
    memory_text
):

    score = score_memory(
        memory_text
    )

    save = (
        score >=
        MEMORY_SAVE_THRESHOLD
    )

    if DEBUG_MEMORY:
        print(f"[MEMORY] score={score} accepted={save}")

    return save, score
