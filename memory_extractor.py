from ollama_client import chat as ollama_chat

from memory import save_memory
from memory_filter import (
    should_save_memory
)

from config import (
    MEMORY_EXTRACTOR_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    DEBUG_MEMORY,
    DEBUG_PROMPTS
)


def extract_memory(
    user_text
):

    prompt = f"""
You are a memory extraction system.

Determine whether the user message
contains information useful for future
conversations.

If yes:

Return ONE memory.

Examples:

Input:
I own a 1980 Corvette.

Output:
User owns a 1980 Corvette.

Input:
My daughter owns a barber shop.

Output:
User's daughter owns a barber shop.

Input:
I manage ten engineers.

Output:
User manages ten engineers.

Input:
What's the weather?

Output:
NONE

User message:

{user_text}
"""

    if DEBUG_PROMPTS:
        print(f"[MEMORY EXTRACTION PROMPT]\n{prompt}")

    response = ollama_chat(
        model=MEMORY_EXTRACTOR_MODEL,
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

    memory = (
        response["message"]["content"]
        .strip()
    )

    if not memory:
        return None

    if memory.upper() == "NONE":
        return None

    if len(memory) < 5:
        return None

    return memory


def auto_remember(
    user_text
):

    candidate = extract_memory(
        user_text
    )

    if not candidate:
        if DEBUG_MEMORY:
            print("[MEMORY] no candidate extracted")
        return None

    save, score = (
        should_save_memory(
            candidate
        )
    )

    if not save:
        if DEBUG_MEMORY:
            print(f"[MEMORY] candidate rejected (score={score})")
        return None

    saved = save_memory(
        candidate
    )

    if not saved:
        if DEBUG_MEMORY:
            print("[MEMORY] candidate already stored or invalid")
        return None

    if DEBUG_MEMORY:
        print(f"[MEMORY] candidate saved (score={score})")

    return {
        "memory": candidate,
        "score": score
    }


if __name__ == "__main__":

    while True:

        text = input(
            "\nUser Message: "
        )

        if text.lower() in [
            "quit",
            "exit"
        ]:
            break

        result = auto_remember(
            text
        )

        if result:

            print(
                "\nSaved:"
            )

            print(
                result["memory"]
            )

            print(
                f"Score: {result['score']}"
            )

        else:

            print(
                "\nNo memory stored."
            )
