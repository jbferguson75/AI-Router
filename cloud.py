from openai import OpenAI
from dotenv import load_dotenv
from threading import Lock

from models import create_response

load_dotenv()

client = None
_client_lock = Lock()

from config import CLOUD_MODEL, DEBUG_PROMPTS


def get_client():
    global client
    if client is None:
        with _client_lock:
            if client is None:
                client = OpenAI(
                    timeout=90.0,
                    max_retries=2
                )
    return client


def ask_cloud(question, *, approval_confirmed=False):

    if not approval_confirmed:
        raise PermissionError(
            "OpenAI requests must be approved in the privacy review first."
        )

    if DEBUG_PROMPTS:
        print(f"[OPENAI PROMPT]\n{question}")

    response = get_client().chat.completions.create(
        model=CLOUD_MODEL,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = (
        response
        .choices[0]
        .message.content
    )

    return create_response(
        route="CLOUD",
        answer=answer
    )
