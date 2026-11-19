import logging

import streamlit as st

from orchestrator import ask, approve_openai_request

from memory import (
    save_memory,
    get_all_memories,
    delete_memory,
    memory_count
)
from config import (
    SHOW_SOURCES,
    SHOW_MEMORY_NOTIFICATIONS,
    SHOW_CONFIDENCE_SCORES
)
from index_status import get_index_status

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="AI Router",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_openai" not in st.session_state:
    st.session_state.pending_openai = None

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 AI Router")

st.caption(
    "Local RAG • Hybrid Reasoning • Long-Term Memory"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Controls")

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.pending_openai = None
        st.rerun()

    st.divider()

    st.subheader("Routing")

    st.markdown("""
**Automatic**

Let the orchestrator choose.

**Overrides**

- `/local question`
- `/hybrid question`
- `/cloud question`
""")

    st.divider()

    st.subheader("Memory")

    st.markdown("""
**Commands**

- `/remember fact`
- `/forget fact`
- `/memories`
""")

    try:
        stored_memories = memory_count()
    except Exception:
        logger.exception("Could not read memory count")
        stored_memories = "Unavailable"
    st.metric("Stored Memories", stored_memories)

    st.divider()
    st.subheader("Document indexing")
    if st.button("Refresh indexing status", key="refresh_index_status"):
        st.rerun()

    index_status = get_index_status()
    ready_count = sum(
        1 for entry in index_status.values()
        if entry.get("status") == "ready"
    )
    indexing_count = sum(
        1 for entry in index_status.values()
        if entry.get("status") == "indexing"
    )
    online_only_count = sum(
        1 for entry in index_status.values()
        if entry.get("availability") == "online_only"
    )
    failed_entries = [
        (path, entry)
        for path, entry in index_status.items()
        if entry.get("status") == "failed"
    ]
    st.write(f"Indexed files: {ready_count}")
    st.write(f"Currently indexing: {indexing_count}")
    st.write(f"Online-only files: {online_only_count}")
    if failed_entries:
        st.error(f"Indexing errors: {len(failed_entries)}")
        with st.expander("Review indexing errors"):
            for path, entry in failed_entries[:10]:
                st.write(f"**{path}**")
                st.caption(entry.get("error", "Unknown indexing error"))

    st.divider()

    user_count = sum(
        1
        for msg in st.session_state.messages
        if msg["role"] == "user"
    )

    assistant_count = sum(
        1
        for msg in st.session_state.messages
        if msg["role"] == "assistant"
    )

    st.subheader("Session")

    st.metric(
        "Messages",
        len(st.session_state.messages)
    )

    st.write(
        f"User Messages: {user_count}"
    )

    st.write(
        f"Assistant Messages: {assistant_count}"
    )

# --------------------------------------------------
# Existing Conversation
# --------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

        if (
            SHOW_SOURCES
            and
            msg["role"] == "assistant"
            and msg.get("sources")
        ):

            with st.expander(
                "Sources"
            ):

                for source in msg[
                    "sources"
                ]:

                    st.write(
                        f"[SOURCE {source['source_number']}] "
                        f"{source['file']} "
                        f"(chunk {source['chunk']}) "
                        f"distance={source['distance']:.4f}"
                    )

# --------------------------------------------------
# OpenAI privacy review
# --------------------------------------------------

if st.session_state.pending_openai:
    pending = st.session_state.pending_openai
    st.warning(
        "This request will be sent to OpenAI. Review the information below before approving.",
        icon=":material/lock:"
    )

    for item in pending["privacy_items"]:
        st.markdown(f"- {item}")

    with st.expander("Exact information OpenAI will receive", expanded=True):
        st.code(pending["pending_openai"]["prompt"], language="text")

    approve_col, cancel_col = st.columns(2)
    with approve_col:
        approve_clicked = st.button(
            "Approve and send to OpenAI",
            type="primary",
            width="stretch",
            key="approve_openai_request"
        )
    with cancel_col:
        cancel_clicked = st.button(
            "Cancel",
            width="stretch",
            key="cancel_openai_request"
        )

    if approve_clicked:
        try:
            with st.spinner("Sending approved request..."):
                approved_result = approve_openai_request(
                    pending["pending_openai"]
                )
        except Exception:
            logger.exception("Approved OpenAI request failed")
            st.error(
                "OpenAI could not complete the request. "
                "The approval is still pending; you can retry or cancel."
            )
        else:
            answer = (
                f"**Route:** {approved_result['route']}\n\n"
                f"{approved_result['answer']}"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": approved_result["sources"]
            })
            st.session_state.pending_openai = None
            st.rerun()

    if cancel_clicked:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "OpenAI request cancelled. Nothing was sent.",
            "sources": []
        })
        st.session_state.pending_openai = None
        st.rerun()

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

prompt = st.chat_input(
    "Ask a question...",
    disabled=st.session_state.pending_openai is not None
)

if prompt:

    #
    # Store User Message
    #

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(prompt)

    # ==================================================
    # MEMORY COMMANDS
    # ==================================================

    if prompt.startswith(
        "/remember "
    ):

        memory_text = prompt[
            len("/remember "):
        ].strip()

        try:
            saved = save_memory(memory_text)
        except Exception:
            logger.exception("Could not save a memory")
            saved = None

        if saved:

            answer = (
                "✅ Memory saved\n\n"
                f"{memory_text}"
            )

        elif saved is False:

            answer = (
                "⚠️ Memory already exists "
                "or was invalid."
            )
        else:
            answer = "Could not save the memory. Check that Ollama is available."

        with st.chat_message(
            "assistant"
        ):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": []
            }
        )

        st.stop()

    if prompt.strip() == "/memories":

        try:
            memories = get_all_memories()
        except Exception:
            logger.exception("Could not read stored memories")
            memories = None

        if memories:

            answer = (
                "### Stored Memories\n\n"
                +
                "\n".join(
                    [
                        f"- {m}"
                        for m in memories
                    ]
                )
            )

        elif memories == []:

            answer = (
                "No memories stored."
            )
        else:
            answer = "Could not load memories. Check the local database."

        with st.chat_message(
            "assistant"
        ):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": []
            }
        )

        st.stop()

    if prompt.startswith(
        "/forget "
    ):

        memory_text = prompt[
            len("/forget "):
        ].strip()

        try:
            deleted = delete_memory(memory_text)
        except Exception:
            logger.exception("Could not delete a memory")
            deleted = None

        if deleted:

            answer = (
                "🗑️ Memory removed\n\n"
                f"{memory_text}"
            )

        elif deleted is False:

            answer = (
                "Memory not found."
            )
        else:
            answer = "Could not delete the memory. Check the local database."

        with st.chat_message(
            "assistant"
        ):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": []
            }
        )

        st.stop()

    # ==================================================
    # ROUTE OVERRIDES
    # ==================================================

    route_override = None
    question = prompt

    if prompt.startswith(
        "/local "
    ):

        route_override = "LOCAL"

        question = prompt[
            len("/local "):
        ].strip()

    elif prompt.startswith(
        "/hybrid "
    ):

        route_override = "HYBRID"

        question = prompt[
            len("/hybrid "):
        ].strip()

    elif prompt.startswith(
        "/cloud "
    ):

        route_override = "CLOUD"

        question = prompt[
            len("/cloud "):
        ].strip()

    # ==================================================
    # EXECUTE ORCHESTRATOR
    # ==================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            try:
                result = ask(
                    question=question,
                    history=st.session_state.messages,
                    route_override=route_override
                )
            except Exception:
                logger.exception("AI Router request failed")
                st.error(
                    "The request failed. Check that Ollama and the required "
                    "models are available, then review the launcher console."
                )
                result = {
                    "route": "ERROR",
                    "answer": "I couldn't complete that request.",
                    "sources": [],
                    "answer_score": None,
                    "memory_saved": None,
                    "memory_score": None
                }

            answer = (
                f"**Route:** "
                f"{result['route']}\n\n"
                f"{result['answer']}"
            )

            if result.get("confirmation_required"):
                answer = (
                    "OpenAI approval is required. Review the request below before it is sent."
                )
                st.session_state.pending_openai = result
            st.markdown(answer)

            # --------------------------------------
            # Memory Saved Notification
            # --------------------------------------

            if (
                SHOW_MEMORY_NOTIFICATIONS
                and result.get(
                    "memory_saved"
                )
            ):

                score = (
                    result.get(
                        "memory_score"
                    )
                    or "?"
                )

                st.success(
                    f"Memory stored "
                    f"(score {score})\n\n"
                    f"{result['memory_saved']}"
                )

            # --------------------------------------
            # Answer Confidence
            # --------------------------------------

            if (
                SHOW_CONFIDENCE_SCORES
                and
                result.get(
                    "answer_score"
                )
                is not None
            ):

                st.info(
                    f"Answer Confidence: "
                    f"{result['answer_score']}"
                )

            # --------------------------------------
            # RAG Sources
            # --------------------------------------

            if SHOW_SOURCES and result["sources"]:

                with st.expander(
                    "Sources"
                ):

                    for source in result[
                        "sources"
                    ]:

                        st.write(
                            f"[SOURCE {source['source_number']}] "
                            f"{source['file']} "
                            f"(chunk {source['chunk']}) "
                            f"distance={source['distance']:.4f}"
                        )

    # --------------------------------------------------
    # Store Assistant Response
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result["sources"]
        }
    )

    if result.get("confirmation_required"):
        st.rerun()
