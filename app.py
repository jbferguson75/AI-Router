import logging

import streamlit as st

import conversation_store
from orchestrator import (
    ask,
    approve_openai_request,
    cancel_openai_request
)

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

conversation_store.initialize()

if "active_conversation_id" not in st.session_state:
    existing_messages = st.session_state.get("messages", [])
    conversations = conversation_store.list_conversations()
    if conversations:
        initial_conversation = conversations[0]
        conversation_id = initial_conversation["id"]
    else:
        conversation_id = conversation_store.create_conversation()
        initial_conversation = conversation_store.get_conversation(
            conversation_id
        )

    if existing_messages and not conversation_store.get_messages(conversation_id):
        for message in existing_messages:
            conversation_store.add_message(conversation_id, message)
        initial_conversation = conversation_store.get_conversation(
            conversation_id
        )

    st.session_state.active_conversation_id = conversation_id
    st.session_state.messages = conversation_store.get_messages(conversation_id)
    st.session_state.pending_openai = conversation_store.get_pending_request(
        conversation_id
    )
    st.session_state.conversation_selector = conversation_id
    st.session_state.conversation_title_input = initial_conversation["title"]
    st.session_state.confirm_delete_conversation = False

if "messages" not in st.session_state:
    st.session_state.messages = conversation_store.get_messages(
        st.session_state.active_conversation_id
    )

if "pending_openai" not in st.session_state:
    st.session_state.pending_openai = conversation_store.get_pending_request(
        st.session_state.active_conversation_id
    )


def append_message(message):
    st.session_state.messages.append(message)
    conversation_store.add_message(
        st.session_state.active_conversation_id,
        message
    )


def switch_conversation():
    conversation_id = st.session_state.conversation_selector
    if conversation_id == st.session_state.active_conversation_id:
        return

    conversation = conversation_store.get_conversation(conversation_id)
    if conversation is None:
        return

    st.session_state.active_conversation_id = conversation_id
    st.session_state.messages = conversation_store.get_messages(conversation_id)
    st.session_state.pending_openai = conversation_store.get_pending_request(
        conversation_id
    )
    st.session_state.conversation_title_input = conversation["title"]
    st.session_state.confirm_delete_conversation = False


def create_new_conversation():
    conversation_id = conversation_store.create_conversation()
    st.session_state.active_conversation_id = conversation_id
    st.session_state.messages = []
    st.session_state.pending_openai = None
    st.session_state.conversation_selector = conversation_id
    st.session_state.conversation_title_input = conversation_store.DEFAULT_TITLE
    st.session_state.confirm_delete_conversation = False


def rename_active_conversation():
    conversation_store.rename_conversation(
        st.session_state.active_conversation_id,
        st.session_state.conversation_title_input
    )


def request_conversation_delete():
    st.session_state.confirm_delete_conversation = True


def cancel_conversation_delete():
    st.session_state.confirm_delete_conversation = False


def confirm_conversation_delete():
    conversation_store.delete_conversation(
        st.session_state.active_conversation_id
    )
    conversations = conversation_store.list_conversations()
    if conversations:
        conversation_id = conversations[0]["id"]
        conversation = conversations[0]
    else:
        conversation_id = conversation_store.create_conversation()
        conversation = conversation_store.get_conversation(conversation_id)

    st.session_state.active_conversation_id = conversation_id
    st.session_state.messages = conversation_store.get_messages(conversation_id)
    st.session_state.pending_openai = conversation_store.get_pending_request(
        conversation_id
    )
    st.session_state.conversation_selector = conversation_id
    st.session_state.conversation_title_input = conversation["title"]
    st.session_state.confirm_delete_conversation = False

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

    st.subheader("Conversations")
    st.button(
        "New conversation",
        key="new_conversation",
        width="stretch",
        on_click=create_new_conversation
    )

    conversations = conversation_store.list_conversations()
    conversation_titles = {
        conversation["id"]: conversation["title"]
        for conversation in conversations
    }
    if st.session_state.active_conversation_id not in conversation_titles:
        if not conversations:
            conversation_id = conversation_store.create_conversation()
            conversations = conversation_store.list_conversations()
            conversation_titles = {
                conversation["id"]: conversation["title"]
                for conversation in conversations
            }
        replacement_id = conversations[0]["id"]
        st.session_state.conversation_selector = replacement_id
        st.session_state.active_conversation_id = replacement_id
        st.session_state.messages = conversation_store.get_messages(
            replacement_id
        )
        st.session_state.pending_openai = conversation_store.get_pending_request(
            replacement_id
        )
        st.session_state.conversation_title_input = conversations[0]["title"]

    conversation_ids = list(conversation_titles)
    if st.session_state.get("conversation_selector") not in conversation_ids:
        st.session_state.conversation_selector = (
            st.session_state.active_conversation_id
        )
    st.selectbox(
        "Saved conversations",
        options=conversation_ids,
        format_func=lambda conversation_id: conversation_titles.get(
            conversation_id,
            "Conversation"
        ),
        key="conversation_selector",
        on_change=switch_conversation
    )

    active_conversation = conversation_store.get_conversation(
        st.session_state.active_conversation_id
    )
    if "conversation_title_input" not in st.session_state:
        st.session_state.conversation_title_input = active_conversation["title"]
    with st.form("rename_conversation_form"):
        st.text_input(
            "Conversation title",
            key="conversation_title_input"
        )
        st.form_submit_button(
            "Rename conversation",
            on_click=rename_active_conversation
        )

    if not st.session_state.confirm_delete_conversation:
        st.button(
            "Delete conversation",
            key="delete_conversation",
            on_click=request_conversation_delete
        )
    else:
        st.warning(
            f"Delete ‘{active_conversation['title']}’ and its messages?"
        )
        confirm_col, keep_col = st.columns(2)
        with confirm_col:
            st.button(
                "Delete",
                key="confirm_delete_conversation_button",
                type="primary",
                width="stretch",
                on_click=confirm_conversation_delete
            )
        with keep_col:
            st.button(
                "Keep it",
                key="cancel_delete_conversation_button",
                width="stretch",
                on_click=cancel_conversation_delete
            )

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
            append_message({
                "role": "assistant",
                "content": answer,
                "sources": approved_result["sources"]
            })
            st.session_state.pending_openai = None
            conversation_store.set_pending_request(
                st.session_state.active_conversation_id,
                None
            )
            st.rerun()

    if cancel_clicked:
        try:
            with st.spinner("Answering with local documents..."):
                local_result = cancel_openai_request(pending)
            answer = (
                "**Route:** LOCAL\n\n"
                f"{local_result['answer']}"
            )
            sources = local_result.get("sources", [])
        except Exception:
            logger.exception("Could not answer from local documents after cancellation")
            answer = (
                "**Route:** LOCAL\n\n"
                "I couldn't complete a local answer. "
                "The OpenAI request was cancelled and nothing was sent."
            )
            sources = []

        append_message({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.session_state.pending_openai = None
        conversation_store.set_pending_request(
            st.session_state.active_conversation_id,
            None
        )
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

    append_message({
        "role": "user",
        "content": prompt
    })

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

        append_message({
            "role": "assistant",
            "content": answer,
            "sources": []
        })

        st.rerun()

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

        append_message({
            "role": "assistant",
            "content": answer,
            "sources": []
        })

        st.rerun()

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

        append_message({
            "role": "assistant",
            "content": answer,
            "sources": []
        })

        st.rerun()

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
                conversation_store.set_pending_request(
                    st.session_state.active_conversation_id,
                    result
                )
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

    append_message({
        "role": "assistant",
        "content": answer,
        "sources": result["sources"]
    })

    st.rerun()
