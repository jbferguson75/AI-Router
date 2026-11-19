import conversation_store


def test_conversation_store_creates_renames_and_lists_conversations():
    conversation_store.initialize()
    conversation_id = conversation_store.create_conversation()

    assert conversation_store.get_conversation(conversation_id)["title"] == (
        conversation_store.DEFAULT_TITLE
    )
    assert conversation_store.rename_conversation(conversation_id, "  Trip plans  ")
    assert conversation_store.list_conversations()[0]["title"] == "Trip plans"
    assert not conversation_store.rename_conversation(conversation_id, "   ")


def test_first_user_message_sets_title_and_messages_keep_metadata():
    conversation_store.initialize()
    conversation_id = conversation_store.create_conversation()
    conversation_store.add_message(conversation_id, {
        "role": "user",
        "content": "Plan a weekend trip",
        "route": "LOCAL"
    })
    conversation_store.add_message(conversation_id, {
        "role": "assistant",
        "content": "Here is a plan.",
        "sources": [{"file": "travel.txt", "chunk": 1}]
    })

    assert conversation_store.get_conversation(conversation_id)["title"] == (
        "Plan a weekend trip"
    )
    assert conversation_store.get_messages(conversation_id) == [
        {"role": "user", "content": "Plan a weekend trip", "route": "LOCAL"},
        {
            "role": "assistant",
            "content": "Here is a plan.",
            "sources": [{"file": "travel.txt", "chunk": 1}]
        }
    ]


def test_pending_openai_request_survives_store_reload():
    conversation_store.initialize()
    conversation_id = conversation_store.create_conversation()
    pending = {
        "privacy_items": ["The question and retrieved text"],
        "pending_openai": {"prompt": "Review this before sending"}
    }

    conversation_store.set_pending_request(conversation_id, pending)

    assert conversation_store.get_pending_request(conversation_id) == pending
    conversation_store.set_pending_request(conversation_id, None)
    assert conversation_store.get_pending_request(conversation_id) is None


def test_deleting_conversation_deletes_its_messages():
    conversation_store.initialize()
    conversation_id = conversation_store.create_conversation()
    conversation_store.add_message(conversation_id, {
        "role": "user",
        "content": "Delete me"
    })

    assert conversation_store.delete_conversation(conversation_id)
    assert conversation_store.get_conversation(conversation_id) is None
    assert conversation_store.get_messages(conversation_id) == []
