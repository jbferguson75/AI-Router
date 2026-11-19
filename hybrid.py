def build_hybrid_prompt(question, context):
    return f"""
You are a helpful AI assistant.

You have access to retrieved
private documents.

Use those documents whenever possible.

Retrieved Context:

{context}

Question:

{question}

If the retrieved information is incomplete,
you may supplement with general knowledge.

Clearly prioritize information found
in the retrieved documents.
"""
