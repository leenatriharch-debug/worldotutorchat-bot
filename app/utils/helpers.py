def format_chat_history(chat_history):
    if not chat_history:
        return ""

    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-6:]
    )