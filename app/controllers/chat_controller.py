from app.models.chat_model import ChatRequest
from app.services.rag_services import get_answer, get_streaming_answer
from fastapi.responses import StreamingResponse


def chat(request: ChatRequest):
    answer = get_answer(
        request.question,
        request.chat_history
    )

    return {
        "question": request.question,
        "answer": answer
    }


def chat_stream(request: ChatRequest):
    return StreamingResponse(
        get_streaming_answer(
            request.question,
            request.chat_history
        ),
        media_type="text/plain"
    )