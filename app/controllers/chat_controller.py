from fastapi.responses import StreamingResponse

from app.models.chat_model import ChatRequest, ChatResponse
from app.services.rag_services import get_answer, get_streaming_answer


def _history_to_dicts(request: ChatRequest):
    return [{"role": m.role, "content": m.content} for m in request.history]


def handle_chat(request: ChatRequest) -> ChatResponse:
    answer = get_answer(
        request.question.strip(),
        _history_to_dicts(request)
    )
    return ChatResponse(answer=answer)


def handle_chat_stream(request: ChatRequest) -> StreamingResponse:
    generator = get_streaming_answer(
        request.question.strip(),
        _history_to_dicts(request)
    )

    return StreamingResponse(
        generator,
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )