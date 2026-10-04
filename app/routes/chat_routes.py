from fastapi import APIRouter

from app.controllers.chat_controller import handle_chat, handle_chat_stream
from app.models.chat_model import ChatRequest, ChatResponse


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return handle_chat(request)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    return handle_chat_stream(request)


@router.get("/health")
def health():
    return {"status": "ok"}