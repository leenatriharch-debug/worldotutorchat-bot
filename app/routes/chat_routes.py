from fastapi import APIRouter
from app.models.chat_model import ChatRequest
from app.controllers.chat_controller import chat, chat_stream

router = APIRouter()

router.add_api_route("/chat", chat, methods=["POST"])
router.add_api_route("/chat/stream", chat_stream, methods=["POST"])
