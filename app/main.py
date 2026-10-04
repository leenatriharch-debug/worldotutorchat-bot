from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat_routes import router as chat_router
from app.services import rag_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_services.get_reranker()
    except Exception:
        pass
    yield


app = FastAPI(
    title="WorldoTutors AI Assistant",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)


@app.get("/")
def home():
    return {
        "message": "WorldoTutors AI Assistant API is running"
    }