from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from app.routes.chat_routes import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def home():
    return FileResponse("app/static/index.html")