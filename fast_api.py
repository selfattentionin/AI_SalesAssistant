"""FastAPI wrapper for the LG refrigerator sales assistant."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from api import initialize_chatbot

load_dotenv()

app = FastAPI(title="LG Refrigerator Sales Assistant API", version="1.0.0")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


assistant = None


@app.on_event("startup")
def startup_event():
    global assistant
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    assistant = initialize_chatbot(api_key=api_key)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "assistant": "LG refrigerator sales RAG"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    if assistant is None:
        raise RuntimeError("Assistant not initialized")
    return assistant.ask(request.message)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
