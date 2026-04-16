# api.py
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from new_main import chat_async, setup
from career_agent.memory import load_profile, init_db

from typing import Dict, Any, Optional
import uvicorn

# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    student_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()
    await setup()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Load profile if not provided
    context = req.student_context or load_profile(req.user_id) or {}

    # Call async agent pipeline correctly
    result = await chat_async(
        user_id=req.user_id,
        query=req.message,
        student_context=context
    )

    return ChatResponse(response=result)


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    profile = load_profile(user_id)
    return {"user_id": user_id, "profile": profile or {}}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)