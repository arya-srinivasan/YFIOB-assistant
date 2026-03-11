# api.py
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from router_agent import run, init_db, load_profile

# ── Request/Response schemas ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:         str
    user_id:         str  = "anonymous"
    student_context: dict = {}

class ChatResponse(BaseModel):
    response:      str
    agents_called: list[str]
    student_context: dict

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ← your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()  # initialize memory DB on startup

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Load profile if not provided
    context = req.student_context or load_profile(req.user_id) or {}

    result = run(
        query=          req.message,
        student_context=context,
        user_id=        req.user_id,
    )
    return ChatResponse(
        response=        result["response"],
        agents_called=   result["agents_called"],
        student_context= result.get("student_context", context),
    )

@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    profile = load_profile(user_id)
    return {"user_id": user_id, "profile": profile or {}}

@app.get("/health")
def health():
    return {"status": "ok"}

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)