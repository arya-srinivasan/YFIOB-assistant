"""
app.py
RAG agent for YFIOB Assistant — built with Google ADK.
Converts user query → structured query object → filtered Pinecone retrieval → Groq response.

Install deps:
    pip install pinecone sentence-transformers groq python-dotenv google-adk[extensions] litellm
"""

import os
import re
import ast
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
INDEX_NAME       = "yfiob-rag-agent"
EMBED_MODEL      = "avsolatorio/GIST-large-Embedding-v0"
GROQ_MODEL       = "llama-3.3-70b-versatile"
TOP_K            = 4
APP_NAME         = "yfiob_rag"
USER_ID          = "user"
SESSION_ID       = "session"

VALID_INDUSTRY_SECTORS = {
    "Architecture and Engineering",
    "Agriculture and Natural Resources",
    "Marketing, Sales, and Service",
    "Building, Trades, and Construction",
    "Energy, Environment, Utilities",
    "Fashion and Interior Design",
    "Manufacturing and Product Development",
    "Education, Child Development, Family Services",
    "Public and Government Services",
    "Finance and Business",
    "Arts, Media, and Entertainment",
    "Information and Computer Technologies",
    "Hospitality, Tourism, Recreation",
    "Health Services, Sciences, Medical Technology",
}

# ── Singletons ────────────────────────────────────────────────────────────────
_index       = None
_embed_model = None
_groq        = None

def _get_index():
    global _index
    if _index is None:
        _index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
    return _index

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

def _get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


# ── Tool: retrieve from Pinecone ──────────────────────────────────────────────

def retrieve_from_podcasts(query: str, industry: str = "general") -> str:
    """
    Retrieves relevant career advice from real podcast interview transcripts.
    Use this for any question about careers, professions, job experiences, or career advice.

    Args:
        query:    The student's career question
        industry: Optional industry sector to filter by (e.g. 'Health Services, Sciences, Medical Technology')

    Returns:
        Relevant excerpts from podcast interviews as a formatted string
    """
    # Build query object using Groq
    prompt = f"""
You are a query parser for a career guidance assistant.
Given a student's question, return a JSON object with exactly two fields:
1. "content_string_query": a concise search string capturing the core career question
2. "industry_filter": a list of relevant industry sectors from this exact list (use empty list [] if none match):
{sorted(VALID_INDUSTRY_SECTORS)}

Rules:
- Return ONLY the raw JSON object, no markdown, no code fences, no explanation.
- Only include industry sectors from the provided list, spelled exactly as shown.

Student question: {query}
"""
    resp = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = {"content_string_query": query, "industry_filter": []}

    parsed["industry_filter"] = [
        s for s in parsed.get("industry_filter", [])
        if s in VALID_INDUSTRY_SECTORS
    ]

    # Retrieve from Pinecone
    vector  = _get_embed_model().encode([parsed["content_string_query"]])[0].tolist()
    filters = parsed.get("industry_filter", [])
    kwargs  = dict(vector=vector, top_k=TOP_K, include_metadata=True)
    if filters and industry != "general":
        kwargs["filter"] = {"Industry Sectors": {"$in": filters}}

    matches = _get_index().query(**kwargs)["matches"]

    if not matches:
        return "No relevant podcast excerpts found for this query."

    parts = []
    for i, m in enumerate(matches):
        meta = m["metadata"]
        parts.append(
            f"[Excerpt {i+1} — {meta.get('Interviewee', 'Unknown')} "
            f"({', '.join(meta.get('Industry Sectors', []))}) "
            f"relevance: {round(m['score'], 3)}]\n{meta['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ── ADK Agent ─────────────────────────────────────────────────────────────────

groq_model = LiteLlm(model=f"groq/{GROQ_MODEL}")

rag_agent = Agent(
    name="rag_agent",
    model=groq_model,
    instruction="""
        You are a supportive career guidance assistant for high school students.
        When a student asks about careers, professions, or work experiences:
        1. Call retrieve_from_podcasts with their question
        2. Ground your response in the retrieved podcast excerpts
        3. Reference the interviewee naturally, e.g. 'In a conversation with [Name]...'
        4. Keep your tone warm, encouraging, and age-appropriate
        5. Keep your response concise (max 5 sentences)
        If the excerpts don't fully answer the question, say so honestly and offer what you can.
    """,
    tools=[retrieve_from_podcasts],
)


# ── Public interface ──────────────────────────────────────────────────────────

def run(query: str, student_context: dict | None = None) -> dict:
    """
    Main entry point called by the router agent.
    Returns: { response, sources }
    """
    # Inject student context into the query
    full_query = query
    if student_context:
        ctx = ", ".join(f"{k}: {v}" for k, v in student_context.items())
        full_query = f"[Student context: {ctx}]\n{query}"

    async def _run():
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
        runner = Runner(
            agent=rag_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
        content = types.Content(role="user", parts=[types.Part(text=full_query)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    return event.content.parts[0].text
        return None

    response = asyncio.run(_run())
    return {
        "response": response,
        "sources":  [],
    }


# ── CLI chat loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎓 YFIOB RAG Agent — type 'quit' to exit\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = run(query)
        print(f"\nAgent: {result['response']}\n")