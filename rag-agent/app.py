"""
app.py
RAG agent for YFIOB Assistant.
Converts user query → structured query object → filtered Pinecone retrieval → Groq response.

Install deps:
    pip install pinecone sentence-transformers groq python-dotenv
"""

import os
import re
import ast
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
INDEX_NAME       = "yfiob-rag-agent"
EMBED_MODEL      = "avsolatorio/GIST-large-Embedding-v0"
GROQ_MODEL       = "llama-3.3-70b-versatile"
TOP_K            = 4

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(EMBED_MODEL) 

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
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(INDEX_NAME)
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


# ── Step 1: Convert query to structured query object ─────────────────────────

def build_query_object(query: str) -> dict:
    """
    Use Groq to convert a natural language query into a structured object with:
      - content_string_query: refined search string
      - industry_filter: list of matching industry sectors
    """
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

    # Strip markdown fences if model includes them anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        # Fallback: no filter, use raw query
        parsed = {"content_string_query": query, "industry_filter": []}

    # Validate industry filters
    parsed["industry_filter"] = [
        s for s in parsed.get("industry_filter", [])
        if s in VALID_INDUSTRY_SECTORS
    ]
    return parsed


# ── Step 2: Retrieve from Pinecone ───────────────────────────────────────────

def retrieve(query_obj: dict, top_k: int = TOP_K) -> list[dict]:
    vector = _get_embed_model().encode([query_obj["content_string_query"]])[0].tolist()
    filters = query_obj.get("industry_filter", [])

    kwargs = dict(vector=vector, top_k=top_k, include_metadata=True)
    if filters:
        kwargs["filter"] = {"Industry Sectors": {"$in": filters}}

    return _get_index().query(**kwargs)["matches"]


# ── Step 3: Format context ────────────────────────────────────────────────────

def format_context(matches: list[dict]) -> str:
    parts = []
    for i, m in enumerate(matches):
        meta = m["metadata"]
        parts.append(
            f"[Excerpt {i+1} — {meta.get('Interviewee', 'Unknown')} "
            f"({', '.join(meta.get('Industry Sectors', []))}) "
            f"relevance: {round(m['score'], 3)}]\n{meta['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Step 4: Generate response ─────────────────────────────────────────────────

def generate_response(query: str, context: str, student_context: dict | None = None) -> str:
    student_info = ""
    if student_context:
        student_info = "\n\nStudent profile:\n" + "\n".join(
            f"  - {k}: {v}" for k, v in student_context.items()
        )

    system = (
        "You are a supportive career guidance assistant for high school students. "
        "Ground your response in the podcast excerpts provided. "
        "Reference them naturally, e.g. 'In a conversation with [Interviewee], ...'. "
        "Keep your tone warm, encouraging, and age-appropriate. "
        "If excerpts don't fully answer the question, say so and offer what you can."
        "Keep your response concise (max of 5 sentences)."
    )
    user = (
        f"Podcast excerpts:\n\n{context}"
        f"{student_info}"
        f"\n\nStudent question: {query}"
    )

    resp = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    return resp.choices[0].message.content


# ── Public interface ──────────────────────────────────────────────────────────

def run(query: str, student_context: dict | None = None) -> dict:
    """
    Main entry point called by the router agent.
    Returns: { response, sources, industry_filter, top_matches }
    """    
    query_obj = build_query_object(query)
    matches   = retrieve(query_obj)

    if not matches:
        return {
            "response": None,
            "sources":          [],
            "industry_filter":  query_obj["industry_filter"],
            "top_matches":      [],
        }

    context  = format_context(matches)
    response = generate_response(query, context, student_context)
    sources  = list({m["metadata"].get("Interviewee", "Unknown") for m in matches})

    return {
        "response":         response,
        "sources":          sources,
        "industry_filter":  query_obj["industry_filter"],
        "top_matches":      matches,
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
        print(f"\nAgent: {result['response']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        if result["industry_filter"]:
            print(f"Industry filter used: {', '.join(result['industry_filter'])}")
        print()