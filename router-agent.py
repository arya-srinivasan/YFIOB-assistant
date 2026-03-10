"""
router-agent.py — YFIOB Router
Uses Groq to classify the student's message and dispatch to the right subagent(s).

Subagents:
  - rag_agent      (Arshia)   — answers career questions from podcast transcripts
  - memory_agent   (Prithika) — tracks student interests and profile
  - college_agent  (Pariya)   — college planning and admissions
  - events_agent   (Arya)     — recommends nearby events and role models
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "rag-agent"))
sys.path.append(os.path.join(BASE_DIR, "career_agent"))
sys.path.append(os.path.join(BASE_DIR, "college_agent"))
sys.path.append(os.path.join(BASE_DIR, "events_agent"))

# ── Import subagents ──────────────────────────────────────────────────────────
import app as rag_agent
from memory import load_profile, init_db
from agent import run as memory_run
from events_agent.main import run as events_run

# Uncomment when ready:
# import college_agent

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
GROQ_MODEL       = "llama-3.3-70b-versatile"
AVAILABLE_AGENTS = ["rag_agent", "memory_agent", "college_agent", "events_agent"]

_groq = None
def _get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


# ── Step 1: Classify ──────────────────────────────────────────────────────────

def classify(query: str, student_context: dict) -> list[str]:
    prompt = f"""
You are a router for a high school career guidance assistant.
Given a student's message, decide which agents should handle it.

Available agents:
- rag_agent: answers career questions using real podcast transcripts and stories
- memory_agent: updates or retrieves the student's interest profile and preferences
- college_agent: answers questions about colleges, majors, requirements, applications
- events_agent: recommends nearby events, internships, or role models to meet

Student context: {json.dumps(student_context)}
Student message: {query}

Respond with ONLY a JSON array of agent names to call, like: ["rag_agent", "memory_agent"]
Only include agents that are clearly relevant. Never include more than 2.
"""
    resp = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        agents = json.loads(raw)
        return [a for a in agents if a in AVAILABLE_AGENTS]
    except Exception:
        return ["rag_agent"]


# ── Step 2: Dispatch ──────────────────────────────────────────────────────────

def dispatch(query: str, agents: list[str], student_context: dict, user_id: str = "") -> dict:
    results = {}

    for agent in agents:
        try:
            if agent == "rag_agent":
                print(f"🔧 Calling: rag_agent")
                results["rag_agent"] = rag_agent.run(query, student_context)

            elif agent == "memory_agent":
                print(f"🔧 Calling: memory_agent")
                updated_profile = memory_run(user_id, query)
                student_context.update(updated_profile)
                results["memory_agent"] = {"response": None}

            elif agent == "college_agent":
                print(f"🔧 Calling: college_agent")
                # results["college_agent"] = college_agent.run(query, student_context)
                results["college_agent"] = {"response": "College agent coming soon!"}

            elif agent == "events_agent":
                print(f"🔧 Calling: events_agent")
                # Inject location from student context or default to Santa Cruz
                location = student_context.get("location", "Santa Cruz, CA")
                enhanced_query = f"{query} location: {location}"
                results["events_agent"] = events_run(enhanced_query, student_context)

        except Exception as e:
            print(f"❌ {agent} failed: {e}")
            results[agent] = {"error": str(e)}

    return results


# ── Step 3: Synthesize ────────────────────────────────────────────────────────

def synthesize(query: str, results: dict, student_context: dict) -> str:
    responses = {
        k: v["response"] for k, v in results.items()
        if isinstance(v, dict) and v.get("response")
    }

    if not responses:
        # Only memory_agent was called or no responses — answer from student profile
        if student_context:
            profile_str = json.dumps(student_context, indent=2)
            prompt = f"""
You are a warm, encouraging career guidance assistant for high school students.
Answer the student's question using only their profile information below.
Keep your tone conversational and encouraging.

Student profile:
{profile_str}

Student question: {query}
"""
            resp = _get_groq().chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
            )
            return resp.choices[0].message.content
        return "I don't have enough information about you yet. Tell me a bit about yourself — what are you into?"

    if len(responses) == 1:
        return list(responses.values())[0]

    combined = "\n\n".join(
        f"[{agent.replace('_', ' ').title()}]\n{resp}"
        for agent, resp in responses.items()
    )

    prompt = f"""
You are a warm, encouraging career guidance assistant for high school students.
Multiple sources have provided information to answer the student's question.
Combine them into one clear, conversational response.
Do not mention the internal agents or sources.

Student context: {json.dumps(student_context)}
Student question: {query}

Information gathered:
{combined}

Write a single, cohesive response directly to the student.
"""
    resp = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=700,
    )
    return resp.choices[0].message.content


# ── Public interface ──────────────────────────────────────────────────────────

def run(query: str, student_context: dict | None = None, user_id: str = "") -> dict:
    if student_context is None:
        student_context = {}

    agents   = classify(query, student_context)
    results  = dispatch(query, agents, student_context, user_id)
    response = synthesize(query, results, student_context)

    return {
        "response":      response,
        "agents_called": agents,
        "results":       results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("🎓 YFIOB Assistant\n")

    user_id = input("What's your name? ").strip()
    student_context = load_profile(user_id)

    if student_context:
        print(f"Welcome back, {user_id}!\n")
    else:
        print(f"Nice to meet you, {user_id}!\n")

    print("Type 'quit' to exit\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = run(query, student_context, user_id)
        print(f"\nAssistant: {result['response']}")
        print(f"🔀 Agents called: {', '.join(result['agents_called'])}\n")