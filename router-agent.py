"""
router.py — YFIOB Router
Uses Groq to classify the student's message and dispatch to the right subagent(s).
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Imports ───────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "rag-agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "career_agent"))

import app as rag_agent
from memory import load_profile, init_db

# Uncomment when ready:
# import college_agent
# from events_agent.main import run as events_run

# Config
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
GROQ_MODEL       = "llama-3.3-70b-versatile"
AVAILABLE_AGENTS = ["rag_agent", "college_agent", "events_agent"]

_groq = None
def _get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


# Classify 

def classify(query: str, student_context: dict) -> list[str]:
    """Use Groq to decide which agents to call."""
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


# Dispatch  

def dispatch(query: str, agents: list[str], student_context: dict) -> dict:
    """Call each selected agent and collect results."""
    results = {}

    for agent in agents:
        try:
            if agent == "rag_agent":
                print(f"🔧 Calling: rag_agent")
                results["rag_agent"] = rag_agent.run(query, student_context)

            elif agent == "college_agent":
                print(f"🔧 Calling: college_agent")
                # results["college_agent"] = college_agent.run(query, student_context)
                results["college_agent"] = {"response": "College agent coming soon!"}

            elif agent == "events_agent":
                print(f"🔧 Calling: events_agent")
                # results["events_agent"] = events_run(query, student_context)
                results["events_agent"] = {"response": "Events agent coming soon!"}

        except Exception as e:
            print(f"❌ {agent} failed: {e}")
            results[agent] = {"error": str(e)}

    return results


# Synthesize

def synthesize(query: str, results: dict, student_context: dict) -> str:
    """Merge multiple agent responses into one cohesive reply."""
    responses = {
        k: v["response"] for k, v in results.items()
        if isinstance(v, dict) and "response" in v
    }

    if not responses:
        return "I'm sorry, I wasn't able to find an answer. Could you try rephrasing?"

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

def run(query: str, student_context: dict | None = None) -> dict:
    if student_context is None:
        student_context = {}

    agents   = classify(query, student_context)
    results  = dispatch(query, agents, student_context)
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

        result = run(query, student_context)
        print(f"\nAssistant: {result['response']}")
        print(f"🔀 Agents called: {', '.join(result['agents_called'])}\n")