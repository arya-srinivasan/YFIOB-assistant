"""
main.py — YFIOB Main Workflow
Uses Google ADK LlmAgent for dispatcher and summarizer.
Subagents are called directly in Python — no ADK tool calling (avoids Groq incompatibility).
"""

import os
import sys
import json
import re
import asyncio
import time
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ── Path setup 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "rag-agent"))
sys.path.append(os.path.join(BASE_DIR, "career_agent"))

# ── Imports 
from memory import load_profile, init_db
import app as rag_module

from events_agent.main import run as events_run
from college_subagent import run as college_run
# from agent import run as memory_run

# ── Config 
APP_NAME   = "yfiob_assistant"
GROQ_MODEL = "llama-3.3-70b-versatile"
groq       = LiteLlm(model=f"groq/{GROQ_MODEL}")

VALID_AGENTS = ["rag_agent", "memory_agent", "college_agent", "events_agent"]


# ── Subagent functions ────────────────────

def call_rag_agent(query: str) -> str:
    try:
        query_obj = rag_module.build_query_object(query)
        matches   = rag_module.retrieve(query_obj)
        if not matches:
            return "No relevant podcast content found."
        context  = rag_module.format_context(matches)
        return rag_module.generate_response(query, context) or "No response."
    except Exception as e:
        return f"RAG agent error: {str(e)}"

def call_memory_agent(query: str) -> str:
    # return memory_run(user_id, query)
    return None

def call_events_agent(query: str) -> str:
    # return events_run(query).get("response")
    return None

def call_college_agent(query: str) -> str:
    # return college_run(query).get("response")
    return None

AGENT_FUNCTIONS = {
    "rag_agent":     call_rag_agent,
    "memory_agent":  call_memory_agent,
    "events_agent":  call_events_agent,
    "college_agent": call_college_agent,
}


# ── ADK Agents (dispatcher + summarizer)

dispatcher = LlmAgent(
    name="dispatcher",
    model=groq,
    description="Routes the student's message to the appropriate agents.",
    instruction=f"""
        You are a router for a high school career guidance assistant.
        Given a student's message, decide which agents should handle it.

        Available agents:
        - rag_agent: answers career questions using real podcast transcripts and stories
        - memory_agent: updates or retrieves the student's interest profile and preferences
        - college_agent: answers questions about colleges, majors, requirements, applications
        - events_agent: recommends nearby events, internships, or role models to meet

        Respond with ONLY a JSON array of agent names like: ["rag_agent", "memory_agent"]
        Only include agents that are clearly relevant. Never include more than 2.
        Valid agent names: {VALID_AGENTS}
    """,
    output_key="selected_agents",
)

summarizer = LlmAgent(
    name="summarizer",
    model=groq,
    description="Combines all agent responses into one final answer.",
    instruction="You are a warm, encouraging career guidance assistant for high school students. Answer the student's question based on the information provided to you.",
)


# ── Main SequentialAgent ──────────────────────────────────────────────────────

main_agent = SequentialAgent(
    name="yfiob_main",
    description="YFIOB career guidance assistant for high school students.",
    sub_agents=[dispatcher, summarizer],
)


# ── Session & Runner ──────────────────────────────────────────────────────────

session_service = InMemorySessionService()
runner          = None

async def setup(user_id: str, student_context: dict):
    global runner
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id="session",
        state={"student_context": student_context},
    )
    runner = Runner(
        agent=main_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )


async def run_pipeline(user_id: str, message: str) -> str:
    """
    Full pipeline:
    1. ADK dispatcher decides which agents to call
    2. Python directly calls those agents
    3. ADK summarizer combines results
    """
    # Step Runs dispatcher to get selected agents
    content = types.Content(role="user", parts=[types.Part(text=message)])
    selected_agents = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id="session",
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                raw = event.content.parts[0].text.strip()
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
                try:
                    selected_agents = json.loads(raw)
                    selected_agents = [a for a in selected_agents if a in VALID_AGENTS]
                except Exception:
                    selected_agents = ["rag_agent"]
            break

    # calling selected agents directly 
    results = {}
    for agent in selected_agents:
        fn = AGENT_FUNCTIONS.get(agent)
        if fn:
            results[agent] = fn(message)

    # building summarizer prompt with results injected
    rag_result     = results.get("rag_agent") or "SKIP"
    memory_result  = results.get("memory_agent") or "SKIP"
    events_result  = results.get("events_agent") or "SKIP"
    college_result = results.get("college_agent") or "SKIP"

    summary_prompt = f"""
You are a career guidance assistant for high school students.
Below are responses from specialized agents. 
If only one agent has a real response (not None, not SKIP, not "coming soon"), return it EXACTLY as-is without any modification.
If multiple agents have real responses, combine them into one clear conversational response.
If all responses are None or SKIP, answer the student's question yourself in a warm, encouraging way using your own knowledge.
Do not add any intro, outro, or extra commentary.

Student question: {message}

Career Advice (from podcasts): {rag_result}
Student Profile: {memory_result}
Events & Role Models: {events_result}
College Planning: {college_result}
"""
    # fresh session for summarizer to avoid state conflicts
    summary_session_id = f"summary_{id(message)}"
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=summary_session_id,
    )
    summarizer_runner = Runner(
        agent=summarizer,
        app_name=APP_NAME,
        session_service=session_service,
    )
    summary_content = types.Content(role="user", parts=[types.Part(text=summary_prompt)])
    last_response = None
    async for event in summarizer_runner.run_async(
        user_id=user_id,
        session_id=summary_session_id,
        new_message=summary_content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                last_response = event.content.parts[0].text

    return last_response or "I wasn't able to find an answer. Could you try rephrasing?"


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main():
    init_db()
    print("YFIOB Assistant\n")

    user_id = input("What's your name? ").strip()
    student_context = load_profile(user_id)

    if student_context:
        print(f"Welcome back, {user_id}!\n")
    else:
        print(f"Nice to meet you, {user_id}!\n")

    await setup(user_id, student_context)
    print("Type 'quit' to exit\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        response = await run_pipeline(user_id, query)
        print(f"\nAssistant: {response}\n")
        time.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
