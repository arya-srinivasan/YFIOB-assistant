"""
main.py — YFIOB Main Workflow
SequentialAgent: Dispatcher → ParallelAgent → Summarizer

"""

import os
import sys
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

nest_asyncio.apply()
load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "rag-agent"))
sys.path.append(os.path.join(BASE_DIR, "career_agent"))

# ── Imports ───────────────────────────────────────────────────────────────────
from memory import load_profile, init_db
import app as rag_module

# Uncomment as agents become ready:
# from events_agent.main import run as events_run
# from college_subagent import run as college_run
# from agent import run as memory_run

# ── Config 
APP_NAME   = "yfiob_assistant"
GROQ_MODEL = "llama-3.3-70b-versatile"
groq       = LiteLlm(model=f"groq/{GROQ_MODEL}")

VALID_AGENTS = ["rag_agent", "memory_agent", "college_agent", "events_agent"]


#  Tool functions 

def call_rag_agent(query: str) -> str:
    """
    Answers career-related questions using real podcast interview transcripts.
    Use for questions about careers, professions, job experiences, or career advice.

    Args:
        query: The student's career question
    """
    result = rag_module.run(query)
    return result.get("response") or "No response from RAG agent."


def call_events_agent(query: str) -> str:
    """
    Finds upcoming career events, networking opportunities, and role models.
    Use for questions about events, meetups, or people to connect with.

    Args:
        query: The student's question about events or role models
    """
    # return events_run(query).get("response") or "No response."
    return "Events agent coming soon!"


def call_college_agent(query: str) -> str:
    """
    Answers questions about California colleges, majors, and admissions requirements.
    Use for questions about college planning, applications, or what classes to take.

    Args:
        query: The student's college-related question
    """
    # return college_run(query).get("response") or "No response."
    return "College agent coming soon!"


#  1. Dispatcher 

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


# ── 2. Parallel wrapper agents ────────────────────────────────────────────────

rag_wrapper = LlmAgent(
    name="rag_agent",
    model=groq,
    description="Answers career questions from podcast transcripts.",
    instruction="""
        Check session state key 'selected_agents'.
        If 'rag_agent' is in the list, call call_rag_agent with the student's question
        and output *only* the result.
        If 'rag_agent' is NOT in the list, output *only*: SKIP
    """,
    tools=[call_rag_agent],
    output_key="rag_result",
)

memory_wrapper = LlmAgent(
    name="memory_agent",
    model=groq,
    description="Updates and retrieves the student's interest profile.",
    instruction="""
        Check session state key 'selected_agents'.
        If 'memory_agent' is in the list, acknowledge any new interests or goals
        the student mentioned and output a brief summary.
        If 'memory_agent' is NOT in the list, output *only*: SKIP
    """,
    output_key="memory_result",
)

events_wrapper = LlmAgent(
    name="events_agent",
    model=groq,
    description="Finds career events and role models.",
    instruction="""
        Check session state key 'selected_agents'.
        If 'events_agent' is in the list, call call_events_agent with the student's question
        and output *only* the result.
        If 'events_agent' is NOT in the list, output *only*: SKIP
    """,
    tools=[call_events_agent],
    output_key="events_result",
)

college_wrapper = LlmAgent(
    name="college_agent",
    model=groq,
    description="Answers college planning and admissions questions.",
    instruction="""
        Check session state key 'selected_agents'.
        If 'college_agent' is in the list, call call_college_agent with the student's question
        and output *only* the result.
        If 'college_agent' is NOT in the list, output *only*: SKIP
    """,
    tools=[call_college_agent],
    output_key="college_result",
)


# 3. Parallel workflow 

parallel_workflow = ParallelAgent(
    name="parallel_workflow",
    description="Runs all selected subagents concurrently.",
    sub_agents=[rag_wrapper, memory_wrapper, events_wrapper, college_wrapper],
)


#  4. Summarizer 

summarizer = LlmAgent(
    name="summarizer",
    model=groq,
    description="Combines all agent responses into one final answer.",
    instruction="""
        You are a warm, encouraging career guidance assistant for high school students.
        Combine the following agent responses into one clear, conversational response.
        Only use responses that are not SKIP and not empty and not "coming soon".
        Do not mention the agents or any internal system details.
        Keep your tone warm, encouraging.
        If only one response exists, return it directly without modification.
        If all responses are SKIP, say: "I wasn't able to find an answer to that. Could you try rephrasing?"

        Input Responses:

        Career Advice (from podcasts):
          {rag_result}

        Student Profile:
          {memory_result}

        Events & Role Models:
          {events_result}

        College Planning:
          {college_result}

        Output only the final combined response. Do not include any headings or labels.
    """,
)


#  5. Main SequentialAgent 

main_agent = SequentialAgent(
    name="yfiob_main",
    description="YFIOB career guidance assistant for high school students.",
    sub_agents=[dispatcher, parallel_workflow, summarizer],
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


async def chat_async(user_id: str, message: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    last_response = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id="session",
        new_message=content,
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
        response = await chat_async(user_id, query)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())