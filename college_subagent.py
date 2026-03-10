"""
college_agent.py
Clean version of college_subagent for the router .
"""

import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.genai import types

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
APP_NAME   = "college_workflow"
USER_ID    = "user_1"
SESSION_ID = "session_001"

# ── Agent definition (same as Colab) ────────────────────────────────────
college_planning_agent = Agent(
    name="college_planning_agent",
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    instruction="""
You are an agent designed to assist students in exploring colleges based on their career interests.

Given the user's career interest, direct the conversation towards college planning
and helping the user explore colleges based on requirements and career goals.
Keep answers concise and exploratory. Answers should mimic a college advisor tone
and focus on the subject of colleges and college planning. Focus on California schools.
"""
)


# ── run() wrapper for the router ─────────────────────────────────────────────

def run(query: str, student_context: dict = {}) -> dict:
    """
    Single-turn function for the router to call.
    Takes a question and optional student context, returns a response dict.
    """
    # Inject student context into the query if available
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
            app_name=APP_NAME,
            agent=college_planning_agent,
            session_service=session_service
        )
        content = types.Content(role="user", parts=[types.Part(text=full_query)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    return event.content.parts[0].text
        return "No response from college agent."

    response_text = asyncio.run(_run())
    return {"response": response_text}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run(
        query="What California colleges are good for someone interested in computer science?",
        student_context={"grade": "11th", "interests": "coding, math"}
    )
    print(result["response"])