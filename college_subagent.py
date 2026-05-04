"""
college_agent.py
Clean version of college_subagent for the router .
"""

import os
import asyncio
import uuid
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types

from college_cache import redis_client, CACHE_TTL
import sys

sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
APP_NAME   = "college_workflow"
USER_ID    = "user_1"
SESSION_ID = "session_001"


def scrape_college_website(url: str) -> str:
    
    cache_key = f"college:url:{url}"
    cached = redis_client.get(cache_key)
    
    if cached:
        print(f"[SCRAPE CACHE HIT] {url}", flush=True) # just for debugging
        return cached

    try: # this is like the else block but with error handling
        print(f"[SCRAPING] {url}", flush=True)

        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Truncate to avoid token overflow
        text = text[:4000]

        redis_client.setex(cache_key, CACHE_TTL, text)
        return text

    except Exception as e:
        return f"Error scraping {url}: {e}"


# ── Agent definition (same as Colab) ────────────────────────────────────
college_planning_agent = Agent(
    name="college_planning_agent",
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    tools=[FunctionTool(scrape_college_website)],
    instruction="""
        You are an agent designed to assist students in exploring colleges based on their career interests.

        When the user asks about a college, USE the scrape_college_website tool to fetch
        real information from their official admissions pages before answering.

        Useful California college URLs to scrape:
        - UCLA: https://admission.ucla.edu
        - UCSD: https://admissions.ucsd.edu
        - UC Berkeley: https://admissions.berkeley.edu
        - Cal Poly SLO: https://admissions.calpoly.edu
        - UCSB: https://www.admissions.ucsb.edu
        - UC Davis: https://admissions.ucdavis.edu

        Keep answers concise and in a college advisor tone. Focus on requirements,
        programs, and fit for the student's career goals.
    """
)


# ── run() wrapper for the router ─────────────────────────────────────────────

async def run(query: str, student_context: dict = None) -> dict:
    student_context = student_context or {}
    
    full_query = query
    if student_context:
        ctx = ", ".join(f"{k}: {v}" for k, v in student_context.items())
        full_query = f"[Student context: {ctx}]\n{query}"

    session_id = str(uuid.uuid4())
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=college_planning_agent,
        session_service=session_service
    )
    content = types.Content(role="user", parts=[types.Part(text=full_query)])
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                return {"response": event.content.parts[0].text}
    return {"response": "No response from college agent."}


if __name__ == "__main__":
    result = asyncio.run(run(
        query="What California colleges are good for someone interested in computer science?",
        student_context={"grade": "11th", "interests": "coding, math"}
    ))
    print(result["response"])