import os
import asyncio
import atexit
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from .agent import career_events_rag_agent
from .scheduler import start_scheduler

load_dotenv()

LOCATION = os.getenv("EVENTS_LOCATION", "Santa Cruz, CA")
INDUSTRY = os.getenv("EVENTS_INDUSTRY", None) 

APP_NAME = "career_events_rag"
USER_ID = "user_1"
SESSION_ID = "session_1"


session_service = InMemorySessionService()
runner = None
async def setup():
    global runner
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(
        agent=career_events_rag_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )


def chat(user_message: str):
    content = Content(role="user", parts=[Part(text=user_message)])
    events = runner.run(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content
    )
    for event in events:
        if event.is_final_response():
            print(f"\nAgent: {event.content.parts[0].text}\n")


async def main():
    await setup()

    scheduler = start_scheduler(location=LOCATION, industry=INDUSTRY)
    atexit.register(lambda: scheduler.shutdown())

    print(f"Career Events RAG Agent")
    print(f"Location: {LOCATION} | Industry: {INDUSTRY or 'All'}")
    print(f"Pinecone refreshes automatically every 20 minutes.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input:
            chat(user_input)

if __name__ == "__main__":
    asyncio.run(main())