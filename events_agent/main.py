import os
import asyncio
import atexit
import uuid
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from .agent import career_events_rag_agent

load_dotenv()

LOCATION = os.getenv("EVENTS_LOCATION", "Santa Cruz, CA")
INDUSTRY = os.getenv("EVENTS_INDUSTRY", None) 

APP_NAME = "career_events_rag"
USER_ID = "user_1"


runner = None
session_id = None

async def setup():
    global runner, session_id
    session_id      = f"events_{uuid.uuid4()}"
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=career_events_rag_agent,
        session_service=session_service
    )


def chat(user_message: str):
    content = Content(role="user", parts=[Part(text=user_message)])
    for event in runner.run(
        user_id=USER_ID, session_id=session_id,  # ← use dynamic session_id
        new_message=content
    ):
        if event.is_final_response():
            print(f"\nAgent: {event.content.parts[0].text}\n")
        


async def main():
    await setup()
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input:
            chat(user_input)


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

    # Run the async setup and chat in a single event loop
    async def _run():
        local_session_id = f"events_{uuid.uuid4()}"
        local_session_service = InMemorySessionService()
        
        await local_session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=local_session_id
        )

        location = student_context.get("location", LOCATION)
        industry = student_context.get("industry", INDUSTRY)

        from .tools.pinecone_tool import refresh_and_retrieve
        fresh_events = refresh_and_retrieve(
            query=query,
            location=location,
            industry=industry,
        )
        print(f"[events] {len(fresh_events)} fresh events retrieved")

        if fresh_events:
            events_context = "\n\n".join(
                f"- {e.title} ({e.date or 'Date TBD'}) at {e.location}\n  {e.description}\n  Register: {e.registration_url}"
                for e in fresh_events
            )
            full_query_with_events = f"{full_query}\n\nRecently found events:\n{events_context}"
        else:
            full_query_with_events = full_query

        local_runner = Runner(
            app_name=APP_NAME,
            agent=career_events_rag_agent,
            session_service=local_session_service
        )
        content = Content(role="user", parts=[Part(text=full_query_with_events)])
        async for event in local_runner.run_async(
            user_id=USER_ID,
            session_id=local_session_id,
            new_message=content
        ):
            if event.is_final_response():
                return event.content.parts[0].text
        return "No response from events agent."

    response_text = asyncio.run(_run())
    return {"response": response_text}

if __name__ == "__main__":
    asyncio.run(main())