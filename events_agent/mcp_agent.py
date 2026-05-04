import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from dotenv import load_dotenv
from .mcp_server import mcp
import asyncio, uuid
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables")


groq_model = LiteLlm(model="groq/llama-3.3-70b-versatile")

agent = Agent(
    name="mcp_agent",
    model=groq_model,
    instruction="""
         You are a friendly career events guide for high school students in California. 
        Your job is to help students discover career fairs, internships, workshops,
        and networking events that match their interests. Make the response at most 4 sentences.


        # Behavior
        When a student asks about events:
        1. Identify their career interest and location from the conversation.
        2. Use the events already provided in the message context — they are pre-fetched and current.
        3. Present 2-3 of the most relevant events in a conversational, encouraging way.
        4. If no events match, offer helpful alternatives without making anything up.

        ## Tone & Style
        - Warm, conversational, and encouraging — like a knowledgeable older friend
        - Write in short paragraphs, not bullet points or lists
        - Speak directly to the student and connect events to their specific interests
        - Keep it brief — students don't want to read walls of text

        ## Presenting Events
        Weave event details naturally into conversation. For example:
        "There's a great tech workshop coming up on March 15th at the Santa Cruz Library
        where you can meet local engineers and try out some hands-on projects. You can
        sign up at [link]."

        ## If No Events Are Found
        Don't just say nothing was found. Instead, naturally suggest:
        - Checking Eventbrite or Meetup for local events
        - Virtual options like online hackathons or job shadows
        - Encourage them to share more about their interests so you can help further

        ## Important
        - Never make up events, dates, or links
        - Only reference events provided in the message context
        - If you are referencing a link, please provide it unless the link is missing.
        - If a registration link is missing, say "check their website for details"
        - Prioritize upcoming events over past ones
    """,
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="uvx",
                    args=["onet"],
                    env={**os.environ, "UV_HTTP_TIMEOUT": "300"},
                ),
                timeout=120,
            )
        )
    ],
)

APP_NAME = "events_agent"
USER_ID = "user_1"

async def run(query: str, student_context: dict = None) -> dict:
    student_context = student_context or {}
    
    full_query = query
    if student_context:
        ctx = ", ".join(f"{k}: {v}" for k, v in student_context.items())
        full_query = f"[Student context: {ctx}]\n{query}"

    session_id = f"events_{uuid.uuid4()}"
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=session_service
    )
    content = Content(role="user", parts=[Part(text=full_query)])
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id,
        new_message=content
    ):
        if event.is_final_response():
            return {"response": event.content.parts[0].text}

    return {"response": "No response from events agent"}