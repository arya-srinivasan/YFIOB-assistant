import os
from google.adk.agents import Agent
from .tools.pinecone_tool import retrieve_career_events
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

groq_model = LiteLlm(model="groq/llama-3.3-70b-versatile")

career_events_rag_agent = Agent(
    name="career_events_rag_agent",
    model=groq_model,
    instruction="""
        You are a career events assistant. Your job is to help users discover
        relevant upcoming career fairs, job fairs, hiring events, and professional
        networking events based on their interests.

        When a user asks about career events:
        1. Identify their industry interest and location from the conversation.
        2. Call retrieve_career_events with a descriptive query, their industry, 
           and location as filters.
        3. Present the top results clearly, including the event name, date, 
           location, description, and registration link.
        4. Prioritize upcoming events and those most relevant to the user's industry.

        If the user's query is vague, ask for their industry or location to
        improve the results before calling the tool.

        Event data is refreshed automatically every 20 minutes.
    """,
    tools=[retrieve_career_events],
)