"""
main.py — YFIOB Main Workflow
SequentialAgent: Dispatcher → ParallelAgent → Summarizer

Uses a custom GroqLlm(BaseLlm) bridge that bypasses LiteLLM entirely.
"""

import os
import sys
import json
import asyncio
from typing import AsyncIterator
from dotenv import load_dotenv

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types
from evaluatoragent import evaluate_response
from groq import Groq

load_dotenv()

# ── Path setup 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "rag-agent"))
sys.path.append(os.path.join(BASE_DIR, "career_agent"))

# ── Imports 
from memory import load_profile, init_db
import app as rag_module

# Uncomment as agents become ready:
from events_agent.main import run as events_run
from college_subagent import run as college_run
# from agent import run as memory_run

# ── Config ────────────────────────────────────────────────────────────────────
APP_NAME   = "yfiob_assistant"
GROQ_MODEL = "llama-3.3-70b-versatile"
VALID_AGENTS = ["rag_agent", "memory_agent", "college_agent", "events_agent"]



# GROQ - ADK BRIDGE
# Subclasses BaseLlm so ADK agents use Groq directly and dont use LiteLLM.
# Tool calls are handled by converting ADK tool schemas to Groq's format and executing the results back into the ADK response cycle.


class GroqLlm(BaseLlm):

    def __init__(self, model: str = GROQ_MODEL):
        super().__init__(model=model)
        self._groq_model = model
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    @property
    def model(self) -> str:
        return self._groq_model

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncIterator[LlmResponse]:

        # ── Build messages ────────────────────────────────────────────────────
        messages = []

        if llm_request.config and llm_request.config.system_instruction:
            si = llm_request.config.system_instruction
            text = (
                si if isinstance(si, str)
                else "".join(p.text for p in si.parts if hasattr(p, "text"))
            )
            if text:
                messages.append({"role": "system", "content": text})

        for content in llm_request.contents or []:
            role = "assistant" if content.role == "model" else "user"
            text = "".join(
                p.text for p in (content.parts or []) if hasattr(p, "text")
            )
            if text:
                messages.append({"role": role, "content": text})

        # ── Build Groq tool schemas from ADK tools ────────────────────────────
        groq_tools = []
        tool_map = {}

        if llm_request.tools_dict:
            for tool_name, tool_obj in llm_request.tools_dict.items():
                if not hasattr(tool_obj, "func"):
                    continue
                func = tool_obj.func
                import inspect
                sig = inspect.signature(func)
                properties = {}
                required = []
                for param_name, param in sig.parameters.items():
                    properties[param_name] = {"type": "string"}
                    if param.default is inspect.Parameter.empty:
                        required.append(param_name)

                groq_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": func.__doc__ or "",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                })
                tool_map[tool_name] = func

        # ── Call Groq ─────────────────────────────────────────────────────────
        loop = asyncio.get_event_loop()
        kwargs = dict(
            model=self._groq_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        if groq_tools:
            kwargs["tools"] = groq_tools
            kwargs["tool_choice"] = "auto"

        completion = await loop.run_in_executor(
            None, lambda: self._client.chat.completions.create(**kwargs)
        )

        choice = completion.choices[0]
        msg    = choice.message

        # ── Handle tool calls ─────────────────────────────────────────────────
        if msg.tool_calls:
            # Execute each tool call and collect results
            tool_results = []
            for tc in msg.tool_calls:
                fn   = tool_map.get(tc.function.name)
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = fn(**args) if fn else f"Tool {tc.function.name} not found."
                tool_results.append(f"[{tc.function.name}]: {result}")

            # Feed tool results back to Groq for final response
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]})
            for tc, result in zip(msg.tool_calls, tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            final_completion = await loop.run_in_executor(
                None, lambda: self._client.chat.completions.create(
                    model=self._groq_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000,
                )
            )
            text_out = final_completion.choices[0].message.content or ""
        else:
            text_out = msg.content or ""

        # ── Yield ADK response ────────────────────────────────────────────────
        async def _iter():
            yield LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=text_out)],
                ),
            )

        async for chunk in _iter():
            yield chunk


groq_llm = GroqLlm(model=GROQ_MODEL)


# ── Tool functions ────────────────────────────────────────────────────────────

def call_rag_agent(query: str) -> str:
    """Answers career questions using real podcast interview transcripts."""
    try:
        q       = rag_module.build_query_object(query)
        matches = rag_module.retrieve(q)
        if not matches:
            return "No relevant podcast content found."
        ctx = rag_module.format_context(matches)
        return rag_module.generate_response(query, ctx) or "No response."
    except Exception as e:
        return f"RAG error: {str(e)}"


def call_events_agent(query: str) -> str:
    """Finds upcoming career events, networking opportunities, and role models."""
    # return events_run(query).get("response") or "No response."
    return "Events agent coming soon!"


def call_college_agent(query: str) -> str:
    """Answers questions about California colleges, majors, and admissions."""
    # return college_run(query).get("response") or "No response."
    return "College agent coming soon!"


# ── 1. Dispatcher ─────────────────────────────────────────────────────────────

dispatcher = LlmAgent(
    name="dispatcher",
    model=groq_llm,
    description="Routes the student's message to the appropriate agents.",
    instruction=f"""
You are a router for a high school career guidance assistant.
Given a student's message, decide which agents should handle it.

Available agents:
- rag_agent:     answers career questions using real podcast transcripts and stories
- memory_agent:  updates or retrieves the student's interest profile and preferences
- college_agent: answers questions about colleges, majors, requirements, applications
- events_agent:  recommends nearby events, internships, or role models to meet

Respond with ONLY a JSON array of agent names, e.g.: ["rag_agent"]
Only include agents that are clearly relevant. Never include more than 2.
Valid agent names: {VALID_AGENTS}
""",
    output_key="selected_agents",
)


# ── 2. Parallel wrapper agents ────────────────────────────────────────────────

rag_wrapper = LlmAgent(
    name="rag_agent",
    model=groq_llm,
    description="Answers career questions from podcast transcripts.",
    instruction="""
The selected agents are: {selected_agents}
If 'rag_agent' appears in the selected agents, call call_rag_agent with the student's question and output *only* the result.
If 'rag_agent' does NOT appear in the selected agents, output *only*: SKIP
""",
    tools=[FunctionTool(call_rag_agent)],
    output_key="rag_result",
)

memory_wrapper = LlmAgent(
    name="memory_agent",
    model=groq_llm,
    description="Updates and retrieves the student's interest profile.",
    instruction="""
The selected agents are: {selected_agents}
If 'memory_agent' appears in the selected agents, acknowledge any new interests or goals the student mentioned and output a brief summary.
If 'memory_agent' does NOT appear in the selected agents, output *only*: SKIP
""",
    output_key="memory_result",
)

events_wrapper = LlmAgent(
    name="events_agent",
    model=groq_llm,
    description="Finds career events and role models.",
    instruction="""
The selected agents are: {selected_agents}
If 'events_agent' appears in the selected agents, call call_events_agent with the student's question and output *only* the result.
If 'events_agent' does NOT appear in the selected agents, output *only*: SKIP
""",
    tools=[FunctionTool(call_events_agent)],
    output_key="events_result",
)

college_wrapper = LlmAgent(
    name="college_agent",
    model=groq_llm,
    description="Answers college planning and admissions questions.",
    instruction="""
The selected agents are: {selected_agents}
If 'college_agent' appears in the selected agents, call call_college_agent with the student's question and output *only* the result.
If 'college_agent' does NOT appear in the selected agents, output *only*: SKIP
""",
    tools=[FunctionTool(call_college_agent)],
    output_key="college_result",
)


# ── 3. Parallel workflow ──────────────────────────────────────────────────────

parallel_workflow = ParallelAgent(
    name="parallel_workflow",
    description="Runs all selected subagents concurrently.",
    sub_agents=[rag_wrapper, memory_wrapper, events_wrapper, college_wrapper],
)


# ── 4. Summarizer ─────────────────────────────────────────────────────────────

summarizer = LlmAgent(
    name="summarizer",
    model=groq_llm,
    description="Combines all agent responses into one final answer.",
    instruction="""
You are a warm, encouraging career guidance assistant for high school students.
Combine the following agent responses into one clear, conversational response.
Only use responses that are not "SKIP", not empty, and not "coming soon".
Do not mention agents or any internal system details.
Keep your tone warm and encouraging.
If only one response exists, return it directly without modification.
If all responses are SKIP, say: "I wasn't able to find an answer to that. Could you try rephrasing?"

Input responses:

Career Advice (from podcasts):
{rag_result}

Student Profile Update:
{memory_result}

Events & Role Models:
{events_result}

College Planning:
{college_result}

Output only the final combined response. Do not include any headings or labels.
""",
)


# ── 5. Main SequentialAgent ───────────────────────────────────────────────────

main_agent = SequentialAgent(
    name="yfiob_main",
    description="YFIOB career guidance assistant for high school students.",
    sub_agents=[dispatcher, parallel_workflow, summarizer],
)


# ── Session & Runner ──────────────────────────────────────────────────────────

session_service = InMemorySessionService()
runner: Runner | None = None


async def setup(user_id: str, student_context: dict) -> None:
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
    if runner is None:
        raise RuntimeError("Call setup() before chat_async().")

    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
    last_response = None

    async for event in runner.run_async(
        user_id=user_id,
        session_id="session",
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                last_response = event.content.parts[0].text

# removed return statement here
    final_response = last_response or "I wasn't able to find an answer. Could you try rephrasing?"
    
    # EVALUATE FINAL RESPONSE -- integration point
    evaluation = evaluate_response(
    user_query = message,
    agent_response = final_response,
    agent_type = "final"
    )
    return {
        "response": final_response,
        "evaluation": evaluation
    }

# ── CLI ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    init_db()
    print("🎓 YFIOB Assistant\n")

    user_id = input("What's your name? ").strip()
    student_context = load_profile(user_id) or {}

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
        #response = await chat_async(user_id, query)
        #print(f"\nAssistant: {response}\n")
        
        #CHANGE TO EVAL
        result = await chat_async(user_id, query)
        print(f"\nAssistant: {result['response']}\n")
        print(f"Evaluation: {result['evaluation']}\n")

if __name__ == "__main__":
    asyncio.run(main())


