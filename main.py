"""
main.py — YFIOB Main Workflow
SequentialAgent: Dispatcher → ParallelAgent → Summarizer

Uses a custom GroqLlm(BaseLlm) bridge that bypasses LiteLLM entirely.
"""

import os
import sys
import json
import asyncio
import inspect
from typing import AsyncIterator
from dotenv import load_dotenv

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, BaseAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types
from groq import Groq

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
from agent import run as memory_run 

# ── Config ────────────────────────────────────────────────────────────────────
APP_NAME     = "yfiob_assistant"
GROQ_MODEL = "llama-3.1-8b-instant"
VALID_AGENTS = ["rag_agent", "memory_agent", "college_agent", "events_agent"]

# Tracks current user so tool functions can access it without passing through LLM
_current_user_id: str = ""

# ── GroqLlm Bridge ────────────────────────────────────────────────────────────
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

        # ── Build tool schemas ────────────────────────────────────────────────
        groq_tools = []
        tool_map   = {}

        print("DEBUG MODEL =", self._groq_model)

        if llm_request.tools_dict:
            for tool_name, tool_obj in llm_request.tools_dict.items():
                if not hasattr(tool_obj, "func"):
                    continue
                func = tool_obj.func
                sig  = inspect.signature(func)
                properties = {}
                required   = []
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
            kwargs["tools"]       = groq_tools
            kwargs["tool_choice"] = "auto"

        completion = await loop.run_in_executor(
            None, lambda: self._client.chat.completions.create(**kwargs)
        )

        choice = completion.choices[0]
        msg    = choice.message

        # ── Handle tool calls ─────────────────────────────────────────────────
        if msg.tool_calls:
            tool_results = []
            for tc in msg.tool_calls:
                fn   = tool_map.get(tc.function.name)
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = fn(**args) if fn else f"Tool {tc.function.name} not found."
                tool_results.append(f"[{tc.function.name}]: {result}")

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
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

from typing import Callable, Dict, Any, List
from google.adk.events import Event
import asyncio


class ToolAgent(BaseAgent):
    func: Callable
    output_key: str

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, func: Callable, output_key: str, description: str = ""):
        super().__init__(
            name=name,
            description=description,
            func=func,
            output_key=output_key
        )

    # -------------------------
    # CORE LOGIC
    # -------------------------
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        selected = state.get("selected_agents", [])
        query = state.get("query", "")

        if self.name not in selected:
            return {self.output_key: ""}

        try:
            result = self.func(query)
        except Exception as e:
            result = f"{self.name} error: {str(e)}"

        return {self.output_key: result}

    # -------------------------
    # SYNC ENTRY (used by custom flows)
    # -------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._execute(state)

    # -------------------------
    # REQUIRED BY ADK (IMPORTANT FIX)
    # MUST YIELD EVENTS
    # -------------------------
    async def _run_async_impl(self, context):
        result = await asyncio.to_thread(self._execute, context.state)

        # ADK expects EVENT STREAM, not dict
        yield Event(
            author=self.name,
            content=result
        )


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


def call_memory_agent(query: str) -> str:
    """Updates and retrieves the student's career interest profile."""
    try:
        updated_profile = memory_run(_current_user_id, query)
        if not updated_profile:
            return "No profile data found."
        return f"Profile updated: {json.dumps(updated_profile, indent=2)}"
    except Exception as e:
        return f"Memory error: {str(e)}"


def call_events_agent(query: str) -> str:
    """Finds upcoming career events, networking opportunities, and role models."""
    # return events_run(query).get("response") or "No response."
    return "Events agent coming soon!"


def call_college_agent(query: str) -> str:
    """Answers questions about California colleges, majors, and admissions."""
    # return college_run(query).get("response") or "No response."
    return "College agent coming soon!"


AGENT_MAP = {
    "rag_agent": call_rag_agent,
    "memory_agent": call_memory_agent,
    "events_agent": call_events_agent,
    "college_agent": call_college_agent,
}

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

rag_agent = ToolAgent(
    name="rag_agent",
    func=call_rag_agent,
    output_key="rag_result",
    description="Answers career questions from podcast transcripts."
)

memory_agent = ToolAgent(
    name="memory_agent",
    func=call_memory_agent,
    output_key="memory_result",
    description="Updates student profile."
)

events_agent = ToolAgent(
    name="events_agent",
    func=call_events_agent,
    output_key="events_result",
    description="Finds events and role models."
)

college_agent = ToolAgent(
    name="college_agent",
    func=call_college_agent,
    output_key="college_result",
    description="Handles college planning questions."
)


# ── 3. Parallel workflow ──────────────────────────────────────────────────────

parallel_workflow = ParallelAgent(
    name="parallel_workflow",
    sub_agents=[rag_agent, memory_agent, events_agent, college_agent]
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
output_key="draft"
)

evaluator_agent = LlmAgent(
    name="Evaluator",
    model=groq_llm,
    instruction="""
You are given a summarized answer to the user query.
Your task is to evaluate the answer using the following criteria:

EVALUATION CRITERIA:
- Relevance
- Correctness
- Clarity
- Completeness
- Coherence

INSTRUCTIONS:
- Score each criterion from 0 to 1
- Provide 1–2 sentence justification
- Return PASS only if all scores >= 0.7 and no critical issues
- Otherwise return FAIL

OUTPUT FORMAT (JSON ONLY):
{
  "relevance": <float>,
  "correctness": <float>,
  "clarity": <float>,
  "completeness": <float>,
  "coherence": <float>,
  "verdict": "PASS" | "FAIL",
  "feedback": "<concise actionable feedback>"
}

User query:
{query}

Summarized answer:
{draft}
""",
    output_key="feedback",
)

refiner_agent = LlmAgent(
    name="Refiner",
    model=groq_llm,
    instruction="""
You are improving an answer.

Original answer:
{draft}

Evaluation feedback:
{feedback}

TASK:
- Fix ALL issues mentioned in the feedback
- Improve correctness, completeness, and clarity
- Do NOT ignore low-scoring criteria
- Keep the answer concise but complete
- Do not mention the feedback in your response

Return ONLY the improved answer.
""",
    output_key="final_answer"
)

optimizer_loop = LoopAgent(
    name="Optimizer",
    sub_agents=[evaluator_agent, refiner_agent],
)

summarizer_workflow = SequentialAgent(
    name="summarizer_workflow",
    sub_agents=[summarizer, optimizer_loop],
)

# ── 5. Main SequentialAgent ───────────────────────────────────────────────────

main_agent = SequentialAgent(
    name="yfiob_main",
    description="YFIOB career guidance assistant for high school students.",
    sub_agents=[dispatcher, parallel_workflow, summarizer_workflow]
)


# ── Session & Runner ──────────────────────────────────────────────────────────

session_service = InMemorySessionService()
runner: Runner | None = None


async def setup(user_id: str, student_context: dict) -> None:
    global runner, _current_user_id
    _current_user_id = user_id  # ← set before session so tool functions can use it
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

    return last_response or "I wasn't able to find an answer. Could you try rephrasing?"


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
        response = await chat_async(user_id, query)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())