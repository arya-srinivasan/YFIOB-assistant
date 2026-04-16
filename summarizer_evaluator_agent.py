from new_main import summarizer


GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = "llama-3.1-8b-instant"

evaluator_agent = LlmAgent(
    name="Evaluator",
    model=LiteLlm(model=GROQ_MODEL),
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
    model=groq_llm2,
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