summary_agent = LlmAgent(
    name="Generator",
    model=groq_model,
    instruction="Generate a summary of the following agents. If you receive {feedback}, fix the errors and generate again.",
    output_key="draft"
)

evaluation_prompt = f"""
You are given a summarized answer to the user query. Your task is to evaluate the answer with the following criterion:

EVALUTION CRITERION:
Assess the answer on each of the following dimensions:
- Relevance – Does the answer directly address the user’s query?
- Correctness – Is the information factually accurate and free of errors?
- Clarity – Is the answer easy to understand and unambiguous?
- Completeness – Does the answer cover all important aspects of the query?
- Coherence – Is the answer logically structured and easy to follow?

INSTRUCTIONS:
- Read the provided answer.
- For each criterion:
  - Assign a score between 0 and 1 (where 0 = very poor, 1 = excellent).
  - Provide a brief justification (1–2 sentences).
- After evaluating all criteria:
  - Return PASS only if:
    - All scores are >= 0.7
    - AND no critical issues are identified
  - Otherwise return FAIL.

OUTPUT FORMAT: 
Return your evaluation in the following JSON format:

{{
  "relevance": <float>,
  "correctness": <float>,
  "clarity": <float>,
  "completeness": <float>,
  "coherence": <float>,
  "verdict": "PASS" or "FAIL",
  "feedback": "<concise actionable feedback>"
}}

User query: {user_query}
Summarized answer: {draft}
"""

evaluator_agent = LlmAgent(
    name="Evaluator",
    model=groq_model,
    instruction=evaluation_prompt,
    output_key="feedback"
)


refiner_agent = LlmAgent(
    name="Refiner",
    model=groq_model,
    instruction="""
    You are improving an answer.

    Original answer:
    {draft}

    Evaluation feedback:
    {feedback}

    Task:
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
    model=groq_model,
    agents=[evaluator_agent, refiner_agent],
    max_iterations=2,
    exit_condition='verdict == "PASS"'
)

workflow = SequentialAgent(
    model=groq_model,
    sub_agents=[summary_agent, optimizer_loop],
)