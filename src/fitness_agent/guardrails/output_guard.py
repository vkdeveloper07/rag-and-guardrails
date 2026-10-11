from pydantic import BaseModel, Field

from fitness_agent.guardrails.result import GuardrailResult
from fitness_agent.llm import get_guardrail_model

GUARD_NAME = "output"

JUDGE_PROMPT = """You are a strict reviewer of a car owner's manual assistant's answer.

Check two things:
1. grounded: every claim in the answer is supported by the context. Saying "I don't know" counts as grounded.
2. safe: the answer gives no unsafe driving instruction, no dangerous maintenance advice, no invalid repair procedure, and no recommendation that conflicts with the vehicle manual or could create a vehicle safety risk.

Context:
{context}

Question: {question}

Answer: {answer}"""


class Verdict(BaseModel):
    grounded: bool = Field(description="True if the answer is fully supported by the context")
    safe: bool = Field(description="True if the answer contains no unsafe advice")
    reason: str = Field(description="One short sentence explaining the verdict")


def check_output(question: str, context: str, answer: str) -> GuardrailResult:
    judge = get_guardrail_model().with_structured_output(Verdict)
    verdict = judge.invoke(JUDGE_PROMPT.format(context=context, question=question, answer=answer))
    return GuardrailResult(GUARD_NAME, verdict.grounded and verdict.safe, verdict.reason)
