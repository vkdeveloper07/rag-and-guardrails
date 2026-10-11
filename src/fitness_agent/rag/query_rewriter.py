from fitness_agent.llm import get_agent_model

REWRITE_PROMPT = """Rewrite the user's question as a short search query for a car owner's manual knowledge base.
Keep the key vehicle details, such as model, feature, warning light, warning indicator, maintenance task, or system, and expand common automotive abbreviations.
Remove filler words and conversational phrases, but preserve the important manual-related terms.
Return only the search query.

Question: {question}"""


def rewrite_query(question: str) -> str:
    response = get_agent_model().invoke(REWRITE_PROMPT.format(question=question))
    return response.text.strip().strip('"')
