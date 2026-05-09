"""
Coherence scorer — LLM-as-judge for response quality.
Mock mode returns deterministic score based on response length/structure.
"""


class CoherenceScorer:
    def __init__(self, mock: bool = False):
        self.mock = mock

    def score(self, response_text: str) -> float:
        if self.mock:
            words = len(response_text.split())
            if words < 5:
                return 0.4
            if words > 500:
                return 0.7
            return 0.85
        return self._llm_score(response_text)

    def _llm_score(self, response_text: str) -> float:
        try:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Rate the coherence of the following response on a scale from 0.0 to 1.0. Return only the number."),
                ("human", "{response}"),
            ])
            result = (prompt | llm).invoke({"response": response_text})
            return float(result.content.strip())
        except Exception:
            return 0.7
