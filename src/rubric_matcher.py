"""
Instruction-following rubric matcher.
Checks whether the response addresses the key requirements in the instruction.
"""
import json
import os


class RubricMatcher:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self.default_rubric = self._load_default_rubric()

    def _load_default_rubric(self):
        rubric_path = os.path.join(os.path.dirname(__file__), "..", "rubrics", "default_rubric.json")
        try:
            with open(rubric_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def score(self, response_text: str, instruction: str) -> float:
        if self.mock or not instruction:
            return self._heuristic_score(response_text, instruction)
        return self._llm_score(response_text, instruction)

    def _heuristic_score(self, response_text: str, instruction: str) -> float:
        if not instruction:
            return 0.75
        instruction_words = set(instruction.lower().split())
        response_words = set(response_text.lower().split())
        coverage = len(instruction_words & response_words) / max(len(instruction_words), 1)
        return min(1.0, 0.5 + coverage)

    def _llm_score(self, response_text: str, instruction: str) -> float:
        try:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Given an instruction and a response, rate how well the response follows the instruction from 0.0 to 1.0. Return only the number."),
                ("human", "Instruction: {instruction}\n\nResponse: {response}"),
            ])
            result = (prompt | llm).invoke({"instruction": instruction, "response": response_text})
            return float(result.content.strip())
        except Exception:
            return self._heuristic_score(response_text, instruction)
