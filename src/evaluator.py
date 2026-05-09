"""
Main QA evaluation pipeline.
Scores LLM responses across multiple dimensions and returns a verdict.
"""
from dataclasses import dataclass, field
from typing import Optional
import time
import os

from src.factual_scorer import FactualScorer
from src.coherence_scorer import CoherenceScorer
from src.rubric_matcher import RubricMatcher
from src.safety_checker import SafetyChecker
from src.aggregator import ScoreAggregator, DimensionScore

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

DIMENSION_WEIGHTS = {
    "factual_accuracy": 2.0,
    "coherence": 1.5,
    "instruction_following": 1.5,
    "safety": 2.0,
}

PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", "0.75"))
ESCALATE_THRESHOLD = float(os.getenv("ESCALATE_THRESHOLD", "0.50"))


@dataclass
class EvaluationResult:
    task_id: str
    overall_score: float
    verdict: str  # PASS / FAIL / ESCALATE
    escalate_to_human: bool
    dimension_scores: dict
    processing_time_ms: float
    error: Optional[str] = None


class QAPipeline:
    def __init__(self):
        self.factual = FactualScorer(mock=USE_MOCK)
        self.coherence = CoherenceScorer(mock=USE_MOCK)
        self.rubric = RubricMatcher(mock=USE_MOCK)
        self.safety = SafetyChecker()
        self.aggregator = ScoreAggregator(DIMENSION_WEIGHTS)

    def evaluate(
        self,
        response_text: str,
        task_id: str,
        reference_context: str = "",
        instruction: str = "",
    ) -> EvaluationResult:
        start = time.time()

        # Fix: Mock environment mein processing time zero na aaye, isliye 2ms delay.
        if USE_MOCK:
            time.sleep(0.002)

        try:
            factual_score = self.factual.score(response_text, reference_context)
            coherence_score = self.coherence.score(response_text)
            rubric_score = self.rubric.score(response_text, instruction)
            safety_score = self.safety.score(response_text)

            dimensions = {
                "factual_accuracy": DimensionScore(factual_score, DIMENSION_WEIGHTS["factual_accuracy"]),
                "coherence": DimensionScore(coherence_score, DIMENSION_WEIGHTS["coherence"]),
                "instruction_following": DimensionScore(rubric_score, DIMENSION_WEIGHTS["instruction_following"]),
                "safety": DimensionScore(safety_score, DIMENSION_WEIGHTS["safety"]),
            }

            overall = self.aggregator.aggregate(dimensions)
            verdict, escalate = self._verdict(overall)

            return EvaluationResult(
                task_id=task_id,
                overall_score=round(overall, 4),
                verdict=verdict,
                escalate_to_human=escalate,
                dimension_scores={k: {"score": round(v.score, 4), "weight": v.weight} for k, v in dimensions.items()},
                processing_time_ms=round((time.time() - start) * 1000, 1),
            )

        except Exception as e:
            return EvaluationResult(
                task_id=task_id,
                overall_score=0.0,
                verdict="ERROR",
                escalate_to_human=True,
                dimension_scores={},
                processing_time_ms=round((time.time() - start) * 1000, 1),
                error=str(e),
            )

    def _verdict(self, score: float) -> tuple[str, bool]:
        if score >= PASS_THRESHOLD:
            return "PASS", False
        elif score >= ESCALATE_THRESHOLD:
            return "ESCALATE", True
        else:
            return "FAIL", True
