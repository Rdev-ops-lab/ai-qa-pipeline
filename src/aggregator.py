"""
Weighted score aggregation across evaluation dimensions.
"""
from dataclasses import dataclass


@dataclass
class DimensionScore:
    score: float
    weight: float


class ScoreAggregator:
    def __init__(self, weights: dict):
        self.weights = weights

    def aggregate(self, dimensions: dict[str, DimensionScore]) -> float:
        total_weight = sum(d.weight for d in dimensions.values())
        weighted_sum = sum(d.score * d.weight for d in dimensions.values())
        return weighted_sum / total_weight if total_weight > 0 else 0.0
