"""
Sample usage — mock mode, no API key needed.
Run: USE_MOCK=true python examples/sample_usage.py
"""
import os, sys
os.environ["USE_MOCK"] = "true"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluator import QAPipeline

pipeline = QAPipeline()
samples = [
    {"task_id": "sample-001", "response_text": "Python was created by Guido van Rossum in 1991.", "reference_context": "Python is a language created by Guido van Rossum, released in 1991.", "instruction": "Answer factually about Python history."},
    {"task_id": "sample-002", "response_text": "Studies show 94% of Fortune 500 companies use Python.", "reference_context": "Python is widely used in industry. Exact statistics vary.", "instruction": "Provide statistics on Python adoption."},
]
for s in samples:
    r = pipeline.evaluate(**s)
    print(f"\n{r.task_id} → {r.verdict} ({r.overall_score}) | escalate={r.escalate_to_human}")
    for dim, val in r.dimension_scores.items():
        print(f"  {dim}: {val['score']}")
