"""
FastAPI endpoints for the AI QA Pipeline.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import csv
import io

from src.evaluator import QAPipeline

app = FastAPI(title="AI QA Pipeline", version="1.0.0")
pipeline = QAPipeline()
_results_store: dict = {}


class EvaluateRequest(BaseModel):
    response_text: str
    task_id: str
    reference_context: Optional[str] = ""
    instruction: Optional[str] = ""


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    result = pipeline.evaluate(
        response_text=req.response_text,
        task_id=req.task_id,
        reference_context=req.reference_context or "",
        instruction=req.instruction or "",
    )
    _results_store[req.task_id] = result
    return result


@app.post("/evaluate/batch")
def evaluate_batch(requests: List[EvaluateRequest]):
    results = []
    for req in requests:
        result = pipeline.evaluate(
            response_text=req.response_text,
            task_id=req.task_id,
            reference_context=req.reference_context or "",
            instruction=req.instruction or "",
        )
        _results_store[req.task_id] = result
        results.append(result)
    return results


@app.get("/export/csv")
def export_csv():
    if not _results_store:
        raise HTTPException(status_code=404, detail="No results to export")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["task_id", "overall_score", "verdict", "escalate_to_human",
                     "factual_accuracy", "coherence", "instruction_following", "safety", "processing_time_ms"])
    for r in _results_store.values():
        ds = r.dimension_scores
        writer.writerow([
            r.task_id, r.overall_score, r.verdict, r.escalate_to_human,
            ds.get("factual_accuracy", {}).get("score", ""),
            ds.get("coherence", {}).get("score", ""),
            ds.get("instruction_following", {}).get("score", ""),
            ds.get("safety", {}).get("score", ""),
            r.processing_time_ms,
        ])
    return output.getvalue()
