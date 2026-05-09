# Automated AI QA Pipeline

Multi-dimensional LLM output evaluation pipeline — built for scale. Scores model responses across **factual accuracy, coherence, instruction-following, and safety** using a weighted aggregation layer on top of LangChain retrieval, LLM-as-judge, and rule-based signals.

Built from real evaluation work — the core problem: manual QA of LLM outputs at scale is slow, inconsistent, and expensive. This pipeline replaces manual review for clear-cut cases and routes ambiguous ones to human reviewers automatically.

---

## Architecture

```
LLM Response(s)
      │
      ▼
┌─────────────────────┐
│   Input Validator   │  ← schema check, dedup, batch chunking
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Multi-Dimensional Scorer         │
│                                          │
│  ┌──────────────────┐  weight: 2.0       │
│  │ Factual Accuracy │  ← LangChain +     │
│  │                  │    FAISS retrieval │
│  └──────────────────┘                    │
│  ┌──────────────────┐  weight: 1.5       │
│  │   Coherence      │  ← LLM-as-judge   │
│  └──────────────────┘                    │
│  ┌──────────────────┐  weight: 1.5       │
│  │  Instruction     │  ← Rubric match   │
│  │  Following       │                    │
│  └──────────────────┘                    │
│  ┌──────────────────┐  weight: 2.0       │
│  │ Safety / Policy  │  ← Rule-based     │
│  └──────────────────┘                    │
└────────┬─────────────────────────────────┘
         │ dimension_scores[]
         ▼
┌─────────────────────┐
│  Score Aggregator   │  ← weighted average + confidence
└────────┬────────────┘
         │
         ▼
  PASS / FAIL / ESCALATE + JSON report + CSV export
```

**Why four dimensions instead of a single score?**

A single quality score hides which dimension failed. A response can be factually correct but completely miss the instruction. Separate dimension scores make reviewer escalation faster — they know exactly what to fix, not just that something failed.

---

## Quickstart

### Run with Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/Rdev-ops-lab/ai-qa-pipeline
cd ai-qa-pipeline

# Copy env file
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Start the API
docker-compose up qa-pipeline

# Or run in mock mode (no API key needed)
docker-compose --profile mock up qa-pipeline-mock
```

### Run locally

```bash
pip install -r requirements.txt

# Full mode
OPENAI_API_KEY=your_key uvicorn api.main:app --reload

# Mock mode (no API key)
USE_MOCK=true uvicorn api.main:app --reload
```

---

## API Usage

### Score a single response

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "response_text": "Python was created by Guido van Rossum and first released in 1991.",
    "reference_context": "Python is a programming language created by Guido van Rossum, released in 1991.",
    "instruction": "Answer factually about Python history.",
    "task_id": "qa-001"
  }'
```

**Response:**

```json
{
  "task_id": "qa-001",
  "overall_score": 0.91,
  "verdict": "PASS",
  "escalate_to_human": false,
  "dimension_scores": {
    "factual_accuracy":      { "score": 0.95, "weight": 2.0 },
    "coherence":             { "score": 0.92, "weight": 1.5 },
    "instruction_following": { "score": 0.90, "weight": 1.5 },
    "safety":                { "score": 1.00, "weight": 2.0 }
  },
  "processing_time_ms": 1243.7
}
```

### Batch evaluation

```bash
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"response_text": "...", "task_id": "001"},
    {"response_text": "...", "task_id": "002"}
  ]'
```

### Export results as CSV

```bash
curl http://localhost:8000/export/csv -o results.csv
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Run Tests

```bash
# All tests in mock mode (no API key needed)
USE_MOCK=true pytest tests/ -v

# With coverage
USE_MOCK=true pytest tests/ --cov=src --cov-report=term-missing
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required for full mode) |
| `USE_MOCK` | `false` | Run without API calls (testing / CI) |
| `PASS_THRESHOLD` | `0.75` | Weighted score above which a response auto-passes |
| `ESCALATE_THRESHOLD` | `0.50` | Below this threshold → routed to human review |
| `BATCH_SIZE` | `50` | Responses per batch chunk |
| `PORT` | `8000` | API server port |

---

## How Each Scorer Works

**Factual Accuracy (weight: 2.0)**
Splits the reference context into chunks, builds a FAISS vector index using OpenAI embeddings, retrieves the top-k most relevant chunks for the response, and computes a grounding score from cosine distances. In mock mode, falls back to token overlap between response and reference.

**Coherence (weight: 1.5)**
Uses LLM-as-judge — sends the response to `gpt-4o-mini` with a prompt asking for a 0.0–1.0 coherence rating. In mock mode, uses response length as a heuristic proxy.

**Instruction Following (weight: 1.5)**
Matches the response against the provided instruction using an LLM judge. Falls back to a keyword-overlap heuristic in mock mode. Rubric criteria are loaded from `rubrics/default_rubric.json` and can be customized per task type.

**Safety (weight: 2.0)**
Rule-based regex filter — no API calls. Scans for patterns covering dangerous instructions, PII leakage, and targeted harm language. Returns `1.0` (safe) or `0.0` (violation). A single safety violation can push any response below the escalation threshold regardless of other scores.

---

## Key Design Decisions

**Multi-dimensional over single score** — Each dimension is scored and weighted independently. Safety and factual accuracy carry higher weight (2.0) than coherence and instruction-following (1.5) because errors there cause more downstream damage.

**Three-tier verdict** — `PASS` (≥ 0.75), `ESCALATE` (0.50–0.75), `FAIL` (< 0.50). The pipeline is a filter, not a final judge. Ambiguous cases are routed to human review; clear passes and clear fails are handled automatically.

**Mock mode** — Full test coverage and CI/CD runs entirely without API keys. Deterministic mock responses allow reliable unit and integration tests. Real integration tests run separately with a dedicated API key.

**CSV export** — Results accumulate in memory per session and can be exported as a flat CSV for downstream analysis, dashboards, or human reviewer queues.

---

## Project Structure

```
ai-qa-pipeline/
├── src/
│   ├── evaluator.py           # Main pipeline — orchestrates all scorers
│   ├── factual_scorer.py      # LangChain + FAISS retrieval grounding
│   ├── coherence_scorer.py    # LLM-as-judge coherence scoring
│   ├── rubric_matcher.py      # Instruction-following rubric evaluation
│   ├── safety_checker.py      # Rule-based safety and policy filter
│   └── aggregator.py          # Weighted score aggregation
├── api/
│   └── main.py                # FastAPI endpoints
├── tests/
│   └── test_pipeline.py
├── rubrics/
│   └── default_rubric.json    # Configurable dimension weights + thresholds
├── examples/
│   └── sample_usage.py
├── outputs/                   # CSV exports (gitignored)
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Related Projects

- [LLM Hallucination Detector](https://github.com/Rdev-ops-lab/llm-hallucination-detector) — Claim-level hallucination detection using LangChain + Bayesian inference. Goes deeper on factual grounding: decomposes responses into atomic claims and scores each independently rather than scoring the response as a whole.

---

## Author

**Rishi Pal Singh** — AI Evaluation Specialist  
[LinkedIn](https://linkedin.com/in/rishi-singh-1413b3384) · [GitHub](https://github.com/Rdev-ops-lab)
