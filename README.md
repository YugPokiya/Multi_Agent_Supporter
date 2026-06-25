# Multi-Agent Market Regime Analyzer

Phase 1 prototype for a simple multi-agent market analyzer focused on gold and silver.

The app uses plain Python classes, a FastAPI service, an in-memory store, and a CLI workflow. It analyzes user-provided or live-fetched gold/silver market data, detects a simple regime, generates an action plan, evaluates the plan, and persists the session context to `store.json`.

## Components

- `AssessorAgent`: analyzes gold and silver inputs and detects a market regime.
- `TaskGeneratorAgent`: creates market-analysis instructions from the detected regime.
- `EvaluatorAgent`: reviews the generated instruction for quality and risk awareness.
- `SimpleOrchestrator`: coordinates the agent workflow.
- `InMemoryStore`: keeps runtime state and saves snapshots to `store.json`.
- CLI interface: runs the end-to-end market analysis loop.
- FastAPI interface: exposes health, market input, generation, and retrieve endpoints.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run CLI

```bash
python main.py
```

## Run UI

```bash
uvicorn app.main:app --reload
```

Or use the compatibility launcher:

```bash
python ui.py
```

Open http://127.0.0.1:8000 when running Uvicorn directly, or http://127.0.0.1:8080 when using `python ui.py`. Click **Generate result** to fetch latest real-world gold, silver, and dollar-index data, then run the regime analyzer. Click **Retrieve** to show the `news` string.

## FastAPI Endpoints

- `GET /health`
- `GET /api/latest-market-inputs`
- `POST /api/generate`
- `GET /api/retrieve`

## Test

```bash
python -m pytest
```

## Disclaimer

This project is for educational and portfolio purposes only. It does not provide financial advice.
