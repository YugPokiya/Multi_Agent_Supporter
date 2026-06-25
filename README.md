# Multi-Agent Market Regime Analyzer

Phase 1 prototype for a simple multi-agent market analyzer focused on gold and silver.

The app uses plain Python classes, an in-memory store, and a CLI workflow. It analyzes user-provided gold/silver market data, detects a simple regime, generates an action plan, evaluates the plan, and persists the session context to `store.json`.

## Components

- `AssessorAgent`: analyzes gold and silver inputs and detects a market regime.
- `TaskGeneratorAgent`: creates market-analysis instructions from the detected regime.
- `EvaluatorAgent`: reviews the generated instruction for quality and risk awareness.
- `SimpleOrchestrator`: coordinates the agent workflow.
- `InMemoryStore`: keeps runtime state and saves snapshots to `store.json`.
- CLI interface: runs the end-to-end market analysis loop.

## Run CLI

```bash
python main.py
```

## Run UI

```bash
python ui.py
```

Open http://127.0.0.1:8080, click **Generate result** to fetch latest real-world gold, silver, and dollar-index data, then run the regime analyzer. Click **Retrieve** to show the `news` string.

## Test

```bash
python -m pytest
```

## Disclaimer

This project is for educational and portfolio purposes only. It does not provide financial advice.
