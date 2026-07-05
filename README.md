# Multi-Agent Market Regime Analyzer

Phase 1 prototype for a simple multi-agent market analyzer focused on gold and silver, with an added 5-minute order-flow training planner for XAU, XAG, BTC, and NIFTY.

The app uses plain Python classes, a FastAPI service, an in-memory store, and a CLI workflow. It analyzes user-provided or live-fetched gold/silver market data, detects a simple regime, generates an action plan, evaluates the plan, and persists the session context to `store.json`.

## Components

- `AssessorAgent`: analyzes gold and silver inputs and detects a market regime.
- `TaskGeneratorAgent`: creates market-analysis instructions from the detected regime.
- `EvaluatorAgent`: reviews the generated instruction for quality and risk awareness.
- `SimpleOrchestrator`: coordinates the agent workflow.
- `InMemoryStore`: keeps runtime state and saves snapshots to `store.json`.
- CLI interface: runs the end-to-end market analysis loop.
- FastAPI interface: exposes health, market input, generation, retrieve, and 5-minute order-flow plan endpoints.

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
- `POST /api/orderflow-plan`

## Test

```bash
python -m pytest
```

## Disclaimer

This project is for educational and portfolio purposes only. It does not provide financial advice.


## 5-Minute Order-Flow Training Algo

Use `POST /api/orderflow-plan` to generate an educational, rules-based plan for `XAU`, `XAG`, `BTC`, or `NIFTY` on the 5-minute timeframe. The planner scores:

- liquidity sweeps and reclaim/rejection behavior
- market-structure break direction
- cumulative delta and latest delta change
- bid/ask imbalance
- absorption and trapped-trader evidence
- relative volume

Example request:

```json
{
  "instrument": "XAU",
  "last_price": 2350.5,
  "session": "new_york",
  "cumulative_delta": 1200,
  "delta_change": 450,
  "volume_ratio": 1.4,
  "bid_ask_imbalance": 0.22,
  "absorption_score": 8,
  "trapped_trader_score": 7,
  "structure_break": "long",
  "liquidity_sweep": "long",
  "atr_5m": 3.2
}
```

The response includes directional bias, confidence, entry trigger, invalidation, risk plan, targets, management rules, and score breakdown. It is for order-flow training and backtesting only, not financial advice.
