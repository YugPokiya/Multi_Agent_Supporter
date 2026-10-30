# MACE Multi-Agent Market Regime Analyzer

A multi-agent decision system for **gold (XAU), silver (XAG), Bitcoin, NIFTY and WTI crude**. For each instrument:
- an **ML model** and five rule-based agents vote BUY / SELL / HOLD;
- a **LangGraph** workflow combines the votes;
- a **risk agent** sizes the trade or vetoes it;
- an **evaluator** approves the decision or downgrades it to HOLD.

Several instruments are decided in parallel, and every decision is stored with its full reasoning.

> Educational decision support only. It does not place trades and is not financial advice.

## How a decision is made

```
             market data (Yahoo / CSV)  ──►  features  ──►  ML model: P(SELL), P(HOLD), P(BUY)
                                                │
   ┌──────────────┬──────────┬──────────┬───────┴────────┬────────────┐
 ml_signal      regime    momentum   mean_reversion   macro_dxy    order_flow      ← vote in parallel
   └──────────────┴──────────┴──────────┴───────┬────────┴────────────┘
                                         aggregate   weighted score; weights depend on the regime
                                                │
                                              risk   ATR stop/target, position size, vetoes
                                                │
                                          evaluator  ≥2 agents agree, no strong opposing vote, R:R ≥ 1.5
                                                │
                                   final BUY / SELL / HOLD  →  database + API + UI
```

| Agent | Votes from | Abstains when |
|---|---|---|
| `ml_signal` | gradient-boosting model's latest probabilities for the next 5 days | the model did **not** beat the baseline on unseen data, or its top probability is < 45% |
| `regime` | price vs a rising/falling 50-day average; volatility percentile | no clear trend (range-bound) |
| `momentum` | 20-day return (in sigmas) confirmed by MACD | they disagree, or the move is small |
| `mean_reversion` | RSI ≤ 30 → BUY, ≥ 70 → SELL | RSI is neutral |
| `macro_dxy` | 20-day dollar-index change (gold and silver only) | other instruments, or the dollar is flat |
| `order_flow` | the existing 5-minute order-flow planner, when you send order-flow input | no input supplied |

**Weights by regime:**
- Trends give momentum ×1.5 and mean reversion ×0.5. Range-bound regimes do the reverse.
- High volatility raises the decision threshold by 50%.

**Risk agent vetoes when:**
- the data is stale;
- volatility is high and confidence is low;
- the ML model (if it has proven edge) gives ≥ 50% to the opposite direction.

**Sizing:** risk 1% of capital, with a stop at 2×ATR and a target at 3×ATR.

### The ML model is held to a standard

`app/ml/model.py` trains one `HistGradientBoostingClassifier` per instrument.
- **Features:** stationary inputs only (returns, volatility, RSI, MACD, distance from the 50-day average, ATR %).
- **Labels:** volatility-scaled 5-day forward moves.
- **Split:** a time-ordered split, with training labels purged so they don't overlap the test period.
- **Gate:** it is scored on the last 25% of history against two baselines: always predicting the majority class, and always predicting the class frequencies. It is marked `has_edge` only if it beats both. Otherwise the ML agent reports its probabilities but **does not vote**.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # open http://127.0.0.1:8000 → Decision Board
```

```bash
python -m app.cli train    XAU XAG BTC NIFTY CRUDE   # train and show hold-out quality
python -m app.cli decide   XAU XAG BTC NIFTY CRUDE   # one decision per instrument, with every vote
python -m app.cli evaluate XAU                        # replay the whole agent graph over the hold-out period
python -m app.cli decide XAU --source synthetic       # offline demo on random data
```

Models train automatically on first use. They retrain when they are more than 7 days behind the data.

| Setting (env var) | Default | |
|---|---|---|
| `DATA_SOURCE` | `yahoo` | `yahoo` (internet), `csv` (`DATA_DIR/<INSTRUMENT>.csv`), `synthetic` (tests only) |
| `DATABASE_URL` | `sqlite:///decisions.db` | e.g. `postgresql+psycopg://user:pw@host/db` (install the driver) |
| `MODEL_DIR` | `models` | trained models (`<INSTRUMENT>.joblib`) |
| `HORIZON_DAYS` | `5` | prediction and holding horizon |
| `RISK_CAPITAL`, `RISK_PCT` | `100000`, `0.01` | position sizing |
| `COST_BPS` | `5` | per-side cost used by `evaluate` |

## API

| Method | Path | |
|---|---|---|
| POST | `/api/decisions` | `{"instruments": ["XAU","BTC"], "orderflow": {"XAU": {...}}}` → one decision per instrument |
| GET | `/api/decisions/history?instrument=XAU` | stored decisions |
| GET | `/api/decisions/evaluate/{instrument}` | hold-out backtest with a per-agent scorecard |
| POST | `/api/models/train` · GET `/api/models` | train, and list model quality |
| POST | `/api/orderflow-plan` · `/api/generate` | the original order-flow planner and gold/silver regime report |

## Backtest (`evaluate`)

`evaluate` replays the whole graph on the hold-out period, using the model trained only on earlier data.
- **Trading rule:** one decision every 5 days, held for 5 days, with round-trip costs deducted.
- **Output:** hit rate, average trade, total return vs buy-and-hold, Sharpe ratio, and a **scorecard per agent** showing which agents actually helped.
- **Limitation:** stops and targets are not simulated.

## Tests

```bash
python -m pytest -q     # 17 tests, offline (synthetic data, temp SQLite)
```

**What they cover:**
- the features and the model don't peek at future bars;
- each agent's rules;
- regime weighting, risk vetoes and position sizing, and the evaluator's checks;
- a failing data feed for one instrument doesn't break the others;
- the API end to end.

CI runs them on every push.

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
