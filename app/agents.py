from dataclasses import dataclass


@dataclass(frozen=True)
class MarketInput:
    gold_price: float
    gold_change_pct: float
    silver_price: float
    silver_change_pct: float
    dollar_index_change_pct: float
    volatility_score: int


class AssessorAgent:
    """Detects a simple gold/silver market regime from user-provided data."""

    def assess(self, market: MarketInput) -> dict:
        metals_avg_change = (market.gold_change_pct + market.silver_change_pct) / 2
        gold_silver_spread = market.gold_change_pct - market.silver_change_pct

        if market.volatility_score >= 8:
            regime = "high_volatility"
            confidence = 0.78
            reason = "Volatility score is elevated, so risk control is the priority."
        elif metals_avg_change > 0.6 and market.dollar_index_change_pct < 0:
            regime = "bullish_precious_metals"
            confidence = 0.74
            reason = "Gold and silver are rising while the dollar index is weakening."
        elif metals_avg_change < -0.6 and market.dollar_index_change_pct > 0:
            regime = "bearish_precious_metals"
            confidence = 0.74
            reason = "Gold and silver are falling while the dollar index is strengthening."
        elif abs(gold_silver_spread) > 1.0:
            regime = "divergence"
            confidence = 0.66
            reason = "Gold and silver are moving differently, showing possible divergence."
        else:
            regime = "range_bound"
            confidence = 0.62
            reason = "Signals are mixed or mild, suggesting no strong directional regime."

        return {
            "agent": "AssessorAgent",
            "regime": regime,
            "confidence": confidence,
            "reason": reason,
            "features": {
                "metals_avg_change_pct": round(metals_avg_change, 3),
                "gold_silver_spread_pct": round(gold_silver_spread, 3),
                "dollar_index_change_pct": market.dollar_index_change_pct,
                "volatility_score": market.volatility_score,
            },
        }


class TaskGeneratorAgent:
    """Creates analysis instructions based on the detected market regime."""

    def generate(self, assessment: dict) -> dict:
        regime = assessment["regime"]
        instructions_by_regime = {
            "bullish_precious_metals": [
                "Look for confirmation from higher highs in both gold and silver.",
                "Check whether dollar weakness continues before considering bullish bias.",
                "Use tight risk limits because precious metals can reverse quickly.",
            ],
            "bearish_precious_metals": [
                "Check whether the dollar index strength is broad-based.",
                "Avoid long bias until gold or silver shows a recovery signal.",
                "Watch support zones and volume before forming a bearish continuation view.",
            ],
            "high_volatility": [
                "Reduce position-size assumptions in the analysis.",
                "Prioritize risk management and wait for confirmation candles.",
                "Flag the market as unstable for beginner traders.",
            ],
            "divergence": [
                "Compare gold and silver relative strength before making a directional call.",
                "Avoid overconfident conclusions until both metals align.",
                "Track whether silver is leading risk-on or gold is leading safe-haven demand.",
            ],
            "range_bound": [
                "Mark the regime as neutral until a breakout appears.",
                "Focus on support, resistance, and mean-reversion behavior.",
                "Avoid strong buy/sell language without confirmation.",
            ],
        }
        return {
            "agent": "TaskGeneratorAgent",
            "regime": regime,
            "instruction": " ".join(instructions_by_regime[regime]),
            "risk_note": "Educational analysis only; this is not financial advice.",
        }


class EvaluatorAgent:
    """Evaluates generated instructions for clarity, regime fit, and risk awareness."""

    def evaluate(self, assessment: dict, task: dict) -> dict:
        instruction = task["instruction"].lower()
        score = 0
        checks = {
            "mentions_regime_logic": assessment["regime"].replace("_", " ") in instruction
            or assessment["reason"].split(",")[0].lower() in instruction,
            "contains_risk_control": "risk" in instruction or "position-size" in instruction,
            "avoids_financial_advice": "not financial advice" in task["risk_note"].lower(),
        }
        score += 4 if checks["contains_risk_control"] else 0
        score += 3 if checks["avoids_financial_advice"] else 0
        score += 3 if len(task["instruction"]) > 80 else 0

        if score >= 8:
            verdict = "approved"
        elif score >= 5:
            verdict = "needs_revision"
        else:
            verdict = "rejected"

        return {
            "agent": "EvaluatorAgent",
            "score": score,
            "max_score": 10,
            "verdict": verdict,
            "checks": checks,
            "feedback": "Instruction is suitable for beginner market-analysis practice."
            if verdict == "approved"
            else "Instruction needs clearer risk controls and more context.",
        }
