from app.orderflow import OrderFlowInput, build_orderflow_plan


def test_orderflow_plan_builds_long_bias_for_aligned_reclaim():
    plan = build_orderflow_plan(
        OrderFlowInput(
            instrument="XAU",
            last_price=2350.5,
            session="new_york",
            cumulative_delta=1200,
            delta_change=450,
            volume_ratio=1.4,
            bid_ask_imbalance=0.22,
            absorption_score=6,
            trapped_trader_score=8,
            structure_break="long",
            liquidity_sweep="long",
            atr_5m=3.2,
        )
    )

    assert plan.timeframe == "5m"
    assert plan.bias == "long"
    assert plan.instrument == "XAU"
    assert "not financial advice" in plan.disclaimer


def test_orderflow_plan_stays_neutral_when_signals_conflict():
    plan = build_orderflow_plan(
        OrderFlowInput(
            instrument="BTC",
            last_price=65000,
            session="crypto_24x7",
            cumulative_delta=100,
            delta_change=-50,
            volume_ratio=0.8,
            bid_ask_imbalance=0.02,
            absorption_score=3,
            trapped_trader_score=2,
            structure_break="neutral",
            liquidity_sweep="neutral",
            atr_5m=120,
        )
    )

    assert plan.bias == "neutral"
    assert "No-trade" in plan.setup_name
