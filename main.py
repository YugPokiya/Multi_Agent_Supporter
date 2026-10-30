from app.agents import MarketInput
from app.orchestrator import SimpleOrchestrator


def ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Please enter a valid number.")


def ask_int(prompt: str, minimum: int = 1, maximum: int = 10) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Please enter a value from {minimum} to {maximum}.")


def main() -> None:
    print("MACE Phase 1: Gold/Silver Market Regime Analyzer")
    print("Educational analysis only. This is not financial advice.\n")

    market_input = MarketInput(
        gold_price=ask_float("Gold price: "),
        gold_change_pct=ask_float("Gold daily change %: "),
        silver_price=ask_float("Silver price: "),
        silver_change_pct=ask_float("Silver daily change %: "),
        dollar_index_change_pct=ask_float("Dollar index daily change %: "),
        volatility_score=ask_int("Volatility score 1-10: "),
    )

    context = SimpleOrchestrator().run(market_input)
    assessment = context["assessment"]
    task = context["generated_task"]
    evaluation = context["evaluation"]

    print("\n--- Regime Detection ---")
    print(f"Regime: {assessment['regime']}")
    print(f"Confidence: {assessment['confidence']}")
    print(f"Reason: {assessment['reason']}")

    print("\n--- Generated Instruction ---")
    print(task["instruction"])
    print(task["risk_note"])

    print("\n--- Evaluator Result ---")
    print(f"Score: {evaluation['score']}/{evaluation['max_score']}")
    print(f"Verdict: {evaluation['verdict']}")
    print(f"Feedback: {evaluation['feedback']}")
    print("\nContext saved to store.json")


if __name__ == "__main__":
    main()
