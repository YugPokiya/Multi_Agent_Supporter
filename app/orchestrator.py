from app.agents import AssessorAgent, EvaluatorAgent, MarketInput, TaskGeneratorAgent
from app.memory import InMemoryStore


class SimpleOrchestrator:
    """Coordinates the market-analysis agents and stores generated context."""

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store = store or InMemoryStore()
        self.assessor = AssessorAgent()
        self.task_generator = TaskGeneratorAgent()
        self.evaluator = EvaluatorAgent()

    def run(self, market_input: MarketInput) -> dict:
        assessment = self.assessor.assess(market_input)
        task = self.task_generator.generate(assessment)
        evaluation = self.evaluator.evaluate(assessment, task)
        context = {
            "market_input": market_input.__dict__,
            "assessment": assessment,
            "generated_task": task,
            "evaluation": evaluation,
        }
        self.store.save_context(context)
        return context
