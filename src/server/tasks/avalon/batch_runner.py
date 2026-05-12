import asyncio
import json

from .task import AvalonBench
from .direct_session import DirectSession
from src.typings.output import TaskOutput

class AvalonBatchRunner:
    """
    Runs N Avalon games concurrently using asyncio.gather().
    AsyncLLMEngine automatically batches all concurrent LLM calls
    into single forward passes — no manual batching code needed.
    """
    def __init__(self, task: AvalonBench, concurrent_games: int = 8, log_dir: str = "logs"):
        self.task = task
        self.concurrent_games = concurrent_games
        self.log_dir = log_dir

    async def run_index(self, index: int, agent) -> TaskOutput:
        from .utils import game_logger_context
        print(f"[Game {index}] Started")
        async with game_logger_context(index, log_dir=self.log_dir):
            session = DirectSession(agent)
            result = await self.task.start_sample(index, session)
            print(f"[Game {index}] Finished (Status: {result.status})")
            return TaskOutput(index=index, status=result.status, result=result.result)

    async def run_all(self, agent, output_path: str, max_samples: int = None):
        indices = self.task.get_indices()
        if max_samples is not None:
            indices = indices[:max_samples]
        sem = asyncio.Semaphore(self.concurrent_games)

        async def run_with_sem(idx):
            async with sem:
                return await self.run_index(idx, agent)

        results = await asyncio.gather(
            *[run_with_sem(i) for i in indices],
            return_exceptions=True
        )
        valid = [r for r in results if not isinstance(r, Exception)]
        with open(output_path, 'w') as f:
            for r in valid:
                f.write(json.dumps({"index": r.index, **(r.result or {})}) + '\n')
        return self.task.calculate_overall(valid)
