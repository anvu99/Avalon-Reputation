import sys
import asyncio
import json
import os

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

        # Clear the output file first
        with open(output_path, 'w') as f:
            pass

        per_batch_metrics = None

        # Force sequential batch execution if LTM or Public Reputation is enabled
        if getattr(self.task, 'long_term_memories', None) or getattr(self.task, 'use_public_reputation', False):
            results, per_batch_metrics = await self._run_batched(indices, agent, output_path)
        else:
            sem = asyncio.Semaphore(self.concurrent_games)

            async def run_with_sem(idx):
                async with sem:
                    res = await self.run_index(idx, agent)
                    if not isinstance(res, Exception):
                        with open(output_path, 'a') as f:
                            f.write(json.dumps({"index": res.index, **(res.result or {})}) + '\n')
                    return res

            results = await asyncio.gather(
                *[run_with_sem(i) for i in indices],
                return_exceptions=True
            )

        valid = [r for r in results if not isinstance(r, Exception)]
        summary = self.task.calculate_overall(valid, per_batch_metrics=per_batch_metrics)

        # Print a human-readable summary to stdout
        _print_summary(summary)

        return summary

    async def _run_batched(self, indices, agent, output_path):
        """Run games in sequential chunks (required for LTM), writing results and
        per-batch metrics incrementally to disk after every chunk."""

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        base, ext = os.path.splitext(output_path)
        metrics_path = base + '_batch_metrics' + (ext or '.jsonl')
        # Clear the batch metrics file
        with open(metrics_path, 'w') as f:
            pass

        all_results = []
        per_batch_metrics = []

        for batch_num, chunk in enumerate(chunks(indices, self.concurrent_games)):
            # Record the LTM size entering this batch for each tracked agent
            ltms = self.task.long_term_memories
            # Use Player 0's LTM size for the batch metrics (backward compat)
            ltm_p0 = ltms.get(0)
            ltm_size_entering = len(ltm_p0.memory_text) if (ltm_p0 and ltm_p0.memory_text) else 0
            # Log sizes for all tracked agents
            ltm_size_str = ", ".join(
                f"P{pid}={len(ltm.memory_text) if ltm.memory_text else 0}c"
                for pid, ltm in sorted(ltms.items())
            )

            chunk_results = await asyncio.gather(
                *[self.run_index(i, agent) for i in chunk],
                return_exceptions=True,
            )
            all_results.extend(chunk_results)
            for r in chunk_results:
                if isinstance(r, Exception):
                    import traceback
                    print('CAUGHT EXCEPTION:', r, file=sys.stderr)
                    traceback.print_exception(type(r), r, r.__traceback__, file=sys.stderr)

            # Write chunk results to main output immediately (with batch_num tag)
            valid_chunk = [r for r in chunk_results if not isinstance(r, Exception)]
            with open(output_path, 'a') as f:
                for res in valid_chunk:
                    f.write(json.dumps({"index": res.index, "batch_num": batch_num, **(res.result or {})}) + '\n')

            # Compute and persist per-batch metrics before synthesizing memory
            batch_metrics = self.task.compute_batch_metrics(
                valid_chunk, batch_num=batch_num, ltm_size_chars=ltm_size_entering
            )
            per_batch_metrics.append(batch_metrics)

            with open(metrics_path, 'a') as f:
                f.write(json.dumps(batch_metrics) + '\n')

            print(
                f"[Batch {batch_num}] "
                f"Games={batch_metrics['n_valid_games']} | "
                f"LTM={ltm_size_str} | "
                f"WinRate={batch_metrics['win_rate']:.3f} | "
                f"DeducAcc={batch_metrics['avg_deduction_acc']:.3f} | "
                f"MerlinDet={batch_metrics.get('merlin_detection_acc')}"
            )

            for pid, ltm in self.task.long_term_memories.items():
                await self._synthesize_memory(agent, ltm, n_games=len(chunk), player_id=pid)

        return all_results, per_batch_metrics

    async def _synthesize_memory(self, agent, ltm, n_games: int, player_id: int = 0):
        from .prompts import LONG_TERM_SYNTHESIS_PROMPT, LONG_TERM_SYNTHESIS_PROMPT_COUNTER_NORM
        from .utils import get_game_logger

        if not ltm.pending_lessons:
            return

        lessons = "\n\n---\n\n".join(ltm.pending_lessons)
        current_memory = ltm.memory_text if ltm.memory_text else "(No memory yet)"

        num_players = getattr(self.task, 'num_players', 5)
        other_pids = [i for i in range(num_players) if i != player_id]
        other_player_ids = ", ".join(f"Player {i}" for i in other_pids)

        base_prompt = LONG_TERM_SYNTHESIS_PROMPT_COUNTER_NORM if getattr(self.task, 'ltm_counter_norm', False) else LONG_TERM_SYNTHESIS_PROMPT
        prompt = base_prompt.format(
            n=n_games,
            current_memory=current_memory,
            lessons=lessons,
            player_id=player_id,
            other_player_ids=other_player_ids
        )

        session = DirectSession(agent)
        session.inject({"role": "user", "content": prompt})

        try:
            response = await session.action()
            new_memory = response.content if response.content else ""
            ltm.memory_text = new_memory.strip()
            get_game_logger().info(f"##### [LTM Synthesis for Player {player_id}] #####\n{ltm.memory_text}")
        except Exception as e:
            get_game_logger().warning(f"[LTM Synthesis Player {player_id}] LLM call failed: {e}")
        finally:
            ltm.pending_lessons.clear()


def _print_summary(summary: dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print("=== EVALUATION SUMMARY ===")
    print("=" * 60)
    print(f"  Valid games       : {summary.get('n_valid_games', '?')}")
    print(f"  Win rate (overall): {summary.get('win_rate', '?'):.4f}")
    print(f"  Win rate as Good  : {summary.get('win_rate_as_good', '?')}")
    print(f"  Win rate as Evil  : {summary.get('win_rate_as_evil', '?')}")
    print(f"  Avg deduction acc : {summary.get('avg_deduction_acc', '?'):.4f}")
    print(f"  Merlin detect acc : {summary.get('merlin_detection_acc', '?')}")

    curve = summary.get("per_batch_learning_curve")
    if curve:
        print("\n--- Per-Batch Learning Curve (LTM mode) ---")
        print(f"  {'Batch':>5}  {'LTM_in':>8}  {'Games':>6}  {'WinRate':>8}  {'DeducAcc':>9}  {'MerlinDet':>10}  {'WinGood':>8}  {'WinEvil':>8}")
        print("  " + "-" * 72)
        for m in curve:
            merlin_str = f"{m['merlin_detection_acc']:.3f}" if m.get('merlin_detection_acc') is not None else "  N/A "
            good_str   = f"{m['win_rate_as_good']:.3f}"    if m.get('win_rate_as_good')    is not None else "  N/A "
            evil_str   = f"{m['win_rate_as_evil']:.3f}"    if m.get('win_rate_as_evil')    is not None else "  N/A "
            print(
                f"  {m['batch_num']:>5}  "
                f"{m['ltm_size_chars']:>8}  "
                f"{m['n_valid_games']:>6}  "
                f"{m['win_rate']:>8.3f}  "
                f"{m['avg_deduction_acc']:>9.3f}  "
                f"{merlin_str:>10}  "
                f"{good_str:>8}  "
                f"{evil_str:>8}"
            )
    print("=" * 60 + "\n")
