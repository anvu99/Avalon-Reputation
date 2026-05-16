import asyncio
import argparse
import os
from src.server.tasks.avalon.task import AvalonBench
from src.server.tasks.avalon.batch_runner import AvalonBatchRunner
from src.client.agents.vllm_agent import VLLMAgentClient


def parse_args():
    parser = argparse.ArgumentParser(description="Run Avalon LLM evaluation.")
    parser.add_argument(
        "--use-reputation-memory",
        action="store_true",
        default=False,
        help="Enable ReputationMemory tracking for Player 0 (LLM agent). "
             "When set, Player 0 will maintain a structured memory of peer "
             "observations updated once per round.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/avalon/all_dev.json",
        help="Path to the evaluation data file.",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=64,
        help="Number of games to run concurrently.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="vLLM model name to use for all agents.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="(4)",
        help="Suffix appended to the output file name, e.g. '(3)'.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    agent = VLLMAgentClient(
        model_name=args.model,
        max_tokens=1024,
        temperature=0.0,
        tensor_parallel_size=1,
    )

    data_file = args.data_file
    data_name = os.path.splitext(os.path.basename(data_file))[0]

    task = AvalonBench(
        num_players=5,
        agent_list=["llm", "llm", "llm", "llm", "llm"],
        discussion=True,
        data_file=data_file,
        use_reputation_memory=args.use_reputation_memory,
    )

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    base_suffix = args.output_suffix

    # Append '_rep' to log/output names when reputation memory is active
    rep_tag = "_rep" if args.use_reputation_memory else ""
    
    suffix = base_suffix
    log_dir = f"logs/{data_name}_{job_id}_{suffix}{rep_tag}"
    output_path = f"outputs/results_{data_name}_{job_id}_{suffix}{rep_tag}.jsonl"

    # Auto-increment suffix to prevent overwriting
    counter = 1
    while os.path.exists(output_path) or os.path.exists(log_dir):
        suffix = f"{base_suffix}_v{counter}"
        log_dir = f"logs/freedom_{data_name}_{job_id}_{suffix}{rep_tag}"
        output_path = f"outputs/results_freedom_{data_name}_{job_id}_{suffix}{rep_tag}.jsonl"
        counter += 1

    os.makedirs("outputs", exist_ok=True)
    runner = AvalonBatchRunner(task, concurrent_games=args.concurrent_games, log_dir=log_dir)

    results = await runner.run_all(agent, output_path=output_path)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
