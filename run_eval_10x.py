import asyncio
import argparse
import os
from src.server.tasks.avalon.task import AvalonBench
from src.server.tasks.avalon.batch_runner import AvalonBatchRunner
from src.server.tasks.avalon.long_term_memory import LongTermMemory
from src.client.agents.vllm_agent import VLLMAgentClient


def parse_args():
    parser = argparse.ArgumentParser(description="Run Avalon LLM batch evaluation.")
    parser.add_argument(
        "--use-reputation-memory",
        action="store_true",
        default=False,
        help="Enable ReputationMemory tracking for Player 0.",
    )
    parser.add_argument(
        "--use-bayesian-prediction",
        action="store_true",
        default=False,
        help="Enable Bayesian-like reasoning mode for periodic predictions.",
    )
    parser.add_argument(
        "--use-long-term-memory",
        action="store_true",
        default=False,
        help="Enable LongTermMemory tracking and cross-game learning.",
    )
    parser.add_argument(
        "--ltm-agents",
        type=int,
        nargs="+",
        default=[0],
        help="Player IDs to equip with Long-Term Memory (default: 0). e.g. --ltm-agents 0 1",
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
        default="(10x)",
        help="Suffix appended to the output file name.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=10,
        help="Number of times to repeat the dataset.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature. Must be > 0.0 for non-deterministic results across repeats.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for slicing the dataset.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for slicing the dataset.",
    )
    parser.add_argument(
        "--no-memory-snapshot",
        action="store_true",
        default=False,
        help="Disable in-game memory snapshot logging for all players.",
    )
    parser.add_argument(
        "--no-periodic-prediction",
        action="store_true",
        default=False,
        help="Disable periodic good/evil and Merlin prediction calls.",
    )
    parser.add_argument(
        "--personality-list",
        type=str,
        nargs=5,
        default=["default"] * 5,
        metavar=("P0", "P1", "P2", "P3", "P4"),
        help="Per-player personality: naive, deceptive, or default. "
             "E.g. --personality-list default default naive deceptive naive",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    agent = VLLMAgentClient(
        model_name=args.model,
        max_tokens=1024,
        temperature=args.temperature,
        tensor_parallel_size=1,
    )

    data_file = args.data_file
    data_name = os.path.splitext(os.path.basename(data_file))[0]

    # Build per-agent LTM dict (each agent gets its own independent LongTermMemory)
    if args.use_long_term_memory:
        long_term_memories = {pid: LongTermMemory() for pid in args.ltm_agents}
    else:
        long_term_memories = {}

    task = AvalonBench(
        num_players=5,
        agent_list=["llm", "llm", "llm", "llm", "llm"],
        discussion=True,
        data_file=data_file,
        use_reputation_memory=args.use_reputation_memory,
        long_term_memories=long_term_memories,
        num_repeats=args.num_repeats,
        use_bayesian_prediction=args.use_bayesian_prediction,
        log_memory_snapshots_for=[] if args.no_memory_snapshot else None,
        predict_for=[] if args.no_periodic_prediction else None,
        personality_list=args.personality_list,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    base_suffix = args.output_suffix

    # Append tags to log/output names
    rep_tag = "_rep" if args.use_reputation_memory else ""
    bayes_tag = "_bayes" if args.use_bayesian_prediction else ""
    ltm_tag = "_ltm" if args.use_long_term_memory else ""
    multi_tag = "_multi" if (args.use_long_term_memory and len(args.ltm_agents) > 1) else ""
    personality_tag = "_" + "-".join(args.personality_list) if any(p != "default" for p in args.personality_list) else ""
    
    suffix = base_suffix
    log_dir = f"logs/{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{multi_tag}{personality_tag}"
    output_path = f"outputs/results_{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{multi_tag}{personality_tag}.jsonl"

    # Auto-increment suffix to prevent overwriting
    counter = 1
    while os.path.exists(output_path) or os.path.exists(log_dir):
        suffix = f"{base_suffix}_v{counter}"
        log_dir = f"logs/freedom_{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{multi_tag}{personality_tag}"
        output_path = f"outputs/results_freedom_{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{multi_tag}{personality_tag}.jsonl"
        counter += 1

    os.makedirs("outputs", exist_ok=True)
    runner = AvalonBatchRunner(task, concurrent_games=args.concurrent_games, log_dir=log_dir)

    results = await runner.run_all(agent, output_path=output_path)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
