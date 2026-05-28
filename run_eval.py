import asyncio
import argparse
import os
from src.server.tasks.avalon.task import AvalonBench
from src.server.tasks.avalon.batch_runner import AvalonBatchRunner
from src.server.tasks.avalon.long_term_memory import LongTermMemory
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
        "--use-bayesian-prediction",
        action="store_true",
        default=False,
        help="Enable Bayesian-like reasoning mode for periodic predictions.",
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
    parser.add_argument(
        "--no-memory-snapshot",
        action="store_true",
        default=False,
        help="Disable in-game memory snapshot logging and summarization for all players.",
    )
    parser.add_argument(
        "--no-periodic-prediction",
        action="store_true",
        default=False,
        help="Disable periodic good/evil and Merlin prediction calls for all players.",
    )
    parser.add_argument(
        "--personality-list",
        type=str,
        nargs=5,
        default=["default"] * 5,
        metavar=("P0", "P1", "P2", "P3", "P4"),
        help="Per-player personality, one of: naive, deceptive, default. "
             "Provide exactly 5 values, e.g.: --personality-list deceptive naive default default default",
    )
    parser.add_argument(
        "--ltm-agents",
        type=int,
        nargs="+",
        default=[],
        metavar="PLAYER_ID",
        help="Player IDs that should use Long-Term Memory (LTM). "
             "Games are run in batches; memory is synthesized after each batch. "
             "E.g. --ltm-agents 0 1",
    )
    parser.add_argument(
        "--use-discrete-rating",
        action="store_true",
        default=False,
        help="Use discrete 1-5 Likert scale ratings instead of continuous probabilities for beliefs.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallel execution (vLLM).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    agent = VLLMAgentClient(
        model_name=args.model,
        max_tokens=1024,
        temperature=0.0,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=16384,
        disable_custom_all_reduce=True,
    )

    data_file = args.data_file
    data_name = os.path.splitext(os.path.basename(data_file))[0]

    # Build LTM objects for each requested agent
    long_term_memories = {pid: LongTermMemory() for pid in args.ltm_agents}

    task = AvalonBench(
        num_players=5,
        agent_list=["llm", "llm", "llm", "llm", "llm"],
        discussion=True,
        data_file=data_file,
        use_reputation_memory=args.use_reputation_memory,
        use_discrete_rating=args.use_discrete_rating,  # NEW
        use_bayesian_prediction=args.use_bayesian_prediction,
        log_memory_snapshots_for=[] if args.no_memory_snapshot else None,
        predict_for=[] if args.no_periodic_prediction else None,
        personality_list=args.personality_list,
        long_term_memories=long_term_memories if long_term_memories else None,
    )

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    base_suffix = args.output_suffix

    # Append tags to log/output names
    rep_tag = "_rep" if args.use_reputation_memory else ""
    bayes_tag = "_bayes" if args.use_bayesian_prediction else ""
    ltm_tag = "_ltm" + "-".join(str(p) for p in sorted(args.ltm_agents)) if args.ltm_agents else ""
    discrete_tag = "_discrete" if args.use_discrete_rating else ""
    personality_tag = "_" + "-".join(args.personality_list) if any(p != "default" for p in args.personality_list) else ""
    
    suffix = base_suffix
    log_dir = f"logs/{data_name}/{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{discrete_tag}{personality_tag}"
    output_path = f"outputs/results_{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{discrete_tag}{personality_tag}.jsonl"

    # Auto-increment suffix to prevent overwriting
    counter = 1
    while os.path.exists(output_path) or os.path.exists(log_dir):
        suffix = f"{base_suffix}_v{counter}"
        log_dir = f"logs/{data_name}/{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{discrete_tag}{personality_tag}"
        output_path = f"outputs/results_{data_name}_{job_id}_{suffix}{rep_tag}{bayes_tag}{ltm_tag}{discrete_tag}{personality_tag}.jsonl"
        counter += 1

    os.makedirs("outputs", exist_ok=True)
    runner = AvalonBatchRunner(task, concurrent_games=args.concurrent_games, log_dir=log_dir)

    results = await runner.run_all(agent, output_path=output_path)
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
