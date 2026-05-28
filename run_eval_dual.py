import asyncio
import argparse
import os
import sys
from src.server.tasks.avalon.task import AvalonBench
from src.server.tasks.avalon.batch_runner import AvalonBatchRunner
from src.server.tasks.avalon.long_term_memory import LongTermMemory
from src.client.agents.vllm_agent import VLLMAgentClient


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dual Avalon LLM batch evaluations sequentially.")
    
    # Common Engine & Dataset settings
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="vLLM model name to use for all agents.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallel execution (vLLM).",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/avalon/dev_servant.json",
        help="Path to the evaluation data file.",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=1,
        help="Number of games to run concurrently inside each batch.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=2,
        help="Number of times to repeat the dataset.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for LLM generation.",
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
        "--sequential",
        action="store_true",
        default=False,
        help="Run experiments sequentially instead of concurrently (simultaneously).",
    )

    # Experiment 1 Settings
    parser.add_argument(
        "--exp1-suffix",
        type=str,
        default="(exp1)",
        help="Output naming suffix for Experiment 1.",
    )
    parser.add_argument(
        "--exp1-use-pubrep",
        action="store_true",
        default=False,
        help="Enable public reputation database for Experiment 1.",
    )
    parser.add_argument(
        "--exp1-personality",
        type=str,
        nargs=5,
        default=["default"] * 5,
        metavar=("P0", "P1", "P2", "P3", "P4"),
        help="Experiment 1 per-player personalities: naive, deceptive, default.",
    )

    # Experiment 2 Settings
    parser.add_argument(
        "--exp2-suffix",
        type=str,
        default="(exp2)",
        help="Output naming suffix for Experiment 2.",
    )
    parser.add_argument(
        "--exp2-use-pubrep",
        action="store_true",
        default=False,
        help="Enable public reputation database for Experiment 2.",
    )
    parser.add_argument(
        "--exp2-personality",
        type=str,
        nargs=5,
        default=["default"] * 5,
        metavar=("P0", "P1", "P2", "P3", "P4"),
        help="Experiment 2 per-player personalities: naive, deceptive, default.",
    )
    parser.add_argument(
        "--use-discrete-rating",
        action="store_true",
        default=False,
        help="Use discrete 1-5 Likert scale ratings instead of continuous probabilities for beliefs.",
    )

    return parser.parse_args()


async def run_single_experiment(args, agent, is_exp1=True):
    # Select exp-specific settings
    suffix_tag = args.exp1_suffix if is_exp1 else args.exp2_suffix
    use_pubrep = args.exp1_use_pubrep if is_exp1 else args.exp2_use_pubrep
    personality = args.exp1_personality if is_exp1 else args.exp2_personality
    
    exp_name = "Experiment 1" if is_exp1 else "Experiment 2"
    print(f"\n========================================================")
    print(f" STARTING {exp_name.upper()}: {suffix_tag}")
    print(f" - Public Reputation: {use_pubrep}")
    print(f" - Discrete Rating Mode: {args.use_discrete_rating}")
    print(f" - Personalities: {'-'.join(personality)}")
    print(f"========================================================\n")

    data_file = args.data_file
    data_name = os.path.splitext(os.path.basename(data_file))[0]

    # Instantiate the benchmark task
    task = AvalonBench(
        num_players=5,
        agent_list=["llm", "llm", "llm", "llm", "llm"],
        discussion=True,
        data_file=data_file,
        use_reputation_memory=False,
        use_public_reputation=use_pubrep,
        use_discrete_rating=args.use_discrete_rating,  # NEW
        long_term_memories={},  # Empty since public reputation is independent
        num_repeats=args.num_repeats,
        use_bayesian_prediction=False,
        log_memory_snapshots_for=[] if args.no_memory_snapshot else [0, 1, 2, 3, 4],
        predict_for=[] if args.no_periodic_prediction else None,
        personality_list=personality,
        ltm_counter_norm=False,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Construct filenames
    pubrep_tag = "_pubrep" if use_pubrep else ""
    discrete_tag = "_discrete" if args.use_discrete_rating else ""
    personality_tag = "_" + "-".join(personality) if any(p != "default" for p in personality) else ""
    
    log_dir = f"logs/{data_name}_{job_id}_{suffix_tag}{pubrep_tag}{discrete_tag}{personality_tag}"
    output_path = f"outputs/results_{data_name}_{job_id}_{suffix_tag}{pubrep_tag}{discrete_tag}{personality_tag}.jsonl"

    # Auto-increment suffix to prevent overwriting
    counter = 1
    while os.path.exists(output_path) or os.path.exists(log_dir):
        alt_suffix = f"{suffix_tag}_v{counter}"
        log_dir = f"logs/{data_name}_{job_id}_{alt_suffix}{pubrep_tag}{discrete_tag}{personality_tag}"
        output_path = f"outputs/results_{data_name}_{job_id}_{alt_suffix}{pubrep_tag}{discrete_tag}{personality_tag}.jsonl"
        counter += 1

    os.makedirs("outputs", exist_ok=True)
    runner = AvalonBatchRunner(task, concurrent_games=args.concurrent_games, log_dir=log_dir)

    results = await runner.run_all(agent, output_path=output_path)
    print(f"\n========================================================")
    print(f" FINISHED {exp_name.upper()}")
    print(f" - Log Directory: {log_dir}")
    print(f" - Output File: {output_path}")
    print(f" - Results: {results}")
    print(f"========================================================\n")


async def main():
    args = parse_args()

    # Load vLLM client ONCE
    print(">>> Initializing vLLM Client (loading model into VRAM)...")
    agent = VLLMAgentClient(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=16384,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    try:
        if args.sequential:
            # Run Experiment 1 and 2 sequentially (one after another)
            await run_single_experiment(args, agent, is_exp1=True)
            await run_single_experiment(args, agent, is_exp1=False)
        else:
            # Run both Experiment 1 and 2 concurrently (simultaneously)
            await asyncio.gather(
                run_single_experiment(args, agent, is_exp1=True),
                run_single_experiment(args, agent, is_exp1=False)
            )

    finally:
        # Cleanup agent client resources if needed
        pass


if __name__ == "__main__":
    asyncio.run(main())
