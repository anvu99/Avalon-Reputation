import asyncio
import os
from src.server.tasks.avalon.task import AvalonBench
from src.server.tasks.avalon.batch_runner import AvalonBatchRunner
from src.client.agents.vllm_agent import VLLMAgentClient

async def main():
    agent = VLLMAgentClient(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=1024,
        temperature=0.0,
        tensor_parallel_size=1,
    )
    data_file = "data/avalon/all_dev.json"
    data_name = os.path.splitext(os.path.basename(data_file))[0]

    task = AvalonBench(
        num_players=5,
        agent_list=["llm", "llm", "llm", "llm", "llm"],
        discussion=True,
        data_file=data_file,
    )
    
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log_dir = f"logs/{data_name}_{job_id}"
    
    runner = AvalonBatchRunner(task, concurrent_games=64, log_dir=log_dir)
    
    output_path = f"outputs/results_{data_name}_{job_id}.jsonl"
    os.makedirs("outputs", exist_ok=True)
    
    results = await runner.run_all(agent, output_path=output_path)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
