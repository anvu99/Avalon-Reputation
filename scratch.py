import asyncio
from src.client.agents.vllm_agent import VLLMAgentClient
async def test():
    agent = VLLMAgentClient(model_name="Qwen/Qwen2.5-7B-Instruct", max_tokens=1024, temperature=0.0)
    prompt = "Yes\n\nBased on the information, does the player approve the team? Please answer with the following template:\n\nAnswer: {Yes|No}"
    res = await agent.generate(prompt)
    print("RAW OUTPUT:", repr(res))
asyncio.run(test())
