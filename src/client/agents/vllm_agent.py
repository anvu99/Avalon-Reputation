from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict
import asyncio
from ..agent import AgentClient

class VLLMEngine:
    """Multiton: one AsyncLLMEngine per model_name, shared across all agents."""
    _instances: Dict[str, AsyncLLMEngine] = {}
    _tokenizers: Dict[str, AutoTokenizer] = {}

    @classmethod
    def get_instance(cls, model_name: str, **engine_kwargs) -> AsyncLLMEngine:
        if model_name not in cls._instances:
            engine_args = AsyncEngineArgs(model=model_name, **engine_kwargs)
            cls._instances[model_name] = AsyncLLMEngine.from_engine_args(engine_args)
            cls._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls._instances[model_name]

    @classmethod
    def get_tokenizer(cls, model_name: str) -> AutoTokenizer:
        return cls._tokenizers[model_name]


class VLLMAgentClient(AgentClient):
    """Direct vLLM inference client — replaces HTTPAgent."""

    def __init__(self, model_name: str, max_tokens: int = 1024,
                 temperature: float = 0.1, **engine_kwargs):
        self.model_name = model_name
        self.engine = VLLMEngine.get_instance(model_name, **engine_kwargs)
        self.tokenizer = VLLMEngine.get_tokenizer(model_name)
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )
        super().__init__()

    def _format_messages(self, history: List[Dict[str, str]]) -> str:
        """Apply the model's native chat template (HuggingFace-standard)."""
        messages = [
            {"role": "user" if m["role"] == "user" else "assistant",
             "content": m["content"]}
            for m in history
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def inference(self, history: List[Dict]) -> str:
        prompt = self._format_messages(history)
        request_id = str(id(asyncio.current_task()))
        results = self.engine.generate(prompt, self.sampling_params, request_id)
        final = None
        async for result in results:
            final = result
        if final:
            return final.outputs[0].text
        return ""
