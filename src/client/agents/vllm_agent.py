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
            # Configure reasoning parser and config for thinking models to support thinking_token_budget
            is_qwen3 = "qwen" in model_name.lower() and "qwen3" in model_name.lower().replace("/", "")
            if is_qwen3:
                try:
                    from vllm.config import ReasoningConfig
                    if "reasoning_config" not in engine_kwargs:
                        engine_kwargs["reasoning_config"] = ReasoningConfig()
                except ImportError:
                    pass
                if "reasoning_parser" not in engine_kwargs:
                    engine_kwargs["reasoning_parser"] = "qwen3"
            elif "deepseek" in model_name.lower():
                try:
                    from vllm.config import ReasoningConfig
                    if "reasoning_config" not in engine_kwargs:
                        engine_kwargs["reasoning_config"] = ReasoningConfig()
                except ImportError:
                    pass
                if "reasoning_parser" not in engine_kwargs:
                    engine_kwargs["reasoning_parser"] = "deepseek_r1"
                
            engine_args = AsyncEngineArgs(model=model_name, **engine_kwargs)
            cls._instances[model_name] = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Override model_max_length to match the configured max context length
            max_len = engine_kwargs.get("max_model_len", 16384)
            cls._tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=max_len
            )
            cls._tokenizers[model_name].model_max_length = max_len
        return cls._instances[model_name]

    @classmethod
    def get_tokenizer(cls, model_name: str) -> AutoTokenizer:
        return cls._tokenizers[model_name]


class AgentResponse(str):
    def __new__(cls, content, thinking="", finish_reason=""):
        obj = str.__new__(cls, content)
        obj.thinking = thinking
        obj.finish_reason = finish_reason
        return obj


class VLLMAgentClient(AgentClient):
    """Direct vLLM inference client — replaces HTTPAgent."""

    def __init__(self, model_name: str, max_tokens: int = 4096,
                 temperature: float = 0.1, thinking_token_budget: int = 1024,
                 enable_thinking: bool = True,
                 **engine_kwargs):
        self.model_name = model_name
        self.engine = VLLMEngine.get_instance(model_name, **engine_kwargs)
        self.tokenizer = VLLMEngine.get_tokenizer(model_name)
        # Match any Qwen3 thinking model (Qwen3-8B, Qwen3-14B, Qwen3.6-35B-A3B, etc.)
        is_qwen3 = "qwen" in model_name.lower() and "qwen3" in model_name.lower().replace("/", "")
        self.is_qwen3 = is_qwen3
        self.enable_thinking = enable_thinking if is_qwen3 else False
        self.thinking_token_budget = thinking_token_budget
        if is_qwen3 and enable_thinking:
            self.sampling_params = SamplingParams(
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                presence_penalty=1.5,
                repetition_penalty=1.0,
                max_tokens=max_tokens,
                thinking_token_budget=thinking_token_budget,
            )
        elif is_qwen3:
            # No-thinking mode: Qwen3 recommended non-thinking sampling params
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                presence_penalty=1.5,
                repetition_penalty=1.0,
                max_tokens=max_tokens,
            )
        else:
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
        if self.is_qwen3:
            # enable_thinking=False suppresses the <think> prompt, disabling reasoning mode
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def inference(self, history: List[Dict], max_tokens: int = None) -> str:
        prompt = self._format_messages(history)
        request_id = str(id(asyncio.current_task()))
        
        sampling_params = self.sampling_params
        if max_tokens is not None:
            kwargs = dict(
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                top_k=self.sampling_params.top_k,
                min_p=self.sampling_params.min_p,
                presence_penalty=self.sampling_params.presence_penalty,
                repetition_penalty=self.sampling_params.repetition_penalty,
                max_tokens=max_tokens,
            )
            if self.is_qwen3 and self.enable_thinking:
                kwargs["thinking_token_budget"] = self.thinking_token_budget
            sampling_params = SamplingParams(**kwargs)
            
        results = self.engine.generate(prompt, sampling_params, request_id)
        final = None
        async for result in results:
            final = result
        if final:
            text = final.outputs[0].text
            finish_reason = final.outputs[0].finish_reason
            import re
            
            thinking_content = ""
            cleaned_text = text

            prompt_ends_with_think = "<think>" in prompt[-100:]

            last_think_end = text.rfind('</think>')
            if last_think_end != -1:
                # Everything before the LAST </think> is thinking; everything after is content
                thinking_content = text[:last_think_end].replace('<think>', '').strip()
                cleaned_text = text[last_think_end + len('</think>'):].strip()
            elif '<think>' in text:
                # Truncated: <think> opened but never closed — all output is thinking
                first_think_start = text.find('<think>')
                thinking_content = text[first_think_start + len('<think>'):].strip()
                cleaned_text = text[:first_think_start].strip()
            elif prompt_ends_with_think:
                # Prompt ended with <think> — model output is the thinking continuation
                thinking_content = text.strip()
                cleaned_text = ""

            # Fallback: if parsed content is empty but output is non-empty, use raw output
            if not cleaned_text.strip() and text.strip():
                cleaned_text = text.strip()
                
            return AgentResponse(cleaned_text, thinking=thinking_content, finish_reason=finish_reason)
        return AgentResponse("")
