from src.client.agents.vllm_agent import VLLMAgentClient
from src.typings import AgentOutput, AgentOutputStatus

from src.server.task import Session

class DirectSession(Session):
    """Replaces the HTTP-based Session. Calls VLLMAgentClient directly."""

    def __init__(self, agent: VLLMAgentClient):
        self.agent = agent
        self.history = []

    def inject(self, message: dict):
        """Append a message to history without calling the LLM."""
        if 'role' in message and 'content' in message:
            self.history.append({'role': message['role'], 'content': message['content']})

    async def action(self, **kwargs) -> AgentOutput:
        """Call the LLM with the current history and return its response."""
        response_text = await self.agent.inference(self.history, **kwargs)
        self.history.append({'role': 'agent', 'content': response_text})
        return AgentOutput(status=AgentOutputStatus.NORMAL, content=response_text)
