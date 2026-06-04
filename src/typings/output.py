from typing import Union, List, Any

from pydantic import BaseModel, model_validator

from . import ChatHistoryItem
from .general import JSONSerializable, SampleIndex
from .status import SampleStatus, AgentOutputStatus


class TaskOutput(BaseModel):
    index: Union[None, SampleIndex] = None
    status: SampleStatus = SampleStatus.RUNNING
    result: JSONSerializable = None
    history: Union[None, List[ChatHistoryItem]] = None


class TaskSampleExecutionResult(BaseModel):
    status: SampleStatus = SampleStatus.COMPLETED
    result: JSONSerializable = None


class AgentOutput(BaseModel):
    status: AgentOutputStatus = AgentOutputStatus.NORMAL
    content: Any = None

    # at least one of them should be not None
    @model_validator(mode="after")
    def post_validate(self):
        assert (
            self.status is not AgentOutputStatus.NORMAL
            or self.content is not None
        ), "If status is NORMAL, content should not be None"
        return self


class TaskClientOutput(BaseModel):
    error: Union[str, None] = None
    info: Union[str, None] = None
    output: Union[TaskOutput, None] = None
