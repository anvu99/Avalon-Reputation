from dataclasses import dataclass, field
from typing import List

from .prompts import LONG_TERM_MEMORY_INJECTION_PROMPT

@dataclass
class LongTermMemory:
    memory_text: str = ""
    pending_lessons: List[str] = field(default_factory=list)

    def add_lesson(self, lesson: str, won: bool = None) -> None:
        if lesson and lesson.strip():
            if won is not None:
                label = "[WIN]" if won else "[LOSS]"
                self.pending_lessons.append(f"{label}\n{lesson.strip()}")
            else:
                self.pending_lessons.append(lesson.strip())

    def is_empty(self) -> bool:
        return not bool(self.memory_text.strip())

    def to_prompt_block(self) -> str:
        if self.is_empty():
            return ""
        return LONG_TERM_MEMORY_INJECTION_PROMPT.format(memory_text=self.memory_text)
