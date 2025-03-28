from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

class AiLLM(ABC):  # better name to come
    @abstractmethod
    def answer(self, system_prompt: str, prompt: str):
        ...

@dataclass(frozen=True)
class Service:
    name: str
    requires_api_key: bool
    online: bool
    model: Optional[str] = None
