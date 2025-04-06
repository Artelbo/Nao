from logging import getLogger, Logger
from typing import Deque, Tuple, Dict, TypedDict
from collections import deque


class MemoryLocale(TypedDict):
    question: str
    answer: str


class Memory:
    """
    Represents a memory for an LLM, helps to contextualize.
    """
    def __init__(self, capacity: int = 5, locale: Dict = None):
        self.__logger: Logger = getLogger('ai')

        if locale is None or not isinstance(locale, dict):
            locale = {'question': 'Domanda', 'answer': 'Risposta'}
        if locale.get('question') is None or locale.get('answer') is None:
            raise ValueError('locale should follow MemoryLocale')
        self.locale = locale

        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity should be a positive int not '{type(capacity)}' with value '{capacity}'")
        self.memory: Deque[Tuple[str, str]] = deque(maxlen=capacity)

    def add(self, question: str, answer: str) -> None:
        """
        Adds a question-answer pair to the memory.

        Args:
            question: The question string.
            answer: The answer string.

        Raises:
            ValueError: If either the question or the answer is not a string.
        """
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError('Both question and answer should be strings')
        self.memory.append((question, answer))

    def get_memory_string(self) -> str:
        """
        Returns a formatted string representation of the memory.

        Returns:
            str
        """
        return '\n\n'.join([f'{self.locale["question"]}: {item[0]}\n{self.locale["answer"]}: {item[1]}'
                            for item in self.memory])

    __str__ = get_memory_string