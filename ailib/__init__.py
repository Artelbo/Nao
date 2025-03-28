from .base import AiLLM, Service
from enum import Enum


class Services(Enum):
    GoogleGemini = Service(name='Google Gemini',
                           requires_api_key=True,
                           online=True)

    LocalLLama3_2 = Service(name='Llama 3.2',
                            requires_api_key=False,
                            online=False)


class AiLLMException(Exception):
    """Some exception"""
    pass


class AiLLM(AiLLM):
    def __init__(self, service: Services, api_key: str = None):
        if not isinstance(service, Services):
            raise ValueError('Service must be an instance of Services.')
        self.service: Services = service

        if not isinstance(api_key, str) and api_key is not None:
            raise ValueError('API key must be a string or None.')
        self.api_key: str | None = api_key

    def answer(self):
