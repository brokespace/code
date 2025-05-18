import os
import requests
from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Optional

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class BaseMessage(BaseModel):
    role: Role
    content: str
    
    class Config:
        use_enum_values = True


class ToolCallMessage(BaseMessage):
    role: Role = Role.TOOL
    tool_call_id: str
    name: str
    content: str


class Edit(BaseModel):
    file_name: str
    line_number: int
    line_content: str
    new_line_content: str


class Patch(BaseModel):
    edits: list[Edit]


class ToolCall(BaseModel):
    name: str
    args: dict

class Response(BaseModel):
    result: str
    total_tokens: int
    tool_calls: Optional[list[ToolCall]] = None

# if host ip is localhost itll fail, need to get docker host ip
class LLMClient:
    def __init__(
        self,
        base_url: str = f"http://{os.getenv('HOST_IP', 'localhost')}:25000",
        api_key: str = os.getenv("OPENROUTER_API_KEY", ""),
    ):
        """Initialize LLM client with API server URL"""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def __call__(
        self,
        query: str,
        llm_name: str,
        temperature: float = 0.7,
        max_tokens: int = 16384,
    ) -> tuple[str, int]:
        """
        Call LLM API endpoint

        Args:
            query (str): The prompt/query to send to the LLM
            llm_name (str): Name of LLM model to use (e.g. "gpt-4", "claude-3-sonnet")
            temperature (float): Temperature for the LLM
        Returns:
            tuple[str, int]: (Generated response text, Total tokens used for this key)

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {
            "query": query,
            "llm_name": llm_name,
            "temperature": temperature,
            "api_key": self.api_key,
            "max_tokens": max_tokens,
        }

        response = requests.post(f"{self.base_url}/call", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["result"], result["total_tokens"]

    def call(
        self,
        messages: list[BaseMessage],
        tools: list[dict] = [],
        temperature: float = 0.7,
        max_tokens: int = 16384,
        model: str = "gpt-4o",
    ) -> Response:
        payload = {
            "messages": [message.model_dump() for message in messages],
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "llm_name": model,
            "api_key": self.api_key,
        }

        response = requests.post(f"{self.base_url}/call", json=payload)
        response.raise_for_status()

        result = response.json()
        return Response(**result)

    def embed(self, query: str) -> list[float]:
        """
        Get embeddings for text using the embedding API endpoint

        Args:
            query (str): The text to get embeddings for

        Returns:
            list[float]: Vector embedding of the input text

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {"query": query}

        response = requests.post(f"{self.base_url}/embed", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["vector"]

    def embed_documents(self, queries: list[str]) -> list[list[float]]:
        """
        Get embeddings for text using the embedding API endpoint

        Args:
            queries (list[str]): The list of texts to get embeddings for

        Returns:
            list[list[float]]: Vector embedding of the input text

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {"queries": queries}

        response = requests.post(f"{self.base_url}/embed/batch", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["vectors"]


class SWEBase(ABC):
    def __init__(self):
        self.llm = LLMClient()

    @abstractmethod
    def __call__(self, repo_location: str, issue_description: str) -> Patch:
        pass
