import os
import requests

from pydantic import BaseModel
from enum import Enum

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class BaseMessage(BaseModel):
    role: Role
    content: str

class ToolCallMessage(BaseMessage):
    role: Role = Role.TOOL
    tool_call_id: str
    name: str
    content: str

class LLMClient:
    def __init__(
        self, base_url: str = f"http://{os.getenv('DOCKER_HOST_IP', 'localhost')}:25000"
    ):
        """Initialize LLM client with API server URL"""
        self.base_url = base_url.rstrip("/")

    def __call__(self, query: str, llm_name: str) -> tuple[str, int]:
        """
        Call LLM API endpoint

        Args:
            query (str): The prompt/query to send to the LLM
            llm_name (str): Name of LLM model to use (e.g. "gpt-4", "claude-3-sonnet")

        Returns:
            tuple[str, int]: (Generated response text, Total tokens used for this key)

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {"query": query, "llm_name": llm_name}

        response = requests.post(f"{self.base_url}/call", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["result"], result["total_tokens"]

    def 
    
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
