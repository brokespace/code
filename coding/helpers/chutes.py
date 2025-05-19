import os
import openai
import time
import httpx
import asyncio
import threading
from typing import Union, Dict, List
from pydantic import BaseModel

from .model import get_model_max_len
from .vram import calculate_model_gpu_vram


class EngineArgs(BaseModel):
    max_model_len: int = 4096
    num_scheduler_steps: int = 24
    enforce_eager: bool = False


class NodeSelector(BaseModel):
    gpu_count: int = 1
    min_vram_gb_per_gpu: int = 24


def generate_node_selector(model_id: str) -> NodeSelector:
    num_gpus, vram_per_gpu = calculate_model_gpu_vram(model_id)
    return NodeSelector(
        gpu_count=num_gpus,
        min_vram_gb_per_gpu=vram_per_gpu,
    )


def generate_engine_args(model_id: str) -> EngineArgs:
    return EngineArgs(
        num_scheduler_steps=24,
        enforce_eager=False,
        max_model_len=get_model_max_len(model_id),
    )


class ChutesAPIClient:
    BASE_CHUTES_URL = "https://api.chutes.ai/chutes"
    LLM_BASE_URL = "https://llm.chutes.ai/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Set up sync client for backward compatibility
        self._sync_client = httpx.Client(headers=self.headers)

    async def _async_request(self, method, url, **kwargs):
        """Helper method for async HTTP requests"""
        timeout = httpx.Timeout(kwargs.pop("timeout", 30.0))
        async with httpx.AsyncClient(headers=self.headers, timeout=timeout) as client:
            return await getattr(client, method)(url, **kwargs)

    @staticmethod
    async def verify_api_key(api_key: str):
        """Verify API key asynchronously"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.get(f"{ChutesAPIClient.BASE_CHUTES_URL}/")
            return response.status_code == 200


class VLLMChute(ChutesAPIClient):
    """
    Manages a vLLM chute.

    On instantiation, it checks whether a chute exists for the given model.
    If not, it creates one and waits until it's ready (i.e. an instance is active).
    The invoke() method ensures the chute is available before sending a query.

    The chute is automatically deleted on object deletion or when used as a context manager.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        tagline: str = None,
        engine_args: Union[Dict, EngineArgs] = None,
        node_selector: Union[Dict, NodeSelector] = None,
    ):
        super().__init__(api_key)
        self.model = model
        self.tagline = tagline or model.split("/")[-1].replace("-", " ").title()

        # Normalize configuration arguments
        self.engine_args = (
            EngineArgs(**engine_args)
            if isinstance(engine_args, dict)
            else (engine_args or EngineArgs())
        )
        self.node_selector = (
            NodeSelector(**node_selector)
            if isinstance(node_selector, dict)
            else (node_selector or NodeSelector())
        )

        # Using model as the chute identifier
        self.chute_id = self.model
    
        # Ensure the chute is created and ready.
        self.ensure_chute()

    def ensure_chute(self):
        if not self.chute_exists():
            self.create_chute()
            self.wait_until_ready()
        else:
            self.wait_until_ready()
            
    async def ensure_chute_async(self):
        exists = await self.chute_exists_async()
        if not exists:
            await self.create_chute_async()
            await self.wait_until_ready_async()
        else:
            await self.wait_until_ready_async()

    def chute_exists(self) -> bool:
        response = self._sync_client.get(f"{self.BASE_CHUTES_URL}/{self.model}")
        return response.status_code == 200
        
    async def chute_exists_async(self) -> bool:
        try:
            response = await self._async_request("get", f"{self.BASE_CHUTES_URL}/{self.model}")
            print("Response:", response.status_code)
            return response.status_code == 200
        except Exception as e:
            print(f"Error checking chute existence: {e}")
            return False

    def create_chute(self) -> Dict:
        data = {
            "tagline": self.tagline,
            "model": self.model,
            "public": True,
            "node_selector": self.node_selector.model_dump(),
            "engine_args": self.engine_args.model_dump(),
        }
        response = self._sync_client.post(f"{self.BASE_CHUTES_URL}/vllm", json=data)
        if response.status_code != 200:
            raise Exception("Error creating chute: " + response.text)
        return response.json()
        
    async def create_chute_async(self) -> Dict:
        data = {
            "tagline": self.tagline,
            "model": self.model,
            "public": True,
            "node_selector": self.node_selector.model_dump(),
            "engine_args": self.engine_args.model_dump(),
        }
        response = await self._async_request("post", f"{self.BASE_CHUTES_URL}/vllm", json=data)
        if response.status_code != 200:
            raise Exception("Error creating chute: " + response.text)
        return response.json()

    def wait_until_ready(self, timeout: int = 180, poll_interval: int = 5):
        """
        Polls the chute until it appears ready (i.e. a test invocation returns a non-empty result).
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self._sync_client.get(f"{self.BASE_CHUTES_URL}/{self.model}")
            if response.status_code == 200:
                try:
                    # Call invoke in test mode so it skips the readiness check.
                    if self.invoke(
                        "Hello, world!",
                        temperature=0.5,
                        max_tokens=150,
                        skip_readiness_check=True,
                    ):
                        return
                except Exception:
                    pass
            time.sleep(poll_interval)
        raise TimeoutError("Chute did not become ready within the timeout period.")
        
    async def wait_until_ready_async(self, timeout: int = 180, poll_interval: int = 5):
        """
        Polls the chute asynchronously until it appears ready.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = await self._async_request("get", f"{self.BASE_CHUTES_URL}/{self.model}")
            if response.status_code == 200:
                try:
                    # Call invoke in test mode so it skips the readiness check.
                    result = await self.invoke_async(
                        "Hello, world!",
                        temperature=0.5,
                        max_tokens=150,
                        skip_readiness_check=True,
                    )
                    if result:
                        return
                except Exception:
                    pass
            await asyncio.sleep(poll_interval)
        raise TimeoutError("Chute did not become ready within the timeout period.")

    def invoke(
        self,
        messages: Union[str, List[Dict]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        skip_readiness_check: bool = False,
        timeout: int = 45,
    ) -> str:
        """
        Invokes the LLM via the chute.

        Args:
            messages: Input prompt (string or list of message dicts).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            skip_readiness_check: If True, bypasses the check to recreate the chute.
            timeout: Timeout in seconds for the API request.

        Returns:
            The generated response as a string.
            
        Raises:
            TimeoutError: If the request times out.
            Exception: If there's an error invoking the chute.
        """
        if not skip_readiness_check and not self.chute_exists():
            self.create_chute()
            self.wait_until_ready()

        endpoint = f"{self.LLM_BASE_URL}/chat/completions"
        payload = {
            "model": self.model,
            "messages": (
                [{"role": "user", "content": messages}]
                if isinstance(messages, str)
                else messages
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = self._sync_client.post(endpoint, json=payload, timeout=timeout)
            if response.status_code != 200:
                raise Exception("Error invoking chute: " + response.text)
            return response.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            raise TimeoutError(f"Request to chute {self.model} timed out after {timeout} seconds")
        except httpx.HTTPError as e:
            raise Exception(f"Network error when invoking chute {self.model}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error invoking chute {self.model}: {str(e)}")
            
    async def invoke_async(
        self,
        messages: Union[str, List[Dict]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        skip_readiness_check: bool = False,
        timeout: int = 45,
    ) -> str:
        """
        Invokes the LLM via the chute asynchronously.

        Args:
            messages: Input prompt (string or list of message dicts).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            skip_readiness_check: If True, bypasses the check to recreate the chute.
            timeout: Timeout in seconds for the API request.

        Returns:
            The generated response as a string.
            
        Raises:
            TimeoutError: If the request times out.
            Exception: If there's an error invoking the chute.
        """
        if not skip_readiness_check:
            exists = await self.chute_exists_async()
            if not exists:
                await self.create_chute_async()
                await self.wait_until_ready_async()

        endpoint = f"{self.LLM_BASE_URL}/chat/completions"
        payload = {
            "model": self.model,
            "messages": (
                [{"role": "user", "content": messages}]
                if isinstance(messages, str)
                else messages
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = await self._async_request("post", endpoint, json=payload, timeout=timeout)
            if response.status_code != 200:
                raise Exception("Error invoking chute: " + response.text)
            return response.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            raise TimeoutError(f"Request to chute {self.model} timed out after {timeout} seconds")
        except httpx.HTTPError as e:
            raise Exception(f"Network error when invoking chute {self.model}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error invoking chute {self.model}: {str(e)}")

    def delete_chute(self):
        delete_url = f"{self.BASE_CHUTES_URL}/{self.chute_id}"
        response = self._sync_client.delete(delete_url)
        if response.status_code != 200:
            print("Error deleting chute:", response.text)
        else:
            print("Chute deleted successfully.")
            
    async def delete_chute_async(self):
        delete_url = f"{self.BASE_CHUTES_URL}/{self.chute_id}"
        response = await self._async_request("delete", delete_url)
        if response.status_code != 200:
            print("Error deleting chute:", response.text)
        else:
            print("Chute deleted successfully.")

    def __del__(self):
        try:
            self.delete_chute()
            self._sync_client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.delete_chute()


class Chutes(ChutesAPIClient):
    def __init__(self, api_key: str, model_timeout: float = 60*60, create_chute: bool = True):
        """
        :param api_key: API key for authentication.
        :param model_timeout: Timeout in seconds after which a created VLLMChute
                              will be automatically removed from the cache.
        :param create_chute: If True, a VLLMChute will be created if the model is not available directly.
        """
        super().__init__(api_key)
        self.model_timeout = model_timeout
        self.chutes = {}  # Cache of created VLLMChute instances
        self.create_chute = create_chute
        self.client = openai.OpenAI(api_key=api_key, base_url="https://llm.chutes.ai/v1")
        
    def schedule_deletion(self, model: str):
        """
        Schedules deletion of the given model from the cache after model_timeout seconds.
        """

        def delete_model():
            # Optionally, add logging here if needed.
            self.chutes.pop(model, None)

        timer = threading.Timer(self.model_timeout, delete_model)
        timer.daemon = True  # So the timer thread won't block program exit.
        timer.start()

    def list_chutes(self):
        response = self._sync_client.get(f"{self.BASE_CHUTES_URL}/")
        return response.json()
        
    async def list_chutes_async(self):
        response = await self._async_request("get", f"{self.BASE_CHUTES_URL}/")
        return response.json()

    def get_chute(self, model: str):
        response = self._sync_client.get(f"{self.BASE_CHUTES_URL}/{model}")
        return response.json()
        
    async def get_chute_async(self, model: str):
        response = await self._async_request("get", f"{self.BASE_CHUTES_URL}/{model}")
        return response.json()

    def delete_chute(self, model: str):
        response = self._sync_client.delete(f"{self.BASE_CHUTES_URL}/{model}")
        return response.json()
        
    async def delete_chute_async(self, model: str):
        response = await self._async_request("delete", f"{self.BASE_CHUTES_URL}/{model}")
        return response.json()

    @property
    def models(self) -> List[str]:
        # Check if we have a cached result that is still valid
        current_time = time.time()
        if hasattr(self, '_models_cache') and hasattr(self, '_models_cache_time'):
            if current_time - self._models_cache_time < 30 * 60:  # 30 minutes in seconds
                return self._models_cache
        
        try:
            response = self._sync_client.get(f"{self.LLM_BASE_URL}/models")
            if response.status_code == 200:
                models_list = [m["id"] for m in response.json().get("data", [])]
                # Cache the successful result
                self._models_cache = models_list
                self._models_cache_time = current_time
                return models_list
            else:
                print(f"Error fetching models: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
        
    async def models_async(self) -> List[str]:
        # Check if we have a cached result that is still valid
        current_time = time.time()
        if hasattr(self, '_models_cache') and hasattr(self, '_models_cache_time'):
            if current_time - self._models_cache_time < 30 * 60:  # 30 minutes in seconds
                return self._models_cache

        try:
            response = await self._async_request("get", f"{self.LLM_BASE_URL}/models")
            if response.status_code == 200:
                models_list = [m["id"] for m in response.json().get("data", [])]
                # Cache the successful result
                self._models_cache = models_list
                self._models_cache_time = current_time
                return models_list
            else:
                print(f"Error fetching models asynchronously: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error fetching models asynchronously: {e}")
            return []

    def model_exists(self, model: str) -> bool:
        """
        Checks if a model is available directly via the API.
        """
        return model in self.models
        
    async def model_exists_async(self, model: str) -> bool:
        """
        Checks if a model is available directly via the API asynchronously.
        """
        models = await self.models_async()
        return model in models

    def invoke(
        self,
        model: str,
        messages: Union[str, List[Dict]],
        temperature: float = 0.7,
        tools: List[Dict] = None,
        max_tokens: int = 256,
        timeout: int = 30,
    ) -> str:
        """
        Invokes a model. If the model isn't available directly,
        it will fall back to using a VLLMChute.
        
        Args:
            model: The model to invoke.
            messages: Input prompt (string or list of message dicts).
            temperature: Sampling temperature.
            tools: List of tools for function calling.
            max_tokens: Maximum tokens in the response.
            timeout: Timeout in seconds for the API request.
            
        Returns:
            The generated response as a string.
            
        Raises:
            TimeoutError: If the request times out.
            Exception: If there's an error invoking the model.
        """
        if not self.model_exists(model):
            if not self.create_chute:
                raise Exception(f"Model {model} not found and create_chute is False.")
            if model not in self.chutes:
                # Create the chute and schedule its deletion after model_timeout seconds.
                self.chutes[model] = VLLMChute(
                    model,
                    api_key=self.api_key,
                    engine_args=generate_engine_args(model),
                    node_selector=generate_node_selector(model),
                )
                self.schedule_deletion(model)
            try:
                return self.chutes[model].invoke(messages, temperature, max_tokens)
            except Exception as e:
                # If the VLLMChute invoke fails, remove it from the cache to avoid future failures
                self.chutes.pop(model, None)
                raise e

        # Prepare the messages format
        formatted_messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        
        # Prepare the payload
        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        
        try:
            # Use the OpenAI client to call the chat completions endpoint
            response = self.client.chat.completions.create(**payload, timeout=timeout)
            return response.choices[0].message.content
        except openai.APITimeoutError:
            raise TimeoutError(f"Request to {model} timed out after {timeout} seconds")
        except openai.APIConnectionError as e:
            raise Exception(f"Network error when invoking {model}: {str(e)}")
        except openai.APIError as e:
            raise Exception(f"Error invoking {model}: {str(e)}")
    async def invoke_async(
        self,
        model: str,
        messages: Union[str, List[Dict]],
        temperature: float = 0.7,
        tools: List[Dict] = None,
        max_tokens: int = 256,
        timeout: int = 30,
    ) -> str:
        """
        Invokes a model asynchronously. If the model isn't available directly,
        it will fall back to using a VLLMChute.
        
        Args:
            model: The model to invoke.
            messages: Input prompt (string or list of message dicts).
            temperature: Sampling temperature.
            tools: List of tools for function calling.
            max_tokens: Maximum tokens in the response.
            timeout: Timeout in seconds for the API request.
            
        Returns:
            The generated response as a string.
            
        Raises:
            TimeoutError: If the request times out.
            Exception: If there's an error invoking the model.
        """
        model_exists = await self.model_exists_async(model)
        if not model_exists:
            if not self.create_chute:
                raise Exception(f"Model {model} not found and create_chute is False.")
            if model not in self.chutes:
                # Create the chute and schedule its deletion after model_timeout seconds.
                self.chutes[model] = VLLMChute(
                    model,
                    api_key=self.api_key,
                    engine_args=generate_engine_args(model),
                    node_selector=generate_node_selector(model),
                )
                self.schedule_deletion(model)
            try:
                return await self.chutes[model].invoke_async(messages, temperature, max_tokens, timeout=timeout)
            except Exception as e:
                # If the VLLMChute invoke fails, remove it from the cache to avoid future failures
                self.chutes.pop(model, None)
                raise e

        endpoint = f"{self.LLM_BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": (
                [{"role": "user", "content": messages}]
                if isinstance(messages, str)
                else messages
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        
        try:
            response = await self._async_request("post", endpoint, json=payload, timeout=timeout)
            if response.status_code != 200:
                raise Exception("Error invoking model: " + response.text)
            return response.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            raise TimeoutError(f"Request to {model} timed out after {timeout} seconds")
        except httpx.HTTPError as e:
            raise Exception(f"Network error when invoking {model}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error invoking {model}: {str(e)}")


if __name__ == "__main__":
    MODEL = "princeton-nlp/SWE-Llama-7b"
    TAGLINE = "SWE Llama 7b"
    API_KEY = os.getenv("CHUTES_API_KEY")
    if not API_KEY:
        raise Exception("CHUTES_API_KEY environment variable not set.")

    chute = VLLMChute(MODEL, API_KEY, TAGLINE)
    try:
        response = chute.invoke("Hello, world!", temperature=0.5, max_tokens=150)
        print("Response:", response)
    finally:
        chute.delete_chute()
