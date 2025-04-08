import os
import time
import requests
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
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def verify_api_key(api_key: str):
        response = requests.get(
            f"{ChutesAPIClient.BASE_CHUTES_URL}/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        return response.status_code == 200


class VLLMChute(ChutesAPIClient):
    """
    Manages a vLLM chute.

    On instantiation, it checks whether a chute exists for the given model.
    If not, it creates one and waits until it’s ready (i.e. an instance is active).
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

    def chute_exists(self) -> bool:
        response = self.session.get(f"{self.BASE_CHUTES_URL}/{self.model}")
        return response.status_code == 200

    def create_chute(self) -> Dict:
        data = {
            "tagline": self.tagline,
            "model": self.model,
            "public": True,
            "node_selector": self.node_selector.model_dump(),
            "engine_args": self.engine_args.model_dump(),
        }
        response = self.session.post(f"{self.BASE_CHUTES_URL}/vllm", json=data)
        if response.status_code != 200:
            raise Exception("Error creating chute: " + response.text)
        return response.json()

    def wait_until_ready(self, timeout: int = 180, poll_interval: int = 5):
        """
        Polls the chute until it appears ready (i.e. a test invocation returns a non-empty result).
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.BASE_CHUTES_URL}/{self.model}")
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

    def invoke(
        self,
        messages: Union[str, List[Dict]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        skip_readiness_check: bool = False,
    ) -> str:
        """
        Invokes the LLM via the chute.

        Args:
            messages: Input prompt (string or list of message dicts).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            skip_readiness_check: If True, bypasses the check to recreate the chute.

        Returns:
            The generated response as a string.
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
        response = self.session.post(endpoint, json=payload)
        if response.status_code != 200:
            raise Exception("Error invoking chute: " + response.text)
        return response.json()["choices"][0]["message"]["content"]

    def delete_chute(self):
        delete_url = f"{self.BASE_CHUTES_URL}/{self.chute_id}"
        response = self.session.delete(delete_url)
        if response.status_code != 200:
            print("Error deleting chute:", response.text)
        else:
            print("Chute deleted successfully.")

    def __del__(self):
        try:
            self.delete_chute()
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
        response = self.session.get(f"{self.BASE_CHUTES_URL}/")
        return response.json()

    def get_chute(self, model: str):
        response = self.session.get(f"{self.BASE_CHUTES_URL}/{model}")
        return response.json()

    def delete_chute(self, model: str):
        response = self.session.delete(f"{self.BASE_CHUTES_URL}/{model}")
        return response.json()

    def model_exists(self, model: str) -> bool:
        """
        Checks if a model is available directly via the API.
        """
        response = self.session.get(f"{self.LLM_BASE_URL}/models")
        if response.status_code != 200:
            return False
        models = response.json()
        return any(m["id"] == model for m in models.get("data", []))

    def invoke(
        self,
        model: str,
        messages: Union[str, List[Dict]],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        """
        Invokes a model. If the model isn’t available directly,
        it will fall back to using a VLLMChute.
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
            return self.chutes[model].invoke(messages, temperature, max_tokens)

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
        response = self.session.post(endpoint, json=payload)
        if response.status_code != 200:
            raise Exception("Error invoking model: " + response.text)
        return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    MODEL = "princeton-nlp/SWE-Llama-7b"
    TAGLINE = "SWE Llama 7b"
    API_KEY = os.getenv("CHUTES_API_KEY")
    if not API_KEY:
        raise Exception("CHUTES_API_KEY environment variable not set.")

    chute = VLLMChute(MODEL, TAGLINE, API_KEY)
    try:
        response = chute.invoke("Hello, world!", temperature=0.5, max_tokens=150)
        print("Response:", response)
    finally:
        chute.delete_chute()
