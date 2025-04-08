from dotenv import load_dotenv

load_dotenv("../../../.env", override=False)  # Don't override existing env vars

import os

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List

# ------------------------------
#    Import Provider Libraries
# ------------------------------
import openai
import anthropic
from google import genai
from google.genai import types
from langchain_openai import OpenAIEmbeddings

from coding.helpers.chutes import Chutes

token_usage: Dict[str, int] = {}
current_key: Optional[str] = None

# FastAPI App
app = FastAPI()

# Instead of LangChain instances we now map model names to a config that includes:
# - provider: one of "openai", "anthropic", "google"
# - model: the actual model name/ID used by the API
# - max_tokens: maximum tokens to request (used for each API call)
models = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o", "max_tokens": 16384},
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "max_tokens": 16384,
    },
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 16384},
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "model": "claude-3.5-sonnet",
        "max_tokens": 8192,
    },
    "claude-3-7-sonnet": {
        "provider": "anthropic",
        "model": "claude-3.7-sonnet",
        "max_tokens": 8192,
    },
    "gemini-2.0-flash-exp": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "max_tokens": 8192,
    },
}
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)


class InitRequest(BaseModel):
    key: str


class LLMRequest(BaseModel):
    query: str
    api_key: str
    llm_name: str
    temperature: Optional[float] = 0.7  # default temperature value
    max_tokens: Optional[int] = None


class LLMResponse(BaseModel):
    result: str
    total_tokens: int


class EmbeddingRequest(BaseModel):
    query: str


class EmbeddingResponse(BaseModel):
    vector: List[float]


# New models for batch embedding support
class BatchEmbeddingRequest(BaseModel):
    queries: List[str]


class BatchEmbeddingResponse(BaseModel):
    vectors: List[List[float]]


class BatchEmbeddingResponse(BaseModel):
    vectors: List[List[float]]


# ------------------------------
#       Auth Dependency
# ------------------------------
async def verify_auth(auth_key: str = Depends(lambda: os.getenv("LLM_AUTH_KEY"))):
    if not auth_key:
        raise HTTPException(
            status_code=500, detail="LLM_AUTH_KEY environment variable not set"
        )
    return auth_key


# ------------------------------
#   Initialize / Reset / Count
# ------------------------------
@app.post("/init")
async def init_key(request: InitRequest, auth_key: str = Depends(verify_auth)):
    global current_key
    if request.key not in token_usage:
        token_usage[request.key] = 0
    current_key = request.key
    return {"message": f"Set active key to {request.key}"}


@app.post("/reset")
async def reset_count(auth_key: str = Depends(verify_auth)):
    global current_key
    if not current_key:
        raise HTTPException(
            status_code=400, detail="No active key. Call /init endpoint first."
        )
    token_usage[current_key] = 0
    return {"message": f"Reset token count for key {current_key}"}


@app.get("/count")
async def get_count(auth_key: str = Depends(verify_auth)):
    global current_key
    if not current_key:
        raise HTTPException(
            status_code=400, detail="No active key. Call /init endpoint first."
        )
    return {"key": current_key, "count": token_usage[current_key]}


async def call_openai(
    query: str,
    model: str,
    temperature: float,
    max_tokens: int = 16384,
    api_key: str = None,
):
    if not api_key:
        print("No API key provided")
        return {"content": "", "usage": {"total_tokens": 0}}
    openai.api_key = api_key

    def sync_call():
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"provider": {"sort": "throughput"}}
        )
        return response

    response = await asyncio.to_thread(sync_call)
    result = response.choices[0].message.content
    tokens = response.usage.prompt_tokens + response.usage.completion_tokens
    return {"content": result, "usage": {"total_tokens": tokens}}


async def ainvoke_with_retry(
    model: str,
    query: str,
    temperature: float,
    api_key: str,
    max_retries: int = 50,
    initial_delay: int = 1,
    max_tokens: int = 16384,
):
    chutes = Chutes(api_key=os.getenv("CHUTES_API_KEY"), model_timeout=180, create_chute=False)
    delay = initial_delay
    last_exception = None
    for attempt in range(max_retries):
        if chutes.model_exists(model):
            response = chutes.invoke(model, query, temperature, max_tokens)
            return {"content": response, "usage": {"total_tokens": 0}}
        else:
            try:
                response = await call_openai(
                    query,
                    model,
                    temperature,
                    max_tokens,
                    api_key,
            )
                return response
            except Exception as e:
                print("Error in ainvoke_with_retry:", e, "when calling", model)
                # Retry on rate-limit or server errors
                if "429" in str(e) or "529" in str(e) or "list index out of range" in str(e):
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        raise
                else:
                    raise
    if last_exception:
        raise last_exception
    else:
        raise HTTPException(status_code=500, detail="Unknown error invoking LLM")


@app.post("/call", response_model=LLMResponse)
async def call_llm(request: LLMRequest):
    print("Calling LLM", flush=True)
    global current_key, token_usage
    try:
        if not current_key:
            # If no key is initialized, default to "test"
            current_key = "test"
            token_usage[current_key] = 0

        # Attempt the requested LLM; fall back to "gpt-4o" if not found.
        requested_llm = request.llm_name
        if request.llm_name in models:
            requested_llm = f"{models[request.llm_name]['provider']}/{models[request.llm_name]['model']}"
            max_tokens = models[request.llm_name]["max_tokens"]
        if request.max_tokens:
            max_tokens = request.max_tokens

        response = await ainvoke_with_retry(
            requested_llm,
            request.query,
            request.temperature,
            request.api_key,
            max_tokens=max_tokens,
        )

        # Update token usage (if provided by the API)
        tokens = response["usage"].get("total_tokens", 0)
        token_usage[current_key] += tokens

        return LLMResponse(
            result=response["content"], total_tokens=token_usage[current_key]
        )
    except Exception as e:
        print("Error in call_llm endpoint:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#          Embedding Endpoint
# ------------------------------
@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Get embeddings for the given query using OpenAI's embeddings API.
    """
    try:
        response = await embedder.aembed_query(request.query)
        vector = response
        return EmbeddingResponse(vector=vector)
    except Exception as e:
        print("Error in get_embeddings endpoint:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#      Batch Embeddings
# ------------------------------
@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    """
    Returns embedding vectors for a batch of input queries.
    """
    try:
        # Run embedding tasks concurrently for all queries in the batch.
        vectors = await embedder.aembed_documents(request.queries)
        return BatchEmbeddingResponse(vectors=vectors)
    except Exception as e:
        print("An error occurred in get_batch_embeddings", e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
#      Run via Uvicorn
# ------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=25000)
