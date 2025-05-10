from dotenv import load_dotenv

load_dotenv("../../../.env", override=False)  # Don't override existing env vars
CHUTES_ONLY = False
import os
import traceback
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
import json

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
    query: Optional[str] = None
    messages: Optional[List[dict]] = None
    tools: Optional[List[dict]] = None
    api_key: str
    llm_name: str
    temperature: Optional[float] = 0.7  # default temperature value
    max_tokens: Optional[int] = None


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict

class LLMResponse(BaseModel):
    result: str
    total_tokens: int
    tool_calls: Optional[List[ToolCall]] = None


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
    query: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    tools: Optional[List[dict]] = None,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 16384,
    api_key: str = None,
):
    print("Calling OpenAI", flush=True)
    print({"query": query, "messages": messages, "tools": tools, "model": model, "temperature": temperature, "max_tokens": max_tokens, "api_key": api_key})
    
    if not api_key:
        print("No API key provided")
        return {"content": "", "usage": {"total_tokens": 0}, "tool_calls": None}
    
    # Prepare arguments, conditionally adding tools if they exist
    kwargs = {
        "model": model,
        "messages": messages if messages else [{"role": "user", "content": query}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {"provider": {"order": ["Anthropic"]}},
    }
    
    # Only add tools if the list is non-empty to avoid validation errors
    if tools and len(tools) > 0:
        kwargs["tools"] = tools
    
    try:
        # Create an async client for OpenAI
        async_client = openai.AsyncOpenAI(api_key=api_key)
        
        # Use the streaming API to get the response
        full_content = ""
        final_tool_calls = {}
        total_tokens = 0
        response_id = None
        
        async for chunk in await async_client.chat.completions.create(stream=True, **kwargs):
            # print(chunk, flush=True)
            if not response_id and hasattr(chunk, 'id'):
                response_id = chunk.id
                
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
            
            # Handle tool calls
            if chunk.choices and hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    index = tool_call.index
                    if tool_call.type == 'function':
                        if index not in final_tool_calls:
                            final_tool_calls[index] = {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments or ""
                                }
                            }
                        elif tool_call.function.arguments:
                            final_tool_calls[index]["function"]["arguments"] += tool_call.function.arguments
                            
            # if hasattr(chunk, 'usage'):
                # total_tokens = chunk.usage.total_tokens
        
        # Convert tool calls to list format
        tool_calls_list = list(final_tool_calls.values()) if final_tool_calls else None
        
        return {
            "content": full_content, 
            "usage": {"total_tokens": total_tokens}, 
            "tool_calls": tool_calls_list
        }
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        traceback.print_exc()
        return {"content": f"API Error: {str(e)}", "usage": {"total_tokens": 0}, "tool_calls": None}

async def ainvoke_with_retry(
    model: str,
    query: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    tools: Optional[List[dict]] = None,
    temperature: float = 0.7,
    api_key: str = None,
    max_retries: int = 5,
    initial_delay: int = 1,
    max_tokens: int = 16384,
):
    chutes = Chutes(api_key=os.getenv("CHUTES_API_KEY"), model_timeout=180, create_chute=False)
    delay = initial_delay
    last_exception = None
    for attempt in range(max_retries):
        if query and not messages:
            messages = [{"role": "user", "content": query}]
            
        try:
            # Check if model exists in Chutes
            model_exists = await chutes.model_exists_async(model)
            if model_exists:
                print("Using Chutes for ", model, flush=True)
                response = await chutes.invoke_async(model, messages, temperature, tools, max_tokens, timeout=60)
                return {"content": response, "usage": {"total_tokens": 0}}
            elif not CHUTES_ONLY:
                response = await call_openai(
                    query,
                    messages,
                    tools,
                    model,
                    temperature,
                    max_tokens,
                    api_key,
                )
                return response
            else:
                raise Exception(f"Model {model} not found in Chutes and CHUTES_ONLY is set")
                
        except TimeoutError as e:
            print(f"Timeout error: {e}")
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                print("Max retries reached.")
                raise HTTPException(
                    status_code=408, 
                    detail=f"Timeout when calling {model} after {max_retries} attempts"
                )
        except Exception as e:
            print("Error in ainvoke_with_retry:", e, "when calling", model)
            traceback.print_exc()
            # Retry on rate-limit or server errors
            # Added check for potential None response or missing keys which might cause errors
            if isinstance(e, (openai.RateLimitError, openai.APIStatusError)) or \
            (isinstance(e, IndexError) and "list index out of range" in str(e)) or \
            "529" in str(e): # Check common retryable error types/codes
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    print("Max retries reached.")
                    raise # Re-raise the last exception after max retries
            else:
                print("Non-retryable error encountered.")
                raise # Re-raise immediately for non-retryable errors
                
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
        max_tokens = models.get("gpt-4o", {}).get("max_tokens", 16384) # Default max_tokens
        if request.llm_name in models:
            model_config = models[request.llm_name]
            # Assuming provider info is part of the model name passed to call_openai now
            # Let's keep requested_llm as the key for models dict for simplicity here.
            # The actual API model name is fetched within call_openai or ainvoke_with_retry based on the key
            # For openrouter, the format is usually "provider/model"
            requested_llm = f"{model_config['provider']}/{model_config['model']}" # Construct full model name if needed by API
            max_tokens = model_config["max_tokens"]
        else:
            # Handle case where llm_name is not in models dict - maybe it's a direct model string?
            # We might need a default provider or assume it's openai compatible
            print(f"Warning: llm_name '{request.llm_name}' not found in configured models. Attempting direct call.")
            # If provider isn't specified, how does call_openai know? Let's assume it's openai format or similar
            # Or maybe the name already includes the provider like "openai/gpt-4o"
            requested_llm = request.llm_name # Use the name directly


        if request.max_tokens:
            max_tokens = request.max_tokens # Override if provided in request

        response = await ainvoke_with_retry(
            requested_llm,
            request.query,
            request.messages,
            request.tools,
            request.temperature,
            request.api_key,
            max_tokens=max_tokens,
        )

        # Update token usage (if provided by the API)
        tokens = response.get("usage", {}).get("total_tokens", 0)
        token_usage[current_key] += tokens

        # Process tool calls if present
        tool_calls_response = None
        if response.get("tool_calls"):
            tool_calls_response = [
                ToolCall(id=tc["id"], name=tc["function"]["name"], args=json.loads(tc["function"]["arguments"]))
                for tc in response["tool_calls"]
            ]

        # Ensure result is a string, provide default if content is None
        result_content = response.get("content", "") or ""

        return LLMResponse(
            result=result_content, # Use the processed result_content
            total_tokens=token_usage[current_key],
            tool_calls=tool_calls_response # Pass the processed tool calls
        )
    except Exception as e:
        print("Error in call_llm endpoint:", e)
        traceback.print_exc() # Print full traceback for debugging
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
