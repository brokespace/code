# SWE Finetuning

## Task Outline

The task is to create a patch that fixes an issue in the repository. You will be provided the location to a repository and a description of the issue. This will be a real git repository as well as a real issue and you will be graded against the real patch. 

### What is a patch?

A patch is a list of edits to the repository. Each edit is an edit of a file, containing the file name, line number, line content, and new line content. As defined in the `Patch` class below.

```python
class Edit(BaseModel):
    file_name: str
    line_number: int
    line_content: str
    new_line_content: str

class Patch(BaseModel):
    edits: list[Edit]
```

## Things available to you

### Repository

You will be provided a repository to work on. There is a conda env created `testbed` that has all the dependencies installed, allowing you to call `pytest -rA` or other commands to run tests, scripts, etc. 

To access this environment, you will need to ensure that you have the `testbed` environment activated. 

```bash
source /opt/miniconda3/bin/activate # this cmd may not be needed, but it's good to have 
conda activate testbed
```

### Size Limits

You will have access to the `NUM_ALLOWED_CHARACTERS` variable in the `coding/constants.py` file. This is the maximum number of characters that can be used in your submission.

### LLM Models

You will have access to the following LLM models:

- "gpt-4o"
- "gpt-3.5-turbo"
- "gpt-4o-mini"
- "claude-3-5-sonnet"
- "gemini-2.0-flash-exp"

- Along with that any model hosted from OpenRouter.

You will also have access to the following embedding models:

- "text-embedding-3-small"

#### How to use the models

You can use the models by calling the `llm` property of the `SWEBase` class. For example:

```python
from coding.finetune.swe-server.swebase import SWEBase

swe = SWEBase()
response, tokens = swe.llm("gpt-4o", "What is the capital of France?")
embeddings = swe.llm.embed("What is the capital of France?")
```

#### Tool Calling

We utilize OpenRouter to handle tool calling, as such they have a formalized input for ALL LLM's, namely they use the OpenAI format. You can see more [here](https://openrouter.ai/docs/features/tool-calling).

Utilizing the LLMClient, you can call tools by passing in a list of tools. It must be a list of dictionaries, an example of which is below:

```python
tools = [
  {
    "type": "function",
    "function": {
      "name": "search_gutenberg_books",
      "description": "Search for books in the Project Gutenberg library based on specified search terms",
      "parameters": {
        "type": "object",
        "properties": {
          "search_terms": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "List of search terms to find books in the Gutenberg library (e.g. ['dickens', 'great'] to search for books by Dickens with 'great' in the title)"
          }
        },
        "required": ["search_terms"]
      }
    }
  }
]
```

The following response type is provided by the LLM:

```python
class ToolCall(BaseModel):
    name: str
    args: dict

class Response(BaseModel):
    result: str
    total_tokens: int
    tool_calls: Optional[list[ToolCall]] = None
```

See [swebase.py](../../coding/finetune/swe-server/swebase.py) for more details, and the up to date method.


#### Reminders

- The server that hosts your code is restricted to not allow for internet access. You should not try to use it as you will likely fail.

## Submission

Locate the `coding/miners/swe.py` file. This is where your miner will go to grab your submission.

Your submission must initiate a class `SWE` that inherits from `SWEBase`. This will be called with a `repo_location` and `issue_description`. 

The `SWE` class must return a `Patch` object. This will be used to evaluate your submission.

## Testing

Use the notebook `notebooks/sample-swe-task.ipynb` to test your submission.

You need to verify your logic using the notebook `notebooks/logic-verification.ipynb`. 
