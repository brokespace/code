from typing import List
from coding.tasks.task import Task
from coding.finetune.model import load_model_and_tokenizer, evaluate, cleanup
from coding.rewards.codesim import CodeSimModel

def score(validator, model_name: str, prompt_tokens: dict, tasks: List[Task], codesim: CodeSimModel) -> float:
    """
    Calculate the average score across multiple tasks for a given model.

    Args:
        model_name (str): Name or path of the model to evaluate
        prompt_tokens (dict): Dictionary containing FIM prompt tokens:
            - "prefix": the prefix of the prompt
            - "middle": the middle of the prompt
            - "suffix": the suffix of the prompt
        tasks (List[Task]): List of Task objects to evaluate the model on. Task must be of the FIM type.

    Returns:
        float: Average score across all tasks, where each task score is between 0 and 1

    The function:
    1. Loads the model and tokenizer
    2. For each task:
        - Evaluates the model's response on the task query
        - Calculates a score for that response
    3. Cleans up model resources
    4. Returns mean score across all tasks
    """
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, validator.config.finetune_gpu_id)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}") # TODO change to logging
        return 0.0
    scores = []
    responses = [evaluate(model, tokenizer, prompt_tokens, task.query) for task in tasks]
    references = [task.reference for task in tasks]
    scores = codesim.batch_similarity(references, responses)
    cleanup(model, tokenizer)
    return sum(scores) / len(scores)

