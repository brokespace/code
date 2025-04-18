# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Macrocosmos
# Copyright © 2024 Brokespace


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Callable, Any

from coding.tasks import TASKS
from .reward import BaseRewardModel
from .codesim import CodeSimModel
from .speed import SpeedModel
from .diffsim import DiffSimModel
from .validcode import ValidCodeModel

REWARD_MODELS = {
    "codesim": CodeSimModel,
    DiffSimModel().name: DiffSimModel,
    SpeedModel().name: SpeedModel,
    ValidCodeModel().name: ValidCodeModel,
    "self": None,
}


class RewardPipeline:
    def __init__(self, selected_tasks: List[str], device: str, code_scorer):
        self.selected_tasks = selected_tasks
        self.device = device
        self.code_scorer = code_scorer
        self.validate_tasks()
        self.load_reward_pipeline()

    def __getitem__(self, __key: str) -> BaseRewardModel:
        return self.reward_models.get(__key)

    def get(self, __key: str) -> BaseRewardModel:
        return self.reward_models.get(__key)

    def keys(self) -> List[str]:  # TODO this might not be the right return type
        return self.reward_models.keys()

    def __repr__(self):
        return f"RewardPipeline({self.reward_models})"

    def validate_tasks(self):
        for task in self.selected_tasks:
            if task not in TASKS:
                raise ValueError(
                    f"Task {task} not supported. Please choose from {TASKS.keys()}"
                )
            # Check that the reward_definition and penalty_definition are lists of dictionaries whose weights sum to one
            self._check_weights(task, "reward_definition", expected_weight=1)
            self._check_weights(task, "penalty_definition", expected_weight=None)

    def _check_weights(self, task, definition, expected_weight):
        total_weight = 0

        model_infos = getattr(TASKS[task], definition)

        for model_info in model_infos:
            if not isinstance(model_info, dict):
                raise ValueError(
                    f"{definition} model {model_info} is not a dictionary."
                )
            if "weight" not in model_info:
                raise ValueError(
                    f"{definition} model {model_info} does not have a weight."
                )

            weight = model_info["weight"]
            if not isinstance(weight, (float, int)):
                raise ValueError(
                    f"{definition} model {model_info} weight is not a float."
                )
            if not 0 <= weight <= 1:
                raise ValueError(
                    f"{definition} model {model_info} weight is not between 0 and 1."
                )

            total_weight += weight

        if (
            model_infos
            and expected_weight is not None
            and total_weight != expected_weight
        ):
            raise ValueError(
                f"{definition} model {model_infos} weights do not sum to {expected_weight} (sum={total_weight})"
            )

    def load_reward_pipeline(self):
        """Dynamically loads the reward models required by the selected tasks so that we only use the necessary resources."""
        active_reward_models = []

        for task in self.selected_tasks:
            active_reward_models += TASKS[task].reward_definition
            active_reward_models += TASKS[task].penalty_definition

        # Instantiate only the required reward models
        reward_models = {}
        for model in active_reward_models:
            name = model.get("name")
            if not name:
                raise ValueError(f"Reward model {model} does not have a name. ")
            if name not in REWARD_MODELS.keys():
                raise ValueError(
                    f"Reward model {name} not supported. Please choose from {REWARD_MODELS.keys()}"
                )
            elif name in reward_models:  # Prevents duplicate reward models
                continue

            cls = REWARD_MODELS[name]
            if name == "self":
                reward_models[name] = "self"
                continue
            params = {k: v for k, v in model.items() if k not in ["name", "weight"]}
            reward_models[name] = cls(
                device=self.device, code_scorer=self.code_scorer, **params
            )

        self.reward_models = reward_models
