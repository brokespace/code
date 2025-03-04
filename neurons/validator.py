# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Broke

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
import dotenv

dotenv.load_dotenv()
import os
import sys
import time
import traceback
from time import sleep
import bittensor as bt
from typing import Awaitable, Tuple
from concurrent.futures import ThreadPoolExecutor

from coding.validator.forward import forward
from coding.protocol import EvaluationSynapse
from coding.constants import HONEST_VALIDATOR_HOTKEYS
# import base validator class which takes care of most of the boilerplate
from coding.utils.config import config as util_config
from coding.base.validator import BaseValidatorNeuron
from coding.utils.logging import init_wandb_if_not_exists
from coding.finetune.coordinator import FinetuneCoordinator
from coding.finetune.dockerutil import test_docker_container
from coding.helpers.containers import DockerServer, test_docker_server


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        if not config:
            config = util_config(self)
        self.finetune_results = {}
        super(Validator, self).__init__(config=config)

        self.last_task_update = 0
        self.last_wandb_clean = self.block
        bt.logging.info("load_state()")
        self.load_state()
        if self.last_task_update == 0:
            self.last_task_update = self.block
        self.last_finetune_eval_time = 0
        init_wandb_if_not_exists(self)
        # self.active_tasks = [
        #     task
        #     for task, p in zip(
        #         self.config.neuron.tasks, self.config.neuron.task_weights
        #     )
        #     if p > 0
        # ]
        self.executor = ThreadPoolExecutor()
        # Load the reward pipeline
        # self.reward_pipeline = RewardPipeline(
            # selected_tasks=self.active_tasks,
            # device=self.device,
            # code_scorer=None,
        # )
        self.docker_server = DockerServer(
            remote_host_url=os.getenv("REMOTE_DOCKER_HOST"),
            remote_host_registry=f"{os.getenv('DOCKER_HOST_IP')}:5000"
        )
        try:
            self.docker_server.remote.run("registry:2", ports={"5000/tcp": 5000}, name="swe-registry")
        except Exception as e:
            bt.logging.error(f"Error running registry: {e}")
            print(traceback.format_exc())
        test_result = test_docker_container(os.getenv("REMOTE_DOCKER_HOST"))
        if not test_result:
            bt.logging.error("Docker container test failed, exiting.")
            sys.exit(1)
        while True:
            docker_server_test = test_docker_server()
            if docker_server_test:
                break
            bt.logging.error("Docker server test failed, waiting 3 minutes and trying again.")
            sleep(60*3)
        
        self.coordinator = FinetuneCoordinator(self.config, self.wallet)

    def _forward(
        self, synapse: EvaluationSynapse
    ) -> (
        EvaluationSynapse
    ): 
        """
        forward method that is called when the validator is queried with an axon
        """
        if synapse is not None:
            print("Synapse requested for model. ", synapse)
            status = self.coordinator.get_model_status(synapse.model_hash)
            if status:
                synapse.in_progress = status.in_progress
                synapse.completed = status.completed
                synapse.score = status.score
                synapse.started_at = status.started_at
                synapse.completed_at = status.completed_at
                synapse.server_id = status.server_id
            synapse.alive = True
            print("Synapse returned. ", synapse)
            return synapse
        return forward(self, synapse)
    
    async def forward(self, synapse: EvaluationSynapse) -> Awaitable:
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        if synapse is not None:
            print("Synapse requested for model. ", synapse)
            status = self.coordinator.get_model_status(synapse.model_hash)
            if status:
                synapse.in_progress = status.in_progress
                synapse.completed = status.completed
                synapse.score = status.score
                synapse.started_at = status.started_at
                synapse.completed_at = status.completed_at
                synapse.server_id = status.server_id
            print("Synapse returned. ", synapse)
            synapse.alive = True
            return synapse
        return forward(self, synapse)

    # TODO make it so that the only thing accepted is the subnet owners hotkey + the validators coldkey
    async def blacklist(self, synapse: EvaluationSynapse) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite.hotkey == "5Fy7c6skhxBifdPPEs3TyytxFc7Rq6UdLqysNPZ5AMAUbRQx":
            return False, "Subnet owner hotkey"
        if synapse.dendrite.hotkey in HONEST_VALIDATOR_HOTKEYS:
            return False, "Honest validator hotkey"
        return True, "Blacklisted"

    async def priority(self, synapse: EvaluationSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            if not validator.thread.is_alive():
                bt.logging.error("Child thread has exited, terminating parent thread.")
                sys.exit(1)  # Exit the parent thread if the child thread dies
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
