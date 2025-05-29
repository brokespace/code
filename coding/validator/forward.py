import os
from time import sleep
import bittensor as bt
from datetime import datetime, timezone, timedelta

from coding.constants import COMPETITION_ID
from coding.protocol import StreamCodeSynapse
from coding.helpers.results import forward_results
from coding.finetune.pipeline import FinetunePipeline
from coding.utils.logging import log_event, clean_wandb
from coding.finetune.dockerutil import delete_all_containers

async def forward(self, synapse: StreamCodeSynapse):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    bt.logging.info("🚀 Starting forward loop...")
    if not FinetunePipeline.tasks_exist(self.config):
        FinetunePipeline.generate_tasks(self.config)
    
    if self.last_task_update + 10800 < self.block: # every 1.5 days replace 50(half) the tasks
        FinetunePipeline.update_tasks(self.config, 50, 100)
        self.last_task_update = self.block
    
    if not hasattr(self, 'finetune_eval_future'):
        if self.last_model_clear + 14400 * 3 < self.block:
            self.model_store.clear_all()
            # delete trackers
            if os.path.exists(f"{self.config.neuron.full_path}/trackers_{COMPETITION_ID}.pkl"):
                os.remove(f"{self.config.neuron.full_path}/trackers_{COMPETITION_ID}.pkl")
            self.last_model_clear = self.block
        delete_all_containers(os.getenv("REMOTE_DOCKER_HOST", None))
        sleep(10) # wait for containers to be truly deleted
        print("Creating finetune pipeline")
        finetune_pipeline = FinetunePipeline(
            config=self.config,
            use_remote=True,
            model_store=self.model_store,
        )
        print("Finetune pipeline created, submitting evaluation")
        self.finetune_eval_future = self.executor.submit(finetune_pipeline.evaluate)
    # Check if evaluation is complete
    if hasattr(self, "finetune_eval_future") and self.finetune_eval_future.done():
        self.finetune_results[COMPETITION_ID] = self.finetune_eval_future.result()
        delattr(self, "finetune_eval_future")  # Remove the future after getting results

    self.update_scores()

    log_event(
        self,
        {
            "step": self.step,
            **(
                self.finetune_results[COMPETITION_ID].__state_dict__()
                if COMPETITION_ID in self.finetune_results
                else {}
            ),
        },
    )

    # Call clean_wandb once every day
    if self.last_wandb_clean + 7200 < self.block:
        try:    
            clean_wandb(self)
            self.last_wandb_clean = self.block
        except Exception as e:
            bt.logging.error(f"Error cleaning wandb: {e}")

    try:
        await forward_results(self)
    except Exception as e:
        bt.logging.error(f"Error during forward while waiting for results: {e}")
    
    sleep(60*5)