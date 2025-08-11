# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

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

import os
import sys
import copy
import asyncio
import argparse
import threading
import numpy as np
import bittensor as bt
from scoring_utils import score_spreading

from traceback import print_exception

from coding.mock import MockDendrite
from coding.base.neuron import BaseNeuron
from coding.constants import BLACKLISTED_COLDKEYS
from coding.utils.config import add_validator_args
from coding.utils.exceptions import MaxRetryError
from coding.base.utils.score_grabbing import gather_scores

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)
        self.load_state()
        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        if not hasattr(self, "scores") or self.scores is None:
            self.scores = np.zeros(self.metagraph.n)
        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.axon.attach(
                    forward_fn=self._forward,
                    blacklist_fn=self.blacklist,
                    priority_fn=self.priority,
                )
                self.axon.serve(
                    netuid=self.config.netuid,
                    subtensor=self.subtensor,
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.

        try:
            self.sync()
        except Exception as e:  # Broken pipe handling
            bt.logging.error("Error while syncing, killing self to restart", str(e))
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))
            sys.exit(1)
        if not self.config.neuron.axon_off:
            try:
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
                try:
                    # serve the axon
                    self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
                except Exception as e:
                    bt.logging.error(f"Failed to serve Axon with exception: {e}")
                try:
                    self.axon.start()
                except Exception as e:
                    bt.logging.error(f"Failed to start Axon with exception: {e}")
            except Exception as e:
                bt.logging.error(
                    f"Failed to serve and then start Axon with exception: {e}"
                )
        else:
            bt.logging.info(
                f"Running validator on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
            )

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                forward_timeout = 60*60 # 1 hour
                try:
                    tasks = [
                        self.loop.create_task(asyncio.run(self.forward(synapse=None)))
                        for _ in range(self.config.neuron.num_concurrent_forwards)
                    ]
                    
                    self.loop.run_until_complete(
                        asyncio.wait_for(asyncio.gather(*tasks), timeout=forward_timeout)
                    )
                except MaxRetryError as e:
                    bt.logging.error(f"MaxRetryError: {e}")
                    continue
                except asyncio.TimeoutError as e:
                    bt.logging.error(
                        f"Forward timeout: Task execution exceeded {forward_timeout} seconds and was cancelled.: {e}"
                    )
                    continue
                except (
                    Exception
                ) as e:  # TODO this wasnt here previously, but any errors were cancelling the forward loop so i added it
                    bt.logging.error("Error during validation", str(e))
                    bt.logging.debug(print_exception(type(e), e, e.__traceback__))
                    sys.exit(1)

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()
                if self.step is None:
                    self.step = 0
                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            sys.exit()

        # In case of unforeseen errors, the validator will log the error and quit
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))
            # self.should_exit = True
            sys.exit()

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        # check to be sure self.scores is not all 0's
        for uid in self.metagraph.uids:
            if self.metagraph.coldkeys[uid] in BLACKLISTED_COLDKEYS:
                self.scores[uid] = 0
        if np.all(self.scores == 0):
            bt.logging.warning("self.scores is all 0's, skipping set_weights.")
            return
        divisions = int(np.floor(self.block / 1000))
        print("scores", self.scores)
        # Check if self.scores contains any NaN values and log a warning if it does.
        raw_weights = np.divide(self.scores, np.sum(self.scores, axis=0))
        print("raw_weights", raw_weights)
        raw_weights[raw_weights < 0] = 0
        current_scores = raw_weights
        weighted_scores = score_spreading(
            current_scores,
            divisions,
            0.002,
            0.0025,
            kurtosis_factor=0.5,
            divisions=np.random.randint(2, 9),
        )
        # find the uids of the top 10 weights
        top_10_uids = np.argsort(weighted_scores)[-10:]
        print("top_10_uids", top_10_uids)
        # set all but the top 10 to 0
        weighted_scores[~np.isin(np.arange(len(weighted_scores)), top_10_uids)] = 0
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=weighted_scores,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        print("processed_weights", processed_weights)
        print("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        print("uint_weights", uint_weights)
        print("uint_uids", uint_uids)
        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
            return
        else:
            bt.logging.error(f"set_weights failed {msg}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    async def update_scores(self):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""
        bt.logging.info(f"update_scores()")
        if self.config.neuron.audit:
            tracked_scores = await gather_scores(self)
        if self.config.neuron.audit:
            bt.logging.info("Audit was set, using tracked scores")
            # Temporary override to use tracked scores for audit
            self.scores = np.array(tracked_scores)
            return
        if not self.finetune_results:
            if self.config.neuron.audit:    
                self.scores = np.array(tracked_scores)
            return
        latest_competition_id = max(self.finetune_results.keys())
        bt.logging.info(
            f"latest_competition_id: {latest_competition_id} from {self.finetune_results.keys()}"
        )
        finetune_scores = np.zeros(self.metagraph.n)
        max_score = max(self.finetune_results[latest_competition_id].trackers, key=lambda x: x.score).score
        # group the trackers by if theyre not the same logic. only do this for trackers that have the max score
        tracker_groups = {}
        for tracker in self.finetune_results[latest_competition_id].trackers:
            finetune_scores[tracker.uid] = tracker.score
            if tracker.score != max_score:
                continue
            # Convert dict to tuple of sorted items for hashing
            logic_key = tuple(sorted(tracker.logic.items()))
            if logic_key not in tracker_groups:
                tracker_groups[logic_key] = []
            tracker_groups[logic_key].append(tracker)
        # Calculate how many trackers to select from each group
        trackers_per_group = 10 // len(tracker_groups)
        remainder = 10 % len(tracker_groups)

        # Select trackers from each group
        selected_trackers = []
        not_selected_trackers = []
        for group_trackers in tracker_groups.values():
            # Get number of trackers to select from this group (including remainder distribution)
            n_select = trackers_per_group
            if remainder > 0:
                n_select += 1
                remainder -= 1
                
            # Randomly select trackers from group
            group_selected = group_trackers[:n_select] if len(group_trackers) >= n_select else group_trackers
            selected_trackers.extend(group_selected)
            not_selected_trackers.extend(group_trackers[n_select:])
        # Set scores for selected trackers
        for tracker in selected_trackers:
            finetune_scores[tracker.uid] = max_score
        
        # set scores for not selected trackers to a slighly lower score
        for tracker in not_selected_trackers:
            finetune_scores[tracker.uid] = tracker.score - 0.01
        

        threshold = max_score - 0.17  # within 0.18 of max score
        finetune_scores[finetune_scores < threshold] = 0
        if np.all(finetune_scores == 0):
            bt.logging.warning("finetune_scores is all 0's")
            if self.config.neuron.audit:
                bt.logging.warning("finetune_scores is all 0's, using tracked scores")
                self.scores = np.array(tracked_scores)
                # print out the scores
                print("self.scores after being set to tracked scores", self.scores)
            return
        self.scores = finetune_scores
        bt.logging.info(f"Updated moving avg scores: {self.scores}")
        if self.config.neuron.audit:
            bt.logging.info("Audit was set, comparing scores")
            # Only compare uids that have non-zero scores in both arrays
            non_zero_mask = (self.scores > 0) & (np.array(tracked_scores) > 0)
            common_uids = np.where(non_zero_mask)[0]

            if len(common_uids) > 0:
                scores_subset = self.scores[common_uids]
                tracked_subset = np.array(tracked_scores)[common_uids]
                
                # Calculate relative differences
                relative_diff = np.abs(scores_subset - tracked_subset) / np.maximum(scores_subset, tracked_subset)
                
                # Check if scores are within 10% of each other for at least 80% of common uids
                similar_scores = relative_diff < 0.10
                if np.mean(similar_scores) >= 0.50:
                    bt.logging.info("Using tracked scores - local scores verified similar")
                    self.scores = np.array(tracked_scores)
                else:
                    bt.logging.warning("Using local scores - tracked scores differ significantly")
            else:
                bt.logging.warning("No common non-zero scores to compare, using local scores")
        

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Convert finetune_results to a numpy array of tuples for saving
        finetune_items = np.array(list(self.finetune_results.items()), dtype=object)

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            finetune_items=finetune_items,
            last_task_update=self.last_task_update,
            last_wandb_clean=self.last_wandb_clean,
            last_model_clear=self.last_model_clear,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        state_path = self.config.neuron.full_path + "/state.npz"

        # Check if the state file exists before loading.
        if not os.path.exists(state_path):
            bt.warning("State file not found. Loading default state.")
            self.step = None
            self.scores = None
            self.hotkeys = None
            self.finetune_results = {}
            self.last_task_update = 0
            self.last_wandb_clean = 0
            self.last_model_clear = 0
            return

        # Load the state of the validator from file.
        state = np.load(state_path, allow_pickle=True)

        # Set attributes, using default values if they don't exist in the state file.
        self.step = state["step"].item() if "step" in state else None
        self.scores = state["scores"] if "scores" in state else None
        self.hotkeys = state["hotkeys"] if "hotkeys" in state else None
        self.last_task_update = (
            state["last_task_update"].item() if "last_task_update" in state else 0
        )
        self.last_wandb_clean = (
            state["last_wandb_clean"].item() if "last_wandb_clean" in state else 0
        )
        self.last_model_clear = (
            state["last_model_clear"].item() if "last_model_clear" in state else 0
        )
        # Convert finetune_items back to dictionary
        self.finetune_results = {}
        if "finetune_items" in state:
            for key, value in state["finetune_items"]:
                self.finetune_results[key] = value
