import os
import time
import json
import random
import pickle
import difflib
import traceback
import threading
import concurrent.futures
import bittensor as bt
from typing import List
from pydantic import BaseModel
from .tracker import gather_all_logics
from concurrent.futures import ThreadPoolExecutor, as_completed

from .dockerutil import run_docker_container_from_base

from coding.schemas import Patch
from coding.schemas.context import Context
from coding.constants import COMPETITION_ID
from coding.rewards.codesim import CodeSimModel
from coding.schemas.tracking import TrackingInfo
from coding.constants import (
    COMPETITION_ID,
    ALLOWED_MODULES,
    NUM_ALLOWED_CHARACTERS,
    ALLOWED_IMPORTS,
)

from coding.tasks.swe import SWEBenchTask
from coding.datasets.swefull import SWEFullDataset
from coding.finetune.llm.manager import LLMManager
from coding.helpers.containers import DockerServer
from coding.finetune.model import ModelStore, validate_logic


class FinetuneEventResults(BaseModel):
    trackers: List[TrackingInfo]
    competition_id: int = COMPETITION_ID

    def __state_dict__(self):
        return {
            # "trackers": [tracker.model_dump() for tracker in self.trackers],
            "competition_id": COMPETITION_ID,
        }

    def public_state_dict(self):
        trackers = [tracker.model_dump() for tracker in self.trackers]
        for tracker in trackers:
            tracker["model"] = None
        return {
            "trackers": trackers,
            "competition_id": COMPETITION_ID,
        }


def should_evaluate(tracker: TrackingInfo, block: int) -> bool:
    """
    Check if the tracker should be evaluated at the given block number.

    Conditions:
    - If there have been fewer than 3 evaluations in the last 7 days, return True.
    - Otherwise, return False.
    """
    # Calculate blocks in 7 days
    blocks_in_7_days = 7 * 24 * 60 * 60 // 12
    block_7_days_ago = block - blocks_in_7_days
    # Get evaluations within the last 7 days
    recent_evals = [b for b in tracker.score_timestamps if b > block_7_days_ago]

    # Return True if there are fewer than 3 evaluations in the last 7 days
    return len(recent_evals) < 3


def generate_swe_tasks(
    ds, n: int = 1000, docker_server=None, use_remote: bool = False, use_threading: bool = True
) -> List[SWEBenchTask]:
    tasks = []
    fail_count = 0
    ds_lock = threading.Lock()
    
    def create_task():
        nonlocal fail_count
        try:
            with ds_lock:
                context_data = ds.get()
            return SWEBenchTask(
                llm=None,
                context=Context(**context_data),
                docker_server=docker_server,
                use_remote=use_remote,
            )
        except Exception as e:
            bt.logging.error(f"Error generating task: {e}")
            print(traceback.format_exc())
            fail_count += 1
            return None
    
    if use_threading:
        max_workers = min(16, n)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create more tasks than needed to account for failures
            future_tasks = [executor.submit(create_task) for _ in range(n * 2)]
            
            for future in concurrent.futures.as_completed(future_tasks):
                if len(tasks) >= n:
                    break
                    
                task = future.result()
                if task is not None:
                    tasks.append(task)
                
                if fail_count > 100:
                    raise Exception("Failed to generate tasks")
    else:
        # Sequential task creation
        attempts = 0
        while len(tasks) < n and attempts < n * 2 and fail_count <= 100:
            task = create_task()
            attempts += 1
            if task is not None:
                tasks.append(task)
            
            if fail_count > 100:
                raise Exception("Failed to generate tasks")
    
    return tasks[:n]
def bittensor_injector(self):
    self.wallet = bt.wallet(config=self.config)
    self.dendrite = bt.dendrite(wallet=self.wallet)
    self.subtensor = bt.subtensor(config=self.config)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)


class FinetunePipeline:
    def __init__(
        self,
        config,
        tracking_logics: List[TrackingInfo] = None,
        use_remote: bool = False,
        coordinator = None
    ):
        self.config = config
        self.use_remote = use_remote
        try:
            bittensor_injector(self)
        except Exception as e:
            bt.logging.error(f"Error injecting bittensor: {e}")
            print(traceback.format_exc())
        self.docker_server = DockerServer(
            remote_host_url=os.getenv("REMOTE_DOCKER_HOST") if use_remote else None,
            remote_host_registry=(
                f"{os.getenv('DOCKER_HOST_IP')}:5000" if use_remote else None
            ),
        )
        self.graded_trackers = []
        self.ungraded_trackers = []
        self.dataset = SWEFullDataset()
        self.llm_manager = LLMManager()
        self.load_model_store()
        if tracking_logics is None:
            self.load_logics()
        else:
            self.ungraded_trackers = tracking_logics
        self.load_tasks()

    def load_tasks(self):
        print(
            f"Loading tasks from {self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl"
        )
        if os.path.exists(f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl"):
            with open(
                f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "rb"
            ) as f:
                self.tasks = pickle.load(f)[: self.config.neuron.finetune_test_size]
                for task in self.tasks:
                    task.docker_server = self.docker_server
        else:
            self.tasks = generate_swe_tasks(
                self.dataset,
                self.config.neuron.finetune_test_size,
                docker_server=self.docker_server,
                use_remote=self.use_remote,
            )
            self.store_tasks()
        print(f"Loaded {len(self.tasks)} tasks")

    def load_logics(self):
        grabbed_trackers = gather_all_logics(self)
        print(f"Grabbed {len(grabbed_trackers)} logics")
        saved_trackers = self.load_trackers()
        graded_trackers = []
        ungraded_trackers = []
        for tracker in grabbed_trackers:
            self.model_store.upsert(tracker.logic)
            exists = False
            for saved_tracker in saved_trackers:
                if len(saved_tracker.score_timestamps) == 0:
                    saved_tracker.score_timestamps.append(saved_tracker.block)
                if tracker.hotkey == saved_tracker.hotkey:
                    saved_tracker.uid = tracker.uid
                    tracker.score = saved_tracker.score
                    tracker.score_timestamps = saved_tracker.score_timestamps
                    if (
                        len(saved_tracker.score_timestamps) > 0
                        and saved_tracker.score_timestamps[-1]
                        < self.subtensor.block - 14400
                    ):
                        break
                    exists = True
                    if saved_tracker.score == 0:
                        ungraded_trackers.append(tracker)
                        break
                    if (
                        tracker.logic != {}
                        and difflib.SequenceMatcher(
                            None,
                            json.dumps(tracker.logic, sort_keys=True),
                            json.dumps(saved_tracker.logic, sort_keys=True),
                        ).quick_ratio()
                        > 0.90
                    ):
                        model = self.model_store.get(tracker.logic)
                        if not model or not model.valid:
                            tracker.score = 0
                            ungraded_trackers.append(tracker)
                        else:
                            graded_trackers.append(saved_tracker)
                    else:
                        if (
                            tracker.logic == {}
                            and saved_tracker.logic != {}
                        ):
                            model = self.model_store.get(saved_tracker.logic)
                            if not model or not model.valid:
                                saved_tracker.score = 0
                            graded_trackers.append(saved_tracker)
                        else:
                            model = self.model_store.get(tracker.logic)
                            if not model or not model.valid:
                                tracker.score = 0
                            ungraded_trackers.append(tracker)
                    break
            if not exists:
                ungraded_trackers.append(tracker)
        self.graded_trackers = graded_trackers
        self.ungraded_trackers = ungraded_trackers
        for tracker in self.graded_trackers:
            model = self.model_store.get(tracker.logic)
            if model:
                self.coordinator.mark_evaluation_started(model.hash)
        self.store_model_store()
        print(
            f"Loaded {len(self.graded_trackers)} graded and {len(self.ungraded_trackers)} ungraded trackers"
        )

    @property
    def results(self) -> FinetuneEventResults:
        return FinetuneEventResults(trackers=self.graded_trackers)

    def evaluate(self, n_tasks: int = None, store_results: bool = True) -> FinetuneEventResults:
        # gather all logics
        bt.logging.info("Verifying and building docker containers for each logic...")
        for tracker in self.ungraded_trackers:
            bt.logging.info(f"Verifying logic for hotkey {tracker.hotkey}...")
            model = self.model_store.upsert(tracker.logic)
            if model and not model.valid:
                bt.logging.info(
                    f"Logic for hotkey {tracker.hotkey} is invalid, skipping..."
                )
                tracker.logic = {}
                continue
            bt.logging.info(f"Logic for hotkey {tracker.hotkey} passed verification.")
        bt.logging.info(f"Beginning evaluation of {len(self.tasks)} tasks...")
        random.shuffle(self.ungraded_trackers)
        remote_graded_trackers = []
        for tracker_idx, tracker in enumerate(self.ungraded_trackers):
            bt.logging.info(
                f"Processing tracker {tracker_idx + 1}/{len(self.ungraded_trackers)}"
            )
            # Skip if no logic provided
            if not tracker.logic:
                bt.logging.info(
                    f"No logic provided for tracker {tracker.hotkey}, skipping..."
                )
                self.graded_trackers.append(tracker)
                continue
            if not should_evaluate(tracker, self.metagraph.block):
                bt.logging.info(
                    f"Not enough blocks have passed since the last evaluation for tracker {tracker.hotkey}, skipping..."
                )
                self.graded_trackers.append(tracker)
                continue
            model = self.model_store.upsert(tracker.logic)
            if not self.coordinator.should_evaluate_model(model.hash):
                remote_graded_trackers.append(tracker)
                continue
            
            previous_tracker = next(
                (
                    t
                    for t in self.graded_trackers
                    if difflib.SequenceMatcher(
                        None,
                        json.dumps(tracker.logic, sort_keys=True),
                        json.dumps(t.logic, sort_keys=True),
                    ).quick_ratio()
                    > 0.90
                ),
                None,
            )
            if previous_tracker is not None:
                if (
                    len(previous_tracker.score_timestamps) > 0
                    and len(tracker.score_timestamps) == 0
                ):
                    tracker.score_timestamps = previous_tracker.score_timestamps
                bt.logging.info(
                    f"Finetune: Using previously evaluated score for hotkey: {tracker.hotkey}"
                )
                # if a tracker had a score before, add the block number to the score_timestamps
                if tracker.score > 0 or len(tracker.score_timestamps) == 0:
                    tracker.score_timestamps.append(self.metagraph.block)
                tracker.score = previous_tracker.score
                self.graded_trackers.append(tracker)
                # if tracker.hotkey != previous_tracker.hotkey:
                # self.trackers.append(tracker)
                self.coordinator.mark_evaluation_complete(model.hash, tracker.score)
                continue

            # Otherwise, evaluate the logic
            self.coordinator.mark_evaluation_started(model.hash)
            bt.logging.info(f"Initializing LLM key for hotkey {tracker.hotkey}...")
            self.llm_manager.init_key(tracker.hotkey)
            bt.logging.info(f"Starting docker container for hotkey {tracker.hotkey}...")
            scores = []
            # Create a thread pool to process tasks in parallel
            bt.logging.info("Starting thread pool for task processing...")
            with ThreadPoolExecutor() as executor:
                bt.logging.info("Thread pool started.")

                def process_task(task_data):
                    bt.logging.info(f"Processing task...")
                    task_idx, task = task_data
                    try:
                        bt.logging.info(
                            f"Making request to container for hotkey {tracker.hotkey}, task index {task_idx}..."
                        )
                        result = run_docker_container_from_base(
                            image_name=task.image_name,
                            container_name=f"swe-logic-{str(tracker.hotkey)}-{COMPETITION_ID}-{task_idx}".lower(),
                            repo=task.repo,
                            hotkey=tracker.hotkey,
                            issue_description=task.query,
                            base_commit=task.row["base_commit"],
                            logic_files=tracker.logic,
                            client=(
                                self.docker_server._remote_client
                                if self.use_remote
                                else self.docker_server._local_client
                            ),
                            remote_host_url=(
                                os.getenv("REMOTE_DOCKER_HOST")
                                if self.use_remote
                                else None
                            ),
                        )
                        patch = Patch(**result)
                        bt.logging.info(
                            f"Scoring response for hotkey {tracker.hotkey}, task index {task_idx}..."
                        )
                        # TODO in the next comp uncomment the below
                        # score = task.score(patch, self.llm_manager.get_count())
                        score = task.score(patch)
                        self.llm_manager.reset_count()
                        bt.logging.info(
                            f"Score for hotkey {tracker.hotkey}, task index {task_idx}: {score}"
                        )
                        return score
                    except Exception as e:
                        bt.logging.error(
                            f"Request failed for hotkey {tracker.hotkey}, task index {task_idx}: {e}"
                        )
                        print(traceback.format_exc())
                        return 0

                # Keep track of active futures and tasks
                active_futures = {}
                task_queue = list(enumerate(self.tasks))
                if n_tasks is not None:
                    task_queue = task_queue[:n_tasks]
                task_idx = 0

                # Start initial batch of 8 tasks
                bt.logging.info("Starting initial batch of 8 tasks...")
                while len(active_futures) < 8 and task_queue:
                    task_data = task_queue.pop(0)
                    future = executor.submit(process_task, task_data)
                    active_futures[future] = task_data

                bt.logging.info(
                    f"Task queue drained, active futures left: {len(active_futures)}"
                )
                # Process remaining tasks as others complete
                while active_futures:
                    completed_future = next(as_completed(active_futures))
                    task_data = active_futures.pop(completed_future)

                    # Get score from completed task
                    score = completed_future.result()
                    scores.append(score)
                    bt.logging.info(
                        f"Average score for hotkey {tracker.hotkey}: {sum(scores) / len(scores)}"
                    )

                    # Start next task if any remain
                    if task_queue:
                        task_data = task_queue.pop(0)
                        future = executor.submit(process_task, task_data)
                        active_futures[future] = task_data

                    task_idx += 1
                    bt.logging.info(
                        f"Completed task {task_idx}/{len(self.tasks)} for hotkey {tracker.hotkey}"
                    )
            tracker.score = sum(scores) / len(scores)
            tracker.score_timestamps.append(self.metagraph.block)
            self.graded_trackers.append(tracker)
            if store_results:
                self.store_trackers()

            bt.logging.info(f"Cleaning up container for hotkey {tracker.hotkey}...")
            bt.logging.info(f"Final score for hotkey {tracker.hotkey}: {tracker.score}")
            self.coordinator.mark_evaluation_complete(model.hash, tracker.score)
        bt.logging.info("Evaluation complete!")
        if store_results:
            self.store_trackers()
        
        # Wait for all evaluations to complete
        while True:
            statuses = self.coordinator.get_all_statuses()
            if all(status.completed for status in statuses):
                break
            bt.logging.info("Waiting for all evaluations to complete...")
            time.sleep(60)
        
        for tracker in remote_graded_trackers:
            model = self.model_store.get(tracker.logic)
            model_status = self.coordinator.get_model_status(model.hash)
            tracker.score = model_status.score
            tracker.score_timestamps.append(model_status.completed_at)
            self.graded_trackers.append(tracker)
            if store_results:
                self.store_trackers()
        
        return self.results

    def __str__(self):
        return f"{self.__class__.__name__}(scores={self.scores!r})"

    def __repr__(self):
        return self.__str__()

    def __state_dict__(self):
        return {
            "scores": self.scores,
        }

    @staticmethod
    def start(
        config,
    ) -> FinetuneEventResults:
        pipeline = FinetunePipeline(config)
        result = pipeline.evaluate()
        pipeline.cleanup()  # Ensure cleanup is called after evaluation
        return result

    def load_model_store(self):
        if os.path.exists(
            f"{self.config.neuron.full_path}/models_{COMPETITION_ID}.pkl"
        ):
            with open(
                f"{self.config.neuron.full_path}/models_{COMPETITION_ID}.pkl", "rb"
            ) as f:
                self.model_store = pickle.load(f)
        else:
            self.model_store = ModelStore()

    def store_model_store(self):
        with open(
            f"{self.config.neuron.full_path}/models_{COMPETITION_ID}.pkl", "wb"
        ) as f:
            pickle.dump(self.model_store, f)

    def load_trackers(self):
        loaded_trackers = []
        store_file = f"{self.config.neuron.full_path}/trackers_{COMPETITION_ID}.pkl"
        if os.path.exists(store_file):
            with open(store_file, "rb") as f:
                saved_results = pickle.load(f)
                loaded_trackers = saved_results.get("trackers", [])
        return loaded_trackers

    def store_tasks(self):
        with open(
            f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "wb"
        ) as f:
            pickle.dump(self.tasks, f)

    def store_trackers(self):
        store_file = f"{self.config.neuron.full_path}/trackers_{COMPETITION_ID}.pkl"
        temp_file = store_file + ".tmp"

        # Write to a temp file first
        with open(temp_file, "wb") as f:
            pickle.dump({"trackers": self.graded_trackers}, f)

        # Replace the old file with the new
        os.replace(temp_file, store_file)

    @staticmethod
    def generate_tasks(config) -> List[SWEBenchTask]:
        dataset = SWEFullDataset()
        tasks = generate_swe_tasks(
            dataset,
            config.neuron.finetune_test_size,
            docker_server=DockerServer(
                remote_host_url=os.getenv("REMOTE_DOCKER_HOST"),
                remote_host_registry=f"{os.getenv('DOCKER_HOST_IP')}:5000",
            ),
            use_remote=True,
        )
        with open(f"{config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "wb") as f:
            for task in tasks:
                task.docker_server = None
            pickle.dump(tasks, f)

    @staticmethod
    def update_tasks(config, num_tasks_to_keep: int, num_tasks_wanted: int):
        if os.path.exists(f"{config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl"):
            with open(
                f"{config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "rb"
            ) as f:
                tasks = pickle.load(f)
                # Clean up tasks that will be removed
                for task in tasks[:num_tasks_to_keep]:
                    task._cleanup()
                tasks = tasks[num_tasks_to_keep:]  # Remove the first N tasks
        else:
            tasks = []
        dataset = SWEFullDataset()
        if len(tasks) < num_tasks_wanted:
            new_tasks = generate_swe_tasks(
                dataset,
                num_tasks_wanted - len(tasks),
                docker_server=DockerServer(
                    remote_host_url=os.getenv("REMOTE_DOCKER_HOST"),
                    remote_host_registry=f"{os.getenv('DOCKER_HOST_IP')}:5000",
                ),
                use_remote=True,
            )
            tasks.extend(new_tasks)  # Append N new tasks
        with open(f"{config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl", "wb") as f:
            for task in tasks:
                task.docker_server = None
            pickle.dump(tasks, f)

    @staticmethod
    def tasks_exist(config):
        return os.path.exists(f"{config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl")

    @staticmethod
    def empty_logics_exist(config):
        # load the logics file
        if not os.path.exists(f"{config.neuron.full_path}/logics_{COMPETITION_ID}.pkl"):
            return False
        with open(f"{config.neuron.full_path}/logics_{COMPETITION_ID}.pkl", "rb") as f:
            logics = pickle.load(f)
        return any(logic == {} for logic in logics)

    def verify_results(self):
        scores = []
        for tracker in self.graded_trackers:
            scores.append(tracker.score)
        # if all the scores are 0 then we need to rerun the tasks
        self.graded_trackers = []
        # delete the results file
        if all(score == 0 for score in scores):
            if os.path.exists(
                f"{self.config.neuron.full_path}/results_{COMPETITION_ID}.pkl"
            ):
                os.remove(
                    f"{self.config.neuron.full_path}/results_{COMPETITION_ID}.pkl"
                )
            self.evaluate()

    def cleanup(self):
        """
        Delete the tasks file and any other task files
        """
        os.remove(f"{self.config.neuron.full_path}/tasks_{COMPETITION_ID}.pkl")
        # check if tasks_*.pkl exists and delete it if it does
        for file in os.listdir(self.config.neuron.full_path):
            if file.startswith("tasks_") and file.endswith(".pkl"):
                os.remove(os.path.join(self.config.neuron.full_path, file))
            if file.startswith("results_") and file.endswith(".pkl"):
                os.remove(os.path.join(self.config.neuron.full_path, file))
