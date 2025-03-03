import threading
import bittensor as bt
from typing import List, Dict, Optional
from pydantic import BaseModel


from coding.protocol import EvaluationSynapse
from coding.utils.uids import get_uid_from_hotkey
from coding.constants import HONEST_VALIDATOR_HOTKEYS
from coding.finetune.tracker import run_async_in_thread
# Validator reaches out to other validator to see if it is scoring the same model
# if so it will mark that the model is being evaluated by another server and wait
# if not it will start evaluating the model


class ModelEvaluationStatus(BaseModel):
    """Status of a model evaluation"""
    model_hash: str
    in_progress: bool
    completed: bool
    server_id: Optional[str] = None
    started_at: Optional[int] = None # block number
    completed_at: Optional[int] = None # block number
    score: Optional[float] = None


class FinetuneCoordinator:
    """
    Coordinates model evaluation across multiple servers to prevent duplicate work.
    Each server should create an instance of this class and register itself.
    """
    def __init__(self, config: bt.Config, wallet):
        """
        Initialize the coordinator.
        
        """
        self.server_id = wallet.hotkey.ss58_address
        self.dendrite = bt.dendrite(wallet=wallet)
        self.servers: List[str] = []  # List of server hotkeys
        self.evaluation_statuses: Dict[str, ModelEvaluationStatus] = {}  # hotkey -> status
        self.lock = threading.Lock()
        self.subtensor = bt.subtensor()
        self.metagraph = self.subtensor.metagraph(netuid=config.netuid)
        self.gather_servers()
    
    @property
    def block(self) -> int:
        """
        Get the current block number
        """
        return self.subtensor.block
    
    def gather_servers(self):
        """
        Gather other servers that this coordinator can communicate with.
        """
        self.servers = []
        for hotkey in HONEST_VALIDATOR_HOTKEYS:
            uid = get_uid_from_hotkey(self, hotkey)
            if uid is not None:
                synapse = EvaluationSynapse()
                try:
                    response = run_async_in_thread(
                        self.dendrite.aquery(
                            axons=self.metagraph.axons[uid],
                            synapse=synapse,
                            timeout=10,
                            deserialize=False,
                        )
                    )
                    if response and response.alive:
                        self.servers.append(hotkey)
                except Exception as e:
                    bt.logging.warning(f"Failed to query {hotkey}: {e}")
                    continue
        bt.logging.info(f"Registered {len(self.servers)} servers: {self.servers}")
        
    def get_status(self, model_hash: str) -> ModelEvaluationStatus:
        """
        Get the status of a model evaluation
        """
        self.gather_servers()
        with self.lock:
            # Check if we have a local status for this model
            if model_hash in self.evaluation_statuses:
                status = self.evaluation_statuses[model_hash]
                # If it's being scored locally, return it
                if status.server_id == self.server_id:
                    return status
                
                # If it exists but isn't being scored locally, update the info
                for server in self.servers:
                    try:
                        synapse = EvaluationSynapse()
                        synapse.model_hash = model_hash
                        uid = get_uid_from_hotkey(self, server)
                        response = run_async_in_thread(
                            self.dendrite.aquery(
                                axons=self.metagraph.axons[uid],
                                synapse=synapse,
                                timeout=30,
                                deserialize=False,
                            )
                        )
                        if response.alive and response.model_hash == model_hash and response.server_id == server and (response.in_progress or response.completed):
                            # Update our local status with the latest info
                            self.evaluation_statuses[model_hash] = ModelEvaluationStatus(
                                model_hash=model_hash,
                                in_progress=response.in_progress,
                                completed=response.completed,
                                server_id=server,
                                started_at=response.started_at,
                                completed_at=response.completed_at,
                                score=response.score
                            )
                            return self.evaluation_statuses[model_hash]
                    except Exception as e:
                        bt.logging.warning(f"Failed to check status with {server}: {e}")
            
            # If it doesn't exist, check with all servers
            for server in self.servers:
                try:
                    synapse = EvaluationSynapse()
                    synapse.model_hash = model_hash
                    uid = get_uid_from_hotkey(self, server)
                    response = run_async_in_thread(
                        self.dendrite.aquery(
                            axons=self.metagraph.axons[uid],
                            synapse=synapse,
                            timeout=10,
                            deserialize=False,
                        )
                    )
                    if response.alive and response.model_hash == model_hash and response.server_id == server and (response.in_progress or response.completed):
                        # Store the server that's evaluating it
                        status = ModelEvaluationStatus(
                            model_hash=model_hash,
                            in_progress=response.in_progress,
                            completed=response.completed,
                            server_id=server,
                            started_at=response.started_at,
                            completed_at=response.completed_at,
                            score=response.score
                        )
                        self.evaluation_statuses[model_hash] = status
                        return status
                except Exception as e:
                    bt.logging.warning(f"Failed to check status with {server}: {e}")
            
            # If no server is evaluating it, return None or a default status
            return None
    
    def should_evaluate_model(self, model_hash: str) -> bool:
        """
        Determine if this server should evaluate the given model.
        
        Args:
            model_hash: The model hash to evaluate
            
        Returns:
            True if this server should evaluate the model, False otherwise
        """
        with self.lock:
            # Check if any server is already evaluating this model
            status = self.get_status(model_hash)
            
            if status:
                # If the model is being evaluated and not completed
                if status.in_progress and not status.completed:
                    # If evaluation has been running for more than 1.5 hr, consider it stalled
                    if status.started_at and (self.block - status.started_at) > 1800:
                        bt.logging.warning(f"Evaluation for {model_hash} appears stalled, taking over")
                    else:
                        bt.logging.info(f"Model with the hash {model_hash} is being evaluated by server {status.server_id}")
                        return False
                # If the model evaluation is already completed
                elif status.completed:
                    bt.logging.info(f"Model with the hash {model_hash} has already been evaluated")
                    return False
            
            # No one is evaluating this model, or the previous evaluation was stalled,
            # so we'll do it
            self.evaluation_statuses[model_hash] = ModelEvaluationStatus(
                model_hash=model_hash,
                in_progress=True,
                completed=False,
                server_id=self.server_id,
                started_at=self.block
            )
            
            return True
    
    def mark_evaluation_started(self, model_hash: str):
        """
        Mark a model evaluation as started.
        """
        with self.lock:
            self.evaluation_statuses[model_hash] = ModelEvaluationStatus(
                model_hash=model_hash,
                in_progress=True,
                completed=False,
                server_id=self.server_id,
                started_at=self.block
            )
    
    def mark_evaluation_complete(self, model_hash: str, score: float):
        """
        Mark a model evaluation as complete.
        
        Args:
            model_hash: The model hash that was evaluated
            score: The evaluation score
        """
        with self.lock:
            status = self.evaluation_statuses.get(model_hash)
            if status and status.server_id == self.server_id:
                status.completed = True
                status.completed_at = self.block
                status.score = score
                status.in_progress = False
                self.evaluation_statuses[model_hash] = status
                
    def get_model_status(self, model_hash: str) -> ModelEvaluationStatus | None:
        """Get the evaluation status for a model"""
        with self.lock:
            status = self.evaluation_statuses.get(model_hash, None)
            if status:
                return status
            else:
                return None
    
    def get_all_statuses(self):
        """Get all evaluation statuses"""
        with self.lock:
            for model_hash in self.evaluation_statuses:
                status = self.get_model_status(model_hash)
                if status:
                    yield status
