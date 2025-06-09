import bittensor as bt

from coding.finetune.tracker import run_async_in_thread
from coding.utils.uids import get_uid_from_hotkey
from coding.protocol import ScoresSynapse
from coding.constants import TRACKED_VALIDATOR_HOTKEY

def gather_scores(self) -> list[float]:
    try:
        uid = get_uid_from_hotkey(self, TRACKED_VALIDATOR_HOTKEY)
        axons = [self.metagraph.axons[uid]]
        synapse = ScoresSynapse()
        responses = run_async_in_thread(
            self.dendrite.aquery(
                axons=axons, synapse=synapse, timeout=45, deserialize=False
            )
        )
        return responses[0].scores
    except Exception as e:
        bt.logging.error(f"Error gathering scores: {e}")
        return []