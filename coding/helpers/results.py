import asyncio
import threading
import bittensor as bt

from coding.protocol import ResultSynapse
from coding.utils.uids import get_miner_uids, get_hotkey_from_uid



def run_async_in_thread(coro):
    """
    Runs an async coroutine in a separate thread and returns its result.
    """
    result_container = []
    exception_container = []

    def target():
        try:
            result = asyncio.run(coro)
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    if exception_container:
        raise exception_container[0]
    return result_container[0]


async def forward_results(validator) -> None:
    """
    Forward the results of the validator to the miners.
    """
    bt.logging.info("Forwarding results to miners...")
    uids = get_miner_uids(validator)
    axons = [validator.metagraph.axons[uid] for uid in uids]
    hotkeys = [get_hotkey_from_uid(validator, uid) for uid in uids]
    results = [validator.model_store.get_results_string(hotkey) for hotkey in hotkeys]
    results = [result if result is not None else "" for result in results]
    synapses = [ResultSynapse(result=result) for result in results]
    dendrite = bt.dendrite(wallet=validator.wallet)
    for axon, synapse in zip(axons, synapses):
        await dendrite.forward(
            axons=[axon], synapse=synapse, timeout=5, deserialize=False
        )
    bt.logging.info("Results forwarded to miners.")
