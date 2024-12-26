# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Macrocosmos
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

import sys
import random
import asyncio
import traceback
import bittensor as bt
from datetime import datetime, timezone
from functools import partial
from starlette.types import Send
from dataclasses import dataclass
from typing import Awaitable, List, Dict

from coding.rewards import RewardResult
from coding.utils.logging import log_event
from coding.finetune import FinetunePipeline
from coding.utils.uids import get_random_uids
from coding.protocol import StreamCodeSynapse
from coding.rewards.codesim import CodeSimModel
from coding.dendrite import DendriteResponseEvent
from coding.tasks import create_task, create_organic_task
from coding.finetune.pipeline import FinetuneEventResults
from coding.constants import COMPETITION_END_DATE, COMPETITION_ID

@dataclass
class StreamResult:
    synapse: StreamCodeSynapse = None
    exception: BaseException = None
    uid: int = None


async def process_response(uid: int, async_generator: Awaitable):
    """Process a single response asynchronously."""
    try:
        buffer = ""
        chunk = None  # Initialize chunk with a default value
        async for (
            chunk
        ) in (
            async_generator
        ):  # most important loop, as this is where we acquire the final synapse.
            if isinstance(chunk, str):
                buffer += chunk

        if chunk is not None:
            synapse = chunk  # last object yielded is the synapse itself with completion filled

            # bt.logging.debug(f"process_time: synapse.dendrite.process_time}")
            # Assuming chunk holds the last value yielded which should be a synapse
            if isinstance(synapse, StreamCodeSynapse):
                return synapse
            else:
                raise ValueError(
                    f"Expected a StreamCodeSynapse but received {type(synapse)}"
                )

        bt.logging.debug(
            f"Synapse is not StreamCodeSynapse. Miner uid {uid} completion set to '' "
        )
    except Exception as e:
        # bt.logging.error(f"Error in generating reference or handling responses: {e}", exc_info=True)
        traceback_details = traceback.format_exc()
        bt.logging.error(
            f"Error in generating reference or handling responses for uid {uid}: {e}\n{traceback_details}"
        )

        failed_synapse = StreamCodeSynapse(
            roles=["user"], messages=["failure"], completion=""
        )

        return failed_synapse


async def handle_response(responses: Dict[int, Awaitable]) -> List[StreamResult]:
    """The handle_response function is responsible for creating asyncio tasks around acquiring streamed miner chunks
    and processing them asynchronously. It then pairs the results with their original UIDs and returns a list of StreamResults.

    Args:
        responses (Dict[int, Awaitable]): Responses contains awaitables that are used to acquire streamed miner chunks.

    Raises:
        ValueError

    Returns:
        List[StreamResult]: DataClass containing the synapse, exception, and uid
    """
    tasks_with_uid = [
        (uid, responses[uid]) for uid, _ in responses.items()
    ]  # Pair UIDs with their tasks

    # Start tasks, preserving order and their associated UIDs
    tasks = [process_response(uid, resp) for uid, resp in tasks_with_uid]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    mapped_results = []
    # Pair each result with its original uid
    for (uid, _), result in zip(tasks_with_uid, results):
        # If the result is a StreamCodeSynapse, the response was successful and the stream result is added without exceptions
        if isinstance(result, StreamCodeSynapse):
            mapped_results.append(StreamResult(synapse=result, uid=uid))

        # If the result is an exception, the response was unsuccessful and the stream result is added with the exception and an empty synapse
        elif isinstance(result, BaseException):
            failed_synapse = StreamCodeSynapse(
                roles=["user"], messages=["failure"], completion=""
            )
            mapped_results.append(
                StreamResult(synapse=failed_synapse, exception=result, uid=uid)
            )

        # If the result is neither an error or a StreamSynapse, log the error and raise a ValueError
        else:
            bt.logging.error(f"Unexpected result type for UID {uid}: {result}")
            raise ValueError(f"Unexpected result type for UID {uid}: {result}")

    return mapped_results


def forward_organic_synapse(self, synapse: StreamCodeSynapse):
    async def _forward(synapse, send: Send):
        bt.logging.info(f"Sending {synapse} request to uid: {synapse.uid}, ")

        async def handle_response(responses):
            for resp in responses:
                async for chunk in resp:
                    if isinstance(chunk, str):
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        bt.logging.info(f"Streamed text: {chunk}")
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )

        axon = self.metagraph.axons[synapse.uid]
        bt.logging.info(f"Forwarding {synapse} request to axon: {axon}")
        responses = self.dendrite.query(
            axons=[axon],
            synapse=synapse,
            deserialize=False,
            timeout=synapse.timeout,
            streaming=True,
        )
        return await handle_response(responses)

    token_streamer = partial(_forward, synapse)
    return synapse.create_streaming_response(token_streamer)


async def forward(self, synapse: StreamCodeSynapse):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    bt.logging.info("🚀 Starting forward loop...")
    
    # check if the competition has ended and evaluation not started
    if datetime.now(timezone.utc) > datetime.strptime(COMPETITION_END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc):
        if not COMPETITION_ID in self.finetune_results and not hasattr(self, 'finetune_eval_future'):
            finetune_pipeline = FinetunePipeline(
                config=self.config,
                competition_id=COMPETITION_ID,
                code_sim_model=CodeSimModel(code_scorer=self.code_scorer),
            )
            self.finetune_eval_future = self.executor.submit(finetune_pipeline.evaluate)
    # Check if evaluation is complete
    if hasattr(self, 'finetune_eval_future') and self.finetune_eval_future.done():
        self.finetune_results[COMPETITION_ID] = FinetuneEventResults(**self.finetune_eval_future.result())
        delattr(self, 'finetune_eval_future')  # Remove the future after getting results
    
    # TODO remove this once we are sure that we only want to do finetuning
    if not synapse:
        while True:
            # Create a specific task
            task_name = random.choices(
                self.config.neuron.tasks, self.config.neuron.task_weights
            )[0]
            bt.logging.info(f"📋 Creating {task_name} task... ")
            try:
                task = create_task(llm=self.llm, task_name=task_name, repl=self.repl, code_scorer=self.code_scorer, dataset_manager=self.dataset_manager)
                synapse = StreamCodeSynapse(
                    query=task.query,
                    files=task.files,
                    attachments=task.attachments,
                    messages=task.messages,
                )
                break
            except Exception as e:
                bt.logging.debug(
                    f"Failed to create {task_name} task. {sys.exc_info()}. Skipping to next task."
                )
                bt.logging.debug(traceback.format_exc())
                continue
    else:
        try:
            task = create_organic_task(llm=self.llm, synapse=synapse)
        except:
            bt.logging.error(f"Failed to create organic task. {sys.exc_info()}")
            return

    uids = get_random_uids(self, k=self.config.neuron.sample_size)
    uids_cpu = uids.tolist()
    axons = [self.metagraph.axons[uid] for uid in uids]
    # The dendrite client queries the network.
    streams_responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        timeout=task.timeout,
        deserialize=False,
        streaming=True,
    )
    #  Prepare the task for handling stream responses
    handle_stream_responses_task = asyncio.create_task(
        handle_response(responses=dict(zip(uids_cpu, streams_responses)))
    )
    stream_results = await handle_stream_responses_task
    all_synapses_results = [stream_result.synapse for stream_result in stream_results]
    if len(all_synapses_results) == 0:
        bt.logging.error(f"No synapse results were gotten")
        return

    response_event = DendriteResponseEvent(
        responses=all_synapses_results, uids=uids, timeout=task.timeout, axons=axons
    )
    reward_result = RewardResult(
        self.reward_pipeline,
        task=task,
        response_event=response_event,
        device=self.device,
    )

    self.update_scores(reward_result.rewards, uids)

    log_event(
        self,
        {
            "step": self.step,
            **reward_result.__state_dict__(),
            **response_event.__state_dict__(),
            # **({"trackers": [tracker.model_dump() for tracker in self.finetune_results[COMPETITION_ID].trackers]} if hasattr(self, 'finetune_results') and COMPETITION_ID in self.finetune_results else {}),
        },
    )