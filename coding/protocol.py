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

import json
import pydantic
import bittensor as bt

from starlette.responses import StreamingResponse
from typing import List, AsyncIterator, Any, Optional

from coding.schemas import ChatMessage, File
from coding.constants import COMPETITION_ID


class ScoresSynapse(bt.Synapse):
    """
    ScoresSynapse is a Synapse that is used to get the scores of the miner.
    """
    scores: List[float] = pydantic.Field(
        [],
        title="scores",
        description="The scores of the miners ordered by uid",
    )

class ProvisionKeySynapse(bt.Synapse):
    """
    ProvisionKeySynapse is a Synapse that is used to get the provisioning key of the miner.
    """
    api_key: Optional[str] = None
    key_hash: Optional[str] = None
    action: Optional[str] = None # "create" or "delete"

class ResultSynapse(bt.Synapse):
    """
    ResultSynapse is a Synapse that is used to get the result of the miner.
    """
    result: str = pydantic.Field(
        "",
        title="result",
        description="The result of the miner",
    )

class LogicSynapse(bt.Synapse):
    """
    LogicSynapse is a Synapse that is used to get the logic of the miner. 
    
    Attributes:
        logic (dict): A dictionary where the key is a filename and the value is the file contents
    """
    logic: dict = pydantic.Field(
        {},
        title="logic",
        description="A dictionary where the key is a filename and the value is the file contents",
    )

class HFModelSynapse(bt.Synapse):
    """
    HFModelSynapse is a Synapse that is used to get the HF model name that this miner published to HF
    
    Attributes:
        model_name (Optional[str]): The HF model name that this miner published to HF
        prompt_tokens (Optional[dict]): Dictionary containing FIM prompt tokens:
            - "prefix": the prefix of the prompt
            - "middle": the middle of the prompt
            - "suffix": the suffix of the prompt
        
    """
    model_name: Optional[str] = ""
    competition_id: Optional[int] = COMPETITION_ID
    # prompt_tokens: Optional[dict] = None


class StreamCodeSynapse(bt.StreamingSynapse):
    """
    StreamPromptingSynapse is a specialized implementation of the `StreamingSynapse` tailored for prompting functionalities within
    the Bittensor network. This class is intended to interact with a streaming response that contains a sequence of tokens,
    which represent prompts or messages in a certain scenario.

    As a developer, when using or extending the `StreamPromptingSynapse` class, you should be primarily focused on the structure
    and behavior of the prompts you are working with. The class has been designed to seamlessly handle the streaming,
    decoding, and accumulation of tokens that represent these prompts.

    Attributes:
    - `roles` (List[str]): A list of roles involved in the prompting scenario. This could represent different entities
                           or agents involved in the conversation or use-case. They are immutable to ensure consistent
                           interaction throughout the lifetime of the object.

    - `messages` (List[str]): These represent the actual prompts or messages in the prompting scenario. They are also
                              immutable to ensure consistent behavior during processing.

    - `completion` (str): Stores the processed result of the streaming tokens. As tokens are streamed, decoded, and
                          processed, they are accumulated in the completion attribute. This represents the "final"
                          product or result of the streaming process.
    - `required_hash_fields` (List[str]): A list of fields that are required for the hash.

    Methods:
    - `process_streaming_response`: This method asynchronously processes the incoming streaming response by decoding
                                    the tokens and accumulating them in the `completion` attribute.

    - `deserialize`: Converts the `completion` attribute into its desired data format, in this case, a string.

    - `extract_response_json`: Extracts relevant JSON data from the response, useful for gaining insights on the response's
                               metadata or for debugging purposes.

    Note: While you can directly use the `StreamPromptingSynapse` class, it's designed to be extensible. Thus, you can create
    subclasses to further customize behavior for specific prompting scenarios or requirements.
    """


    

    
    query: str = pydantic.Field(
        "",
        title="query",
        description="The query",
    )
    
    script: str = pydantic.Field(
        "",
        title="script",
        description="A python script that is being worked with",
    )
    
    messages: List[ChatMessage] = pydantic.Field(
        [],
        title="messages",
        description="A list of messages",
    )
     
    attachments: List[Any] = pydantic.Field(
        [],
        title="attachments",
        description="Attachments to be sent alongside the query",
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current CodeSynapse object. This attribute is mutable and can be updated.",
    )

    files: List[File] = pydantic.Field(
        [],
        title="Files",
        description="Files",
    )
    
    uid: int = pydantic.Field(
        9999,
        title="UID",
        description="Miner uid to send task to",
    )
    
    async def process_streaming_response(
        self, response: StreamingResponse
    ) -> AsyncIterator[str]:
        """
        `process_streaming_response` is an asynchronous method designed to process the incoming streaming response from the
        Bittensor network. It's the heart of the StreamPromptingSynapse class, ensuring that streaming tokens, which represent
        prompts or messages, are decoded and appropriately managed.

        As the streaming response is consumed, the tokens are decoded from their 'utf-8' encoded format, split based on
        newline characters, and concatenated into the `completion` attribute. This accumulation of decoded tokens in the
        `completion` attribute allows for a continuous and coherent accumulation of the streaming content.

        Args:
            response: The streaming response object containing the content chunks to be processed. Each chunk in this
                      response is expected to be a set of tokens that can be decoded and split into individual messages or prompts.
        """
        if self.completion is None:
            self.completion = ""

        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8")
            
            try:
                data = json.loads(tokens)
                if isinstance(data, dict) or isinstance(data, list):
                    # Process the dictionary data as needed
                    self.completion = self.completion + json.dumps(data)
                    yield json.dumps(data)
                else:
                    self.completion = self.completion + tokens
                    yield tokens
            except json.JSONDecodeError:
                self.completion = self.completion + tokens
                yield tokens
        # if self.completion is None: #TODO remove this once confirm that above works
        #     self.completion = ""

        # async for chunk in response.content.iter_any():
        #     tokens = chunk.decode("utf-8")

        #     self.completion = self.completion + "".join([t for t in tokens if t])
        #     yield tokens

    def deserialize(self) -> str:
        """
        Deserializes the response by returning the completion attribute.

        Returns:
            str: The completion result.
        """
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
        """
        `extract_response_json` is a method that performs the crucial task of extracting pertinent JSON data from the given
        response. The method is especially useful when you need a detailed insight into the streaming response's metadata
        or when debugging response-related issues.

        Beyond just extracting the JSON data, the method also processes and structures the data for easier consumption
        and understanding. For instance, it extracts specific headers related to dendrite and axon, offering insights
        about the Bittensor network's internal processes. The method ultimately returns a dictionary with a structured
        view of the extracted data.

        Args:
            response: The response object from which to extract the JSON data. This object typically includes headers and
                      content which can be used to glean insights about the response.

        Returns:
            dict: A structured dictionary containing:
                - Basic response metadata such as name, timeout, total_size, and header_size.
                - Dendrite and Axon related information extracted from headers.
                - Roles and Messages pertaining to the current StreamPromptingSynapse instance.
                - The accumulated completion.
        """
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "query": self.query, 
            "attachments": self.attachments,
            "completion": self.completion,
        }
