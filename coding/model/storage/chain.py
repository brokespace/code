import os
import asyncio
import functools
import bittensor as bt
from typing import Optional

from coding.model.schema import Model


class ChainModelMetadataStore():
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        subnet_uid: int,
        wallet: Optional[bt.wallet] = None,
    ):
        self.subtensor = subtensor
        self.wallet = wallet  # Wallet is only needed to write to the chain, not to read.
        self.subnet_uid = subnet_uid

    async def store_model_metadata(self, model: Model):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            self.subtensor.commit,
            self.wallet,
            self.subnet_uid,
            model.to_compressed_str(),
        )
        
        # run the partial function
        await partial()
        

    async def retrieve_model_metadata(self, hotkey: str) -> Optional[Model]:
        """Retrieves model metadata on this subnet for specific hotkey"""

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(bt.extrinsics.serving.get_metadata, self.subtensor, self.subnet_uid, hotkey)

        metadata = await partial()

        if not metadata:
            return None

        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]

        chain_str = bytes.fromhex(hex_data).decode()

        try:
            model = Model.from_compressed_str(chain_str)
        except:
            # If the metadata format is not correct on the chain then we return None.
            bt.logging.trace(f"Failed to parse the metadata on the chain for hotkey {hotkey}.")
            return None

        return model


# Can only commit data every ~20 minutes.
async def test_store_model_metadata():
    """Verifies that the ChainModelMetadataStore can store data on the chain."""
    model = Model(model_name="TestModel", hash="TestHash1", competition_id="1.0", block=1)

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(subtensor=subtensor, wallet=wallet, subnet_uid=net_uid)

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(model=model)

    print(f"Finished storing {model} on the chain.")


async def test_retrieve_model_metadata():
    """Verifies that the ChainModelMetadataStore can retrieve data from the chain."""
    expected_model = Model(model_name="TestModel", hash="TestHash1", competition_id="1.0", block=1)

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured hotkey/uid for the test.
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    hotkey_address = os.getenv("TEST_HOTKEY_ADDRESS")

    # Do not require a wallet for retrieving data.
    metadata_store = ChainModelMetadataStore(subtensor=subtensor, wallet=None, subnet_uid=net_uid)

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey_address)

    print(f"Expecting matching model: {expected_model == model_metadata}")


# Can only commit data every ~20 minutes.
async def test_roundtrip_model_metadata():
    """Verifies that the ChainModelMetadataStore can roundtrip data on the chain."""
    model = Model(model_name="TestModel", hash="TestHash1", competition_id="1.0", block=1)

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(subtensor=subtensor, wallet=wallet, subnet_uid=net_uid)

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(model=model)

    # May need to use the underlying publish_metadata function with wait_for_inclusion: True to pass here.
    # Otherwise it defaults to False and we only wait for finalization not necessarily inclusion.

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)

    print(f"Expecting matching metadata: {model == model_metadata}")


if __name__ == "__main__":
    # Can only commit data every ~20 minutes.
    # asyncio.run(test_roundtrip_model_metadata())
    # asyncio.run(test_store_model_metadata())
    asyncio.run(test_retrieve_model_metadata())