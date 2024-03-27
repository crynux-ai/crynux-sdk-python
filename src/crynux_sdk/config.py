from typing import TypedDict

from web3.types import Wei
from web3 import Web3


class TxOption(TypedDict, total=False):
    chainId: int
    gas: int
    gasPrice: Wei
    maxFeePerGas: Wei
    maxPriorityFeePerGas: Wei

def get_default_tx_option() -> TxOption:
    return {
        "chainId": 42,
        "gas": 4294967,
        "gasPrice": Web3.to_wei(1, "wei")
    }


def get_default_contract_config():
    return {
        "token": "0x95E7e7Ed5463Ff482f61585605a0ff278e0E1FFb",
        "node": "0xc674d7d3599Cb566eC8027767f410dd8cD7Bd36D",
        "task": "0x9b483dc4D18a35802DD4fB0fE9f02A8b32FaD906",
        "qos": "0x91754172B22b4ba8ff2F34C2A7C90cA7ce96B806",
        "task_queue": "0xeA44D3565B48e4791529F591C0bBDA2AC8958258",
        "netstats": "0xaa0F19cb42a19415591003Ed9D99c40cE69B0224",
    }


def get_default_provider_path() -> str:
    return "https://block-node.crynux.ai/rpc"


def get_default_relay_url() -> str:
    return "https://relay.h.crynux.ai"