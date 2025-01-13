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
        "gasPrice": Web3.to_wei(10, "wei"),
        "gas": 5000000
    }


def get_default_contract_config():
    return {
        "node": "0xFc317b2e4649D5208c5CE6f2968338ef66841642",
        "task": "0x44bD7b511c1B9960AB95c8347DE9C3adCC0811B3",
        "qos": "0xC3E755AB19183faFD1C55478bCa23d565Ec83eeB",
        "task_queue": "0x6bbd9ed30685A9064C1BEfa344d0D7F912316125",
        "netstats": "0x54bE38d014c56B990091b62Bc43380879436DC61",
    }


def get_default_provider_path() -> str:
    return "https://json-rpc.testnet.crynux.ai"


def get_default_relay_url() -> str:
    return "https://vss.relay.crynux.ai"