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
        "node": "0x988b6886db67227Ada21A05b3F1A4744DE129515",
        "task": "0x25a3A73E1A9c1CB0fA5366368723b57Eb55eF678",
        "qos": "0x595a0Df849A3a19aec515ED9714CAdd204E7E299",
        "task_queue": "0x9Ac6723e6bABE819f23f15e8Fb3D88A339b4Ed22",
        "netstats": "0x3dE7Cd5C8F32c438bD63e751ACdA88bc7F8F25ff",
    }


def get_default_provider_path() -> str:
    return "https://json-rpc.testnet.crynux.ai"


def get_default_relay_url() -> str:
    return "https://vss.relay.crynux.ai"