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
        "node": "0x662d296cae3f1Be2ed5803227dAd6435f1ffC438",
        "task": "0x07E149A0e372C2F54Df6358d021d700703D222D1",
        "qos": "0x95E7e7Ed5463Ff482f61585605a0ff278e0E1FFb",
        "task_queue": "0xeD4cbf24978AD18d73ee6190e361E71095E857A7",
        "netstats": "0xC2c060f8C46640394E0937D75Ea977207E6df130",
    }


def get_default_provider_path() -> str:
    return "https://json-rpc.crynux.evm.ra.blumbus.noisnemyd.xyz/"


def get_default_relay_url() -> str:
    return "https://dy.relay.crynux.ai"