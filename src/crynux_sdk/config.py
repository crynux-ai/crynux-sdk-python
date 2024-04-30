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
        "gasPrice": Web3.to_wei(1, "wei"),
        "gas": 4294967
    }


def get_default_contract_config():
    return {
        "token": "0xB47E277aE7Cbb93949D7202b6e29e33f541EC262",
        "node": "0x7334e4EA8D6328108fcA0bE7B3042458f058a74b",
        "task": "0x02700Ae3Cc6927a1c957ff48F0D6262236924f82",
        "qos": "0xE15b5DD09f9867C8dD0FbC0f57216b440300c99d",
        "task_queue": "0x719f8f1e106BeF85f2ffC1D23e86C9cbCb7ddB67",
        "netstats": "0xd14f963B54Deff1993FF7987954602c9593d36A4",
    }


def get_default_provider_path() -> str:
    return "https://json-rpc.rolx.evm.ra.blumbus.noisnemyd.xyz"


def get_default_relay_url() -> str:
    return "https://dy.relay.crynux.ai"