import secrets
from typing import Tuple

import vrf
from web3 import Web3


def get_task_hash(task_args: str):
    res = Web3.keccak(task_args.encode("utf-8"))
    return res.hex()


def generate_task_id() -> bytes:
    return secrets.token_bytes(32)


def generate_task_id_commitment(task_id: bytes) -> Tuple[bytes, bytes]:
    nonce = secrets.token_bytes(32)
    task_id_commitment = Web3.solidity_keccak(["bytes32", "bytes32"], [task_id, nonce])
    return nonce, task_id_commitment


def vrf_prove(seed: bytes, private_key: bytes) -> Tuple[int, bytes]:
    beta, proof = vrf.prove(sk=private_key, alpha=seed)
    return Web3.to_int(beta), proof
