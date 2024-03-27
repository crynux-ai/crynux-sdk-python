from web3 import Web3


def get_task_hash(task_args: str):
    res = Web3.keccak(task_args.encode("utf-8"))
    return res.hex()
