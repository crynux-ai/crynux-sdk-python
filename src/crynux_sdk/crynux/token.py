import logging
from typing import Optional

from web3 import Web3
from crynux_sdk.contracts import Contracts
from crynux_sdk.config import TxOption

_logger = logging.getLogger(__name__)


class Token(object):
    def __init__(self, contracts: Contracts, option: Optional[TxOption] = None) -> None:
        self._contracts = contracts
        self._option = option

    async def transfer_eth(self, address: str, eth: int):
        if eth > 0:
            address = Web3.to_checksum_address(address)
            await self._contracts.transfer(
                to=address, amount=eth, option=self._option
            )
            _logger.info(f"transfer {eth} eth to {address}")

    async def eth_balance(self, address: str) -> int:
        address = Web3.to_checksum_address(address)
        token = await self._contracts.get_balance(address)
        _logger.info(f"{address} eth balance {token}")
        return token
