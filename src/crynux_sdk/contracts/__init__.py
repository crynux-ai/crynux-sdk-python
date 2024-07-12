import logging
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional, cast

from anyio import create_memory_object_stream, create_task_group
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from web3.contract.async_contract import AsyncContractEvent
from web3.logs import DISCARD, WARN
from web3.providers.async_base import AsyncBaseProvider
from web3.types import (BlockData, BlockIdentifier, EventData, TxParams,
                        TxReceipt)

from crynux_sdk.config import TxOption

from . import network_stats, node, qos, task, task_queue
from .exceptions import TxRevertedError
from .utils import ContractWrapper, TxWaiter
from .w3_pool import W3Pool

__all__ = ["TxRevertedError", "Contracts", "TxWaiter", "get_contracts", "set_contracts", "ContractWrapper"]

_logger = logging.getLogger(__name__)


class ProviderType(IntEnum):
    HTTP = 0
    WS = 1
    Other = 2


class Contracts(object):
    node_contract: node.NodeContract
    task_contract: task.TaskContract
    qos_contract: qos.QOSContract
    task_queue_contract: task_queue.TaskQueueContract
    netstats_contract: network_stats.NetworkStatsContract

    def __init__(
        self,
        provider: Optional[AsyncBaseProvider] = None,
        provider_path: Optional[str] = None,
        privkey: str = "",
        default_account_index: Optional[int] = None,
        pool_size: int = 5,
        timeout: int = 10,
    ):
        if provider is not None:
            pool_size = 1

        self._w3_pool = W3Pool(
            provider=provider,
            provider_path=provider_path,
            privkey=privkey,
            default_account_index=default_account_index,
            pool_size=pool_size,
            timeout=timeout,
        )

        self._initialized = False
        self._closed = False

    async def init(
        self,
        node_contract_address: Optional[str] = None,
        task_contract_address: Optional[str] = None,
        qos_contract_address: Optional[str] = None,
        task_queue_contract_address: Optional[str] = None,
        netstats_contract_address: Optional[str] = None,
        *,
        option: "Optional[TxOption]" = None,
    ):
        try:
            async with await self._w3_pool.get() as w3:
                assert w3.eth.default_account, "Wallet address is empty"
                self._account = w3.eth.default_account
                _logger.info(f"Wallet address is {w3.eth.default_account}")

                if qos_contract_address is not None:
                    self.qos_contract = qos.QOSContract(
                        self._w3_pool, w3.to_checksum_address(qos_contract_address)
                    )
                elif task_contract_address is None:
                    # task contract has not been deployed, need deploy qos contract
                    self.qos_contract = qos.QOSContract(self._w3_pool)
                    await self.qos_contract.deploy(option=option, w3=w3)
                    qos_contract_address = self.qos_contract.address

                if task_queue_contract_address is not None:
                    self.task_queue_contract = task_queue.TaskQueueContract(
                        self._w3_pool, w3.to_checksum_address(task_queue_contract_address)
                    )
                elif task_contract_address is None:
                    # task contract has not been deployed, need deploy qos contract
                    self.task_queue_contract = task_queue.TaskQueueContract(self._w3_pool)
                    await self.task_queue_contract.deploy(option=option, w3=w3)
                    task_queue_contract_address = self.task_queue_contract.address

                if netstats_contract_address is not None:
                    self.netstats_contract = network_stats.NetworkStatsContract(
                        self._w3_pool, w3.to_checksum_address(netstats_contract_address)
                    )
                elif task_contract_address is None:
                    # task contract has not been deployed, need deploy qos contract
                    self.netstats_contract = network_stats.NetworkStatsContract(
                        self._w3_pool
                    )
                    await self.netstats_contract.deploy(option=option, w3=w3)
                    netstats_contract_address = self.netstats_contract.address

                if node_contract_address is not None:
                    self.node_contract = node.NodeContract(
                        self._w3_pool, w3.to_checksum_address(node_contract_address)
                    )
                else:
                    assert qos_contract_address is not None, "QOS contract address is None"
                    assert (
                        netstats_contract_address is not None
                    ), "NetworkStats contract address is None"
                    self.node_contract = node.NodeContract(self._w3_pool)
                    await self.node_contract.deploy(
                        qos_contract_address,
                        netstats_contract_address,
                        option=option,
                        w3=w3
                    )
                    node_contract_address = self.node_contract.address
                    await self.qos_contract.update_node_contract_address(
                        node_contract_address, option=option, w3=w3
                    )
                    await self.netstats_contract.update_node_contract_address(
                        node_contract_address, option=option, w3=w3
                    )

                if task_contract_address is not None:
                    self.task_contract = task.TaskContract(
                        self._w3_pool, w3.to_checksum_address(task_contract_address)
                    )
                else:
                    assert qos_contract_address is not None, "QOS contract address is None"
                    assert (
                        task_queue_contract_address is not None
                    ), "Task queue contract address is None"
                    assert (
                        netstats_contract_address is not None
                    ), "NetworkStats contract address is None"

                    self.task_contract = task.TaskContract(self._w3_pool)
                    await self.task_contract.deploy(
                        node_contract_address,
                        qos_contract_address,
                        task_queue_contract_address,
                        netstats_contract_address,
                        option=option,
                        w3=w3
                    )
                    task_contract_address = self.task_contract.address

                    await self.node_contract.update_task_contract_address(
                        task_contract_address, option=option, w3=w3
                    )
                    await self.qos_contract.update_task_contract_address(
                        task_contract_address, option=option, w3=w3
                    )
                    await self.task_queue_contract.update_task_contract_address(
                        task_contract_address, option=option, w3=w3
                    )
                    await self.netstats_contract.update_task_contract_address(
                        task_contract_address, option=option, w3=w3
                    )

                self._initialized = True

                return self
        except Exception:
            await self.close()
            raise

    async def close(self):
        if not self._closed:
            await self._w3_pool.close()
            self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.close()

    def get_contract(self, name: str):
        if name == "node":
            return self.node_contract
        elif name == "task":
            return self.task_contract
        elif name == "qos":
            return self.qos_contract
        elif name == "task_queue":
            return self.task_queue_contract
        elif name == "netstats":
            return self.netstats_contract
        else:
            raise ValueError(f"unknown contract name {name}")

    async def get_block(self, block_identifier: BlockIdentifier) -> BlockData:
        async with await self._w3_pool.get() as w3:
            return await w3.eth.get_block(block_identifier)

    async def get_tx_receipt(self, tx_hash: HexBytes) -> TxReceipt:
        async with await self._w3_pool.get() as w3:
            receipt = await w3.eth.get_transaction_receipt(tx_hash)
            return receipt

    async def get_tx_receipts(self, from_block: Optional[int] = None, to_block: Optional[int] = None, concurrency: int = 4):
        async with await self._w3_pool.get() as w3:
            if from_block is None or to_block is None:
                current_blocknum = await w3.eth.get_block_number()
                if from_block is None:
                    from_block = current_blocknum
                if to_block is None:
                    to_block = current_blocknum

            async def _process_block(block_receiver: ObjectReceiveStream[int], tx_sender: ObjectSendStream[HexBytes]):
                async with block_receiver, tx_sender:
                    async for blocknum in block_receiver:
                        block = await w3.eth.get_block(blocknum)
                        assert "transactions" in block
                        for tx_hash in block["transactions"]:
                            assert isinstance(tx_hash, bytes)
                            await tx_sender.send(tx_hash)
                        assert "timestamp" in block
                        blocktime = datetime.fromtimestamp(
                            block["timestamp"]
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        tx_count = len(block["transactions"])
                        _logger.debug(
                            f"block {blocknum} produced at {blocktime}, {tx_count} txs"
                        )

            async def _process_tx_receipts(tx_receiver: ObjectReceiveStream[HexBytes], receipt_sender: ObjectSendStream[TxReceipt]):
                async with tx_receiver, receipt_sender:
                    async for tx_hash in tx_receiver:
                        tx_receipt = await w3.eth.get_transaction_receipt(tx_hash)
                        await receipt_sender.send(tx_receipt)
                        blocknum = tx_receipt["blockNumber"]
                        tx_index = tx_receipt["transactionIndex"]
                        _logger.debug(
                            f"process receipt {tx_index} of block {blocknum}"
                        )

            results: List[TxReceipt] = []

            async with create_task_group() as tg:
                block_sender, block_receiver = create_memory_object_stream(100, item_type=int)
                tx_sender, tx_receiver = create_memory_object_stream(100, item_type=HexBytes)
                receipt_sender, receipt_receiver = create_memory_object_stream(100, item_type=TxReceipt)

                for _ in range(concurrency):
                    tg.start_soon(_process_tx_receipts, tx_receiver.clone(), receipt_sender.clone())
                tx_receiver.close()
                receipt_sender.close()

                for _ in range(concurrency):
                    tg.start_soon(_process_block, block_receiver.clone(), tx_sender.clone())
                block_receiver.close()
                tx_sender.close()

                async with block_sender:
                    for blocknum in range(from_block, to_block + 1):
                        await block_sender.send(blocknum)
                
                async with receipt_receiver:
                    async for receipt in receipt_receiver:
                        results.append(receipt)
            
            return results

    async def get_events(
        self,
        contract_name: str,
        event_name: str,
        filter_args: Optional[Dict[str, Any]] = None,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        concurrency: int = 4,
    ):

        def _filter_event(
            event: EventData, filter_args: Optional[Dict[str, Any]] = None
        ) -> bool:
            if filter_args is None:
                return True
            for key, val in filter_args.items():
                if key in event["args"]:
                    real_val = event["args"][key]
                    if real_val != val:
                        return False
            return True
    
        tx_receipts = await self.get_tx_receipts(from_block=from_block, to_block=to_block, concurrency=concurrency)
        async with await self._w3_pool.get() as w3:
            contract = self.get_contract(contract_name)
            c = w3.eth.contract(address=contract.address, abi=contract.abi)
            event = c.events[event_name]()
            event = cast(AsyncContractEvent, event)


            results: List[EventData] = []
            for receipt in tx_receipts:
                for event_data in event.process_receipt(receipt, errors=DISCARD):
                    if _filter_event(event_data, filter_args):
                        results.append(event_data)
            return results


    async def event_process_receipt(
        self, contract_name: str, event_name: str, recepit: TxReceipt, errors=WARN
    ):
        contract = self.get_contract(contract_name)
        return await contract.event_process_receipt(
            event_name=event_name, recepit=recepit, errors=errors
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def account(self) -> ChecksumAddress:
        return self._w3_pool.account

    async def get_current_block_number(self) -> int:
        async with await self._w3_pool.get() as w3:
            return await w3.eth.get_block_number()

    async def get_balance(self, account: ChecksumAddress) -> int:
        async with await self._w3_pool.get() as w3:
            return await w3.eth.get_balance(account)

    async def transfer(
        self, to: str, amount: int, *, option: "Optional[TxOption]" = None
    ):
        async with await self._w3_pool.get() as w3:
            opt: TxParams = {}
            if option is not None:
                opt.update(**option)
            opt["to"] = w3.to_checksum_address(to)
            opt["from"] = self._w3_pool.account
            opt["value"] = w3.to_wei(amount, "Wei")

            tx_hash = await w3.eth.send_transaction(opt)
            receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
            return receipt


_default_contracts: Optional[Contracts] = None


def get_contracts() -> Contracts:
    assert _default_contracts is not None, "Contracts has not been set."

    return _default_contracts


def set_contracts(contracts: Contracts):
    global _default_contracts

    _default_contracts = contracts
