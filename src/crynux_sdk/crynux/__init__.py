from __future__ import annotations

import logging
import pathlib
from typing import Optional, Union

from anyio import fail_after
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_cause_type,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from web3 import Web3

from crynux_sdk.config import (
    get_default_contract_config,
    get_default_provider_path,
    get_default_relay_url,
    get_default_tx_option,
)
from crynux_sdk.contracts import Contracts, TxRevertedError
from crynux_sdk.models import sd_args
from crynux_sdk.models.contracts import TaskType
from crynux_sdk.relay import Relay, WebRelay

from .exceptions import TaskAbortedError, TaskCancelError, TaskGetResultTimeout
from .task import Task
from .token import Token

__all__ = ["Crynux"]

_logger = logging.getLogger(__name__)


class Crynux(object):
    contracts: Contracts
    relay: Relay

    task: Task
    token: Token

    def __init__(
        self,
        privkey: Optional[str] = None,
        chain_provider_path: Optional[str] = None,
        relay_url: Optional[str] = None,
        token_contract_address: Optional[str] = None,
        node_contract_address: Optional[str] = None,
        task_contract_address: Optional[str] = None,
        qos_contract_address: Optional[str] = None,
        task_queue_contract_address: Optional[str] = None,
        netstats_contract_address: Optional[str] = None,
        chain_id: Optional[int] = None,
        gas: Optional[int] = None,
        gas_price: Optional[int] = None,
        max_fee_per_gas: Optional[int] = None,
        max_priority_fee_per_gas: Optional[int] = None,
        contracts: Optional[Contracts] = None,
        relay: Optional[Relay] = None,
        contracts_timeout: float = 30,
        relay_timeout: float = 30,
    ) -> None:
        if contracts is not None:
            self.contracts = contracts
        else:
            assert privkey is not None, "private key is empty"
            chain_provider_path = chain_provider_path or get_default_provider_path()
            self.contracts = Contracts(
                provider_path=chain_provider_path,
                privkey=privkey,
                timeout=contracts_timeout,
            )

        default_contract_config = get_default_contract_config()

        self.token_contract_address = (
            token_contract_address or default_contract_config["token"]
        )
        self.node_contract_address = (
            node_contract_address or default_contract_config["node"]
        )
        self.task_contract_address = (
            task_contract_address or default_contract_config["task"]
        )
        self.qos_contract_address = (
            qos_contract_address or default_contract_config["qos"]
        )
        self.task_queue_contract_address = (
            task_queue_contract_address or default_contract_config["task_queue"]
        )
        self.netstats_contract_address = (
            netstats_contract_address or default_contract_config["netstats"]
        )

        self.tx_option = get_default_tx_option()
        if chain_id is not None:
            self.tx_option["chainId"] = chain_id
        if gas is not None:
            self.tx_option["gas"] = gas
        if gas_price is not None:
            self.tx_option["gasPrice"] = Web3.to_wei(gas_price, "wei")
        if max_fee_per_gas is not None:
            self.tx_option["maxFeePerGas"] = Web3.to_wei(max_fee_per_gas, "wei")
        if max_priority_fee_per_gas is not None:
            self.tx_option["maxPriorityFeePerGas"] = Web3.to_wei(
                max_priority_fee_per_gas, "wei"
            )

        if relay is not None:
            self.relay = relay
        else:
            relay_url = relay_url or get_default_relay_url()
            assert privkey is not None, "private key is empty"
            self.relay = WebRelay(
                base_url=relay_url,
                privkey=privkey,
                timeout=relay_timeout,
            )

        self.token = Token(self.contracts, self.tx_option)
        self.task = Task(self.contracts, self.relay, self.tx_option)

        self._initialized = False
        self._closed = False

    async def init(self):
        if not self.contracts.initialized:
            await self.contracts.init(
                token_contract_address=self.token_contract_address,
                node_contract_address=self.node_contract_address,
                task_contract_address=self.task_contract_address,
                qos_contract_address=self.qos_contract_address,
                task_queue_contract_address=self.task_queue_contract_address,
                netstats_contract_address=self.netstats_contract_address,
                option=self.tx_option,
            )
        self._initialized = True
        return self

    async def close(self):
        if not self._closed:
            await self.contracts.close()
            await self.relay.close()
            self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def deposit(self, address: str, eth: int, cnx: int, unit: str = "ether"):
        assert self._initialized, "Crynux sdk hasn't been initialized"
        assert not self._closed, "Crynux sdk has been closed"

        eth_wei = Web3.to_wei(eth, unit)
        await self.token.transfer_eth(address=address, eth=eth_wei)
        cnx_wei = Web3.to_wei(cnx, unit)
        await self.token.transfer_cnx(address=address, cnx=cnx_wei)

    async def generate_images(
        self,
        dst_dir: Union[str, pathlib.Path],
        task_fee: int,
        prompt: str,
        vram_limit: Optional[int] = None,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        negative_prompt: str = "",
        task_optional_args: Optional[sd_args.TaskOptionalArgs] = None,
        task_fee_unit: str = "ether",
        max_retries: int = 5,
        max_timeout_retries: int = 3,
        timeout: Optional[float] = None,
        wait_interval: int = 1,
        auto_cancel: bool = True,
    ):
        assert self._initialized, "Crynux sdk hasn't been initialized"
        assert not self._closed, "Crynux sdk has been closed"

        task_fee = Web3.to_wei(task_fee, task_fee_unit)

        async def _run_task():
            task_id = 0
            task_created = False
            task_success = False
            try:
                with fail_after(timeout):
                    blocknum, task_id, cap = await self.task.create_sd_task(
                        task_fee=task_fee,
                        prompt=prompt,
                        vram_limit=vram_limit,
                        base_model=base_model,
                        negative_prompt=negative_prompt,
                        task_optional_args=task_optional_args,
                        max_retries=max_retries,
                    )
                    task_created = True

                    _logger.info(f"waiting task {task_id} to complete")
                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        retry=retry_if_not_exception_type(TaskAbortedError),
                        reraise=True,
                    ):
                        with attemp:
                            await self.task.wait_task_finish(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )
                    task_success = True

                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        reraise=True,
                    ):
                        with attemp:
                            await self.task.wait_task_result_uploaded(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )

                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        reraise=True,
                    ):
                        with attemp:
                            await self.task.get_task_result(
                                task_id=task_id,
                                task_type=TaskType.SD,
                                count=cap,
                                dst_dir=dst_dir,
                            )
            except TimeoutError as timeout_exc:
                if auto_cancel and task_id > 0 and task_created:
                    if not task_success:
                        _logger.error(
                            f"task {task_id} is not successful after {timeout} seconds"
                        )
                        _logger.info(f"try to cancel task {task_id}")
                        try:
                            async for attemp in AsyncRetrying(
                                wait=wait_fixed(2),
                                stop=stop_after_attempt(max_retries),
                                retry=retry_if_not_exception_type(TxRevertedError),
                                reraise=True,
                            ):
                                with attemp:
                                    await self.task.cancel_task(task_id=task_id)
                                    _logger.info(f"cancel task {task_id} successfully")
                        except TxRevertedError as e:
                            _logger.error(
                                f"cannot cancel task {task_id} due to tx reverted: {e.reason}"
                            )
                            raise TaskCancelError(
                                task_id=task_id, reason=e.reason
                            ) from timeout_exc
                        except Exception as e:
                            _logger.error(
                                f"cannot cancel task {task_id} due to {str(e)}"
                            )
                            raise TaskCancelError(
                                task_id=task_id, reason=str(e)
                            ) from timeout_exc
                    else:
                        e = TaskGetResultTimeout(task_id=task_id)
                        _logger.error(str(e))
                        raise TaskGetResultTimeout(task_id=task_id) from timeout_exc
                else:
                    raise timeout_exc

        def _log_before_retry(retry_state: RetryCallState):
            if retry_state.outcome is not None and retry_state.outcome.failed:
                exc: Exception = retry_state.outcome.exception()
                if isinstance(exc, TaskAbortedError):
                    msg = f"image generation failed due to {exc.reason}, "
                else:
                    msg = f"image generation doesn't complete in {timeout} seconds, "
                
                retry_times = max_timeout_retries - retry_state.attempt_number
                msg += f"retry the image generation, remaining retring times {retry_times}"
                _logger.error(msg)

        async for attemp in AsyncRetrying(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_timeout_retries),
            retry=retry_if_exception_type((TaskAbortedError, TimeoutError))
            | retry_if_exception_cause_type(TimeoutError),
            before_sleep=_log_before_retry,
            reraise=True,
        ):
            with attemp:
                await _run_task()
