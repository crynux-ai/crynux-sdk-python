from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

from anyio import create_task_group, sleep
from tenacity import (AsyncRetrying, retry_if_exception,
                      retry_if_not_exception_type, stop_after_attempt,
                      wait_fixed)
from web3.logs import DISCARD

from crynux_sdk.config import TxOption
from crynux_sdk.contracts import Contracts, TxRevertedError
from crynux_sdk.models import sd_args
from crynux_sdk.models.contracts import (TaskAborted, TaskResultUploaded,
                                         TaskStarted, TaskSuccess, TaskType,
                                         load_event_from_contracts)
from crynux_sdk.relay import Relay, RelayError
from crynux_sdk.utils import get_task_hash

from .exceptions import TaskAbortedError

_logger = logging.getLogger(__name__)


class Task(object):
    def __init__(
        self, contracts: Contracts, relay: Relay, option: Optional[TxOption] = None
    ) -> None:
        self._contracts = contracts
        self._relay = relay
        self._option = option

    async def _create_task(
        self,
        task_args: str,
        task_type: TaskType,
        task_fee: int,
        vram_limit: int,
        cap: int,
        max_retries: int = 5,
    ) -> Tuple[int, int]:
        task_id: int = 0
        blocknum: int = 0

        # check allowance of task contract
        async for attemp in AsyncRetrying(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_not_exception_type(TxRevertedError),
            reraise=True,
        ):
            with attemp:
                allowance = await self._contracts.token_contract.allowance(
                    self._contracts.task_contract.address
                )
                if allowance < task_fee:
                    waiter = await self._contracts.token_contract.approve(
                        self._contracts.task_contract.address,
                        task_fee,
                        option=self._option,
                    )
                    await waiter.wait()
                    _logger.info(f"approve task contract {task_fee} cnx")

                task_hash = get_task_hash(task_args)
                data_hash = bytes([0] * 32)

                # create task on chain
                waiter = await self._contracts.task_contract.create_task(
                    task_type=task_type,
                    task_hash=task_hash,
                    data_hash=data_hash,
                    vram_limit=vram_limit,
                    task_fee=task_fee,
                    cap=cap,
                    option=self._option,
                )
                receipt = await waiter.wait()

                blocknum = receipt["blockNumber"]

                events = self._contracts.task_contract.contract.events.TaskPending().process_receipt(
                    receipt, errors=DISCARD
                )
                assert len(events) == 1

                task_id = events[0]["args"]["taskId"]
                _logger.info(f"create {str(task_type)} type task {task_id}")

        # sleep to wait the relay server to receive TaskPending event
        await sleep(2)

        def need_retry(exc: BaseException) -> bool:
            if isinstance(exc, RelayError):
                if (
                    exc.status_code == 400
                    and exc.message == "Task not found on the Blockchain"
                ):
                    return True
                else:
                    return False
            return True

        # upload task args to relay
        async for attemp in AsyncRetrying(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception(need_retry),
            reraise=True,
        ):
            with attemp:
                await self._relay.create_task(task_id=task_id, task_args=task_args)
        _logger.info(f"upload task args of task {task_id} to relay")

        return blocknum, task_id

    async def cancel_task(self, task_id: int):
        waiter = await self._contracts.task_contract.cancel_task(
            task_id=task_id, option=self._option
        )
        await waiter.wait()

    async def create_sd_task(
        self,
        task_fee: int,
        prompt: str,
        vram_limit: Optional[int] = None,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        negative_prompt: str = "",
        max_retries: int = 5,
        task_optional_args: Optional[sd_args.TaskOptionalArgs] = None,
    ) -> Tuple[int, int, int]:
        task_args_obj: Dict[str, Any] = {
            "prompt": prompt,
            "base_model": base_model,
            "negative_prompt": negative_prompt,
        }
        if task_optional_args is not None:
            task_args_obj.update(task_optional_args)

        task_args = sd_args.TaskArgs.model_validate(task_args_obj)
        task_args_str = task_args.model_dump_json()

        cap = task_args.task_config.num_images

        if vram_limit is None:
            if base_model == "runwayml/stable-diffusion-v1-5":
                vram_limit = 8
            else:
                vram_limit = 10

        blocknum, task_id = await self._create_task(
            task_args=task_args_str,
            task_type=TaskType.SD,
            task_fee=task_fee,
            vram_limit=vram_limit,
            cap=cap,
            max_retries=max_retries,
        )
        return blocknum, task_id, cap

    async def wait_task_started(
        self, task_id: int, from_block: int, interval: int
    ) -> TaskStarted:
        while True:
            events = await self._contracts.task_contract.get_events(
                event_name="TaskStarted",
                filter_args={"taskId": task_id},
                from_block=from_block,
            )
            if len(events) > 0:
                res = load_event_from_contracts(events[0])
                assert isinstance(res, TaskStarted)
                assert res.task_id == task_id
                _logger.info(f"task {task_id} started")
                return res

            await sleep(interval)

    async def wait_task_success(
        self, task_id: int, from_block: int, interval: int
    ) -> TaskSuccess:
        while True:
            events = await self._contracts.task_contract.get_events(
                event_name="TaskSuccess",
                filter_args={"taskId": task_id},
                from_block=from_block,
            )
            if len(events) > 0:
                res = load_event_from_contracts(events[0])
                assert isinstance(res, TaskSuccess)
                assert res.task_id == task_id
                _logger.info(f"task {task_id} success")
                return res

            await sleep(interval)

    async def wait_task_aborted(
        self, task_id: int, from_block: int, interval: int
    ) -> TaskAborted:
        while True:
            events = await self._contracts.task_contract.get_events(
                event_name="TaskAborted",
                filter_args={"taskId": task_id},
                from_block=from_block,
            )
            if len(events) > 0:
                res = load_event_from_contracts(events[0])
                assert isinstance(res, TaskAborted)
                assert res.task_id == task_id
                _logger.info(f"task {task_id} aborted, reason: {res.reason}")
                return res

            await sleep(interval)

    async def wait_task_result_uploaded(
        self, task_id: int, from_block: int, interval: int
    ) -> TaskResultUploaded:
        while True:
            events = await self._contracts.task_contract.get_events(
                event_name="TaskResultUploaded",
                filter_args={"taskId": task_id},
                from_block=from_block,
            )
            if len(events) > 0:
                res = load_event_from_contracts(events[0])
                assert isinstance(res, TaskResultUploaded)
                assert res.task_id == task_id
                _logger.info(f"task {task_id} result is uploaded")
                return res

            await sleep(interval)

    async def wait_task_finish(self, task_id: int, from_block: int, interval: int = 1):

        async def raise_when_task_aborted():
            event = await self.wait_task_aborted(
                task_id=task_id, from_block=from_block, interval=interval
            )
            raise TaskAbortedError(task_id, event.reason)

        async with create_task_group() as tg:
            tg.start_soon(raise_when_task_aborted)

            await self.wait_task_success(
                task_id=task_id, from_block=from_block, interval=interval
            )
            tg.cancel_scope.cancel()

    async def get_task_result(
        self,
        task_id: int,
        task_type: TaskType,
        count: int,
        dst_dir: Union[str, pathlib.Path],
    ) -> List[pathlib.Path]:
        ext = "png" if task_type == TaskType.SD else "json"

        async def _download_result(task_id: int, index: int, file: pathlib.Path):
            with file.open(mode="wb") as f:
                await self._relay.get_result(task_id=task_id, index=index, dst=f)
            _logger.info(f"download task {task_id} result {index}.{ext} success")

        dst_dir = pathlib.Path(dst_dir)
        if not dst_dir.exists():
            raise ValueError(f"Result dir {dst_dir} does not exist")

        res = []
        async with create_task_group() as tg:
            for i in range(count):
                file = dst_dir / f"{i}.{ext}"
                tg.start_soon(_download_result, task_id, i, file)
                res.append(file)
        return res

