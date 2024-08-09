from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

from anyio import create_task_group, sleep, Lock
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from web3.logs import DISCARD
from hexbytes import HexBytes

from crynux_sdk.config import TxOption
from crynux_sdk.contracts import Contracts, TxRevertedError
from crynux_sdk.models import sd_args, sd_ft_args
from crynux_sdk.models.contracts import (
    TaskAborted,
    TaskResultUploaded,
    TaskStarted,
    TaskSuccess,
    TaskType,
    load_event_from_contracts,
)
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

        self._create_task_lock = Lock()

    async def _create_task(
        self,
        task_args: str,
        task_type: TaskType,
        task_fee: int,
        vram_limit: int,
        cap: int,
        gpu_name: str,
        gpu_vram: int,
        checkpoint_dir: str = "",
        max_retries: int = 5,
    ) -> Tuple[int, HexBytes, int]:
        task_id: int = 0
        blocknum: int = 0
        tx_hash: HexBytes = HexBytes(b"")

        # check allowance of task contract
        async for attemp in AsyncRetrying(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_not_exception_type(TxRevertedError),
            reraise=True,
        ):
            with attemp:
                async with self._create_task_lock:
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
                        gpu_name=gpu_name,
                        gpu_vram=gpu_vram,
                        option=self._option,
                    )
                    receipt = await waiter.wait()

                blocknum = receipt["blockNumber"]
                tx_hash = receipt["transactionHash"]

                events = await self._contracts.event_process_receipt(
                    "task", "TaskPending", receipt, errors=DISCARD
                )
                assert len(events) == 1

                task_id = events[0]["args"]["taskId"]
                _logger.info(f"create {str(task_type)} type task {task_id}")

        # sleep to wait the relay server to receive TaskPending event
        await sleep(5)

        def need_retry(exc: BaseException) -> bool:
            if isinstance(exc, RelayError):
                if exc.status_code == 400:
                    return True
                else:
                    return False
            return True

        # upload task args to relay
        if checkpoint_dir is not None:
            async for attemp in AsyncRetrying(
                wait=wait_fixed(2),
                stop=stop_after_attempt(max_retries),
                retry=retry_if_exception(need_retry),
                reraise=True,
            ):
                with attemp:
                    await self._relay.upload_checkpoint(task_id=task_id, checkpoint_dir=checkpoint_dir)
        async for attemp in AsyncRetrying(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception(need_retry),
            reraise=True,
        ):
            with attemp:
                await self._relay.create_task(task_id=task_id, task_args=task_args)
        _logger.info(f"upload task args of task {task_id} to relay")

        return blocknum, tx_hash, task_id

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
    ) -> Tuple[int, HexBytes, int, int]:
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

        blocknum, tx_hash, task_id = await self._create_task(
            task_args=task_args_str,
            task_type=TaskType.SD,
            task_fee=task_fee,
            vram_limit=vram_limit,
            cap=cap,
            gpu_name="",
            gpu_vram=0,
            max_retries=max_retries,
        )
        return blocknum, tx_hash, task_id, cap
    
    async def create_sd_finetune_lora_task(
        self,
        task_fee: int,
        gpu_name: str,
        gpu_vram: int,
        model_name: str,
        dataset_name: str,
        model_variant: Optional[str] = None,
        model_revision: str = "main",
        dataset_config_name: Optional[str] = None,
        dataset_image_column: str = "image",
        dataset_caption_column: str = "text",
        validation_prompt: Optional[str] = None,
        validation_num_images: int = 4,
        center_crop: bool = False,
        random_flip: bool = False,
        rank: int = 8,
        init_lora_weights: Union[bool, Literal["gaussian", "loftq"]] = True,
        target_modules: Union[List[str], str, None] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        prediction_type: Optional[Literal["epsilon", "v_prediction"]] = None,
        max_grad_norm: float = 1.0,
        num_train_epochs: int = 1,
        num_train_steps: Optional[int] = None,
        max_train_epochs: int = 1,
        max_train_steps: Optional[int] = None,
        scale_lr: bool = True,
        resolution: int = 512,
        noise_offset: float = 0,
        snr_gamma: Optional[float] = None,
        lr_scheduler: Literal["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"] = "constant",
        lr_warmup_steps: int = 500,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-8,
        dataloader_num_workers: int = 0,
        mixed_precision: Literal["no", "fp16", "bf16"] = "no",
        seed: int = 0,
        checkpoint: Optional[str] = None,
        max_retries: int = 5,
    ):
        task_args = sd_ft_args.FinetuneLoraTaskArgs(
            model=sd_ft_args.ModelArgs(name=model_name, variant=model_variant, revision=model_revision),
            dataset=sd_ft_args.DatasetArgs(name=dataset_name, config_name=dataset_config_name, image_column=dataset_image_column, caption_column=dataset_caption_column),
            validation=sd_ft_args.ValidationArgs(prompt=validation_prompt, num_images=validation_num_images),
            train_args=sd_ft_args.TrainArgs(
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                prediction_type=prediction_type,
                max_grad_norm=max_grad_norm,
                num_train_epochs=num_train_epochs,
                num_train_steps=num_train_steps,
                max_train_epochs=max_train_epochs,
                max_train_steps=max_train_steps,
                scale_lr=scale_lr,
                resolution=resolution,
                noise_offset=noise_offset,
                snr_gamma=snr_gamma,
                lr_scheduler=sd_ft_args.LRSchedulerArgs(lr_scheduler=lr_scheduler, lr_warmup_steps=lr_warmup_steps),
                adam_args=sd_ft_args.AdamOptimizerArgs(
                    beta1=adam_beta1, beta2=adam_beta2, weight_decay=adam_weight_decay, epsilon=adam_epsilon,
                )
            ),
            transforms=sd_ft_args.TransformArgs(center_crop=center_crop, random_flip=random_flip),
            lora=sd_ft_args.LoraArgs(rank=rank, init_lora_weights=init_lora_weights, target_modules=target_modules),
            dataloader_num_workers=dataloader_num_workers,
            mixed_precision=mixed_precision,
            seed=seed,
            checkpoint="checkpoint.zip" if checkpoint is not None else None
        )
        task_args_str = task_args.model_dump_json()
        blocknum, tx_hash, task_id = await self._create_task(
            task_args=task_args_str,
            task_type=TaskType.SD_FT,
            task_fee=task_fee,
            vram_limit=gpu_vram,
            cap=1,
            gpu_name=gpu_name,
            gpu_vram=gpu_vram,
            max_retries=max_retries,
        )
        return blocknum, tx_hash, task_id, 1


    async def wait_task_started(
        self, task_id: int, from_block: int, interval: int
    ) -> Tuple[int, HexBytes, TaskStarted]:
        while True:
            to_block = await self._contracts.get_current_block_number()
            if from_block <= to_block:
                events = await self._contracts.get_events(
                    contract_name="task",
                    event_name="TaskStarted",
                    from_block=from_block,
                    to_block=to_block,
                    filter_args={"taskId": task_id}
                )
                if len(events) > 0:
                    res = load_event_from_contracts(events[0])
                    assert isinstance(res, TaskStarted)
                    assert res.task_id == task_id
                    return events[0]["blockNumber"], events[0]["transactionHash"], res

                from_block = to_block + 1
            await sleep(interval)

    async def wait_task_success(
        self, task_id: int, from_block: int, interval: int
    ) -> Tuple[int, HexBytes, TaskSuccess]:
        while True:
            to_block = await self._contracts.get_current_block_number()
            if from_block <= to_block:
                events = await self._contracts.get_events(
                    contract_name="task",
                    event_name="TaskSuccess",
                    filter_args={"taskId": task_id},
                    from_block=from_block,
                    to_block=to_block,
                )
                if len(events) > 0:
                    res = load_event_from_contracts(events[0])
                    assert isinstance(res, TaskSuccess)
                    assert res.task_id == task_id
                    _logger.info(f"task {task_id} success")
                    return events[0]["blockNumber"], events[0]["transactionHash"], res

                from_block = to_block + 1
            await sleep(interval)

    async def wait_task_aborted(
        self, task_id: int, from_block: int, interval: int
    ) -> Tuple[int, HexBytes, TaskAborted]:
        while True:
            to_block = await self._contracts.get_current_block_number()
            if from_block < to_block:
                events = await self._contracts.get_events(
                    contract_name="task",
                    event_name="TaskAborted",
                    filter_args={"taskId": task_id},
                    from_block=from_block,
                    to_block=to_block,
                )
                if len(events) > 0:
                    res = load_event_from_contracts(events[0])
                    assert isinstance(res, TaskAborted)
                    assert res.task_id == task_id
                    _logger.info(f"task {task_id} aborted, reason: {res.reason}")
                    return events[0]["blockNumber"], events[0]["transactionHash"], res

                from_block = to_block + 1
            await sleep(interval)

    async def wait_task_result_uploaded(
        self, task_id: int, from_block: int, interval: int
    ) -> Tuple[int, HexBytes, TaskResultUploaded]:
        while True:
            to_block = await self._contracts.get_current_block_number()
            if from_block <= to_block:
                events = await self._contracts.get_events(
                    contract_name="task",
                    event_name="TaskResultUploaded",
                    filter_args={"taskId": task_id},
                    from_block=from_block,
                    to_block=to_block,
                )
                if len(events) > 0:
                    res = load_event_from_contracts(events[0])
                    assert isinstance(res, TaskResultUploaded)
                    assert res.task_id == task_id
                    _logger.info(f"task {task_id} result is uploaded")
                    return events[0]["blockNumber"], events[0]["transactionHash"], res

                from_block = to_block + 1
            await sleep(interval)

    async def wait_task_finish(self, task_id: int, from_block: int, interval: int = 1):

        async def raise_when_task_aborted():
            _, _, event = await self.wait_task_aborted(
                task_id=task_id, from_block=from_block, interval=interval
            )
            raise TaskAbortedError(task_id, event.reason)

        async with create_task_group() as tg:
            tg.start_soon(raise_when_task_aborted)

            res = await self.wait_task_success(
                task_id=task_id, from_block=from_block, interval=interval
            )
            tg.cancel_scope.cancel()
        return res

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

    async def get_task_result_checkpoint(
        self,
        task_id: int,
        checkpoint_dir: Union[str, pathlib.Path]
    ):
        if not isinstance(checkpoint_dir, str):
            checkpoint_dir = str(checkpoint_dir)
        await self._relay.get_result_checkpoint(
            task_id=task_id,
            result_checkpoint_dir=checkpoint_dir
        )