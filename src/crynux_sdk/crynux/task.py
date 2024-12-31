from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from anyio import Lock, create_task_group, sleep
from hexbytes import HexBytes
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from web3.logs import DISCARD

from crynux_sdk import utils
from crynux_sdk.config import TxOption
from crynux_sdk.contracts import Contracts, TxRevertedError
from crynux_sdk.models import sd_args, sd_ft_lora_args
from crynux_sdk.models.contracts import (
    ChainTask,
    TaskAbortReason,
    TaskStatus,
    TaskType,
    load_event_from_contracts,
)
from crynux_sdk.relay import Relay, RelayError

from .exceptions import TaskAbortedError

_logger = logging.getLogger(__name__)


def _relay_need_retry(exc: BaseException) -> bool:
    if isinstance(exc, RelayError):
        if exc.status_code == 400:
            return True
        else:
            return False
    return True


class Task(object):
    def __init__(
        self, contracts: Contracts, relay: Relay, option: Optional[TxOption] = None
    ) -> None:
        self._contracts = contracts
        self._relay = relay
        self._option = option

        self._create_task_lock = Lock()

    async def _get_task(self, task_id_commitment: bytes, max_retries: int = 5):
        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
        async def _inner():
            return await self._contracts.task_contract.get_task(task_id_commitment)

        return await _inner()

    async def _get_node_info(self, address: str, max_retries: int = 5):
        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
        async def _inner():
            return await self._contracts.node_contract.get_node_info(address=address)

        return await _inner()

    async def _create_task_on_chain(
        self,
        task_id: bytes,
        task_type: TaskType,
        task_model_ids: List[str],
        task_version: List[int],
        task_fee: int,
        task_size: int,
        min_vram: int,
        required_gpu: str = "",
        required_gpu_vram: int = 0,
        max_retries: int = 5,
    ):
        nonce, task_id_commitment = utils.generate_task_id_commitment(task_id)
        waiter = None

        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_not_exception_type(TxRevertedError),
            reraise=True,
        )
        async def _inner():
            nonlocal waiter
            if waiter is None:
                waiter = await self._contracts.task_contract.create_task(
                    task_fee=task_fee,
                    task_type=task_type,
                    task_id_commitment=task_id_commitment,
                    nonce=nonce,
                    model_ids=task_model_ids,
                    min_vram=min_vram,
                    required_gpu=required_gpu,
                    required_gpu_vram=required_gpu_vram,
                    task_version=task_version,
                    task_size=task_size,
                    option=self._option,
                )
            await waiter.wait()
            _logger.info(
                f"create {str(task_type)} type task, task id commitment: {task_id_commitment.hex()}"
            )
            return task_id_commitment

        return await _inner()

    async def _create_task_on_relay(
        self,
        task_id_commitment: bytes,
        task_args: str,
        checkpoint_dir: Optional[str] = None,
        max_retries: int = 5,
    ):
        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception(_relay_need_retry),
            reraise=True,
        )
        async def _inner():
            await self._relay.create_task(
                task_id_commitment=task_id_commitment,
                task_args=task_args,
                checkpoint_dir=checkpoint_dir,
            )
            _logger.info(
                f"upload task args of task {task_id_commitment.hex()} to relay"
            )

        return await _inner()

    async def _validate_single_task(
        self,
        task_id_commitment: bytes,
        vrf_proof: bytes,
        max_retries: int = 5,
    ):
        waiter = None

        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_not_exception_type(TxRevertedError),
            reraise=True,
        )
        async def _inner():
            nonlocal waiter
            if waiter is None:
                waiter = await self._contracts.task_contract.validate_single_task(
                    task_id_commitment=task_id_commitment,
                    vrf_proof=vrf_proof,
                    public_key=self._contracts.public_key.to_bytes(),
                    option=self._option,
                )
            await waiter.wait()
            _logger.info(f"validate single task {task_id_commitment.hex()}")

        return await _inner()

    async def _validate_task_group(
        self,
        task_id: bytes,
        task_id_commitments: List[bytes],
        vrf_proof: bytes,
        max_retries: int = 5,
    ):
        assert len(task_id_commitments) == 3
        waiter = None

        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_not_exception_type(TxRevertedError),
            reraise=True,
        )
        async def _inner():
            nonlocal waiter
            if waiter is None:
                waiter = await self._contracts.task_contract.validate_task_group(
                    task_id_commitment1=task_id_commitments[0],
                    task_id_commitment2=task_id_commitments[1],
                    task_id_commitment3=task_id_commitments[2],
                    task_id=task_id,
                    vrf_proof=vrf_proof,
                    public_key=self._contracts.public_key.to_bytes(),
                    option=self._option,
                )
            await waiter.wait()
            _logger.info(
                f"validate task group {[commitment.hex() for commitment in task_id_commitments]}"
            )

        return await _inner()

    async def _upload_task_args(
        self,
        task_id_commitment: bytes,
        task_args: str,
        checkpoint_dir: Optional[str] = None,
        max_retries: int = 5,
        wait_interval: int = 1,
    ):
        await self._wait_task_started(
            task_id_commitment=task_id_commitment,
            interval=wait_interval,
            max_retries=max_retries,
        )
        await self._create_task_on_relay(
            task_id_commitment=task_id_commitment,
            task_args=task_args,
            checkpoint_dir=checkpoint_dir,
            max_retries=max_retries,
        )

    async def _create_task(
        self,
        task_args: str,
        task_type: TaskType,
        task_model_ids: List[str],
        task_version: str,
        task_fee: int,
        task_size: int,
        min_vram: int,
        required_gpu: str = "",
        required_gpu_vram: int = 0,
        checkpoint_dir: Optional[str] = None,
        max_retries: int = 5,
        wait_interval: int = 1,
    ):
        task_id = utils.generate_task_id()

        version_list = [int(v) for v in task_version.split(".")]
        assert len(version_list) == 3

        # create the first task
        task_id_commitment = await self._create_task_on_chain(
            task_id=task_id,
            task_type=task_type,
            task_model_ids=task_model_ids,
            task_version=version_list,
            task_fee=task_fee,
            task_size=task_size,
            min_vram=min_vram,
            required_gpu=required_gpu,
            required_gpu_vram=required_gpu_vram,
            max_retries=max_retries,
        )
        task = await self._get_task(task_id_commitment, max_retries=max_retries)
        sampling_seed = task.sampling_seed
        num, vrf_proof = utils.vrf_prove(
            sampling_seed, self._contracts.private_key.to_bytes()
        )

        yield task_id, task_id_commitment, vrf_proof

        await self._upload_task_args(
            task_id_commitment=task_id_commitment,
            task_args=task_args,
            checkpoint_dir=checkpoint_dir,
            max_retries=max_retries,
            wait_interval=wait_interval,
        )

        # check if need 2 additional tasks for validation
        if num % 10 == 0:
            # for llm and sd_ft task, need to keep all three tasks using the same GPU
            if task_type == TaskType.LLM or task_type == TaskType.SD_FT_LORA:
                node_info = await self._get_node_info(
                    address=task.selected_node, max_retries=max_retries
                )
                required_gpu = node_info.gpu.name
                required_gpu_vram = node_info.gpu.vram

            async with create_task_group() as tg:
                for _ in range(2):
                    task_id_commitment = await self._create_task_on_chain(
                        task_id=task_id,
                        task_type=task_type,
                        task_model_ids=task_model_ids,
                        task_version=version_list,
                        task_fee=task_fee,
                        task_size=task_size,
                        min_vram=min_vram,
                        required_gpu=required_gpu,
                        required_gpu_vram=required_gpu_vram,
                        max_retries=max_retries,
                    )
                    tg.start_soon(
                        self._upload_task_args,
                        task_id_commitment,
                        task_args,
                        checkpoint_dir,
                        max_retries,
                        wait_interval,
                    )
                    yield task_id, task_id_commitment, vrf_proof

    async def cancel_task(self, task_id_commitment: bytes, max_retries: int = 5):
        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_not_exception_type(TxRevertedError),
            reraise=True,
        )
        async def _inner():
            waiter = await self._contracts.task_contract.abort_task(
                task_id_commitment=task_id_commitment,
                abort_reason=TaskAbortReason.Timeout,
                option=self._option,
            )
            await waiter.wait()

        return await _inner()

    async def create_sd_task(
        self,
        task_fee: int,
        prompt: str,
        min_vram: Optional[int] = None,
        base_model: str = "crynux-ai/stable-diffusion-v1-5",
        negative_prompt: str = "",
        required_gpu: str = "",
        required_gpu_vram: int = 0,
        task_version: str = "3.0.0",
        max_retries: int = 5,
        wait_interval: int = 1,
        task_optional_args: Optional[sd_args.TaskOptionalArgs] = None,
    ):
        task_args_obj: Dict[str, Any] = {
            "prompt": prompt,
            "base_model": base_model,
            "negative_prompt": negative_prompt,
        }
        if task_optional_args is not None:
            task_args_obj.update(task_optional_args)

        task_args = sd_args.TaskArgs.model_validate(task_args_obj)
        if base_model == "crynux-ai/sdxl-turbo":
            task_args.task_config.cfg = 0
            task_args.task_config.steps = max(4, task_args.task_config.steps)
            task_args.scheduler = sd_args.scheduler_args.EulerAncestralDiscrete(
                args=sd_args.scheduler_args.CommonSchedulerArgs(
                    timestep_spacing="trailing"
                )
            )
        task_args_str = task_args.model_dump_json()
        task_size = task_args.task_config.num_images

        if min_vram is None:
            if base_model == "crynux-ai/stable-diffusion-v1-5":
                min_vram = 8
            elif (
                base_model == "crynux-ai/sdxl-turbo"
                or base_model == "crynux-ai/stable-diffusion-xl-base-1.0"
            ):
                min_vram = 14
            else:
                min_vram = 10

        if len(required_gpu) > 0 and required_gpu_vram > 0:
            min_vram = required_gpu_vram

        async for task_id, task_id_commitment, vrf_proof in  self._create_task(
            task_args=task_args_str,
            task_type=TaskType.SD,
            task_model_ids=task_args.generate_model_ids(),
            task_version=task_version,
            task_fee=task_fee,
            task_size=task_size,
            min_vram=min_vram,
            required_gpu=required_gpu,
            required_gpu_vram=required_gpu_vram,
            max_retries=max_retries,
            wait_interval=wait_interval,
        ):
            yield task_id, task_id_commitment, vrf_proof, task_size

    async def create_sd_finetune_lora_task(
        self,
        task_fee: int,
        required_gpu: str,
        required_gpu_vram: int,
        model_name: str,
        dataset_name: str,
        task_version: str = "3.0.0",
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
        lr_scheduler: Literal[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ] = "constant",
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
        wait_interval: int = 1,
    ):
        task_args = sd_ft_lora_args.FinetuneLoraTaskArgs(
            model=sd_ft_lora_args.ModelArgs(
                name=model_name, variant=model_variant, revision=model_revision
            ),
            dataset=sd_ft_lora_args.DatasetArgs(
                name=dataset_name,
                config_name=dataset_config_name,
                image_column=dataset_image_column,
                caption_column=dataset_caption_column,
            ),
            validation=sd_ft_lora_args.ValidationArgs(
                prompt=validation_prompt, num_images=validation_num_images
            ),
            train_args=sd_ft_lora_args.TrainArgs(
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
                lr_scheduler=sd_ft_lora_args.LRSchedulerArgs(
                    lr_scheduler=lr_scheduler, lr_warmup_steps=lr_warmup_steps
                ),
                adam_args=sd_ft_lora_args.AdamOptimizerArgs(
                    beta1=adam_beta1,
                    beta2=adam_beta2,
                    weight_decay=adam_weight_decay,
                    epsilon=adam_epsilon,
                ),
            ),
            transforms=sd_ft_lora_args.TransformArgs(
                center_crop=center_crop, random_flip=random_flip
            ),
            lora=sd_ft_lora_args.LoraArgs(
                rank=rank,
                init_lora_weights=init_lora_weights,
                target_modules=target_modules,
            ),
            dataloader_num_workers=dataloader_num_workers,
            mixed_precision=mixed_precision,
            seed=seed,
            checkpoint="checkpoint.zip" if checkpoint is not None else None,
        )
        task_args_str = task_args.model_dump_json()
        model_id = f"base:{model_name}"
        if model_variant is not None:
            model_id += f"+{model_variant}"
        async for task_id, task_id_commitment, vrf_proof in self._create_task(
            task_args=task_args_str,
            task_type=TaskType.SD_FT_LORA,
            task_model_ids=[model_id],
            task_version=task_version,
            task_fee=task_fee,
            task_size=1,
            min_vram=required_gpu_vram,
            required_gpu=required_gpu,
            required_gpu_vram=required_gpu_vram,
            max_retries=max_retries,
            checkpoint_dir=checkpoint,
            wait_interval=wait_interval,
        ):
            yield task_id, task_id_commitment, vrf_proof

    async def _wait_task_started(
        self, task_id_commitment: bytes, interval: int, max_retries: int = 5
    ) -> ChainTask:
        while True:
            task = await self._get_task(task_id_commitment, max_retries=max_retries)
            if task.status == TaskStatus.Started:
                _logger.info(f"task {task_id_commitment.hex()} is started")
                return task
            elif task.status == TaskStatus.EndAborted:
                _logger.info(f"task {task_id_commitment.hex()} is aborted")
                raise TaskAbortedError(
                    task_id_commitment=task_id_commitment, reason=task.abort_reason.name
                )
            await sleep(interval)

    async def _wait_task_score_ready(
        self, task_id_commitment: bytes, interval: int, max_retries: int = 5
    ):
        while True:
            task = await self._get_task(task_id_commitment, max_retries=max_retries)
            if task.status == TaskStatus.ScoreReady:
                _logger.info(f"task {task_id_commitment.hex()} score is ready")
                return task
            elif task.status == TaskStatus.ErrorReported:
                _logger.info(f"task {task_id_commitment.hex()} reported error")
                return task
            elif task.status == TaskStatus.EndAborted:
                _logger.info(f"task {task_id_commitment.hex()} is aborted")
                raise TaskAbortedError(
                    task_id_commitment=task_id_commitment, reason=task.abort_reason.name
                )
            await sleep(interval)

    async def _validate_task(
        self,
        task_id: bytes,
        task_id_commitments: List[bytes],
        vrf_proof: bytes,
        max_retries: int = 5,
    ):
        assert len(task_id_commitments) == 1 or len(task_id_commitments) == 3
        if len(task_id_commitments) == 1:
            await self._validate_single_task(
                task_id_commitment=task_id_commitments[0],
                vrf_proof=vrf_proof,
                max_retries=max_retries,
            )
        else:
            await self._validate_task_group(
                task_id=task_id,
                task_id_commitments=task_id_commitments,
                vrf_proof=vrf_proof,
                max_retries=max_retries,
            )

    async def _wait_single_task_validated(
        self, task_id_commitment: bytes, interval: int, max_retries: int = 5
    ):
        while True:
            task = await self._get_task(task_id_commitment, max_retries=max_retries)
            if task.status == TaskStatus.Validated:
                _logger.info(f"task {task_id_commitment.hex()} is validated")
                return task
            elif task.status == TaskStatus.EndAborted:
                _logger.info(f"task {task_id_commitment.hex()} is aborted")
                raise TaskAbortedError(
                    task_id_commitment=task_id_commitment, reason=task.abort_reason.name
                )
            await sleep(interval)

    async def _wait_task_in_group_validated(
        self, task_id_commitment: bytes, interval: int, max_retries: int = 5
    ):
        while True:
            task = await self._get_task(task_id_commitment, max_retries=max_retries)
            if task.status == TaskStatus.GroupValidated:
                _logger.info(f"task {task_id_commitment.hex()} is validated")
                return task
            elif task.status == TaskStatus.EndGroupRefund:
                _logger.info(f"task {task_id_commitment.hex()} is successful")
                return task
            elif task.status == TaskStatus.EndInvalidated:
                _logger.info(f"task {task_id_commitment.hex()} is invalidated")
                return task
            elif task.status == TaskStatus.EndAborted:
                _logger.info(f"task {task_id_commitment.hex()} is aborted")
                raise TaskAbortedError(
                    task_id_commitment=task_id_commitment, reason=task.abort_reason.name
                )
            await sleep(interval)

    async def _wait_task_group_validated(
        self, task_id_commitments: List[bytes], interval: int, max_retries: int = 5
    ):
        result_task_id_commitment = b""
        abort_error: Optional[TaskAbortedError] = None

        async def _wait(task_id_commitment: bytes):
            nonlocal result_task_id_commitment
            nonlocal abort_error
            try:
                task = await self._wait_task_in_group_validated(
                    task_id_commitment=task_id_commitment,
                    interval=interval,
                    max_retries=max_retries,
                )
                if task.status == TaskStatus.GroupValidated:
                    result_task_id_commitment = task_id_commitment
            except TaskAbortedError as e:
                if abort_error is None:
                    abort_error = e

        async with create_task_group() as tg:
            for commitment in task_id_commitments:
                tg.start_soon(_wait, commitment)

        if len(result_task_id_commitment) > 0:
            return result_task_id_commitment
        else:
            assert abort_error is not None
            raise abort_error

    async def _wait_task_validated(
        self, task_id_commitments: List[bytes], interval: int, max_retries: int = 5
    ):
        assert len(task_id_commitments) == 1 or len(task_id_commitments) == 3

        if len(task_id_commitments) == 1:
            await self._wait_single_task_validated(
                task_id_commitment=task_id_commitments[0],
                interval=interval,
                max_retries=max_retries,
            )
            return task_id_commitments[0]
        else:
            return await self._wait_task_group_validated(
                task_id_commitments=task_id_commitments,
                interval=interval,
                max_retries=max_retries,
            )

    async def _wait_task_success(
        self, task_id_commitment: bytes, interval: int, max_retries: int = 5
    ):
        while True:
            task = await self._get_task(task_id_commitment, max_retries=max_retries)
            if (
                task.status == TaskStatus.EndSuccess
                or task.status == TaskStatus.EndGroupSuccess
            ):
                _logger.info(f"task {task_id_commitment.hex()} is successful")
                return task
            elif task.status == TaskStatus.EndAborted:
                _logger.info(f"task {task_id_commitment.hex()} is aborted")
                raise TaskAbortedError(
                    task_id_commitment=task_id_commitment, reason=task.abort_reason.name
                )
            await sleep(interval)

    async def get_task_result(
        self,
        task_id_commitment: bytes,
        task_type: TaskType,
        count: int,
        dst_dir: Union[str, pathlib.Path],
        max_retries: int = 5,
    ) -> List[pathlib.Path]:
        ext = "png"
        if task_type == TaskType.LLM:
            ext = "json"

        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception(_relay_need_retry),
            reraise=True,
        )
        async def _download_result(
            task_id_commitment: bytes, index: int, file: pathlib.Path
        ):
            with file.open(mode="wb") as f:
                await self._relay.get_result(
                    task_id_commitment=task_id_commitment, index=index, dst=f
                )
            _logger.info(
                f"download task {task_id_commitment.hex()} result {index}.{ext} success"
            )

        dst_dir = pathlib.Path(dst_dir)
        if not dst_dir.exists():
            raise ValueError(f"Result dir {dst_dir} does not exist")

        res = []
        async with create_task_group() as tg:
            for i in range(count):
                file = dst_dir / f"{i}.{ext}"
                tg.start_soon(_download_result, task_id_commitment, i, file)
                res.append(file)
        return res

    async def get_task_result_checkpoint(
        self,
        task_id_commitment: bytes,
        checkpoint_dir: Union[str, pathlib.Path],
        max_retries: int = 5,
    ):
        if not isinstance(checkpoint_dir, str):
            checkpoint_dir = str(checkpoint_dir)

        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_retries),
            retry=retry_if_exception(_relay_need_retry),
            reraise=True,
        )
        async def _download_result_checkpoint():
            await self._relay.get_result_checkpoint(
                task_id_commitment=task_id_commitment,
                result_checkpoint_dir=checkpoint_dir,
            )

        return await _download_result_checkpoint()

    async def execute_task(
        self,
        task_id: bytes,
        task_id_commitments: List[bytes],
        vrf_proof: bytes,
        wait_interval: int = 1,
        max_retries: int = 5,
    ):
        # wait task score ready
        abort_errors: List[TaskAbortedError] = []

        async def _wait_task_score_ready(task_id_commitment: bytes):
            try:
                await self._wait_task_score_ready(
                    task_id_commitment=task_id_commitment,
                    interval=wait_interval,
                    max_retries=max_retries,
                )
            except TaskAbortedError as e:
                abort_errors.append(e)

        async with create_task_group() as tg:
            for task_id_commitment in task_id_commitments:
                tg.start_soon(_wait_task_score_ready, task_id_commitment)

        # all tasks aborted, raise TaskAbortError
        if len(abort_errors) == len(task_id_commitments):
            raise abort_errors[0]

        # validate task
        await self._validate_task(
            task_id=task_id,
            task_id_commitments=task_id_commitments,
            vrf_proof=vrf_proof,
            max_retries=max_retries,
        )

        result_task_id_commitment = await self._wait_task_validated(
            task_id_commitments=task_id_commitments,
            interval=wait_interval,
            max_retries=max_retries,
        )

        # wait task success
        await self._wait_task_success(
            result_task_id_commitment, interval=wait_interval, max_retries=max_retries
        )
        return result_task_id_commitment
