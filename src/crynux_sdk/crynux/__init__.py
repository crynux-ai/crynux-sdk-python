from __future__ import annotations

import logging
import os
import pathlib
import shutil
import tempfile
from functools import partial
from typing import List, Literal, Optional, Tuple, Union

from anyio import fail_after, move_on_after, to_thread, create_task_group
from tenacity import (
    retry,
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
    """
    The main entry point of crynux sdk.

    You should call the `init` method before you calling other method of this class.
    And you should call the `close` method after you don't need use of it.

    For example:
    ```
    crynux = Crynux(privkey=privkey)
    await crynux.init()
    try:
        await crynux.generate_images(...)
    finally:
        await crynux.close()
    ```

    This class is also a async context manager. So you can automatically close it by `async with` syntax.
    For example:
    ```
    crynux = Crynux(privkey=privkey)
    await crynux.init()
    async with crynux:
        await crynux.generate_images(...)
    ```
    """

    contracts: Contracts
    relay: Relay

    task: Task
    token: Token

    def __init__(
        self,
        privkey: Optional[str] = None,
        chain_provider_path: Optional[str] = None,
        relay_url: Optional[str] = None,
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
        contracts_timeout: int = 30,
        contracts_rps: int = 10,
        relay_timeout: float = 30,
        contracts: Optional[Contracts] = None,
        relay: Optional[Relay] = None,
    ) -> None:
        """
        privkey: Private key. Need for interacting with the blockchain.
        chain_provider_path: Chain provider path. Can be a json rpc path (starts with http) or a websocket path (starts with ws)
                             Default to None, means using the default provider path.
        relay_url: The relay server url. Default to None, means using the default provider path.
        token_contract_address: Token contract address. Default to None, means using the default token contract address.
        node_contract_address: Node contract address. Default to None, means using the default node contract address.
        task_contract_address: Task contract address. Default to None, means using the default task contract address.
        qos_contract_address: qos contract address. Default to None, means using the default qos contract address.
        task_queue_contract_address: Task queue contract address. Default to None, means using the default task queue contract address.
        netstats_contract_address: Netstats contract address. Default to None, means using the default netstats contract address.
        chain_id: Chain id of crynux blockchain. Default to None, means using the default chain id.
        gas: Gas limit of transaction. Default to None, means using the default gas limit.
        gas_price: Gas price of transaction. Default to None, means using the default gas price.
        max_fee_per_gas: Max fee per gas of transaction. Default to None, means using the default max fee per gas.
        max_priority_fee_per_gas: Max priority fee per gas of transaction. Default to None, means using the default max priority fee per gas.
        contracts_timeout: Timeout for interacting with the blockchain in seconds. Default to 30 seconds.
        relay_timeout: Timeout for interacting with the relay in seconds. Default to 30 seconds.

        contracts: crynux_sdk.contracts.Contracts instance. Used for testing.
        relay: crynux_sdk.relay.Relay instance. Used for testing.
        """
        if contracts is not None:
            self.contracts = contracts
        else:
            assert privkey is not None, "private key is empty"
            chain_provider_path = chain_provider_path or get_default_provider_path()
            self.contracts = Contracts(
                provider_path=chain_provider_path,
                privkey=privkey,
                timeout=contracts_timeout,
                rps=contracts_rps,
            )

        default_contract_config = get_default_contract_config()

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

    async def _cancel_task(self, task_id_commitment: bytes, max_retries: int = 5):
        _logger.info(f"try to cancel task {task_id_commitment.hex()}")
        try:
            await self.task.cancel_task(task_id_commitment=task_id_commitment, max_retries=max_retries)
            _logger.info(f"cancel task {task_id_commitment.hex()} successfully")

        except TxRevertedError as e:
            _logger.error(
                f"cannot cancel task {task_id_commitment.hex()} due to tx reverted: {e.reason}"
            )
            raise TaskCancelError(
                task_id_commitment=task_id_commitment, reason=e.reason
            )
        except Exception as e:
            _logger.error(
                f"cannot cancel task {task_id_commitment.hex()} due to {str(e)}"
            )
            raise TaskCancelError(task_id_commitment=task_id_commitment, reason=str(e))

    async def deposit(self, address: str, amount: int, unit: str = "ether"):
        """
        deposit tokens to the address

        address: Address which deposit tokens to
        amount: Tokens need to deposit, 0 means not to deposit eth
        unit: The unit for eth and cnx tokens, default to "ether"
        """
        assert self._initialized, "Crynux sdk hasn't been initialized"
        assert not self._closed, "Crynux sdk has been closed"

        eth_wei = Web3.to_wei(amount, unit)
        await self.token.transfer(address=address, amount=eth_wei)

    async def generate_images(
        self,
        dst_dir: Union[str, pathlib.Path],
        task_fee: int,
        prompt: str,
        min_vram: Optional[int] = None,
        base_model: str = "crynux-ai/stable-diffusion-v1-5",
        variant: Optional[str] = "fp16",
        negative_prompt: str = "",
        required_gpu: str = "",
        required_gpu_vram: int = 0,
        task_version: str = "2.5.0",
        task_optional_args: Optional[sd_args.TaskOptionalArgs] = None,
        task_fee_unit: str = "ether",
        max_retries: int = 5,
        max_timeout_retries: int = 3,
        timeout: Optional[float] = None,
        wait_interval: int = 1,
        auto_cancel: bool = True,
    ) -> Tuple[bytes, List[bytes], List[pathlib.Path]]:
        """
        generate images by crynux network

        dst_dir: Where to store the generated images, should be a string or a pathlib.Path.
                 The dst_dir should be existed.
                 Generated images will be save in path dst_dir/0.png, dst_dir/1.png and so on.
        task_fee: The cnx tokens you paid for image generation, should be a int.
                  You account must have enough cnx tokens before you call this method,
                  or it will failed.
        prompt: The prompt for image generation.
        vram_limit: The GPU VRAM limit for image generation. Crynux network will select nodes
                    with vram larger than vram_limit to generate image for you.
                    If vram_limit is None, then the sdk will predict it by the base model.
        base_model: The base model used for image generation, default to crynux-ai/stable-diffusion-v1-5.
        variant: The variant of base model, default is fp16
        negative_prompt: The negative prompt for image generation.
        task_optional_args: Optional arguments for image generation. See crynux_sdk.models.sd_args.TaskOptionalArgs for details.
        task_fee_unit: The unit for task fee, default to "ether".
        max_retries: Max retry counts when face network issues, default to 5 times.
        max_timeout_retries: Max retry counts when cannot result images after timeout, default to 3 times.
        timeout: The timeout for image generation in seconds. Default to None, means no timeout.
        wait_interval: The interval in seconds for checking crynux contracts events. Default to 1 second.
        auto_cancel: Whether to cancel the timeout image generation task automatically. Default to True.

        returns: a tuple of task id, task id commitments, and the result image paths
        """
        assert self._initialized, "Crynux sdk hasn't been initialized"
        assert not self._closed, "Crynux sdk has been closed"

        task_fee = Web3.to_wei(task_fee, task_fee_unit)

        def _log_before_retry(retry_state: RetryCallState):
            if retry_state.outcome is not None and retry_state.outcome.failed:
                exc: Exception = retry_state.outcome.exception()
                if isinstance(exc, TaskAbortedError):
                    msg = f"image generation failed due to {exc.reason}, "
                else:
                    msg = f"image generation doesn't complete in {timeout} seconds, "

                retry_times = max_timeout_retries - retry_state.attempt_number
                msg += (
                    f"retry the image generation, remaining retring times {retry_times}"
                )
                _logger.error(msg)

        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_timeout_retries),
            retry=retry_if_exception_type((TaskAbortedError, TimeoutError))
            | retry_if_exception_cause_type(TimeoutError),
            before_sleep=_log_before_retry,
            reraise=True,
        )
        async def _run_task():
            task_id = b""
            vrf_proof = b""
            result_task_id_commitment = b""
            task_id_commitments = []
            task_size = 0
            try:
                with fail_after(timeout):
                    async for (
                        _task_id,
                        _task_id_commitment,
                        _vrf_proof,
                        _task_size,
                    ) in self.task.create_sd_task(
                        task_fee=task_fee,
                        prompt=prompt,
                        min_vram=min_vram,
                        base_model=base_model,
                        variant=variant,
                        negative_prompt=negative_prompt,
                        required_gpu=required_gpu,
                        required_gpu_vram=required_gpu_vram,
                        task_version=task_version,
                        max_retries=max_retries,
                        wait_interval=wait_interval,
                        task_optional_args=task_optional_args,
                    ):
                        if len(task_id) == 0:
                            task_id = _task_id
                        task_id_commitments.append(_task_id_commitment)
                        if len(vrf_proof) == 0:
                            vrf_proof = _vrf_proof
                        if task_size == 0:
                            task_size = _task_size

                    result_task_id_commitment = await self.task.execute_task(
                        task_id=task_id,
                        task_id_commitments=task_id_commitments,
                        vrf_proof=vrf_proof,
                        wait_interval=wait_interval,
                        max_retries=max_retries,
                    )

            except TimeoutError as timeout_exc:
                if auto_cancel and len(task_id_commitments):
                    _logger.error(
                        f"task {task_id.hex()} {[c.hex() for c in task_id_commitments]} is not successful after {timeout} seconds"
                    )
                    # try cancel the task
                    try:
                        with move_on_after(30, shield=True):
                            async with create_task_group() as tg:
                                for task_id_commitment in task_id_commitments:
                                    tg.start_soon(self._cancel_task, task_id_commitment, max_retries)
                    except Exception as e:
                        raise e from timeout_exc
                    raise timeout_exc
                else:
                    raise timeout_exc
            except Exception as e:
                _logger.error(f"task {[t.hex() for t in task_id_commitments]} error")
                _logger.exception(e)

            assert len(result_task_id_commitment) > 0
            try:
                with fail_after(timeout):
                    async with create_task_group() as tg:
                        # wait task success
                        tg.start_soon(
                            self.task.wait_task_success,
                            result_task_id_commitment,
                            wait_interval,
                            max_retries,
                        )
                        files = await self.task.get_task_result(
                            task_id_commitment=result_task_id_commitment,
                            task_type=TaskType.SD,
                            count=task_size,
                            dst_dir=dst_dir,
                        )
            except TimeoutError as timeout_exc:
                e = TaskGetResultTimeout(task_id_commitment=result_task_id_commitment)
                _logger.error(str(e))
                raise e from timeout_exc

            return task_id, task_id_commitments, files

        return await _run_task()

    async def finetune_sd_lora(
        self,
        result_checkpoint_path: Union[str, pathlib.Path],
        task_fee: int,
        required_gpu: str,
        required_gpu_vram: int,
        model_name: str,
        dataset_name: str,
        task_version: str = "2.5.0",
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
        input_checkpoint_path: Union[str, pathlib.Path, None] = None,
        task_fee_unit: str = "ether",
        max_retries: int = 5,
        max_timeout_retries: int = 3,
        timeout: Optional[float] = None,
        wait_interval: int = 1,
        auto_cancel: bool = True,
    ) -> Tuple[List[bytes], List[List[bytes]]]:
        """
        Finetune a lora model for stable diffusion by crynux network.
        Due to the training time of whole task may be very long, this task will split the whole task
        into several sub-task to perform.

        result_checkpoint_path: Should be a string or a pathlib.Path. The directory where the result checkpoint files are stored.
                                The result lora weight file is `pytorch_lora_weights.safetensors`.
        task_fee: The cnx tokens you paid for each finetune task, should be an int.
                  You account must have enough cnx tokens before you call this method,
                  or it will failed.
        gpu_name: The specified GPU name to run this finetune task, should be a string.
        gpu_vram: The specified GPU VRAM size to run this finetune task, should be an int, in unit GB.
        model_name: The pretrained stable diffusion model to finetune, should be a model identifier from huggingface.co/models.
        dataset_name: The name of the Dataset (from the HuggingFace hub) to train on, should be a string.
        model_variant: Variant of the model files of the pretrained model identifier from huggingface.co/models,
                       default is None, means no variant.
        model_revision: Revision of pretrained model identifier from huggingface.co/models, default is main.
        dataset_config_name: The config of the Dataset, default is None means there's only one config.
        dataset_image_column: The column of the dataset containing an image, should be a string, default is 'image'.
        dataset_caption_column: The column of the dataset containing a caption or a list of captions, should be a string,
                                default is 'text'.
        validation_prompt: A prompt that is used for validation inference during training, should be a string or None.
                           Default is None, means the the prompt is sampled from dataset.
        validation_num_images: Number of images that should be generated during validation, should be an int, in range [1, 10].
        center_crop: Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped.
                     Default is false.
        random_flip: Whether to randomly flip images horizontally. Default is false.
        rank: Lora attention dimension, should be an int, default is 8.
        init_lora_weights: How to initialize the weights of the LoRA layers.Passing True (default) results in the default
                           initialization from the reference implementation from Microsoft. Passing 'gaussian' results
                           in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization
                           to False leads to completely random initialization and is discouraged.
                           Pass 'loftq' to use LoftQ initialization.
        target_modules: List of module names or regex expression of the module names to replace with Lora.
        learning_rate: Initial learning rate to use. Default is 1e-4.
        batch_size: Batch size for the training dataloader. Default is 16.
        gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass. Default is 1.
        prediction_type: The prediction_type that shall be used for training.
                         Choose between 'epsilon' or 'v_prediction' or leave `None`.
                         If left to `None` the default prediction type of the scheduler:
                         `noise_scheduler.config.prediction_type` is chosen. Default is None.
        max_grad_norm: Max gradient norm. Default is 1.0.
        num_train_epochs: Number of training epochs to perform in one task. Default is 1.
        num_train_steps: Number of training steps to perform in one task. Should be an int or None.
                         Default is None. If not None, overrides 'num_train_epochs'.
        max_train_epochs: Total number of training epochs to perform. Default is 1.
        max_train_steps: Total number of training steps to perform. Should be an int or None.
                         Default is None. If not None, overrides 'max_train_epochs'.
        scale_lr: Whether to scale the learning rate by the number of gradient accumulation steps, and batch size. Default is true.
        resolution: The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.
                    Default is 512.
        noise_offset: The scale of noise offset. Default is 0.
        snr_gamma: SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. Default is None,
                   means to disable rebalancing the loss.
        lr_scheduler: The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",
                      "constant", "constant_with_warmup"]. Default is "constant".
        lr_warmup_steps: Number of steps for the warmup in the lr scheduler. Default is 500.
        adam_beta1: The beta1 parameter for the Adam optimizer. Default is 0.9.
        adam_beta2: The beta2 parameter for the Adam optimizer. Default is 0.999.
        adam_weight_decay: Weight decay to use. Default is 1e-2.
        adam_epsilon: Epsilon value for the Adam optimizer. Default is 1e-8.
        dataloader_num_workers: Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
                                Default is 0.
        mixed_precision: Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).
                         Default is 'no', means disable mixed precision.
        seed: A seed for reproducible training. Default is 0.
        input_checkpoint_path: Whether training should be resumed from a previous checkpoint. Should be a path of the previous checkpoint.
                               Default is None, means no previous checkpoint.
        task_fee_unit: The unit for task fee, default to "ether".
        max_retries: Max retry counts when face network issues, default to 5 times.
        max_timeout_retries: Max retry counts when cannot result images after timeout, default to 3 times.
        timeout: The timeout for image generation in seconds. Default to None, means no timeout.
        wait_interval: The interval in seconds for checking crynux contracts events. Default to 1 second.
        auto_cancel: Whether to cancel the timeout image generation task automatically. Default to True.
        """

        assert self._initialized, "Crynux sdk hasn't been initialized"
        assert not self._closed, "Crynux sdk has been closed"

        task_fee = Web3.to_wei(task_fee, task_fee_unit)

        def _log_before_retry(retry_state: RetryCallState):
            if retry_state.outcome is not None and retry_state.outcome.failed:
                exc: Exception = retry_state.outcome.exception()
                if isinstance(exc, TaskAbortedError):
                    msg = f"image generation failed due to {exc.reason}, "
                else:
                    msg = f"image generation doesn't complete in {timeout} seconds, "

                retry_times = max_timeout_retries - retry_state.attempt_number
                msg += (
                    f"retry the image generation, remaining retring times {retry_times}"
                )
                _logger.error(msg)

        @retry(
            wait=wait_fixed(2),
            stop=stop_after_attempt(max_timeout_retries),
            retry=retry_if_exception_type((TaskAbortedError, TimeoutError))
            | retry_if_exception_cause_type(TimeoutError),
            before_sleep=_log_before_retry,
            reraise=True,
        )
        async def _run_task(
            result_checkpoint: str, input_checkpoint: Optional[str] = None
        ):
            task_id = b""
            vrf_proof = b""
            result_task_id_commitment = b""
            task_id_commitments = []

            try:
                with fail_after(timeout):
                    async for (
                        _task_id,
                        _task_id_commitment,
                        _vrf_proof,
                    ) in self.task.create_sd_finetune_lora_task(
                        task_fee=task_fee,
                        required_gpu=required_gpu,
                        required_gpu_vram=required_gpu_vram,
                        task_version=task_version,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        model_variant=model_variant,
                        model_revision=model_revision,
                        dataset_config_name=dataset_config_name,
                        dataset_image_column=dataset_image_column,
                        dataset_caption_column=dataset_caption_column,
                        validation_prompt=validation_prompt,
                        validation_num_images=validation_num_images,
                        center_crop=center_crop,
                        random_flip=random_flip,
                        rank=rank,
                        init_lora_weights=init_lora_weights,
                        target_modules=target_modules,
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
                        lr_scheduler=lr_scheduler,
                        lr_warmup_steps=lr_warmup_steps,
                        adam_beta1=adam_beta1,
                        adam_beta2=adam_beta2,
                        adam_weight_decay=adam_weight_decay,
                        adam_epsilon=adam_epsilon,
                        dataloader_num_workers=dataloader_num_workers,
                        mixed_precision=mixed_precision,
                        seed=seed,
                        checkpoint=input_checkpoint,
                        max_retries=max_retries,
                    ):
                        if len(task_id) == 0:
                            task_id = _task_id
                        task_id_commitments.append(_task_id_commitment)
                        if len(vrf_proof) == 0:
                            vrf_proof = _vrf_proof

                    result_task_id_commitment = await self.task.execute_task(
                        task_id=task_id,
                        task_id_commitments=task_id_commitments,
                        vrf_proof=vrf_proof,
                        wait_interval=wait_interval,
                        max_retries=max_retries,
                    )

            except TimeoutError as timeout_exc:
                if auto_cancel and len(task_id_commitments):
                    _logger.error(
                        f"task {task_id.hex()} {[c.hex() for c in task_id_commitments]} is not successful after {timeout} seconds"
                    )
                    # try cancel the task
                    try:
                        with move_on_after(30, shield=True):
                            async with create_task_group() as tg:
                                for task_id_commitment in task_id_commitments:
                                    tg.start_soon(self._cancel_task, task_id_commitment, max_retries)
                    except Exception as e:
                        raise e from timeout_exc
                    raise timeout_exc
                else:
                    raise timeout_exc
            except Exception as e:
                _logger.error(f"task {[t.hex() for t in task_id_commitments]} error")
                _logger.exception(e)

            try:
                with fail_after(timeout):
                    async with create_task_group() as tg:
                        # wait task success
                        tg.start_soon(
                            self.task.wait_task_success,
                            result_task_id_commitment,
                            wait_interval,
                            max_retries,
                        )
                        await self.task.get_task_result_checkpoint(
                            task_id_commitment=result_task_id_commitment,
                            checkpoint_dir=result_checkpoint,
                        )
            except TimeoutError as timeout_exc:
                e = TaskGetResultTimeout(task_id_commitment=result_task_id_commitment)
                _logger.error(str(e))
                raise e from timeout_exc

            return task_id, task_id_commitments

        task_num = 0
        if input_checkpoint_path is not None:
            input_checkpoint = str(input_checkpoint_path)
        else:
            input_checkpoint = None
        task_ids = []
        task_id_commitments_list = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            while True:
                result_checkpoint = os.path.join(tmp_dir, f"checkpoint-{task_num}")
                task_id, task_id_commitments = await _run_task(
                    result_checkpoint, input_checkpoint
                )
                task_ids.append(task_id)
                task_id_commitments_list.append(task_id_commitments)

                finish_file = os.path.join(result_checkpoint, "FINISH")
                if os.path.exists(finish_file):
                    await to_thread.run_sync(
                        partial(
                            shutil.copytree,
                            result_checkpoint,
                            result_checkpoint_path,
                            dirs_exist_ok=True,
                        )
                    )
                    break

                input_checkpoint = result_checkpoint
        return task_ids, task_id_commitments_list
