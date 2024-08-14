from __future__ import annotations

import logging
import math
import os
import pathlib
import shutil
import tempfile
from typing import List, Literal, Optional, Tuple, Union

from anyio import fail_after, to_thread
from tenacity import (AsyncRetrying, RetryCallState,
                      retry_if_exception_cause_type, retry_if_exception_type,
                      retry_if_not_exception_type, stop_after_attempt,
                      wait_fixed)
from web3 import Web3

from crynux_sdk.config import (get_default_contract_config,
                               get_default_provider_path,
                               get_default_relay_url, get_default_tx_option)
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
    ) -> Tuple[int, int, List[pathlib.Path]]:
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
        base_model: The base model used for image generation, default to runwayml/stable-diffusion-v1-5.
        negative_prompt: The negative prompt for image generation.
        task_optional_args: Optional arguments for image generation. See crynux_sdk.models.sd_args.TaskOptionalArgs for details.
        task_fee_unit: The unit for task fee, default to "ether".
        max_retries: Max retry counts when face network issues, default to 5 times.
        max_timeout_retries: Max retry counts when cannot result images after timeout, default to 3 times.
        timeout: The timeout for image generation in seconds. Default to None, means no timeout.
        wait_interval: The interval in seconds for checking crynux contracts events. Default to 1 second.
        auto_cancel: Whether to cancel the timeout image generation task automatically. Default to True.

        returns: a tuple of task id, blocknum when the task starts, and the result image paths
        """
        assert self._initialized, "Crynux sdk hasn't been initialized"
        assert not self._closed, "Crynux sdk has been closed"

        task_fee = Web3.to_wei(task_fee, task_fee_unit)

        async def _run_task():
            task_id = 0
            start_blocknum = 0
            task_created = False
            task_success = False
            try:
                with fail_after(timeout):
                    blocknum, tx_hash, task_id, cap = await self.task.create_sd_task(
                        task_fee=task_fee,
                        prompt=prompt,
                        vram_limit=vram_limit,
                        base_model=base_model,
                        negative_prompt=negative_prompt,
                        task_optional_args=task_optional_args,
                        max_retries=max_retries,
                    )
                    task_created = True
                    _logger.debug(f"task {task_id} is created at tx {tx_hash.hex()}")

                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        retry=retry_if_not_exception_type(TaskAbortedError),
                        reraise=True,
                    ):
                        with attemp:
                            blocknum, tx_hash, _ = await self.task.wait_task_started(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )
                    start_blocknum = blocknum
                    _logger.debug(f"task {task_id} starts at tx {tx_hash.hex()}")
                    _logger.info(f"task {task_id} starts")

                    _logger.info(f"waiting task {task_id} to complete")
                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        retry=retry_if_not_exception_type(TaskAbortedError),
                        reraise=True,
                    ):
                        with attemp:
                            blocknum, tx_hash, _ = await self.task.wait_task_finish(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )
                    _logger.debug(
                        f"task {task_id} finish successfully at tx {tx_hash.hex()}"
                    )
                    task_success = True

                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        reraise=True,
                    ):
                        with attemp:
                            (
                                blocknum,
                                tx_hash,
                                _,
                            ) = await self.task.wait_task_result_uploaded(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )
                    _logger.debug(
                        f"result of task {task_id} is uploaded at tx {tx_hash.hex()}"
                    )

                    files: List[pathlib.Path] = []
                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        reraise=True,
                    ):
                        with attemp:
                            files = await self.task.get_task_result(
                                task_id=task_id,
                                task_type=TaskType.SD,
                                count=cap,
                                dst_dir=dst_dir,
                            )
                    return task_id, start_blocknum, files

            except TimeoutError as timeout_exc:
                if auto_cancel and task_id > 0 and task_created:
                    if not task_success:
                        _logger.error(
                            f"task {task_id} is not successful after {timeout} seconds"
                        )
                        _logger.info(f"try to cancel task {task_id}")
                        # try cancel the task
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
                        raise timeout_exc
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
                msg += (
                    f"retry the image generation, remaining retring times {retry_times}"
                )
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
                res = await _run_task()
        return res

    async def finetune_sd_lora(
        self,
        result_checkpoint_path: Union[str, pathlib.Path],
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
    ):
        assert self._initialized, "Crynux sdk hasn't been initialized"
        assert not self._closed, "Crynux sdk has been closed"

        task_fee = Web3.to_wei(task_fee, task_fee_unit)

        async def _run_task(
            result_checkpoint: str, input_checkpoint: Optional[str] = None
        ):
            task_id = 0
            start_blocknum = 0
            task_created = False
            task_success = False
            try:
                with fail_after(timeout):
                    blocknum, tx_hash, task_id, cap = (
                        await self.task.create_sd_finetune_lora_task(
                            task_fee=task_fee,
                            gpu_name=gpu_name,
                            gpu_vram=gpu_vram,
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
                        )
                    )
                    task_created = True
                    _logger.debug(f"task {task_id} is created at tx {tx_hash.hex()}")

                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        retry=retry_if_not_exception_type(TaskAbortedError),
                        reraise=True,
                    ):
                        with attemp:
                            blocknum, tx_hash, _ = await self.task.wait_task_started(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )
                    start_blocknum = blocknum
                    _logger.debug(f"task {task_id} starts at tx {tx_hash.hex()}")
                    _logger.info(f"task {task_id} starts")

                    _logger.info(f"waiting task {task_id} to complete")
                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        retry=retry_if_not_exception_type(TaskAbortedError),
                        reraise=True,
                    ):
                        with attemp:
                            blocknum, tx_hash, _ = await self.task.wait_task_finish(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )
                    _logger.debug(
                        f"task {task_id} finish successfully at tx {tx_hash.hex()}"
                    )
                    task_success = True

                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        reraise=True,
                    ):
                        with attemp:
                            (
                                blocknum,
                                tx_hash,
                                _,
                            ) = await self.task.wait_task_result_uploaded(
                                task_id=task_id,
                                from_block=blocknum,
                                interval=wait_interval,
                            )
                    _logger.debug(
                        f"result of task {task_id} is uploaded at tx {tx_hash.hex()}"
                    )

                    async for attemp in AsyncRetrying(
                        wait=wait_fixed(2),
                        stop=stop_after_attempt(max_retries),
                        reraise=True,
                    ):
                        with attemp:
                            await self.task.get_task_result_checkpoint(
                                task_id=task_id, checkpoint_dir=result_checkpoint
                            )
                    return task_id, start_blocknum

            except TimeoutError as timeout_exc:
                if auto_cancel and task_id > 0 and task_created:
                    if not task_success:
                        _logger.error(
                            f"task {task_id} is not successful after {timeout} seconds"
                        )
                        _logger.info(f"try to cancel task {task_id}")
                        # try cancel the task
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
                        raise timeout_exc
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
                msg += (
                    f"retry the image generation, remaining retring times {retry_times}"
                )
                _logger.error(msg)

        task_num = 0
        if input_checkpoint_path is not None:
            input_checkpoint = str(input_checkpoint_path)
        else:
            input_checkpoint = None
        task_ids = []
        start_blocknums = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            while True:
                result_checkpoint = os.path.join(tmp_dir, f"checkpoint-{task_num}")
                async for attemp in AsyncRetrying(
                    wait=wait_fixed(2),
                    stop=stop_after_attempt(max_timeout_retries),
                    retry=retry_if_exception_type((TaskAbortedError, TimeoutError))
                    | retry_if_exception_cause_type(TimeoutError),
                    before_sleep=_log_before_retry,
                    reraise=True,
                ):
                    with attemp:
                        task_id, start_blocknum = await _run_task(
                            result_checkpoint, input_checkpoint
                        )
                        task_ids.append(task_id)
                        start_blocknums.append(start_blocknum)

                finish_file = os.path.join(result_checkpoint, "FINISH")
                if os.path.exists(finish_file):
                    await to_thread.run_sync(
                        shutil.copytree, result_checkpoint, result_checkpoint_path
                    )
                    break

                input_checkpoint = result_checkpoint
        return task_ids, start_blocknums
