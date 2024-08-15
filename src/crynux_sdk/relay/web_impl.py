import json
import os
import shutil
import ssl
import tempfile
from contextlib import ExitStack
from typing import BinaryIO, List, Optional

import certifi
from aiohttp import (ClientResponse, ClientResponseError, ClientSession,
                     ClientTimeout, TCPConnector, FormData)
from anyio import open_file, to_thread, wrap_file

from crynux_sdk.models.relay import RelayTask

from .abc import Relay
from .exceptions import RelayError
from .sign import Signer


async def _process_resp(resp: ClientResponse, method: str):
    try:
        resp.raise_for_status()
        return resp
    except ClientResponseError as e:
        message = str(e)
        if resp.status == 400:
            try:
                content = await resp.json()
                if "data" in content:
                    data = content["data"]
                    message = json.dumps(data)
                elif "message" in content:
                    message = content["message"]
                else:
                    message = await resp.text()
            except Exception:
                pass
        raise RelayError(resp.status, method, message)


class WebRelay(Relay):
    def __init__(self, base_url: str, privkey: str, timeout: float = 30) -> None:
        super().__init__()
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.client = ClientSession(
            base_url=base_url,
            timeout=ClientTimeout(timeout),
            connector=TCPConnector(ssl=ssl_context),
        )
        self.signer = Signer(privkey=privkey)

    async def create_task(self, task_id: int, task_args: str) -> RelayTask:
        input = {"task_id": task_id, "task_args": task_args}
        timestamp, signature = self.signer.sign(input)
        input.update({"timestamp": timestamp, "signature": signature})

        async with await self.client.post("/v1/inference_tasks", json=input) as resp:
            resp = await _process_resp(resp, "createTask")
            content = await resp.json()
            data = content["data"]
            return RelayTask.model_validate(data)

    async def upload_checkpoint(self, task_id: int, checkpoint_dir: str):
        input = {"task_id": task_id}
        timestamp, signature = self.signer.sign(input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # zip checkpoint dir to a archive file
            checkpoint_file = os.path.join(tmp_dir, "checkpoint.zip")
            await to_thread.run_sync(
                shutil.make_archive, checkpoint_file[:-4], "zip", checkpoint_dir
            )

            with ExitStack() as stack:
                filename = os.path.basename(checkpoint_file)
                file_obj = stack.enter_context(open(checkpoint_file, "rb"))

                data = FormData()
                data.add_field("timestamp", str(timestamp))
                data.add_field("signature", signature)
                data.add_field("checkpoint", file_obj, filename=filename)
                async with await self.client.post(
                    f"/v1/inference_tasks/{task_id}/checkpoint",
                    data=data,
                    timeout=None,
                ) as resp:
                    resp = await _process_resp(resp, "uploadCheckpoint")
                    content = await resp.json()
                    message = content["message"]
                    if message != "success":
                        raise RelayError(resp.status, "uploadCheckpoint", message)

    async def get_checkpoint(self, task_id: int, result_checkpoint_dir: str):
        input = {"task_id": task_id}
        timestamp, signature = self.signer.sign(input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_file = os.path.join(tmp_dir, "checkpoint.zip")
            async with await open_file(checkpoint_file, mode="wb") as f:
                async with self.client.get(
                    f"/v1/inference_tasks/{task_id}/checkpoint",
                    params={"timestamp": timestamp, "signature": signature},
                ) as resp:
                    resp = await _process_resp(resp, "getCheckpoint")
                    async for chunk in resp.content.iter_chunked(4096):
                        await f.write(chunk)

            await to_thread.run_sync(
                shutil.unpack_archive, checkpoint_file, result_checkpoint_dir
            )

    async def get_task(self, task_id: int) -> RelayTask:
        input = {"task_id": task_id}
        timestamp, signature = self.signer.sign(input)

        async with await self.client.get(
            f"/v1/inference_tasks/{task_id}",
            params={"timestamp": timestamp, "signature": signature},
        ) as resp:
            resp = await _process_resp(resp, "getTask")
            content = await resp.json()
            data = content["data"]
            return RelayTask.model_validate(data)

    async def upload_task_result(self, task_id: int, file_paths: List[str], checkpoint_dir: Optional[str] = None):
        input = {"task_id": task_id}
        timestamp, signature = self.signer.sign(input)

        with ExitStack() as stack:
            data = FormData()
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                file_obj = stack.enter_context(open(file_path, "rb"))
                data.add_field("images", file_obj, filename=filename)

            if checkpoint_dir is not None:
                tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                checkpoint_file = os.path.join(tmp_dir, "checkpoint.zip")
                await to_thread.run_sync(shutil.make_archive, checkpoint_file[:-4], "zip", checkpoint_dir)
                filename = os.path.basename(checkpoint_file)
                file_obj = stack.enter_context(open(checkpoint_file, "rb"))
                data.add_field("checkpoint", file_obj, filename=filename)

            data.add_field("timestamp", str(timestamp))
            data.add_field("signature", signature)
            # disable timeout because there may be many images or image size may be very large
            async with await self.client.post(
                f"/v1/inference_tasks/{task_id}/results",
                data=data,
                timeout=None,
            ) as resp:
                resp = await _process_resp(resp, "uploadTaskResult")
                content = await resp.json()
                message = content["message"]
                if message != "success":
                    raise RelayError(resp.status, "uploadTaskResult", message)

    async def get_result(self, task_id: int, index: int, dst: BinaryIO):
        input = {"task_id": task_id, "image_num": str(index)}
        timestamp, signature = self.signer.sign(input)

        async_dst = wrap_file(dst)

        async with self.client.get(
            f"/v1/inference_tasks/{task_id}/results/{index}",
            params={"timestamp": timestamp, "signature": signature},
        ) as resp:
            resp = await _process_resp(resp, "getResult")
            async for chunk in resp.content.iter_chunked(4096):
                await async_dst.write(chunk)

    async def get_result_checkpoint(self, task_id: int, result_checkpoint_dir: str):
        input = {"task_id": task_id}
        timestamp, signature = self.signer.sign(input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_file = os.path.join(tmp_dir, "checkpoint.zip")
            async with await open_file(checkpoint_file, mode="wb") as f:
                async with self.client.get(
                    f"/v1/inference_tasks/{task_id}/results/checkpoint",
                    params={"timestamp": timestamp, "signature": signature},
                ) as resp:
                    resp = await _process_resp(resp, "getResultCheckpoint")
                    async for chunk in resp.content.iter_chunked(4096):
                        await f.write(chunk)

            await to_thread.run_sync(
                shutil.unpack_archive, checkpoint_file, result_checkpoint_dir
            )

    async def close(self):
        await self.client.close()
