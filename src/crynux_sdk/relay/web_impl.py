import json
import os
import ssl
from contextlib import ExitStack
from typing import BinaryIO, List

import certifi
from aiohttp import (ClientResponseError, ClientResponse, ClientSession,
                     ClientTimeout, TCPConnector)
from anyio import wrap_file

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

    async def upload_task_result(self, task_id: int, file_paths: List[str]):
        input = {"task_id": task_id}
        timestamp, signature = self.signer.sign(input)

        with ExitStack() as stack:
            files = []
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                file_obj = stack.enter_context(open(file_path, "rb"))
                files.append(("images", (filename, file_obj)))

            # disable timeout because there may be many images or image size may be very large
            async with await self.client.post(
                f"/v1/inference_tasks/{task_id}/results",
                data={"timestamp": timestamp, "signature": signature},
                files=files,
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
            resp = await _process_resp(resp, "getTask")
            async for chunk in resp.content.iter_chunked(4096):
                await async_dst.write(chunk)

    async def close(self):
        await self.client.close()
