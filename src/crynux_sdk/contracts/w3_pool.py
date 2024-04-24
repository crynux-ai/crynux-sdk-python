import logging
import ssl
import warnings
from collections import deque
from contextlib import asynccontextmanager
from enum import IntEnum
from typing import Optional

import certifi
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from anyio import Condition, Lock
from eth_account.signers.local import LocalAccount
from web3 import AsyncHTTPProvider, AsyncWeb3, WebsocketProviderV2
from web3.middleware.signing import \
    async_construct_sign_and_send_raw_middleware
from web3.providers.async_base import AsyncBaseProvider
from websockets import ConnectionClosed

_logger = logging.getLogger(__name__)


class ProviderType(IntEnum):
    HTTP = 0
    WS = 1
    Other = 2


class W3Pool(object):
    def __init__(
        self,
        provider: Optional[AsyncBaseProvider] = None,
        provider_path: Optional[str] = None,
        privkey: str = "",
        default_account_index: Optional[int] = None,
        pool_size: int = 1,
        timeout: int = 10,
    ) -> None:
        self._privkey = privkey
        self._pool_size = pool_size
        self._provider_path = provider_path
        self._timeout = timeout
        self._default_account_index = default_account_index
        self._provider = None

        if provider is None:
            if provider_path is None:
                raise ValueError("provider and provider_path cannot be all None.")
            if provider_path.startswith("http"):
                self.provider_type = ProviderType.HTTP
            elif provider_path.startswith("ws"):
                self.provider_type = ProviderType.WS
            else:
                raise ValueError(f"unsupported provider {provider_path}")
        else:
            self.provider_type = ProviderType.Other
            self._provider = provider
            self._pool_size = 1
            if pool_size != 1:
                warnings.warn("Pool size can only be 1 when provider type is other")

        self._idle_pool = deque(maxlen=self._pool_size)

        self._condition = Condition()
        self._nonce_lock = Lock()

        self._current_size = 0

    async def _new_w3(self) -> AsyncWeb3:
        if self.provider_type == ProviderType.HTTP:
            assert self._provider_path is not None
            provider = AsyncHTTPProvider(self._provider_path)
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            session = ClientSession(
                timeout=ClientTimeout(self._timeout),
                connector=TCPConnector(ssl=ssl_context),
            )
            await provider.cache_async_session(session)
            w3 = AsyncWeb3(provider)
        elif self.provider_type == ProviderType.WS:
            provider = WebsocketProviderV2(self._provider_path)
            w3 = AsyncWeb3.persistent_websocket(provider)
            await w3.provider.connect()
        else:
            assert self._provider is not None
            w3 = AsyncWeb3(self._provider)

        if self._privkey != "":
            account: LocalAccount = w3.eth.account.from_key(self._privkey)
            middleware = await async_construct_sign_and_send_raw_middleware(account)
            w3.middleware_onion.add(middleware)
            w3.eth.default_account = account.address
        elif self._default_account_index is not None:
            w3.eth.default_account = (await w3.eth.accounts)[
                self._default_account_index
            ]

        _logger.debug("new web3 eth instance")

        return w3

    async def _get(self) -> AsyncWeb3:
        async with self._condition:
            if len(self._idle_pool) == 0 and self._current_size < self._pool_size:
                w3 = await self._new_w3()
                self._current_size += 1
                _logger.debug("get web3 eth instance")
                return w3
            while len(self._idle_pool) == 0:
                await self._condition.wait()
            w3 = self._idle_pool.popleft()
            _logger.debug("get web3 eth instance")
            return w3

    async def _put(self, w3: AsyncWeb3):
        async with self._condition:
            self._idle_pool.append(w3)
            self._condition.notify(1)
            _logger.debug("put web3 eth instance")

    @asynccontextmanager
    async def get(self):
        w3: Optional[AsyncWeb3] = None
        try:
            w3 = await self._get()
            yield w3
        except ConnectionClosed:
            async with self._condition:
                self._current_size -= 1
                w3 = None
        finally:
            if w3 is not None:
                await self._put(w3)

    @asynccontextmanager
    async def with_nonce_lock(self):
        async with self._nonce_lock:
            yield

    async def close(self):
        pass
