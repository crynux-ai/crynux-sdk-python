from typing import TYPE_CHECKING, List, Optional

from eth_typing import ChecksumAddress
from web3 import AsyncWeb3, Web3

from crynux_sdk.models.contracts import ChainNodeInfo, ChainNodeStatus, GpuInfo

from .utils import ContractWrapper, TxWaiter
from .w3_pool import W3Pool

if TYPE_CHECKING:
    from crynux_sdk.config import TxOption


__all__ = ["NodeContract"]


_default_stake_amount = Web3.to_wei(400, "ether")


class NodeContract(ContractWrapper):
    def __init__(
        self, w3_pool: W3Pool, contract_address: Optional[ChecksumAddress] = None
    ):
        super().__init__(w3_pool, "Node", contract_address)

    async def join(
        self,
        gpu_name: str,
        gpu_vram: int,
        *,
        stake_amount: int = _default_stake_amount,
        option: "Optional[TxOption]" = None,
        w3: Optional[AsyncWeb3] = None,
    ) -> TxWaiter:
        return await self._transaction_call(
            "join",
            gpuName=gpu_name,
            gpuVram=gpu_vram,
            value=stake_amount,
            option=option,
            w3=w3,
        )

    async def quit(
        self, *, option: "Optional[TxOption]" = None, w3: Optional[AsyncWeb3] = None
    ) -> TxWaiter:
        return await self._transaction_call("quit", option=option, w3=w3)

    async def pause(
        self, *, option: "Optional[TxOption]" = None, w3: Optional[AsyncWeb3] = None
    ) -> TxWaiter:
        return await self._transaction_call("pause", option=option, w3=w3)

    async def resume(
        self, *, option: "Optional[TxOption]" = None, w3: Optional[AsyncWeb3] = None
    ) -> TxWaiter:
        return await self._transaction_call("resume", option=option, w3=w3)

    async def update_task_contract_address(
        self,
        address: str,
        *,
        option: "Optional[TxOption]" = None,
        w3: Optional[AsyncWeb3] = None,
    ) -> TxWaiter:
        return await self._transaction_call(
            "updateTaskContractAddress", option=option, taskContract=address, w3=w3
        )

    async def get_node_status(
        self, address: str, *, w3: Optional[AsyncWeb3] = None
    ) -> ChainNodeStatus:
        res = await self._function_call("getNodeStatus", nodeAddress=address, w3=w3)
        return ChainNodeStatus(res)

    async def get_node_info(
        self, address: str, *, w3: Optional[AsyncWeb3] = None
    ) -> ChainNodeInfo:
        res = await self._function_call("getNodeInfo", nodeAddress=address, w3=w3)
        info = ChainNodeInfo(
            status=ChainNodeStatus(res[0]),
            gpu_id=res[1],
            gpu=GpuInfo(name=res[2][0], vram=res[2][1]),
        )
        return info

    async def get_available_nodes(self, *, w3: Optional[AsyncWeb3] = None) -> List[str]:
        res = await self._function_call("getAvailableNodes", w3=w3)
        return res

    async def get_available_gpus(
        self, *, w3: Optional[AsyncWeb3] = None
    ) -> List[GpuInfo]:
        res = await self._function_call("getAvailableGPUs", w3=w3)
        return [GpuInfo(name=item[0], vram=item[1]) for item in res]

    async def get_staked_amount(self, *, w3: Optional[AsyncWeb3] = None) -> int:
        res = await self._function_call("getStakedAmount", w3=w3)
        return res
