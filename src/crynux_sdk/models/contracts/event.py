from typing import Literal

from eth_typing import ChecksumAddress
from pydantic import BaseModel, Field
from web3 import Web3
from web3.types import EventData

from .task import TaskType

TaskKind = Literal[
    "TaskPending",
    "TaskStarted",
    "TaskResultCommitmentsReady",
    "TaskSuccess",
    "TaskAborted",
    "TaskResultUploaded",
    "TaskNodeSuccess",
    "TaskNodeSlashed",
    "TaskNodeCancelled",
    "load_event_from_contracts"
]


class TaskEvent(BaseModel):
    kind: TaskKind = Field(init_var=False)
    task_id: int


class TaskPending(TaskEvent):
    kind: TaskKind = Field(default="TaskPending", init_var=False, frozen=True)
    task_type: TaskType
    creator: ChecksumAddress
    task_hash: str
    data_hash: str


class TaskStarted(TaskEvent):
    kind: TaskKind = Field(default="TaskStarted", init_var=False, frozen=True)
    task_type: TaskType
    creator: ChecksumAddress
    selected_node: ChecksumAddress
    task_hash: str
    data_hash: str
    round: int


class TaskResultCommitmentsReady(TaskEvent):
    kind: TaskKind = Field(
        default="TaskResultCommitmentsReady", init_var=False, frozen=True
    )


class TaskSuccess(TaskEvent):
    kind: TaskKind = Field(default="TaskSuccess", init_var=False, frozen=True)
    result: str
    result_node: ChecksumAddress


class TaskAborted(TaskEvent):
    kind: TaskKind = Field(default="TaskAborted", init_var=False, frozen=True)
    reason: str


class TaskResultUploaded(TaskEvent):
    kind: TaskKind = Field(default="TaskResultUploaded", init_var=False, frozen=True)


class TaskNodeSuccess(TaskEvent):
    kind: TaskKind = Field(default="TaskNodeSuccess", init_var=False, frozen=True)
    node_address: ChecksumAddress
    fee: int


class TaskNodeSlashed(TaskEvent):
    kind: TaskKind = Field(default="TaskNodeSlashed", init_var=False, frozen=True)
    node_address: ChecksumAddress


class TaskNodeCancelled(TaskEvent):
    kind: TaskKind = Field(default="TaskNodeCancelled", init_var=False, frozen=True)
    node_address: ChecksumAddress


def load_event_from_contracts(event_data: EventData) -> TaskEvent:
    name = event_data["event"]
    if name == "TaskPending":
        return TaskPending(
            task_id=event_data["args"]["taskId"],
            task_type=event_data["args"]["taskType"],
            creator=Web3.to_checksum_address(event_data["args"]["creator"]),
            task_hash=Web3.to_hex(event_data["args"]["taskHash"]),
            data_hash=Web3.to_hex(event_data["args"]["dataHash"]),
        )
    elif name == "TaskStarted":
        return TaskStarted(
            task_id=event_data["args"]["taskId"],
            task_type=event_data["args"]["taskType"],
            creator=Web3.to_checksum_address(event_data["args"]["creator"]),
            selected_node=Web3.to_checksum_address(event_data["args"]["selectedNode"]),
            task_hash=Web3.to_hex(event_data["args"]["taskHash"]),
            data_hash=Web3.to_hex(event_data["args"]["dataHash"]),
            round=event_data["args"]["round"],
        )
    elif name == "TaskResultCommitmentsReady":
        return TaskResultCommitmentsReady(task_id=event_data["args"]["taskId"])
    elif name == "TaskSuccess":
        return TaskSuccess(
            task_id=event_data["args"]["taskId"],
            result=Web3.to_hex(event_data["args"]["result"]),
            result_node=Web3.to_checksum_address(event_data["args"]["resultNode"]),
        )
    elif name == "TaskAborted":
        return TaskAborted(
            task_id=event_data["args"]["taskId"], reason=event_data["args"]["reason"]
        )
    elif name == "TaskResultUploaded":
        return TaskResultUploaded(task_id=event_data["args"]["taskId"])
    elif name == "TaskNodeSuccess":
        return TaskNodeSuccess(
            task_id=event_data["args"]["taskId"],
            node_address=Web3.to_checksum_address(event_data["args"]["nodeAddress"]),
            fee=event_data["args"]["fee"],
        )
    elif name == "TaskNodeSlashed":
        return TaskNodeSlashed(
            task_id=event_data["args"]["taskId"],
            node_address=Web3.to_checksum_address(event_data["args"]["nodeAddress"]),
        )
    elif name == "TaskNodeCancelled":
        return TaskNodeCancelled(
            task_id=event_data["args"]["taskId"],
            node_address=Web3.to_checksum_address(event_data["args"]["nodeAddress"]),
        )
    else:
        raise ValueError(f"unknown event kind {name} from contracts")
