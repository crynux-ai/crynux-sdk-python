from enum import IntEnum
from typing import List

from pydantic import BaseModel


class TaskType(IntEnum):
    SD = 0
    LLM = 1
    SD_FT_LORA = 2


class TaskError(IntEnum):
    NONE = 0
    ParametersValidationFailed = 1


class TaskAbortReason(IntEnum):
    NONE = 0
    Timeout = 1
    ModelDownloadFailed = 2
    IncorrectResult = 3
    TaskFeeTooLow = 4


class TaskStatus(IntEnum):
    Queued = 0
    Started = 1
    ParametersUploaded = 2
    ErrorReported = 3
    ScoreReady = 4
    Validated = 5
    GroupValidated = 6
    EndInvalidated = 7
    EndSuccess = 8
    EndAborted = 9
    EndGroupRefund = 10
    EndGroupSuccess = 11


class ChainTask(BaseModel):
    task_type: TaskType
    creator: str
    task_id_commitment: bytes
    sampling_seed: bytes
    nonce: bytes
    sequence: int
    status: TaskStatus
    selected_node: str
    timeout: int
    score: bytes
    task_fee: int
    task_size: int
    task_model_ids: List[str]
    min_vram: int
    required_gpu: str
    required_gpu_vram: int
    task_version: List[int]
    abort_reason: TaskAbortReason
    error: TaskError
    payment_addresses: List[str]
    payments: List[int]
    create_timestamp: int
    start_timestamp: int
    score_ready_timestamp: int
