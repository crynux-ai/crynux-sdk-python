from pydantic import BaseModel


class RelayTask(BaseModel):
    task_id: int
    creator: str
    task_hash: str
    data_hash: str
    task_args: str
