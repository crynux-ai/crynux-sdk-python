from pydantic import BaseModel


class RelayTask(BaseModel):
    task_id_commitment: bytes
    creator: str
    task_args: str

