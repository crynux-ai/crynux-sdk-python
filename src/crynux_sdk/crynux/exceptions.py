class TaskError(Exception):
    def __init__(self, task_id_commitment: bytes) -> None:
        super().__init__(task_id_commitment)
        self.task_id_commitment = task_id_commitment


class TaskAbortedError(TaskError):
    def __init__(self, task_id_commitment: bytes, reason: str) -> None:
        super().__init__(task_id_commitment)
        self.reason = reason

    def __str__(self) -> str:
        return f"Task {self.task_id_commitment.hex()} aborted for {self.reason}"


class TaskCancelError(TaskError):
    def __init__(self, task_id_commitment: bytes, reason: str) -> None:
        super().__init__(task_id_commitment)
        self.reason = reason

    def __str__(self) -> str:
        return f"Cannot cancel task {self.task_id_commitment.hex()}, reason {self.reason}"


class TaskGetResultTimeout(TaskError):
    def __str__(self) -> str:
        return (
            f"Geting result of task {self.task_id_commitment.hex()} timeout! "
            "The task is finished successfully, "
            "you can try to get the task result again."
        )
