class TaskError(Exception):
    def __init__(self, task_id: int) -> None:
        super().__init__(task_id)
        self.task_id = task_id


class TaskAbortedError(TaskError):
    def __init__(self, task_id: int, reason: str) -> None:
        super().__init__(task_id)
        self.reason = reason

    def __str__(self) -> str:
        return f"Task {self.task_id} aborted for {self.reason}"


class TaskCancelError(TaskError):
    def __init__(self, task_id: int, reason: str) -> None:
        super().__init__(task_id)
        self.reason = reason

    def __str__(self) -> str:
        return f"Cannot cancel task {self.task_id}, reason {self.reason}"


class TaskGetResultTimeout(TaskError):
    def __str__(self) -> str:
        return (
            f"Geting result of task {self.task_id} timeout! "
            "The task is finished successfully, "
            "you can try to get the task result again."
        )
