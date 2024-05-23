"""Module for custom errors."""


class GeneratorError(Exception):
    """Exception raise when encountering an error related to generators."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg
