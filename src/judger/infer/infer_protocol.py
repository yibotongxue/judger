from typing import Protocol, runtime_checkable

from judger.types import InferParameters, InferResult, MessageDataItem


@runtime_checkable
class InferProtocol(Protocol):
    async def infer(
        self, messages: list[MessageDataItem], parameters: InferParameters
    ) -> InferResult: ...
