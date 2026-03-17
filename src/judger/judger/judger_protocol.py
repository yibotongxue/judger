from __future__ import annotations

from typing import Protocol, runtime_checkable

from judger.types import InputItem, JudgeResult


@runtime_checkable
class JudgeProtocol(Protocol):
    async def judge(self, inputs: list[InputItem]) -> list[JudgeResult]: ...
