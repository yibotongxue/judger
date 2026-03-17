from typing import Any, Literal, TypedDict

from pydantic import BaseModel


class MessageDataItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class InferParameters(BaseModel):
    top_p: float = 0.95
    temperature: float = 1.0


class InferMeta(TypedDict):
    model_id: str
    infer_parameters: InferParameters


class InferResult(BaseModel):
    response: str
    next_messages: list[MessageDataItem]
    meta: InferMeta


class InputItem(BaseModel):
    id: str
    question: str
    response: str
    ref_answer: str | None = None
    meta: dict[str, Any]


class _JudgeMeta(TypedDict):
    question: str
    response: str
    ref_answer: str


class JudgeResult(BaseModel):
    id: str
    score: float
    reason: str
    meta: _JudgeMeta
