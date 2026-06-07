from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


TaskName = Literal["code", "math", "general"]
RoleName = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: RoleName
    content: str = Field(min_length=1)


class TaskChatRequest(BaseModel):
    task: TaskName
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def must_include_user_message(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not any(message.role == "user" for message in messages):
            raise ValueError("messages must include at least one user message")
        return messages

    @field_validator("stream")
    @classmethod
    def streaming_not_supported_in_v1(cls, stream: bool) -> bool:
        if stream:
            raise ValueError("streaming responses are not supported by the V1 gateway")
        return stream


class TaskChatResponse(BaseModel):
    ok: bool
    task: str
    adapter: str
    model: str
    latency_ms: float
    response: dict
