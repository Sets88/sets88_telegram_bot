from __future__ import annotations
from datetime import datetime
from typing import Any
from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.agents import AgentTool


class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


@dataclass
class LLMModel:
    provider: AIProvider
    name: str
    title: str | None = None
    thinking: bool = True
    tool_calling: bool = True
    vision: bool = True


@dataclass
class ToolParameter:
    name: str
    description: str
    ptype: type
    required: bool


@dataclass
class UniversalMessage:
    id: str
    role: MessageRole
    content: Any
    content_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=datetime.now)
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_call_params: dict[str, Any] | None = None
    image_id: str | None = None


@dataclass
class ConversationConfig:
    model: LLMModel | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    one_off: bool = False
    thinking: bool = False
    tools: list['AgentTool'] | None = None
    memory: bool = False
    drawing_model: str | None = None
