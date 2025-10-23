from time import time
from enum import Enum
from typing import Any
from dataclasses import dataclass, field, replace
from datetime import datetime
from abc import ABC, abstractmethod
import uuid


class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"


class Tool:
    def __init__(
        self,
        type: str,
        provider: AIProvider,
        name: str | None = None,
        description: str | None = None,
        schema: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None
    ):
        self.name = name
        self.description = description
        self.provider = provider
        self.type = type
        self.schema = schema
        self.params = params or {}


@dataclass
class UniversalMessage:
    id: str
    role: MessageRole
    content: str | bytes
    content_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ConversationConfig:
    model: str
    max_tokens: int | None = None
    system_prompt: str | None = None
    one_off: bool = False
    thinking: bool = False
    tools: list[Tool] | None = None


@dataclass
class RequestData:
    provider: AIProvider
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None
    system_prompt: str | None
    thinking: bool | None
    tools: list[dict[str, Any]] | None


class RequestDataConverter(ABC):
    @abstractmethod
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def message_from_provider_format(self, provider_message: dict[str, Any]) -> UniversalMessage:
        pass

    @abstractmethod
    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        pass

    @abstractmethod
    def make_request_data(self, config: ConversationConfig, messages: list[UniversalMessage]) -> dict[str, Any]:
        pass


class OpenAIConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []

        openai_messages.append({
            "role": MessageRole.SYSTEM.value,
            "content": config.system_prompt or "You are a helpful assistant"
        })

        for msg in messages:
            openai_msg: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content
            }

            if msg.content_type == MessageType.IMAGE:
                openai_msg: dict[str, Any] = {
                    "role": msg.role.value,
                    "content": [{
                        "type": "input_image",
                        "image_url": msg.content
                    }]
                }

            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls

            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            openai_messages.append(openai_msg)

        return openai_messages

    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        tools: list[dict[str, Any]] = []

        for tool in config.tools or []:
            if tool.provider != AIProvider.OPENAI:
                continue

            tool_data: dict[str, Any] = {}

            if tool.type:
                tool_data["type"] = tool.type

            if tool.params:
                for key, value in tool.params.items():
                    tool_data[key] = value

            tools.append(tool_data)
        return tools

    def make_request_data(self, config: ConversationConfig, messages: list[UniversalMessage]) -> dict[str, Any]:
        request_data: dict[str, Any] = {
            "model": config.model,
            "messages": self.messages_to_provider_format(config, messages)
        }

        if config.max_tokens is not None:
            request_data["max_tokens"] = config.max_tokens

        return request_data

    def message_from_provider_format(self, provider_message: dict[str, Any]) -> UniversalMessage:
        return UniversalMessage(
            id=str(uuid.uuid4()),
            role=MessageRole(provider_message["role"]),
            content=provider_message.get("content", ""),
            tool_calls=provider_message.get("tool_calls"),
            tool_call_id=provider_message.get("tool_call_id")
        )

class OllamaConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        ollama_messages: list[dict[str, Any]] = []

        ollama_messages.append({
            "role": MessageRole.SYSTEM.value,
            "content": config.system_prompt or "You are a helpful assistant"
        })

        for msg in messages:
            ollama_msg: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content
            }

            if msg.content_type == MessageType.IMAGE:
                # Remove the "data:image/jpeg;base64," prefix if present
                image = msg.content[msg.content.index(',') +1 :]

                ollama_msg: dict[str, Any] = {
                    "role": msg.role.value,
                    "content": "",
                    "images": [image]
                }

            if msg.tool_calls:
                ollama_msg["tool_calls"] = msg.tool_calls

            if msg.tool_call_id:
                ollama_msg["tool_call_id"] = msg.tool_call_id

            ollama_messages.append(ollama_msg)

        return ollama_messages

    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        pass  # To be implemented

    def make_request_data(self, config: ConversationConfig, messages: list[UniversalMessage]) -> dict[str, Any]:
        request_data: dict[str, Any] = {
            "model": config.model,
            "messages": self.messages_to_provider_format(config, messages)
        }

        if config.max_tokens is not None:
            request_data["max_tokens"] = config.max_tokens

        return request_data

    def message_from_provider_format(self, provider_message: dict[str, Any]) -> UniversalMessage:
        return UniversalMessage(
            id=str(uuid.uuid4()),
            role=MessageRole(provider_message["role"]),
            content=provider_message.get("content", ""),
            tool_calls=provider_message.get("tool_calls"),
            tool_call_id=provider_message.get("tool_call_id")
        )


class AnthropicConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                if msg.content_type == MessageType.TEXT:
                    anthropic_msg: dict[str, Any] = {
                        "role": msg.role.value,
                        "content": msg.content
                    }
                elif msg.content_type == MessageType.IMAGE:
                    # Remove the "data:image/jpeg;base64," prefix if present
                    image = msg.content[msg.content.index(',') +1 :]
                    anthropic_msg: dict[str, Any] = {
                        "role": msg.role.value,
                        "content": [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': 'image/jpeg',
                                    'data': image
                                }
                            }
                        ]
                    }

                anthropic_messages.append(anthropic_msg)
            elif msg.role == MessageRole.TOOL:
                # Для Anthropic tool responses обрабатываются по-особому
                anthropic_messages.append({
                    "role": "user",
                    "content": f"Tool result: {msg.content}"
                })

                anthropic_messages.append(anthropic_msg)
            elif msg.role == MessageRole.TOOL:
                # Для Anthropic tool responses обрабатываются по-особому
                anthropic_messages.append({
                    "role": "user",
                    "content": f"Tool result: {msg.content}"
                })

        return anthropic_messages

    def message_from_provider_format(self, provider_message: dict[str, Any]) -> UniversalMessage:
        return UniversalMessage(
            id=str(uuid.uuid4()),
            role=MessageRole(provider_message["role"]),
            content=provider_message.get("content", "")
        )

    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        tools: list[dict[str, Any]] = []

        for tool in config.tools or []:
            if tool.provider != AIProvider.ANTHROPIC:
                continue

            tool_data: dict[str, Any] = {
                "name": tool.name
            }

            if tool.type:
                tool_data["type"] = tool.type

            if tool.params:
                for key, value in tool.params.items():
                    tool_data[key] = value

            tools.append(tool_data)

        return tools

    def make_request_data(self, config: ConversationConfig, messages: list[UniversalMessage]) -> dict[str, Any]:
        request_data: dict[str, Any] = {
            "model": config.model,
            "messages": self.messages_to_provider_format(config, messages),
            "system": config.system_prompt or "You are a helpful assistant"
        }

        if config.max_tokens is not None:
            request_data["max_tokens"] = config.max_tokens

        return request_data


class ConversationManager:
    def __init__(self):
        self.id = round(time())
        self.converters: dict[AIProvider, RequestDataConverter] = {
            AIProvider.OPENAI: OpenAIConverter(),
            AIProvider.ANTHROPIC: AnthropicConverter(),
            AIProvider.OLLAMA: OllamaConverter(),
        }
        self.messages: list[UniversalMessage] = []
        self.current_provider: AIProvider | None = None
        self.config: ConversationConfig = ConversationConfig(model="")

    def set_provider(self, provider: AIProvider, model: str):
        self.current_provider = provider
        self.config = replace(self.config, model=model)

    def set_config_param(self, param: str, value: Any):
        self.config = replace(self.config, **{param: value})

    def add_message(
        self,
        role: MessageRole,
        content: str,
        content_type: MessageType = MessageType.TEXT,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UniversalMessage:
        if self.config.one_off is True:
            self.messages.clear()

        message = UniversalMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            content_type=content_type,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            metadata=metadata
        )
        self.messages.append(message)
        return message

    def add_ai_response(self, response_content: str, tool_calls: list[dict[str, Any]] | None = None):
        return self.add_message(
            MessageRole.ASSISTANT,
            response_content,
            tool_calls=tool_calls
        )

    def clear_conversation(self):
        self.messages.clear()

    def get_request_data(self) -> RequestData:
        if not self.current_provider or not self.config:
            raise ValueError("AI provider and configuration must be set before getting request data.")

        converter = self.converters[self.current_provider]

        provider_messages: list[dict[str, Any]] = converter.messages_to_provider_format(
            self.config,
            self.messages
        )

        tools = converter.get_tools(self.config)

        return RequestData(
            provider=self.current_provider,
            model=self.config.model,
            messages=provider_messages,
            max_tokens=self.config.max_tokens,
            system_prompt=self.config.system_prompt,
            thinking=self.config.thinking,
            tools=tools
        )
