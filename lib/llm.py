import json
from time import time
from enum import Enum
from typing import Any, Callable, get_args, Literal
from typing import AsyncGenerator, BinaryIO
from dataclasses import dataclass, field, replace
from datetime import datetime
from abc import ABC, abstractmethod
from hashlib import md5
import uuid

import anthropic
import ollama
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseFunctionToolCall
from openai.types.responses import ResponseOutputItemAddedEvent, ResponseFunctionCallArgumentsDeltaEvent

import config
from logger import logger


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
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


@dataclass
class LLMModel:
    provider: AIProvider
    name: str
    thinking_supported: bool
    tool_calling_supported: bool
    image_input_supported: bool


@dataclass
class ToolParameter:
    name: str
    description: str
    ptype: type
    required: bool


@dataclass
class Tool:
    type: str
    providers: list[AIProvider]
    name: str | None = None
    function: Callable | None = None,
    description: str | None = None,
    schema: dict[str, ToolParameter] | None = None,
    params: dict[str, Any] | None = None


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


@dataclass
class ConversationConfig:
    model: LLMModel | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    one_off: bool = False
    thinking: bool = False
    tools: list[Tool] | None = None


class RequestDataConverter(ABC):
    @abstractmethod
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        pass

    @abstractmethod
    def get_request_parameters(self, conversation: 'ConversationManager') -> dict[str, Any]:
        pass


def get_type_data(ptype: type) -> tuple[str, tuple[str] | None]:
    if ptype == str:
        return "string", None
    elif ptype == int:
        return "integer", None
    elif ptype == float:
        return "number", None
    elif ptype == bool:
        return "boolean", None
    elif hasattr(ptype, '__origin__') and ptype.__origin__ == Literal:
        return 'string', list(get_args(ptype))


class OpenAIConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        result_message: list[dict[str, Any]] = []

        result_message.append({
            "role": MessageRole.SYSTEM.value,
            "content": config.system_prompt or "You are a helpful assistant"
        })

        for msg in messages:
            message: dict[str, Any] | None = None

            if msg.content_type == MessageType.TEXT:
                message = {
                    "role": msg.role.value,
                    "content": msg.content
                }

            elif msg.content_type == MessageType.IMAGE and config.model.image_input_supported:
                message = {
                    "role": msg.role.value,
                    "content": [{
                        "type": "input_image",
                        "image_url": msg.content
                    }]
                }

            elif msg.content_type == MessageType.TOOL_USE and config.model.tool_calling_supported:
                message = {
                    "type": "function_call",
                    "call_id": msg.tool_call_id,
                    "name": msg.tool_name,
                    "arguments": json.dumps(msg.tool_call_params)
                }

            if msg.content_type == MessageType.TOOL_RESULT and config.model.tool_calling_supported:
                message = {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content
                }

            if message:
                result_message.append(message)

        return result_message

    def get_parameters(self, tool: Tool) -> dict[str, Any]:
        params: dict[str, Any] = {}

        if not tool.schema:
            return params

        params['type'] = 'object'
        params['properties'] = {}
        params['required'] = []

        for key, value in tool.schema.items():
            ptype, enum_list = get_type_data(value.ptype)
            params['properties'][key] = {
                'type': ptype,
                'description': value.description,
            }
            if enum_list:
                params['properties'][key]['enum'] = enum_list

            if value.required:
                params['required'].append(key)

        return params

    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        if not config.model or not config.model.tool_calling_supported:
            return None

        tools: list[dict[str, Any]] = []

        for tool in config.tools or []:
            if AIProvider.OPENAI not in tool.providers:
                continue

            tool_data: dict[str, Any] = {}

            if tool.type:
                tool_data["type"] = tool.type

            if tool.type == 'function':
                tool_data['name'] = tool.name
                tool_data["parameters"] = self.get_parameters(tool)
                tool_data["description"] = tool.description

            if tool.params:
                for key, value in tool.params.items():
                    tool_data[key] = value

            tools.append(tool_data)
        return tools

    def get_request_parameters(self, conversation: 'ConversationManager') -> dict[str, Any]:
        provider_messages: list[dict[str, Any]] = self.messages_to_provider_format(
            conversation.config,
            conversation.messages
        )

        request_data: dict[str, Any] = {
            "model": conversation.config.model.name,
            "input": provider_messages,
            "max_output_tokens": conversation.config.max_tokens,
            "stream": True
        }

        if conversation.config.model.tool_calling_supported:
            tools = self.get_tools(conversation.config)
            request_data["tools"] = tools

        if conversation.user_id:
            request_data["user"] = md5(f'aaa-{conversation.user_id}-bbb'.encode('utf-8')).hexdigest()

        if conversation.config.thinking and conversation.config.model.thinking_supported:
            request_data['reasoning'] = {'effort': 'medium'}

        return request_data


class OllamaConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        result_messages: list[dict[str, Any]] = []

        result_messages.append({
            "role": MessageRole.SYSTEM.value,
            "content": config.system_prompt or "You are a helpful assistant"
        })

        for msg in messages:
            if msg.content_type == MessageType.TEXT:
                result_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

            elif msg.content_type == MessageType.IMAGE and config.model.image_input_supported:
                # Remove the "data:image/jpeg;base64," prefix if present
                image = msg.content[msg.content.index(',') +1 :]

                result_messages.append({
                    "role": msg.role.value,
                    "content": "",
                    "images": [image]
                })

            elif msg.content_type == MessageType.TOOL_USE and config.model.tool_calling_supported:
                result_messages.append({
                    "role": MessageRole.ASSISTANT.value,
                    "content": "",
                    "tool_use": {
                        "name": msg.tool_name,
                        "input": msg.tool_call_params or {}
                    }
                })

            elif msg.content_type == MessageType.TOOL_RESULT and config.model.tool_calling_supported:
                result_messages.append({
                    "role": MessageRole.TOOL.value,
                    "content": msg.content,
                    "tool_name": msg.tool_name,
                })

        return result_messages

    def get_parameters(self, tool: Tool) -> dict[str, Any]:
        params: dict[str, Any] = {}

        if not tool.schema:
            return params

        params['type'] = 'object'
        params['properties'] = {}
        params['required'] = []

        for key, value in tool.schema.items():
            ptype, enum_list = get_type_data(value.ptype)
            params['properties'][key] = {
                'type': ptype,
                'description': value.description,
            }
            if enum_list:
                params['properties'][key]['enum'] = enum_list
                # Not following the line above usually, so adding it to description as well
                params['properties'][key]['description'] = f"{value.description} Possible values: {', '.join(enum_list)}"

            if value.required:
                params['required'].append(key)

        return params

    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        tools: list[dict[str, Any]] = []

        for tool in config.tools or []:
            if AIProvider.OLLAMA not in tool.providers:
                continue

            tool_data: dict[str, Any] = {}

            if tool.type:
                tool_data["type"] = tool.type

            if tool.type == 'function':
                tool_data['function'] = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self.get_parameters(tool)
                }

            tools.append(tool_data)
        return tools

    def get_request_parameters(self, conversation: 'ConversationManager') -> dict[str, Any]:
        provider_messages: list[dict[str, Any]] = self.messages_to_provider_format(
            conversation.config,
            conversation.messages
        )

        request_data: dict[str, Any] = {
            "messages": provider_messages,
            "model": conversation.config.model.name,
            "stream": True
        }

        if conversation.config.model.tool_calling_supported:
            tools = self.get_tools(conversation.config)
            request_data["tools"] = tools

        if not conversation.config.thinking:
            request_data['think'] = False

        return request_data


class AnthropicConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        config: ConversationConfig,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        result_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.content_type == MessageType.TEXT:
                result_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            elif msg.content_type == MessageType.IMAGE and config.model.image_input_supported:
                # Remove the "data:image/jpeg;base64," prefix if present
                image = msg.content[msg.content.index(',') +1 :]
                result_messages.append({
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
                })
            elif msg.content_type == MessageType.TOOL_USE and config.model.tool_calling_supported:
                result_messages.append({
                    "role": MessageRole.ASSISTANT.value,
                    "content": [
                        {
                            "id": msg.tool_call_id,
                            "input": msg.tool_call_params or {},
                            "name": msg.tool_name,
                            "type": "tool_use"
                        }
                    ]
                })
            elif msg.content_type == MessageType.TOOL_RESULT and config.model.tool_calling_supported:
                result_messages.append({
                    "role": MessageRole.USER.value,
                    "content": [
                        {
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                            "type": "tool_result"
                        }
                    ]
                })

        return result_messages

    def get_parameters(self, tool: Tool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": []
        }

        if not tool.schema:
            return params

        params['type'] = 'object'
        params['properties'] = {}
        params['required'] = []

        for key, value in tool.schema.items():
            ptype, enum_list = get_type_data(value.ptype)
            params['properties'][key] = {
                'type': ptype,
                'description': value.description,
            }
            if enum_list:
                params['properties'][key]['enum'] = enum_list

            if value.required:
                params['required'].append(key)

        return params

    def get_tools(self, config: ConversationConfig) -> list[dict[str, Any]] | None:
        tools: list[dict[str, Any]] = []

        for tool in config.tools or []:
            if AIProvider.ANTHROPIC not in tool.providers:
                continue

            tool_data: dict[str, Any] = {
                "name": tool.name
            }

            if tool.type != 'function':
                tool_data["type"] = tool.type

            if tool.type == 'function':
                tool_data['name'] = tool.name
                tool_data["input_schema"] = self.get_parameters(tool)
                tool_data["description"] = tool.description

            if tool.params:
                for key, value in tool.params.items():
                    tool_data[key] = value

            tools.append(tool_data)

        return tools

    def get_request_parameters(self, conversation: 'ConversationManager') -> dict[str, Any]:
        provider_messages: list[dict[str, Any]] = self.messages_to_provider_format(
            conversation.config,
            conversation.messages
        )

        max_tokens = conversation.config.max_tokens or 4096

        request_data: dict[str, Any] = {
            "max_tokens": max_tokens,
            "messages": provider_messages,
            "model": conversation.config.model.name,
            "system": conversation.config.system_prompt or "You are a helpful assistant",
            "stream": True,
        }

        if conversation.config.model.tool_calling_supported:
            tools = self.get_tools(conversation.config)
            request_data["tools"] = tools

        if conversation.config.thinking and conversation.config.model.thinking_supported:
            request_data['thinking'] = {'type': 'enabled', 'budget_tokens': max_tokens - 500}

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
        self.config: ConversationConfig = ConversationConfig()
        self.user_id: int | None = None

    def set_user_id(self, user_id: int | None):
        self.user_id = user_id

    def set_model(self, model: LLMModel):
        self.config = replace(self.config, model=model)

    def set_config_param(self, param: str, value: Any):
        self.config = replace(self.config, **{param: value})

    def add_message(
        self,
        role: MessageRole,
        content: str,
        content_type: MessageType = MessageType.TEXT,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_call_params: dict[str, Any] | None = None,
    ) -> UniversalMessage:
        if self.config.one_off is True:
            self.messages.clear()

        message = UniversalMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            content_type=content_type,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_call_params=tool_call_params,
        )

        self.messages.append(message)
        return message

    def clear_conversation(self):
        self.messages.clear()

    def has_tool(self, tool_name: str) -> bool:
        if not self.config.tools:
            return False

        for tool in self.config.tools:
            if tool.type == 'function' and tool.name == tool_name:
                return True

        return False

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.config.tools:
            raise ValueError(f"Tool {tool_name} not found in conversation configuration.")

        for tool in self.config.tools:
            if tool.type == 'function' and tool.name == tool_name:
                if not tool.function:
                    raise ValueError(f"Tool {tool_name} has no associated function.")
                return await tool.function(**arguments)

        raise ValueError(f"Tool {tool_name} not found in conversation configuration.")

    def dump(self) -> dict[str, Any]:
        return {
            'messages': [msg for msg in self.messages],
            'config': self.config,
        }

    async def make_request(self, extra_params: dict[str, Any]) -> AsyncGenerator[str|None, None]:
        if not self.config.model:
            raise ValueError("Model is not set for the conversation.")

        current_provider = self.config.model.provider
        converter: RequestDataConverter = self.converters[current_provider]

        if current_provider == AIProvider.OPENAI:
            async for chunk in openai_instance.make_request(self, converter, extra_params):
                yield chunk
        elif current_provider == AIProvider.ANTHROPIC:
            async for chunk in claude_instance.make_request(self, converter, extra_params):
                yield chunk
        elif current_provider == AIProvider.OLLAMA:
            async for chunk in ollama_instance.make_request(self, converter, extra_params):
                yield chunk


class OpenAIInstance:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def execute_tools(
        self,
        conversation: ConversationManager,
        function_call_requests: list[ResponseFunctionToolCall],
        extra_params: dict[str, Any]
    ) -> Any:
        results: list[tuple[ResponseFunctionToolCall, Any]] = []
        for func_call in function_call_requests:
            if not conversation.has_tool(func_call.name):
                continue
            arguments: dict[str, Any] = {}

            if func_call.arguments:
                arguments = json.loads(func_call.arguments)

            result = await conversation.execute_tool(func_call.name, arguments | extra_params)

            results.append((func_call, result))
        return results

    async def whisper_transcribe(self, audio: BinaryIO) -> str:
        response = await self.client.audio.transcriptions.create(
            model='whisper-1',
            file=audio,
        )

        return response.text

    async def make_request(
        self,
        conversation: ConversationManager,
        converter: RequestDataConverter,
        extra_params: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        for _ in range(3):
            request_data = converter.get_request_parameters(conversation)

            try:
                stream = await self.client.responses.create(**request_data)
            except Exception as exc:
                logger.info(request_data)
                raise exc

            full_response: str = ''
            function_calls: dict[ResponseFunctionToolCall] = {}

            async for event in stream:
                if isinstance(event, ResponseOutputItemAddedEvent) and isinstance(event.item, ResponseFunctionToolCall):
                    function_calls[event.item.id] = event.item

                if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                    if not function_calls.get(event.item_id):
                        continue

                    if not function_calls[event.item_id].arguments:
                        function_calls[event.item_id].arguments = ''

                    function_calls[event.item_id].arguments += event.delta

                if isinstance(event, ResponseTextDeltaEvent):
                    content = event.delta
                    if content:
                        full_response += content
                        yield content

            if full_response:
                conversation.add_message(
                    role=MessageRole.ASSISTANT,
                    content=full_response
                )

            executed = await self.execute_tools(conversation, function_calls.values(), extra_params)

            for func_call, result in executed:
                params: dict[str, Any] = {}

                if func_call.arguments:
                    params = json.loads(func_call.arguments)

                conversation.add_message(
                    role=MessageRole.ASSISTANT,
                    content='',
                    content_type=MessageType.TOOL_USE,
                    tool_call_id=func_call.call_id,
                    tool_name=func_call.name,
                    tool_call_params=params
                )

                conversation.add_message(
                    role=MessageRole.USER,
                    content=result,
                    content_type=MessageType.TOOL_RESULT,
                    tool_call_id=func_call.call_id
                )

            if not executed:
                break


class ClaudeInstance:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    async def execute_tools(
        self,
        conversation: ConversationManager,
        function_call_requests: list[anthropic.types.ToolUseBlock],
        extra_params: dict[str, Any]
    ) -> list[tuple[anthropic.types.ToolUseBlock, Any]]:
        results: list[tuple[anthropic.types.ToolUseBlock, Any]] = []
        for func_call in function_call_requests:
            if not conversation.has_tool(func_call.name):
                continue

            arguments: dict[str, Any] = func_call.input

            result = await conversation.execute_tool(func_call.name, arguments | extra_params)

            results.append((func_call, result))
        return results

    # async def process_request(self, conversation: ConversationManager, request: dict[str, Any]) -> Any:
        # [{'role': 'user', 'content': 'сколько время?'}, {'role': 'assistant', 'content': 'Сейчас время 08:50 утра, 24 октября 2025 года. Чем могу помочь?'}, {'role': 'assistant', 'content': 'Сейчас время 08:50. Чем могу помочь?'}, {'role': 'user', 'content': 'сколько время?'}, {'role': 'assistant', 'content': [{'id': 'toolu_01Y5LTQDyFeds3LARjrjpuv8', 'input': {}, 'name': 'get_current_time', 'type': 'tool_use'}]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_01Y5LTQDyFeds3LARjrjpuv8', 'content': '2025-10-24 08:51:23'}]}, {'role': 'assistant', 'content': 'Сейчас **08:51:23** (8 часов 51 минута и 23 секунды), 24 октября 2025 года.'}]

    async def make_request(
        self,
        conversation: ConversationManager,
        converter: RequestDataConverter,
        extra_params: dict[str, Any]
    ) -> AsyncGenerator[str | None, None]:
        for _ in range(3):
            request_data = converter.get_request_parameters(conversation)

            try:
                stream = await self.client.messages.create(**request_data)
            except Exception as exc:
                logger.info(request_data)
                raise exc

            full_response: str = ''
            function_calls: list[anthropic.types.ToolUseBlock] = []
            tool_params_json = ''
            executed: list[tuple[anthropic.types.ToolUseBlock, Any]] = []

            async for event in stream:
                if event.type == 'content_block_stop':
                    if full_response:
                        conversation.add_message(
                            role=MessageRole.ASSISTANT,
                            content=full_response
                        )
                        full_response = ''
                        # Flush command
                        yield None

                    if function_calls:
                        function_calls[-1].input = json.loads(tool_params_json)
                        tool_params_json = ''

                        executed = await self.execute_tools(conversation, function_calls, extra_params)

                        for func_call, result in executed:
                            conversation.add_message(
                                role=MessageRole.ASSISTANT,
                                content='',
                                content_type=MessageType.TOOL_USE,
                                tool_call_id=func_call.id,
                                tool_name=func_call.name,
                                tool_call_params=func_call.input
                            )

                            conversation.add_message(
                                role=MessageRole.USER,
                                content=result,
                                content_type=MessageType.TOOL_RESULT,
                                tool_call_id=func_call.id
                            )
                        function_calls = []

                if event.type == 'content_block_start' and isinstance(event.content_block, anthropic.types.ToolUseBlock):
                    function_calls.append(event.content_block)

                if event.type == 'content_block_delta' and isinstance(event.delta, anthropic.types.InputJSONDelta):
                    tool_params_json += event.delta.partial_json

                elif event.type == 'content_block_delta' and isinstance(event.delta, anthropic.types.TextDelta):
                    content = event.delta.text

                    if content:
                        full_response += content
                        yield content
            if not executed:
                break


class OllamaInstance:
    def __init__(self):
        self.client = ollama.AsyncClient(host=config.OLLAMA_HOST)

    async def execute_tools(
        self,
        conversation: ConversationManager,
        function_call_requests: list[ollama.Message.ToolCall],
        extra_params: dict[str, Any]
    ) -> Any:
        results: list[tuple[ollama.Message.ToolCall, Any]] = []
        for func_call in function_call_requests:
            if not conversation.has_tool(func_call.function.name):
                continue
            arguments: dict[str, Any] = {}

            if func_call.function.arguments:
                arguments = func_call.function.arguments

            result = await conversation.execute_tool(func_call.function.name, arguments | extra_params)

            results.append((func_call, result))
        return results

    async def make_request(
        self,
        conversation: ConversationManager,
        converter: RequestDataConverter,
        extra_params: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        for _ in range(3):
            request_data = converter.get_request_parameters(conversation)

            full_response: str = ''
            function_calls: list[Any] = []

            try:
                stream = await self.client.chat(**request_data)
            except Exception as exc:
                logger.info(request_data)
                raise exc

            async for event in stream:
                if isinstance(event, ollama.ChatResponse):
                    content = event.message.content
                    if content:
                        full_response += content
                        yield content

                    if event.message.tool_calls:
                        function_calls.extend(event.message.tool_calls)

            if full_response:
                conversation.add_message(
                    role=MessageRole.ASSISTANT,
                    content=full_response
                )

            executed = await self.execute_tools(conversation, function_calls, extra_params)

            for func_call, result in executed:
                conversation.add_message(
                    role=MessageRole.ASSISTANT,
                    content='',
                    content_type=MessageType.TOOL_USE,
                    tool_call_id=str(result.__hash__()),
                    tool_name=func_call.function.name,
                    tool_call_params=func_call.function.arguments
                )

                conversation.add_message(
                    role=MessageRole.ASSISTANT,
                    content=result,
                    content_type=MessageType.TOOL_RESULT,
                    tool_name=func_call.function.name,
                    tool_call_id=str(result.__hash__())
                )

            if not executed:
                break


openai_instance = OpenAIInstance()
claude_instance = ClaudeInstance()
ollama_instance = OllamaInstance()
