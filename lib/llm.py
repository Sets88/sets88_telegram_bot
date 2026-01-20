import json
import os
import functools
import base64
from time import time
from datetime import datetime
from typing import Any, get_args, Literal
from typing import AsyncGenerator, BinaryIO
from dataclasses import replace
from abc import ABC, abstractmethod
from hashlib import md5
import uuid

import anthropic
import ollama
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseFunctionToolCall
from openai.types.responses import ResponseOutputItemAddedEvent, ResponseFunctionCallArgumentsDeltaEvent, ResponseCompletedEvent

import config
from lib.agents import AgentTool
from lib.structs import LLMModel, AIProvider, MessageRole, MessageType, UniversalMessage, ConversationConfig
from logger import logger


LLM_RESPONSE_TIMEOUT = 500


class RequestDataConverter(ABC):
    def __init__(self, conversation_manager: 'ConversationManager') -> None:
        self.conversation_manager = conversation_manager

    @abstractmethod
    def messages_to_provider_format(
        self,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_tools(self, extra_params: dict[str, Any]) -> list[dict[str, Any]] | None:
        pass

    @abstractmethod
    def get_request_parameters(self, extra_params: dict[str, Any]) -> dict[str, Any]:
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
    elif ptype == list[str]:
        return "array", None
    elif hasattr(ptype, '__origin__') and ptype.__origin__ == Literal:
        return 'string', list(get_args(ptype))


class OpenAIConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        result_message: list[dict[str, Any]] = []

        system_prompt = self.conversation_manager.get_system_prompt()
        model = self.conversation_manager.config.model

        result_message.append({
            "role": MessageRole.SYSTEM.value,
            "content": system_prompt or "You are a helpful assistant"
        })

        for msg in messages:
            message: dict[str, Any] | None = None

            if msg.content_type == MessageType.TEXT:
                message = {
                    "role": msg.role.value,
                    "content": msg.content
                }

            elif msg.content_type == MessageType.IMAGE:
                if model.vision:
                    message = {
                        "role": msg.role.value,
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"uploaded file with image_id: {msg.image_id} use it as reference"
                            },
                            {
                                "type": "input_image",
                                "image_url": msg.content
                            }
                        ]
                    }
                else:
                    message = {
                        "role": msg.role.value,
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"image been provided, image_id: {msg.image_id} use it as reference"
                            },
                        ]
                    }

            elif msg.content_type == MessageType.TOOL_USE and model.tool_calling:
                message = {
                    "type": "function_call",
                    "call_id": msg.tool_call_id,
                    "name": msg.tool_name,
                    "arguments": json.dumps(msg.tool_call_params)
                }

            if msg.content_type == MessageType.TOOL_RESULT and model.tool_calling:
                message = {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content
                }

            if message:
                result_message.append(message)

        return result_message

    def get_parameters(self, tool: AgentTool) -> dict[str, Any]:
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
            if ptype == "array":
                params['properties'][key]['items'] = {'type': 'string'}

            if enum_list:
                params['properties'][key]['enum'] = enum_list

            if value.required:
                params['required'].append(key)

        return params

    def get_tools(self, extra_params: dict[str, Any]) -> list[dict[str, Any]] | None:
        llm_config = self.conversation_manager.config

        if not llm_config.model or not llm_config.model.tool_calling:
            return None

        tools: list[dict[str, Any]] = []

        for tool in llm_config.tools or []:
            if AIProvider.OPENAI not in tool.providers:
                continue

            if not tool.is_enabled(self.conversation_manager, extra_params):
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

    def get_request_parameters(self, extra_params: dict[str, Any]) -> dict[str, Any]:
        conversation = self.conversation_manager
        model = conversation.config.model

        provider_messages: list[dict[str, Any]] = self.messages_to_provider_format(
            conversation.messages
        )

        request_data: dict[str, Any] = {
            "model": model.name,
            "input": provider_messages,
            "max_output_tokens": conversation.config.max_tokens,
            "stream": True
        }

        if model.tool_calling:
            tools = self.get_tools(extra_params)
            request_data["tools"] = tools

        if conversation.user_id:
            request_data["user"] = md5(f'aaa-{conversation.user_id}-bbb'.encode('utf-8')).hexdigest()

        if conversation.config.thinking and model.thinking:
            request_data['reasoning'] = {'effort': 'medium'}

        return request_data


class OllamaConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        result_messages: list[dict[str, Any]] = []

        system_prompt = self.conversation_manager.get_system_prompt()

        model = self.conversation_manager.config.model

        result_messages.append({
            # ASSISTANT role because Qwen models skip SYSTEM messages on the first round for some reason
            "role": MessageRole.ASSISTANT.value,
            "content": system_prompt or "You are a helpful assistant"
        })

        for msg in messages:
            if msg.content_type == MessageType.TEXT:
                result_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

            elif msg.content_type == MessageType.IMAGE:
                if  model.vision:
                    # Remove the "data:image/jpeg;base64," prefix if present
                    image = msg.content[msg.content.index(',') +1 :]
                    result_messages.append({
                        "role": msg.role.value,
                        "content": f"uploaded file with image_id: {msg.image_id} use it as reference"
                    })
                    result_messages.append({
                        "role": msg.role.value,
                        "content": "",
                        "images": [image]
                    })
                else:
                    result_messages.append({
                        "role": msg.role.value,
                        "content": f"image been provided, image_id: {msg.image_id} use it as reference"
                    })

            elif msg.content_type == MessageType.TOOL_USE and model.tool_calling:
                result_messages.append({
                    "role": MessageRole.ASSISTANT.value,
                    "content": "",
                    "tool_use": {
                        "name": msg.tool_name,
                        "input": msg.tool_call_params or {}
                    }
                })

            elif msg.content_type == MessageType.TOOL_RESULT and model.tool_calling:
                result_messages.append({
                    "role": MessageRole.TOOL.value,
                    "content": msg.content,
                    "tool_name": msg.tool_name,
                })

        return result_messages

    def get_parameters(self, tool: AgentTool) -> dict[str, Any]:
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
            if ptype == "array":
                params['properties'][key]['items'] = {'type': 'string'}
            if enum_list:
                params['properties'][key]['enum'] = enum_list
                # Not following the line above usually, so adding it to description as well
                params['properties'][key]['description'] = f"{value.description} Possible values: {', '.join(enum_list)}"

            if value.required:
                params['required'].append(key)

        return params

    def get_tools(self, extra_params: dict[str, Any]) -> list[dict[str, Any]] | None:
        llm_config = self.conversation_manager.config
        tools: list[dict[str, Any]] = []

        for tool in llm_config.tools or []:
            if AIProvider.OLLAMA not in tool.providers:
                continue

            if not tool.is_enabled(self.conversation_manager, extra_params):
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

    def get_request_parameters(self, extra_params: dict[str, Any]) -> dict[str, Any]:
        conversation = self.conversation_manager
        model = conversation.config.model

        provider_messages: list[dict[str, Any]] = self.messages_to_provider_format(
            conversation.messages
        )

        request_data: dict[str, Any] = {
            "messages": provider_messages,
            "model": model.name,
            "stream": True,
            "options": {"num_ctx": conversation.config.max_tokens}
        }

        if model.tool_calling:
            tools = self.get_tools(extra_params)
            request_data["tools"] = tools

        if not conversation.config.thinking:
            request_data['think'] = False

        return request_data


class OpenRouterConverter(OpenAIConverter):
    pass


class AnthropicConverter(RequestDataConverter):
    def messages_to_provider_format(
        self,
        messages: list[UniversalMessage]
    ) -> list[dict[str, Any]]:
        model = self.conversation_manager.config.model

        result_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.content_type == MessageType.TEXT:
                result_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            elif msg.content_type == MessageType.IMAGE and model.vision:
                # Remove the "data:image/jpeg;base64," prefix if present
                image = msg.content[msg.content.index(',') +1 :]
                result_messages.append({
                    "role": msg.role.value,
                    "content": [
                        {
                            'type': 'text',
                            'text': f"uploaded file with image_id: {msg.image_id} use it as reference"
                        },
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
            elif msg.content_type == MessageType.TOOL_USE and model.tool_calling:
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
            elif msg.content_type == MessageType.TOOL_RESULT and model.tool_calling:
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

    def get_parameters(self, tool: AgentTool) -> dict[str, Any]:
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
            if ptype == "array":
                params['properties'][key]['items'] = {'type': 'string'}
            if enum_list:
                params['properties'][key]['enum'] = enum_list

            if value.required:
                params['required'].append(key)

        return params

    def get_tools(self, extra_params: dict[str, Any]) -> list[dict[str, Any]] | None:
        llm_config = self.conversation_manager.config

        tools: list[dict[str, Any]] = []

        for tool in llm_config.tools or []:
            if AIProvider.ANTHROPIC not in tool.providers:
                continue

            if not tool.is_enabled(self.conversation_manager, extra_params):
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

    def get_request_parameters(self, extra_params: dict[str, Any]) -> dict[str, Any]:
        conversation = self.conversation_manager
        model = conversation.config.model

        provider_messages: list[dict[str, Any]] = self.messages_to_provider_format(
            conversation.messages
        )

        max_tokens = conversation.config.max_tokens or 4096

        request_data: dict[str, Any] = {
            "max_tokens": max_tokens,
            "messages": provider_messages,
            "model": model.name,
            "system": conversation.get_system_prompt(),
            "stream": True,
        }

        if model.tool_calling:
            tools = self.get_tools(extra_params)
            request_data["tools"] = tools

        if conversation.config.thinking and model.thinking:
            request_data['thinking'] = {'type': 'enabled', 'budget_tokens': max_tokens - 500}

        return request_data


class ConversationManager:
    def __init__(self):
        self.parent_manager: 'ConversationManager | None' = None
        self.id = round(time())
        self.converters: dict[AIProvider, RequestDataConverter] = {
            AIProvider.OPENAI: OpenAIConverter(self),
            AIProvider.ANTHROPIC: AnthropicConverter(self),
            AIProvider.OLLAMA: OllamaConverter(self),
            AIProvider.OPENROUTER: OpenRouterConverter(self),
        }
        self.messages: list[UniversalMessage] = []
        self.config: ConversationConfig = ConversationConfig()
        self.user_id: int | None = None
        self.memory: dict[str, Any] | None = None
        self.data_cache: dict[str, str] = {}
        self.openai_instance: OpenAIInstance = openai_instance
        self.claude_instance: ClaudeInstance = claude_instance
        self.ollama_instance: OllamaInstance = ollama_instance
        self.openrouter_instance: OpenRouterInstance = openrouter_instance
        self.last_message_time: float = time()

    def refresh_last_message_time(self) -> None:
        self.last_message_time = time()

    def is_old_conversation(self, threshold_seconds: int) -> bool:
        if self.messages and (self.last_message_time + threshold_seconds < time()):
            return True
        return False

    def invoke_subagent(self) -> 'ConversationManager':
        """Creates a new ConversationManager that copies the configuration, has shared memory and a cache"""
        sub_conversation = ConversationManager()
        sub_conversation.parent_manager = self
        sub_conversation.user_id = self.user_id
        sub_conversation.config = replace(self.config)
        sub_conversation.memory = self.memory
        return sub_conversation

    def set_user_id(self, user_id: int | None):
        self.user_id = user_id

    def cache_data(self, data: str | bytes) -> str:
        data_id = uuid.uuid4().hex

        self.data_cache[data_id] = data

        self.get_cached_data(data_id)

        return data_id

    @functools.lru_cache(maxsize=5)
    def get_cached_data(self, data_id: str) -> str | bytes | None:
        if data_id in self.data_cache:
            return self.data_cache.pop(data_id, None)
        if self.parent_manager:
            return self.parent_manager.get_cached_data(data_id)

    def set_model(self, model: LLMModel):
        self.config = replace(self.config, model=model)

    def set_config_param(self, param: str, value: Any):
        self.config = replace(self.config, **{param: value})

    def get_system_prompt(self) -> str:
        prompt = "You are a helpful assistant"
        if self.config.system_prompt:
            prompt = self.config.system_prompt

        if self.memory is None and self.config.memory:
            self.load_memory()

        if self.memory and self.config.memory:
            prompt += "\n\nYour memory:\n"
            for key, value in self.memory.items():
                prompt += f"- {key}: {value}\n"

        return prompt

    def load_memory(self) -> None:
        if not self.user_id:
            return

        filename = os.path.basename(f'{self.user_id}_memory.json')

        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory', filename))

        self.memory = {}

        if not os.path.exists(full_path):
            return

        try:
            with open(full_path, 'r') as f:
                for key, value in json.load(f).items():
                    self.memory[key] = value
        except (FileNotFoundError, json.JSONDecodeError):
            return
        except Exception as exc:
            logger.error(f"Error loading memory: {exc}")

    def save_memory(self) -> None:
        if self.memory is None:
            self.load_memory()

        if not self.user_id:
            return

        filename = os.path.basename(f'{self.user_id}_memory.json')

        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory', filename))

        try:
            with open(full_path, 'w') as f:
                json.dump(self.memory, f)
        except Exception as exc:
            logger.error(f"Error saving memory: {exc}")

    def add_memory(self, key: str, value: Any) -> None:
        if not self.user_id:
            return

        self.memory[key] = value
        self.save_memory()

    def delete_memory(self, key: str) -> None:
        if not self.user_id:
            return

        if key in self.memory:
            del self.memory[key]
            self.save_memory()

    def add_message(
        self,
        role: MessageRole,
        content: str,
        content_type: MessageType = MessageType.TEXT,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_call_params: dict[str, Any] | None = None,
        image_id: str | None = None
    ) -> UniversalMessage:
        self.last_message_time = time()

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
            image_id=image_id
        )

        self.messages.append(message)
        return message

    def clear_conversation(self, older_than: int | None = None) -> None:
        # regernerate id to keep previous conversation saved
        self.id = round(time())

        if older_than is None:
            self.messages.clear()
            return

        current_time = datetime.now()
        self.messages = [
            msg for msg in self.messages
            if (current_time - msg.timestamp).total_seconds() < older_than
        ]

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
                logger.info(f"Executing tool {tool_name} with arguments {arguments}")
                return await tool.execute(self, arguments)

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
            async for chunk in self.openai_instance.make_request(self, converter, extra_params):
                yield chunk
        elif current_provider == AIProvider.ANTHROPIC:
            async for chunk in self.claude_instance.make_request(self, converter, extra_params):
                yield chunk
        elif current_provider == AIProvider.OLLAMA:
            async for chunk in self.ollama_instance.make_request(self, converter, extra_params):
                yield chunk
        elif current_provider == AIProvider.OPENROUTER:
            async for chunk in self.openrouter_instance.make_request(self, converter, extra_params):
                yield chunk


class OpenAIInstance:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=LLM_RESPONSE_TIMEOUT)

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

            arguments = arguments | extra_params

            result = await conversation.execute_tool(func_call.name, arguments)

            results.append((func_call, result))
        return results

    async def whisper_transcribe(self, audio: BinaryIO) -> str:
        response = await self.client.audio.transcriptions.create(
            model='whisper-1',
            file=audio,
        )

        return response.text

    async def image_generate(self, image_model: str, prompt: str, images: list[BinaryIO]) -> list[str]:
        if images:
            response = await self.client.images.edit(
                model=image_model,
                image=images,
                prompt=prompt
            )
        else:
            response = await self.client.images.generate(
                model=image_model,
                prompt=prompt
            )

        result_images = []

        for image_data in response.data:
            image_base64 = image_data.b64_json
            result_images.append(base64.b64decode(image_base64))

        return result_images

    async def make_request(
        self,
        conversation: ConversationManager,
        converter: RequestDataConverter,
        extra_params: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        for _ in range(3):
            request_data = converter.get_request_parameters(extra_params)

            try:
                stream = await self.client.responses.create(**request_data)
            except Exception as exc:
                logger.info(request_data)
                raise exc

            full_response: str = ''
            function_calls: dict[ResponseFunctionToolCall] = {}

            async for event in stream:
                if event.type == 'response.incomplete':
                    response = f'I am not able to complete your request at the moment due to {event.response.incomplete_details.reason}'
                    logger.info(response)
                    yield response
                    return

                if isinstance(event, ResponseCompletedEvent) and event.response.output:
                    for item in event.response.output:
                        if (
                            isinstance(item, ResponseFunctionToolCall) and
                            item.call_id in function_calls and
                            not function_calls[item.call_id].arguments
                        ):
                            function_calls[item.call_id] = item

                if isinstance(event, ResponseOutputItemAddedEvent) and isinstance(event.item, ResponseFunctionToolCall):
                    function_calls[event.item.call_id] = event.item

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
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY, timeout=LLM_RESPONSE_TIMEOUT)

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

            arguments: dict[str, Any] = func_call.input | extra_params

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
            request_data = converter.get_request_parameters(extra_params)

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
                if (
                    event.type == 'message_delta' and
                    event.delta.stop_reason != 'tool_use' and
                    event.delta.stop_reason != 'end_turn'
                ):
                    response = f'\nI am not able to complete your request at the moment due to {event.delta.stop_reason}'
                    logger.info(response)
                    yield response
                    return

                if event.type == 'content_block_stop':
                    if function_calls:
                        if full_response:
                            conversation.add_message(
                                role=MessageRole.ASSISTANT,
                                content=full_response
                            )
                            full_response = ''
                            # Flush command
                            yield None

                        if tool_params_json:
                            function_calls[-1].input = json.loads(tool_params_json)
                        else:
                            function_calls[-1].input = {}

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

            if full_response:
                conversation.add_message(
                    role=MessageRole.ASSISTANT,
                    content=full_response
                )
                full_response = ''
                # Flush command
                yield None

            if not executed:
                break

class OpenRouterInstance(OpenAIInstance):
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=LLM_RESPONSE_TIMEOUT
        )

class OllamaInstance:
    def __init__(self):
        self.client = ollama.AsyncClient(host=config.OLLAMA_HOST, timeout=LLM_RESPONSE_TIMEOUT)

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

            arguments = arguments | extra_params

            result = await conversation.execute_tool(func_call.function.name, arguments)

            results.append((func_call, result))
        return results

    async def make_request(
        self,
        conversation: ConversationManager,
        converter: RequestDataConverter,
        extra_params: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        for _ in range(3):
            request_data = converter.get_request_parameters(extra_params)

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
                    role=MessageRole.TOOL,
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
openrouter_instance = OpenRouterInstance()
