from __future__ import annotations
from typing import TYPE_CHECKING
import abc
from io import BytesIO
from datetime import datetime
from typing import Any

from telebot.types import Message
from mcp import ClientSession
from mcp.client.sse import sse_client

import config
from replicate_module import replicate_execute_and_send
from lib.permissions import is_replicate_available
from logger import logger
from lib.structs import AIProvider, ToolParameter, MessageRole
from telebot_nav import TeleBotNav, Message

if TYPE_CHECKING:
    from lib.llm import ConversationManager


SUBAGENT_PROMPT = '''You are a transparent tool executor. Your SOLE function is to:
1. Detect which tool the user needs
2. Execute it with correct parameters
3. Return empty response
'''


DEFAULT_DRAWING_MODEL = 'flux-2-pro'


DIFFUSION_MODELS_IMAGE_FIELDS: dict[str, str] = {
    'flux-2-pro': 'input_images',
    'nano-banana-pro': 'image_input',
    'seedream-4.5': 'image_input',
    'qwen-image-edit': 'image',
}

OPENAI_IMAGE_MODELS: set[str] = {
    'gpt-image-1.5'
}


class AgentTool(abc.ABC):
    type: str = ''
    providers: list[AIProvider] = []
    name: str | None = None
    description: str | None = None
    schema: dict[str, ToolParameter] | None = None
    params: dict[str, Any] | None = None

    def dump(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            'type': self.type,
            'name': self.name,
            'description': self.description,
            'schema': {}
        }
        if self.schema:
            for param_name, param in self.schema.items():
                result['schema'][param_name] = {
                    'description': param.description,
                    'type': param.ptype.__name__,
                    'required': param.required
                }
        if self.params:
            result['params'] = self.params

        return result

    def is_enabled(self, manager: 'ConversationManager', params: dict[str, Any]) -> bool:
        return True
    
    @abc.abstractmethod
    async def execute(self, conversation: 'ConversationManager', params: dict[str, Any]) -> Any:
        pass


class OpenAiWebSearchAgentTool(AgentTool):
    type = 'web_search'
    providers = [AIProvider.OPENAI]

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        return 'Not implemented yet'


class AntropicWebSearchAgentTool(AgentTool):
    type = 'web_search_20250305'
    providers = [AIProvider.ANTHROPIC]
    name='web_search'
    params={'max_uses': 5}

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        return 'Not implemented yet'


class GetCurrentTimeAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'get_current_time'
    description = 'Returns the current time in YYYY-MM-DD HH:MM:SS format'
    schema = {}

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class ImageGenerationAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'image_generation'
    description = 'Generates or edits an image and sends it to the user based on the provided prompt;' + \
        'In case of editing, the original image_id list should be provided as a reference. ' + \
        'returns true if the image is successfully sent to the user, ' + \
        'If successful, NEVER respond that you failed to draw the photo.'
    schema = {
            'prompt': ToolParameter(
                name='prompt', description='A richly detailed prompt in English designed for '
                    'a diffusion model to generate an image',
                ptype=str, required=True
            ),
            'images': ToolParameter(
                name='images', description='List of image_id to use as input for image generation, '
                    'if multiple images are provided, a collage will be generated',
                ptype=list[str], required=False
            )
        }

    def is_enabled(self, manager: 'ConversationManager', params: dict[str, Any]) -> bool:
        message: Message = params.get('message')
        botnav: TeleBotNav = params.get('botnav')
        if not message or not botnav:
            return False
        return is_replicate_available(botnav, message)

    def get_images_list(self, conversation: 'ConversationManager', images: list[str] | None) -> list[BytesIO]:
        image_bytes_list: list[BytesIO] = []

        if images:
            for image in images:
                img_data = conversation.get_cached_data(image)

                if not img_data:
                    return []

                image_bytes_list.append(BytesIO(img_data))
        return image_bytes_list

    async def execute_replicate(
        self,
        conversation: 'ConversationManager',
        botnav: TeleBotNav,
        message: Message,
        model: str,
        prompt: str,
        images: list[str] | None = None,
    ) -> str:
        input_data = {
            'prompt': prompt
        }

        image_bytes_list = self.get_images_list(conversation, images)

        if image_bytes_list:
            image_field = DIFFUSION_MODELS_IMAGE_FIELDS.get(model, None)
            input_data[image_field] = image_bytes_list

        try:
            result = await replicate_execute_and_send(botnav, message, model, input_data)

            if result:
                if isinstance(result, list):
                    for res_file in result:
                        image_id = conversation.cache_data(res_file.read())
                        conversation.add_message(
                            MessageRole.ASSISTANT,
                            content=f'I uploaded image with reference image_id is {image_id}'
                        )
                else:
                    image_id = conversation.cache_data(result.read())
                    conversation.add_message(
                        MessageRole.ASSISTANT,
                        content=f'I uploaded image with reference image_id is {image_id}'
                    )

            return 'true'
        except Exception as exc:
            await botnav.bot.send_message(message.chat.id, "Image generation failed, try again later")
            logger.exception(exc)
            return 'false'


    async def execute_openai(
        self,
        conversation: 'ConversationManager',
        botnav: TeleBotNav,
        message: Message,
        model: str,
        prompt: str,
        images: list[str] | None = None,
    ) -> str:
        image_bytes_list: list[BytesIO] = self.get_images_list(conversation, images)

        result_image = await conversation.openai_instance.image_generate(
            image_model=model,
            prompt=prompt,
            images=image_bytes_list
        )
        for image in result_image:
            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_photo(message.chat.id, image),
                'upload_photo'
            )

            conversation.add_message(
                MessageRole.ASSISTANT,
                content=f'I uploaded image with reference image_id is {image}'
            )
        return 'true'

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        if (
            'prompt' not in params or
            'botnav' not in params or
            'message' not in params
        ):
            return 'false'

        botnav: TeleBotNav = params['botnav']
        message: Message = params['message']
        prompt: str = params['prompt']
        images: list[str] | None = params.get('images', None)

        model = conversation.config.drawing_model

        try:
            if model in OPENAI_IMAGE_MODELS:
                return await self.execute_openai(conversation, botnav, message, model, prompt, images)

            return await self.execute_replicate(conversation, botnav, message, model, prompt, images)
        except Exception as exc:
            logger.exception(exc)
            return 'false'

class MemoryAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'memory'
    description = (
        'Saves facts about user as key-value pair to the user memory, where key is a short to ' +
        'represent it to user and value is the actual information to remember with ' + 
        'short description, always call it if user asks to remember something, returns true if successful'
    )
    schema = {
        'key': ToolParameter(
            name='key', description='The key is the short name to represent the memory, '
                'like language, timezone, birthdate etc. max 20 characters latin letters or _',
            ptype=str, required=True
        ),
        'value': ToolParameter(
            name='value', description='The value is the actual information to remember'
                'with short description within 1000 characters max',
            ptype=str, required=True
        )
    }

    def is_enabled(self, manager: ConversationManager, params: dict[str, Any]) -> bool:
        return manager.config.memory

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        if not params or not 'key' in params or not 'value' in params:
            return 'false'

        key = params['key']
        value = params['value']

        if len(key) > 20 or len(value) > 1000:
            return 'false'

        if not key.isascii() or not all(c.isalnum() or c == '_' for c in key):
            return 'false'

        conversation.add_memory(key, value)

        return 'true'

class FetchUrlAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'fetch_http_content'
    description = 'Fetches and returns the textual content of the specified HTTP or HTTPS URL, or ERROR if fetching fails'
    schema = {
        'url': ToolParameter(
            name='url', description='The URL of the webpage to fetch',
            ptype=str, required=True
        )
    }

    def is_enabled(self, manager: ConversationManager, params: dict[str, Any]) -> bool:
        return config.MCP_FETCH_URL is not None

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        if not params or 'url' not in params:
            return 'ERROR'

        url = params['url']

        try:
            async with sse_client(config.MCP_FETCH_URL) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool('fetch', arguments={'url': url, 'max_length': 100000})
                    return result.content[0].text
        except Exception as exc:
            logger.exception(exc)
            return 'ERROR'
        return 'ERROR'


class SubAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'subagent'
    _description = 'An autonomous agent that can use multiple tools to accomplish complex tasks, capabilities are: \n'
    description = ''
    schema = {
        'prompt': ToolParameter(
            name='prompt', description='A detailed prompt describing the task to '
                'be accomplished by the agent with all necessary context',
            ptype=str, required=True
        )
    }

    agent_tools: list[AgentTool] = []

    def __init__(self) -> None:
        super().__init__()

        self.agent_tools = [
           MemoryAgentTool(), FetchUrlAgentTool(), ImageGenerationAgentTool(),
            GetCurrentTimeAgentTool()
        ]

    def get_active_tools(self, manager: 'ConversationManager', params: dict[str, Any]) -> list[AgentTool]:
        active_tools = [tool for tool in self.agent_tools if tool.is_enabled(manager, params)]
        return active_tools

    def is_enabled(self, manager: 'ConversationManager', params: dict[str, Any]) -> bool:
        active_tools = self.get_active_tools(manager, params)
        self.description = self._description + '\n'.join([tool.description for tool in active_tools])
        return bool(active_tools)
            
    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        try:
            if not params or 'botnav' not in params or 'message' not in params or 'prompt' not in params:
                return 'ERROR'

            botnav: TeleBotNav = params['botnav']
            message: Message = params['message']
            prompt: str = params['prompt']

            active_tools = self.get_active_tools(conversation, params)
            sub_conversation = conversation.invoke_subagent()
            sub_conversation.set_config_param('tools', active_tools)
            sub_conversation.set_config_param('system_prompt', SUBAGENT_PROMPT)

            sub_conversation.add_message(MessageRole.USER, content=prompt)
            llm_gen = sub_conversation.make_request(extra_params={'botnav': botnav, 'message': message})

            async for _ in llm_gen:
                await botnav.send_chat_action(message.chat.id, 'typing')

            for msg in reversed(sub_conversation.messages):
                if msg.tool_call_id:
                    return msg.content
        except Exception as exc:
            logger.exception(exc)
            return 'ERROR'

        return 'ERROR'
