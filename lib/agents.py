from __future__ import annotations
from typing import TYPE_CHECKING, Type
import abc
from io import BytesIO
from datetime import datetime
from typing import Any
import json
import os
import uuid

from telebot import types
from telebot.types import Message, WebAppInfo
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


DEFAULT_DRAWING_MODEL = 'flux-2-max'


DIFFUSION_MODELS_IMAGE_FIELDS: dict[str, str] = {
    'flux-2-max': 'input_images',
    'nano-banana-2': 'image_input',
    'seedream-4.5': 'image_input',
    'seedream-5-lite': 'image_input',
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
    top_level_description: str | None = None

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
        user_id: int = params.get('user_id')

        if not user_id:
            return False

        return is_replicate_available(user_id)
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
        user_id: int,
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
            result = await replicate_execute_and_send(botnav, user_id, model, input_data)

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
            await botnav.bot.send_message(user_id, "Image generation failed, try again later")
            logger.exception(exc)
            return 'false'


    async def execute_openai(
        self,
        conversation: 'ConversationManager',
        botnav: TeleBotNav,
        user_id: int,
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
                user_id,
                botnav.bot.send_photo(user_id, image),
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
            'user_id' not in params
        ):
            return 'false'

        botnav: TeleBotNav = params['botnav']
        user_id = params['user_id']
        prompt: str = params['prompt']
        images: list[str] | None = params.get('images', None)

        model = conversation.config.drawing_model

        try:
            if model in OPENAI_IMAGE_MODELS:
                return await self.execute_openai(conversation, botnav, user_id, model, prompt, images)

            return await self.execute_replicate(conversation, botnav, user_id, model, prompt, images)
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


def _apps_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'webapp', 'apps'))


def _app_owner(app_id: str) -> str | None:
    """Extract owner user_id from app_id (format: {user_id}_{uuid})."""
    parts = app_id.split('_', 1)
    return parts[0] if len(parts) == 2 and parts[0].isdigit() else None


class GetWebAppSourceTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'get_web_app_source'
    top_level_description = ''
    description = (
        'Returns the HTML source code of a previously created web app by its app_id. '
        'Use this before editing an existing app (own app_id) or to fork another user\'s app as a starting point.'
    )
    schema = {
        'app_id': ToolParameter(
            name='app_id',
            description='The app_id of the app to read',
            ptype=str,
            required=True,
        ),
    }

    def is_enabled(self, manager: 'ConversationManager', params: dict[str, Any]) -> bool:
        return bool(getattr(config, 'WEBAPP_BASE_URL', None))

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        app_id: str | None = params.get('app_id')
        if not app_id:
            return json.dumps({'error': 'app_id is required'})

        index_path = os.path.join(_apps_root(), app_id, 'index.html')
        if not os.path.exists(index_path):
            return json.dumps({'error': f'App {app_id} not found'})

        try:
            with open(index_path, 'r', encoding='utf-8') as fh:
                return fh.read()
        except Exception as exc:
            logger.exception(exc)
            return json.dumps({'error': str(exc)})


class CreateWebAppAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'create_web_app'
    top_level_description = (
        'BUILT-IN web app builder: Create or edit web mini-apps/games/tools hosted inside this bot. '
        'When the user asks to create, modify, update, fix, or improve any app — '
        'delegate to the subagent with full details and context (what to build or change, existing app_id if editing). '
        'Do NOT search the internet for apps.'
    )
    @property
    def description(self) -> str:
        base = getattr(config, 'WEBAPP_BASE_URL', '').rstrip('/')
        return (
            'Creates or updates a web application from a single self-contained HTML file '
            '(with all JS and CSS embedded inline) and returns the hosted URL plus a Telegram share link. '
            'To create a new app omit app_id. '
            'To update YOUR OWN existing app pass its app_id (use get_web_app_source first). '
            'To build on another user\'s app, read it with get_web_app_source then call this without app_id to fork it. \n'
            'The html parameter must be a complete, standalone HTML document.\n'
            'NEVER forget to import telegram js library <script src="https://telegram.org/js/telegram-web-app.js"></script>\n\n'
            'BACKEND REST APIS (call these from JavaScript inside the web app):\n'
            f'Base URL: {base}\n'
            'All requests require header: X-Telegram-Init-Data: window.Telegram.WebApp.initData\n\n'
            f'POST {base}/api/llm\n'
            '  Body: {"messages": [{"role":"user","content":"..."}], "system":"optional", "model":"optional"}\n'
            '  Response: {"text": "..."}\n\n'
            f'POST {base}/api/replicate\n'
            '  Body: {"model": "stable-diffusion", "input": {"prompt": "..."}}\n'
            '  Available models: stable-diffusion, kandinsky, flux-2-max, nano-banana-2,\n'
            '    seedream-4.5, seedream-5-lite, qwen-image-edit, veo-3.1, veo-3.1-fast,\n'
            '    real-esrgan, speech-2.8-turbo\n'
            '  Response: {"output": "url" or ["url1", "url2"]}\n\n'
            f'GET {base}/api/models\n'
            '  Response: {"llm": [...], "replicate": [...]}\n'
            '  Lists only models the current user is permitted to use.\n\n'
            f'GET {base}/api/settings/get?app_id={{app_id}}\n'
            '  Returns all stored settings for this user as {"key": value, ...}. Empty object on first run.\n\n'
            f'POST {base}/api/settings/set?app_id={{app_id}}\n'
            '  Body: {"key": "settingName", "value": <any JSON value>}\n'
            '  Saves or overwrites a single setting. Response: {"ok": true}\n\n'
            'Settings storage example (put in DOMContentLoaded):\n'
            'const appId = window.location.pathname.split(\'/\')[2];\n'
            'const initData = window.Telegram.WebApp.initData;\n'
            '// Load:\n'
            f'const settings = await fetch(`{base}/api/settings/get?app_id=${{appId}}`,\n'
            '  {headers: {"X-Telegram-Init-Data": initData}}).then(r => r.json());\n'
            '// Save:\n'
            f'await fetch(`{base}/api/settings/set?app_id=${{appId}}`, {{\n'
            '  method: "POST",\n'
            '  headers: {"Content-Type": "application/json", "X-Telegram-Init-Data": initData},\n'
            '  body: JSON.stringify({key: "score", value: 42})\n'
            '});\n\n'
            'JavaScript fetch example:\n'
            'const res = await fetch(`' + base + '/api/llm`, {\n'
            '  method: "POST",\n'
            '  headers: {"Content-Type": "application/json",\n'
            '            "X-Telegram-Init-Data": window.Telegram.WebApp.initData},\n'
            '  body: JSON.stringify({messages: [{role:"user", content: prompt}]})\n'
            '});\n'
            'const {text} = await res.json();'
        )
    schema = {
        'html': ToolParameter(
            name='html',
            description='Complete standalone HTML document with all JS and CSS embedded inline',
            ptype=str,
            required=True,
        ),
        'title': ToolParameter(
            name='title',
            description='Short human-readable title for the app (used in the reply message)',
            ptype=str,
            required=False,
        ),
        'app_id': ToolParameter(
            name='app_id',
            description='Your own app_id to overwrite. Omit to create a new app.',
            ptype=str,
            required=False,
        ),
    }

    def is_enabled(self, manager: 'ConversationManager', params: dict[str, Any]) -> bool:
        return bool(getattr(config, 'WEBAPP_BASE_URL', None))

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        html: str | None = params.get('html')
        title: str = params.get('title', 'Web App')
        botnav: TeleBotNav | None = params.get('botnav')
        user_id = params.get('user_id')

        if not html:
            return json.dumps({'error': 'html parameter is required'})

        current_user_id = str(user_id) if user_id else None
        requested_app_id: str | None = params.get('app_id')

        if requested_app_id:
            requested_app_id = os.path.basename(requested_app_id)
            owner_id = _app_owner(requested_app_id)
            if owner_id != current_user_id:
                return json.dumps({
                    'error': 'Permission denied: you can only edit your own apps. '
                             'Use get_web_app_source + create_web_app (without app_id) to fork it.'
                })
            app_id = requested_app_id
        else:
            uid = current_user_id or 'anon'
            app_id = f"{uid}_{uuid.uuid4()}"

        apps_dir = os.path.join(_apps_root(), app_id)

        try:
            os.makedirs(apps_dir, exist_ok=True)
            with open(os.path.join(apps_dir, 'index.html'), 'w', encoding='utf-8') as fh:
                fh.write(html)
        except Exception as exc:
            logger.exception(exc)
            return json.dumps({'error': f'Failed to save app: {exc}'})

        base_url = config.WEBAPP_BASE_URL.rstrip('/')
        app_url = f"{base_url}/apps/{app_id}/index.html"

        share_link = None
        if botnav:
            try:
                bot_info = await botnav.bot.get_me()
                share_link = f"https://t.me/{bot_info.username}?start=app_{app_id}"
            except Exception as exc:
                logger.exception(exc)

        if botnav:
            markup = types.InlineKeyboardMarkup()
            markup.add(types.InlineKeyboardButton(
                text=f'🌐 Open {title}',
                web_app=WebAppInfo(url=app_url),
            ))

            await botnav.bot.send_message(
                user_id,
                f'✅ {title} is ready! {share_link}',
                reply_markup=markup
            )

        result: dict[str, Any] = {'app_id': app_id}
        if share_link:
            result['share_link'] = share_link
        return json.dumps(result)


class EditWebAppAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'edit_web_app'
    top_level_description = ''
    description = (
        'Applies targeted find-and-replace edits to an existing web app without rewriting the entire file. '
        'Use this instead of create_web_app when making small or medium changes to an app you already own. '
        'You MUST call get_web_app_source first to read the current source before constructing edits.\n\n'
        'The \'edits\' parameter must be a JSON string containing an array of edit objects:\n'
        '[{"old": "<exact substring to find>", "new": "<replacement string>"}, ...]\n\n'
        'Rules:\n'
        '- Each "old" must appear EXACTLY ONCE in the current file; duplicates are rejected.\n'
        '- Edits are applied in array order; later edits operate on already-patched content.\n'
        '- If any edit fails, NO changes are written (all-or-nothing).\n'
        '- To delete a block, set "new" to "".\n'
        '- Include enough surrounding context in "old" to make it unique (e.g. the full line).'
    )
    schema = {
        'app_id': ToolParameter(
            name='app_id',
            description='The app_id of your own app to edit.',
            ptype=str,
            required=True,
        ),
        'edits': ToolParameter(
            name='edits',
            description=(
                'A JSON string: array of {"old": "...", "new": "..."} objects. '
                'Each "old" must be a unique substring of the current file. '
                'Edits are applied in order.'
            ),
            ptype=str,
            required=True,
        ),
    }

    def is_enabled(self, manager: 'ConversationManager', params: dict[str, Any]) -> bool:
        return bool(getattr(config, 'WEBAPP_BASE_URL', None))

    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        app_id: str | None = params.get('app_id')
        if not app_id:
            return json.dumps({'error': 'app_id is required'})

        app_id = os.path.basename(app_id)
        current_user_id = str(params.get('user_id')) if params.get('user_id') else None
        owner_id = _app_owner(app_id)
        if owner_id != current_user_id:
            return json.dumps({
                'error': 'Permission denied: you can only edit your own apps. '
                         'Use get_web_app_source + create_web_app (without app_id) to fork it.'
            })

        index_path = os.path.join(_apps_root(), app_id, 'index.html')
        if not os.path.exists(index_path):
            return json.dumps({'error': f'App {app_id} not found'})

        try:
            with open(index_path, 'r', encoding='utf-8') as fh:
                original = fh.read()
        except Exception as exc:
            logger.exception(exc)
            return json.dumps({'error': f'Failed to read app: {exc}'})

        edits_raw: str | None = params.get('edits')
        if not edits_raw:
            return json.dumps({'error': 'edits parameter is required'})

        try:
            edits = json.loads(edits_raw)
        except json.JSONDecodeError as exc:
            return json.dumps({'error': f'edits must be a valid JSON array: {exc}'})

        if not isinstance(edits, list):
            return json.dumps({'error': 'edits must be a JSON array'})

        patched = original
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict) or 'old' not in edit or 'new' not in edit:
                return json.dumps({'error': f'Edit #{i + 1}: each item must have "old" and "new" keys'})
            old_str: str = edit['old']
            new_str: str = edit['new']
            count = patched.count(old_str)
            if count == 0:
                return json.dumps({'error': f'Edit #{i + 1}: "old" string not found in current content'})
            if count > 1:
                return json.dumps({
                    'error': f'Edit #{i + 1}: "old" string is ambiguous ({count} matches); '
                             'include more surrounding context to make it unique'
                })
            patched = patched.replace(old_str, new_str, 1)

        try:
            with open(index_path, 'w', encoding='utf-8') as fh:
                fh.write(patched)
        except Exception as exc:
            logger.exception(exc)
            return json.dumps({'error': f'Failed to save app: {exc}'})

        return json.dumps({'ok': True, 'app_id': app_id, 'edits_applied': len(edits)})


class SubAgentTool(AgentTool):
    type = 'function'
    providers = [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA, AIProvider.OPENROUTER]
    name = 'subagent'
    description = ''
    schema = {
        'prompt': ToolParameter(
            name='prompt', description='A detailed prompt describing the task to '
                'be accomplished by the agent with all necessary context',
            ptype=str, required=True
        )
    }

    agent_tools_classes: list[Type[AgentTool]] = []
    agent_tools: list[AgentTool] = []

    def __init__(self) -> None:
        super().__init__()
        self.init_tools()

    def init_tools(self) -> None:
        self.agent_tools = [
            tool_class() for tool_class in self.agent_tools_classes
        ]

    def get_active_tools(self, manager: 'ConversationManager', params: dict[str, Any]) -> list[AgentTool]:
        active_tools = [tool for tool in self.agent_tools if tool.is_enabled(manager, params)]
        return active_tools
            
    async def execute(
        self,
        conversation: 'ConversationManager',
        params: dict[str, Any],
    ) -> str:
        try:
            if not params or 'botnav' not in params or 'user_id' not in params or 'prompt' not in params:
                return 'ERROR'

            botnav: TeleBotNav = params['botnav']
            user_id = params['user_id']
            processing_callback = params.get('processing_callback')
            prompt: str = params['prompt']

            active_tools = self.get_active_tools(conversation, params)
            sub_conversation = conversation.invoke_subagent()
            sub_conversation.set_config_param('tools', active_tools)
            sub_conversation.set_config_param('system_prompt', SUBAGENT_PROMPT)

            sub_conversation.add_message(MessageRole.USER, content=prompt)
            llm_gen = sub_conversation.make_request(
                extra_params={
                    'botnav': botnav,
                    'user_id': user_id,
                    'processing_callback': processing_callback
                }
            )

            async for _ in llm_gen:
                await botnav.send_chat_action(user_id, 'typing')

            for msg in reversed(sub_conversation.messages):
                if msg.tool_call_id:
                    return msg.content
        except Exception as exc:
            logger.exception(exc)
            return 'ERROR'

        return 'ERROR'


class SubAgentCommonTool(SubAgentTool):
    _description = 'An autonomous agent that can use multiple tools to accomplish complex tasks, capabilities are: \n'

    agent_tools_classes = [MemoryAgentTool, FetchUrlAgentTool, ImageGenerationAgentTool, GetCurrentTimeAgentTool]

    def is_enabled(self, manager: 'ConversationManager', params: dict[str, Any]) -> bool:
        active_tools = self.get_active_tools(manager, params)
        self.description = self._description + '\n'.join(
            [tool.top_level_description if tool.top_level_description is not None else tool.description for tool in active_tools]
        )

        return bool(active_tools)


class SubAgentWebAppTool(SubAgentCommonTool):
    name = 'subagent_webapp'
    description = (
        'BUILT-IN web app builder: Create or edit web mini-apps/games/tools hosted inside this bot. '
        'When the user asks to create, modify, update, fix, or improve any app — '
        'delegate to the subagent with full details and context (what to build or change, existing app_id if editing). '
        'Do NOT search the internet for apps.\n'
        'Apps can use backend APIs from JavaScript: LLM text generation, Replicate image/video/audio generation models, '
        'and per-user settings storage (save/load arbitrary JSON values).'
    )

    agent_tools_classes = [CreateWebAppAgentTool, GetWebAppSourceTool, EditWebAppAgentTool]
