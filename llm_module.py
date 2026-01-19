import asyncio
import base64
from typing import Callable, Coroutine, Type
import functools
from typing import Any
from io import BytesIO
import json
import os

from telebot.types import Message
from openai import RateLimitError
from pydub import AudioSegment
import telegramify_markdown
from telebot.asyncio_helper import ApiTelegramException

import config
from lib.llm import AIProvider, ConversationManager, MessageRole, MessageType, LLMModel
from lib.llm import openai_instance
from lib.llm import LLM_RESPONSE_TIMEOUT
from lib.permissions import is_llm_model_allowed, is_permitted, is_replicate_available
from lib.utils import ConvEncoder, MessageSplitter
from lib.chat_roles import get_chat_roles
from lib.agents import OpenAiWebSearchAgentTool, AntropicWebSearchAgentTool, SubAgentTool, AgentTool
from lib.agents import DEFAULT_DRAWING_MODEL, DIFFUSION_MODELS_IMAGE_FIELDS, OPENAI_IMAGE_MODELS
from telebot_nav import TeleBotNav
from logger import logger
from help_content import HELP_CONTENT


DEFAULT_MODEL = 'claude-haiku-4-5'

AVAILABLE_LLM_MODELS = {
    'gpt-4.1-mini': LLMModel(AIProvider.OPENAI, 'gpt-4.1-mini', thinking=False),
    'o4-mini': LLMModel(AIProvider.OPENAI, 'o4-mini'),
    'gpt-4.1': LLMModel(AIProvider.OPENAI, 'gpt-4.1', thinking=False),
    'gpt-5-nano': LLMModel(AIProvider.OPENAI, 'gpt-5-nano'),
    'gpt-5': LLMModel(AIProvider.OPENAI, 'gpt-5'),
    'gpt-5.2': LLMModel(AIProvider.OPENAI, 'gpt-5.2'),
    'o3': LLMModel(AIProvider.OPENAI, 'o3'),
    'claude-haiku-4-5': LLMModel(AIProvider.ANTHROPIC, 'claude-haiku-4-5'),
    'claude-sonnet-4-5': LLMModel(AIProvider.ANTHROPIC, 'claude-sonnet-4-5'),
    'claude-opus-4-1': LLMModel(AIProvider.ANTHROPIC, 'claude-opus-4-1'),
    'gpt-oss:20b': LLMModel(AIProvider.OLLAMA, 'gpt-oss:20b', vision=False),
    'gemma3:27b': LLMModel(AIProvider.OLLAMA, 'gemma3:27b', tool_calling=False),
    'qwen3:32b': LLMModel(AIProvider.OLLAMA, 'qwen3:32b', vision=False),
    'granite4:small-h': LLMModel(AIProvider.OLLAMA, 'granite4:small-h', thinking=False, vision=False),
    'ministral-3:14b': LLMModel(AIProvider.OLLAMA, 'ministral-3:14b', thinking=False),
    'qwen3:4b-instruct': LLMModel(AIProvider.OLLAMA, 'qwen3:4b-instruct', vision=False)
}

CHAT_ROLES = get_chat_roles(AVAILABLE_LLM_MODELS, DEFAULT_MODEL)

DEFAULT_TOOLS: list[Type[AgentTool]] = [
    OpenAiWebSearchAgentTool, AntropicWebSearchAgentTool, SubAgentTool
]

SPEECH_MODELS = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer"
]

DEFAULT_ROLE = 'Assistant'
DEFAULT_MAX_TOKENS = 8192

OLD_CONVERSATION_TIMEOUT = 3600 # 1 hour

CONV_PATH = os.path.join(os.path.dirname(__file__), "conv")

conversations: dict[int, ConversationManager] = {}


async def openai_send_speech(botnav: TeleBotNav, message: Message, text: str, speech_model: str) -> None:
    audio_file = await botnav.await_coro_sending_action(
        message.chat.id,
        SpeechHelper.tts_generate_audio(text, speech_model),
        'record_voice'
    )

    await botnav.bot.send_voice(message.chat.id, audio_file)


def get_model_title(model: LLMModel) -> str:
    title = model.name + ' '
    if model.thinking:
        title += "🧠"
    if model.tool_calling:
        title += "🔧"
    if model.vision:
        title += "👁️"
    return title


async def send_md_formated_or_plain(botnav: TeleBotNav, message: Message, text: str) -> None:
    prettify_answers: bool = message.state_data.get('prettify_answers', True)
    speech_model: str = message.state_data.get('speech_model', 'Off')

    if speech_model != 'Off':
        await openai_send_speech(botnav, message, text, speech_model)
        return

    if not prettify_answers:
        await botnav.bot.send_message(message.chat.id, text)
        return
    try:
        md_text = telegramify_markdown.standardize(text)
        await botnav.bot.send_message(message.chat.id, md_text, parse_mode='MarkdownV2')
    except Exception:
        logger.exception("Failed to send md formatted message, sending plain text")
        await botnav.bot.send_message(message.chat.id, text)


def get_available_models(botnav: TeleBotNav, message: Message) -> dict[str, LLMModel]:
    available_models: dict[str, LLMModel] = {}
    for name, model in AVAILABLE_LLM_MODELS.items():
        if model.provider == AIProvider.OPENAI and not config.OPENAI_API_KEY:
            continue
        if model.provider == AIProvider.ANTHROPIC and not config.ANTHROPIC_API_KEY:
            continue
        if model.provider == AIProvider.OLLAMA and not config.OLLAMA_HOST:
            continue

        if not is_llm_model_allowed(botnav, message, model):
            continue

        available_models[name] = model
    return available_models


def get_or_create_conversation(botnav: TeleBotNav, message: Message) -> ConversationManager:
    user_id = botnav.get_user(message).id
    if user_id not in conversations:
        conversations[user_id] = get_new_conversation_manager(botnav, message)
    return conversations[user_id]


def save_conversation_to_file(user: str, conversation: ConversationManager) -> None:
    if not os.path.exists(CONV_PATH):
        os.mkdir(CONV_PATH)
    json.dump(
        conversation.dump(),
        open(os.path.join(CONV_PATH, f"{user}_{conversation.id}.json"), "w"),
        cls=ConvEncoder
    )


def set_role(botnav: TeleBotNav, message: Message, conversation: ConversationManager, role: str) -> None:
    if role not in CHAT_ROLES:
        role = DEFAULT_ROLE

    system_prompt: str = CHAT_ROLES[role]['system_prompt']
    conversation.set_config_param('system_prompt', system_prompt)
    message.state_data['current_role'] = role

    if 'one_off' in CHAT_ROLES[role]:
        conversation.set_config_param('one_off', CHAT_ROLES[role]['one_off'])
    if 'thinking' in CHAT_ROLES[role]:
        conversation.set_config_param('thinking', CHAT_ROLES[role]['thinking'])
    if 'model' in CHAT_ROLES[role]:
        model: LLMModel = CHAT_ROLES[role]['model']

        if is_llm_model_allowed(botnav, message, model):
            conversation.set_model(model)
        else:
            conversation.set_model(AVAILABLE_LLM_MODELS[DEFAULT_MODEL])
    else:
        conversation.set_model(AVAILABLE_LLM_MODELS[DEFAULT_MODEL])


def get_new_conversation_manager(botnav: TeleBotNav, message: Message) -> ConversationManager:
    manager = ConversationManager()

    set_role(botnav, message, manager, DEFAULT_ROLE)
    manager.set_config_param('max_tokens', DEFAULT_MAX_TOKENS)

    if is_permitted(botnav, message, 'can_use_tools'):
        tools = [x() for x in DEFAULT_TOOLS]
        manager.set_config_param('tools', tools)

    if is_permitted(botnav, message, 'can_use_memory_tool'):
        manager.set_config_param('memory', True)

    manager.set_user_id(botnav.get_user(message).id)
    manager.set_config_param('drawing_model', DEFAULT_DRAWING_MODEL)

    return manager


def encode_jpg_image(image: bytes):
    return "data:image/jpeg;base64," + base64.b64encode(image).decode('utf-8')
 

class SpeechHelper:
    @classmethod
    def get_mp3_from_ogg(cls, file_content: bytes) -> BytesIO:
        file = BytesIO(file_content)
        file.seek(0)
        ogg: AudioSegment = AudioSegment.from_ogg(file)
        mp3 = BytesIO()
        ogg.export(mp3, format='mp3')
        mp3.seek(0)
        return mp3

    @classmethod
    async def extract_text_from_voice(cls, botnav: TeleBotNav, message: Message) -> str | None:
        if not message.voice:
            return

        file_info = await botnav.bot.get_file(message.voice.file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        if file_info.file_path.endswith('.ogg') or file_info.file_path.endswith('.oga'):
            file = await asyncio.to_thread(cls.get_mp3_from_ogg, file_content)
            file.name = 'voice.mp3'
        elif file_info.file_path.endswith('.mp3'):
            file = BytesIO(file_content)
            file.name = 'voice.mp3'
        else:
            raise ValueError("Unsupported audio format")

        text = await openai_instance.whisper_transcribe(file)
        return text

    @classmethod
    async def tts_generate_audio(cls, text: str, voice: str) -> BytesIO:
        response = await openai_instance.client.audio.speech.create(
            model='gpt-4o-mini-tts',
            input=text,
            voice=voice,
            response_format='mp3'
        )

        audio_file = BytesIO(response.read())
        audio_file.name = 'speech.mp3'
        audio_file.seek(0)
        return audio_file


class LLMRouter:
    @classmethod
    async def reset_conversation(cls, botnav: TeleBotNav, message: Message) -> None:
        user_id = botnav.get_user(message).id
        conversations[user_id] = get_new_conversation_manager(botnav, message)

        if message.state_data:
            message.state_data.pop('delayed_message', None)
            message.state_data.pop('prettify_answers', None)
            message.state_data.pop('speech_model', None)

        await cls.show_chat_options(botnav, message)

    ## Options handlers
    @classmethod
    async def set_one_off(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        if conversation.config.one_off:
            conversation.set_config_param('one_off', False)
        else:
            conversation.set_config_param('one_off', True)

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def switch_delayed_message_mode(cls, botnav: TeleBotNav, message: Message) -> None:
        if not message.state_data.get('delayed_message', False):
            message.state_data['delayed_message'] = True
        else:
            message.state_data['delayed_message'] = False

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def request_set_max_tokens(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.edit_message_text("Set maximum tokens in response"
            " randomnes in its responses)", message.chat.id, message.message_id)

        botnav.set_next_handler(message, cls.set_max_tokens)

    @classmethod
    async def set_max_tokens(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        try:
            if not message.text:
                raise ValueError()

            max_tokens = int(message.text)

            if max_tokens <= 0:
                raise ValueError()

            conversation.set_config_param('max_tokens', max_tokens)
            print(max_tokens)
            await cls.show_chat_options(botnav, message)
        except (ValueError, TypeError):
            await botnav.bot.send_message(message.chat.id, "Invalid max tokens value, must be a positive integer")
            return

    @classmethod
    async def request_set_system_prompt(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        await botnav.bot.edit_message_text(
            f"Set the prompt of your opponent, current prompt: \n{conversation.config.system_prompt}",
            message.chat.id,
            message.message_id
        )

        botnav.set_next_handler(message, cls.set_system_prompt)

    @classmethod
    async def set_system_prompt(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        conversation.set_config_param('system_prompt', message.text)

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def clean_conversation(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        last_len = len(conversation.messages)
        conversation.clear_conversation()

        if len(conversation.messages) == last_len:
            return

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def show_models_list(cls, botnav: TeleBotNav, message: Message) -> None:
        buttons: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            f"{get_model_title(model)} ({model.provider.value})": functools.partial(
                cls.switch_llm_model,
                model
            ) for name, model in get_available_models(botnav, message).items()
        }
        buttons['⬅️ Back'] = cls.show_chat_options

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            message_to_rewrite=message,
            text='Available models:',
            row_width=1,
        )

    @classmethod
    async def switch_llm_model(cls, model: LLMModel, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)

        if is_llm_model_allowed(botnav, message, model):
            conversation.set_model(model)

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def show_roles_list(cls, botnav: TeleBotNav, message: Message) -> None:
        buttons: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            x: functools.partial(cls.set_role, x) for x in CHAT_ROLES.keys()
        }
        buttons['⬅️ Back'] = cls.show_chat_options

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            message_to_rewrite=message,
            text='Available roles:',
            row_width=2,
        )

    @classmethod
    async def set_role(cls, role: str, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        set_role(botnav, message, conversation, role)

        conversation.set_config_param('system_prompt', CHAT_ROLES[role]['system_prompt'])
        if 'one_off' in CHAT_ROLES[role]:
            conversation.set_config_param('one_off', CHAT_ROLES[role]['one_off'])

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def show_memory_list(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)

        if not conversation.memory:
            conversation.load_memory()

        memory_status = "✅" if conversation.config.memory else "❌"
        is_empty = not conversation.memory

        buttons: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            x: functools.partial(cls.show_memory, x) for x in conversation.memory.keys()
        }
        buttons[f'🧠 Toggle Memory {memory_status}'] = cls.toggle_memory
        buttons['⬅️ Back'] = cls.show_chat_options

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            row_width=1,
            message_to_rewrite=message,
            text='User memory:' if not is_empty else 'User memory is empty.'
        )

    @classmethod
    async def show_memory(cls, key: str, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)

        if not conversation.memory or key not in conversation.memory:
            return

        value = conversation.memory.get(key, '')

        await botnav.print_buttons(
            message.chat.id,
            {
                '🗑️ Delete': functools.partial(cls.delete_memory, key),
                '⬅️ Back': cls.show_memory_list
            },
            row_width=1,
            message_to_rewrite=message,
            text=f"Memory key: {key}\nValue: {value}"
        )

    @classmethod
    async def delete_memory(cls, key: str, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)

        if not conversation.memory or key not in conversation.memory:
            return

        if key in conversation.memory:
            conversation.delete_memory(key)

        await botnav.bot.send_message(
            message.chat.id,
            f"Memory key: {key} deleted"
        )
        await cls.show_memory_list(botnav, message)

    @classmethod
    async def toggle_memory(cls, botnav: TeleBotNav, message: Message) -> None:
        if not is_permitted(botnav, message, 'can_use_memory_tool'):
            return

        conversation = get_or_create_conversation(botnav, message)
        conversation.config.memory = not conversation.config.memory

        await cls.show_memory_list(botnav, message)

    @classmethod
    async def toggle_prettyfy_answers(cls, botnav: TeleBotNav, message: Message) -> None:
        if not message.state_data.get('prettify_answers', True):
            message.state_data['prettify_answers'] = True
        else:
            message.state_data['prettify_answers'] = False
        await cls.show_chat_options(botnav, message)

    @classmethod
    async def show_speech_models_list(cls, botnav: TeleBotNav, message: Message) -> None:
        buttons: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            f"{name}": functools.partial(
                cls.switch_speech_model,
                name
            ) for name in SPEECH_MODELS
        }

        buttons['Off'] = functools.partial(
            cls.switch_speech_model,
            'Off'
        )
        buttons['⬅️ Back'] = cls.show_chat_options

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            message_to_rewrite=message,
            text='Available speech models:',
            row_width=2,
        )

    @classmethod
    async def show_drawing_models_list(cls, botnav: TeleBotNav, message: Message) -> None:
        buttons: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            f"{name}": functools.partial(
                cls.switch_drawing_model,
                name
            ) for name in [*DIFFUSION_MODELS_IMAGE_FIELDS.keys(), *OPENAI_IMAGE_MODELS]
        }

        buttons['⬅️ Back'] = cls.show_chat_options

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            message_to_rewrite=message,
            text='Available drawing models:',
            row_width=2,
        )

    @classmethod
    async def switch_drawing_model(cls, model_name: str, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        conversation.set_config_param('drawing_model', model_name)
        await cls.show_chat_options(botnav, message)

    @classmethod
    async def switch_speech_model(cls, model_name: str, botnav: TeleBotNav, message: Message) -> None:
        message.state_data['speech_model'] = model_name
        await cls.show_chat_options(botnav, message)

    @classmethod
    async def show_help(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.send_message(
            message.chat.id,
            HELP_CONTENT,
            parse_mode='Markdown'
        )

    @classmethod
    async def continue_with_new_conversation(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        conversation.clear_conversation(older_than=OLD_CONVERSATION_TIMEOUT)
        await cls.get_reply(botnav, message)

    @classmethod
    async def continue_with_old_conversation(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        conversation.refresh_last_message_time()
        await cls.get_reply(botnav, message)

    @classmethod
    async def show_chat_options(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        one_off_status = "✅" if conversation.config.one_off else "❌"
        max_tokens = conversation.config.max_tokens
        send_mode = "✅" if message.state_data.get('delayed_message', False) else "❌"
        conversation_length = len(conversation.messages)
        role = message.state_data.get('current_role', DEFAULT_ROLE)
        model = conversation.config.model
        memory_permited = is_permitted(botnav, message, 'can_use_memory_tool')
        memory_enabled = "✅" if conversation.config.memory else "❌"
        prettyfy_answers = "✅" if message.state_data.get('prettify_answers', True) else "❌"
        speech_model = message.state_data.get('speech_model', 'Off')
        drawing_on = conversation.config.drawing_model and is_replicate_available(botnav, message)
        model_title = get_model_title(model)

        try:
            await botnav.print_buttons(
                message.chat.id,
                {
                    f'🎨 Prettify answers {prettyfy_answers}': cls.toggle_prettyfy_answers,
                    f'🎯 One Off {one_off_status}': cls.set_one_off,
                    f'📤 Send upon command {send_mode}': cls.switch_delayed_message_mode,
                    f'🔢 Max tokens({max_tokens})': cls.request_set_max_tokens,
                    '🏁 Set system prompt': cls.request_set_system_prompt,
                    f'🤖 Model({model_title})': cls.show_models_list,
                    f'👥 Role({role})': cls.show_roles_list,
                    f'💾 Memory {memory_enabled}': cls.show_memory_list if memory_permited else None,
                    f'🗣️ Speech Model({speech_model})': cls.show_speech_models_list,
                    f'🎨 Drawing Model({conversation.config.drawing_model})': cls.show_drawing_models_list if drawing_on else None,
                    '🔄 Reset conversation': cls.reset_conversation,
                    f'🧹 Clean conversation({conversation_length})': cls.clean_conversation,
                    '❓ Help': cls.show_help,
                },
                row_width=1,
                message_to_rewrite=message if message.from_user.is_bot else None,
                text='Send message to chat, additional options:'
            )
        except ApiTelegramException as exc:
            if 'Bad Request: message is not modified' in exc.description:
                return
            raise exc

    @classmethod
    async def chat_message_handler(cls, botnav: TeleBotNav, message: Message):
        if message.content_type not in ('text', 'voice', 'photo'):
            return

        text = ''
        image = None

        if message.content_type == 'voice':
            result = await botnav.await_coro_sending_action(
                message.chat.id,
                SpeechHelper.extract_text_from_voice(botnav, message),
                'typing'
            )

            if result:
                text = result
                await botnav.bot.send_message(message.chat.id, f'You said: "{text}"')

        if message.content_type == 'text' and message.text:
            text = message.text

        if message.content_type == 'photo' and message.photo:
            if message.caption:
                text = message.caption
            file_info = await botnav.bot.get_file(message.photo[-1].file_id)
            image = await botnav.bot.download_file(file_info.file_path)

        if not text and not image:
            return

        conversation = get_or_create_conversation(botnav, message)
        is_old_conversation = conversation.is_old_conversation(OLD_CONVERSATION_TIMEOUT)

        if text:
            conversation.add_message(
                role=MessageRole.USER,
                content=text
            )

        if image:
            encoded_image = encode_jpg_image(image)
            image_id = conversation.cache_data(image)

            conversation.add_message(
                role=MessageRole.USER,
                content=encoded_image,
                content_type=MessageType.IMAGE,
                image_id=image_id
            )

        if is_old_conversation:
            await botnav.print_buttons(
                message.chat.id,
                {
                    '🧹 New': cls.continue_with_new_conversation,
                    '📩 Old': cls.continue_with_old_conversation,
                },
                'Do you want to continue the old conversation or start a new one?'
            )
            return

        if message.state_data.get('delayed_message', False):
            await botnav.print_buttons(
                message.chat.id,
                {
                    '📩 Send': cls.get_reply,
                },
                'Press to send'
            )
            return

        if message.content_type == 'photo'and message.media_group_id:
            if message.caption is None:
                return # Send request for the main image in media group

            # wait for possible other images in media group
            await botnav.await_coro_sending_action(
                message.chat.id,
                asyncio.sleep(3),
                'typing'
            )

        await asyncio.wait_for(cls.get_reply(botnav, message), timeout=LLM_RESPONSE_TIMEOUT)

    @classmethod
    async def get_reply(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.send_chat_action(message.chat.id, 'typing')

        try:
            message_splitter = MessageSplitter(2000)
            conversation = get_or_create_conversation(botnav, message)

            llm_gen = conversation.make_request(extra_params={'botnav': botnav, 'message': message})

            async for reply in llm_gen:
                await botnav.send_chat_action(message.chat.id, 'typing')

                # Flush command
                if reply is None:
                    for msg in message_splitter.flush():
                        await send_md_formated_or_plain(botnav, message, msg)
                    continue

                msg = message_splitter.add(reply)

                if msg:
                    await send_md_formated_or_plain(botnav, message, msg)

            for msg in message_splitter.flush():
                if msg:
                    await send_md_formated_or_plain(botnav, message, msg)

            user = botnav.get_user(message)

            save_conversation_to_file(
                f'{user.id}_{user.username}',
                conversation
            )
        except RateLimitError as exc:
            await botnav.bot.send_message(message.chat.id, 'OpenAi servers are overloaded, try again later')
            logger.exception(exc)
            await cls.reset_conversation(botnav, message)
        except Exception as exc:
            if getattr(exc, 'code', None) == 'context_length_exceeded':
                await botnav.bot.send_message(message.chat.id, getattr(exc, 'user_message', "Something went wrong, try again later"))
                await cls.reset_conversation(botnav, message)
                return

            await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
            logger.exception(exc)
            message.state_data.clear()

    @classmethod
    async def run(cls, botnav: TeleBotNav, message: Message) -> None:
        botnav.wipe_commands(message, preserve=['start', 'openai'])
        botnav.add_command(message, 'chat_gpt_reset', '🔄 Reset conversation', cls.reset_conversation)
        botnav.add_command(message, 'chat_gpt_clean', '🧹 Clean conversation', cls.clean_conversation)
        botnav.add_command(message, 'chat_gpt_options', '⚙️ Chat gpt Options', cls.show_chat_options)
        botnav.set_default_handler(message, cls.chat_message_handler)
        botnav.clean_next_handler(message)
        get_or_create_conversation(botnav, message)
        await botnav.send_commands(message)
        await cls.show_chat_options(botnav, message)
