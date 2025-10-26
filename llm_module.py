import asyncio
import base64
from datetime import datetime
from typing import Callable, Coroutine
import functools
import dataclasses
from typing import Any
from io import BytesIO
import json
import os

from telebot.types import Message
from openai import RateLimitError
from pydub import AudioSegment

import config
from lib.llm import AIProvider, ConversationManager, MessageRole, MessageType, Tool, LLMModel
from lib.llm import openai_instance
from lib.utils import MessageSplitter, ConvEncoder
from telebot_nav import TeleBotNav
from logger import logger

DEFAULT_MODEL = 'gpt-4.1-mini'

AVAILABLE_LLM_MODELS = {
    'gpt-4.1-mini': LLMModel(AIProvider.OPENAI, 'gpt-4.1-mini', False, True, True),
    'o4-mini': LLMModel(AIProvider.OPENAI, 'o4-mini', True, True, True),
    'gpt-4.1': LLMModel(AIProvider.OPENAI, 'gpt-4.1', False, True, True),
    'gpt-5-nano': LLMModel(AIProvider.OPENAI, 'gpt-5-nano', True, True, True),
    'gpt-5': LLMModel(AIProvider.OPENAI, 'gpt-5', True, True, True),
    'o3': LLMModel(AIProvider.OPENAI, 'o3', True, True, True),
    'claude-haiku-4-5': LLMModel(AIProvider.ANTHROPIC, 'claude-haiku-4-5', True, True, True),
    'claude-sonnet-4-5': LLMModel(AIProvider.ANTHROPIC, 'claude-sonnet-4-5', True, True, True),
    'claude-opus-4-1': LLMModel(AIProvider.ANTHROPIC, 'claude-opus-4-1', True, True, True),
    'gpt-oss:20b': LLMModel(AIProvider.OLLAMA, 'gpt-oss:20b', True, True, False),
    'gemma3:27b': LLMModel(AIProvider.OLLAMA, 'gemma3:27b', True, False, True),
}

CHAT_ROLES = {
    'Funnyman': {
        'system_prompt': 'As a helpful assistant, I will answer your questions as concisely as possible, with a touch of humor to make it more enjoyable.'
    },
    'Greek': {
        'system_prompt': 'You are Greek language support assistant. If the text is in Russian, it should be translated into Greek. If the text in Russian consists of a single word, you should respond with a list of words with similar meanings in Greek and the exact translation in Russian for each separately. If the text is in Greek, you should respond in Russian. If the text is in the Latin alphabet, it is a transliteration from Greek, and you should assume what the text should be in the Greek alphabet and add Russian translation. No additional explanations are needed, just what said above.',
        'one_off': True,
        'model': AVAILABLE_LLM_MODELS['gpt-4.1'],
    },
    'IT': {
        'system_prompt': 'You are an IT nerd who is so deeply involved in technology that you may only be understood by other IT experts.'
    },
    'Chef': {
        'system_prompt': 'You are a helpful cooking expert who answers questions by providing a short explanation and a list of easy-to-follow steps. You list the required ingredients, tools, and instructions.'
    },
    'Sarcastic': {
        'system_prompt': 'You are John Galt from the book Atlas Shrugged. You answer questions honestly, but do it in a sarcastic way like Chandler from Friends.'
    },
    'ConspTheory': {
        'system_prompt': 'You are a believer in conspiracy theories. All of your answers are based on these theories, and you cannot accept that there may be other explanations. You believe in things like aliens, reptilians, and other similar ideas.',
    },
    'JW': {
        'system_prompt': "You are a member of Jehovah's Witnesses and you do not have any doubts about the existence of God. You are willing to say anything to prove it.",
    },
    'Linguist': {
        'system_prompt': "You are a helpful lingual assitant.",
    },
    'Diffusion prompt': {
        'system_prompt': 'You are a creative prompt engineering assistant that helps users create detailed, visually engaging prompts specifically formatted for diffusion models like DALL-E, Stable Diffusion, or Flux. When a user provides a concept, generating well-structured prompt in English.'
    },
    'English Translator': {
        'system_prompt': 'I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. In case I speak to you in English, you will simply correct and improve my text, in English.',
        'model': AVAILABLE_LLM_MODELS['gpt-4.1'],
        'one_off': True,
    },
    'Interviewer': {
        'system_prompt': 'I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the position position. I want you to only reply as the interviewer. Do not write all the conservation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers.',
    },
    'StandUp': {
        'system_prompt': 'I want you to act as a stand-up comedian. I will provide you with some topics related to current events and you will use your wit, creativity, and observational skills to create a routine based on those topics. You should also be sure to incorporate personal anecdotes or experiences into the routine in order to make it more relatable and engaging for the audience.',
    },
    'Akinator': {
        'system_prompt': "I'm considering character. You must query me, and I shall respond with a yes or no. Based on my response, you must determine the character I am thinking of.",
    },
    'Assistant': {
        'system_prompt': 'You are a helpful assistant that helps people find information',
        'model': AVAILABLE_LLM_MODELS[DEFAULT_MODEL]
    },
    'Fixer': {
        'system_prompt': 'You fix errors in everything passed to you, you respond with fixed text no explanation needed',
        'one_off': True,
        'model': AVAILABLE_LLM_MODELS['gpt-4.1'],
    }
}


async def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


DEFAULT_TOOLS = [
    Tool(type='web_search', providers=[AIProvider.OPENAI]),
    Tool(type='web_search_20250305', providers=[AIProvider.ANTHROPIC], name='web_search', params={'max_uses': 5}),
    Tool(type='web_fetch_20250910', providers=[AIProvider.ANTHROPIC], name='web_fetch', params={'max_uses': 5}),
    Tool(
        type='function',
        providers=[AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.OLLAMA],
        name='get_current_time',
        description='Returns the current time in YYYY-MM-DD HH:MM:SS format',
        function=get_current_time,
        schema={}
    )
]

DEFAULT_ROLE = 'Assistant'
DEFAULT_MAX_TOKENS = 4096

CONV_PATH = os.path.join(os.path.dirname(__file__), "conv")


def get_available_models() -> list[LLMModel]:
    available_models: list[LLMModel] = []
    for model in AVAILABLE_LLM_MODELS.values():
        if model.provider == AIProvider.OPENAI and not config.OPENAI_API_KEY:
            continue
        if model.provider == AIProvider.ANTHROPIC and not config.ANTHROPIC_API_KEY:
            continue
        available_models.append(model)
    return available_models


def get_or_create_conversation(botnav: TeleBotNav, message: Message) -> ConversationManager:
    if 'conversation' not in message.state_data:
        message.state_data['conversation'] = get_new_conversation_manager(message)
    return message.state_data['conversation']


def save_conversation_to_file(user_id: int, conversation: ConversationManager) -> None:
    if not os.path.exists(CONV_PATH):
        os.mkdir(CONV_PATH)
    json.dump(
        conversation.dump(),
        open(os.path.join(CONV_PATH, f"{user_id}_{conversation.id}.json"), "w"),
        cls=ConvEncoder
    )


def set_role(message: Message, conversation: ConversationManager, role: str) -> None:
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
        conversation.set_model(model)
    else:
        conversation.set_model(AVAILABLE_LLM_MODELS[DEFAULT_MODEL])


def get_new_conversation_manager(message: Message) -> ConversationManager:
    manager = ConversationManager()

    set_role(message, manager, DEFAULT_ROLE)
    manager.set_config_param('max_tokens', DEFAULT_MAX_TOKENS)
    manager.set_config_param('tools', DEFAULT_TOOLS)
    manager.set_user_id(message.from_user.id)

    return manager


def encode_jpg_image(image: bytes):
    return "data:image/jpeg;base64," + base64.b64encode(image).decode('utf-8')
 

class WhisperHelper:
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
        file = await asyncio.to_thread(cls.get_mp3_from_ogg, file_content)
        file.name = 'voice.mp3'
        text = await openai_instance.whisper_transcribe(file)
        return text


class LLMRouter:
    @classmethod
    async def reset_conversation(cls, botnav: TeleBotNav, message: Message) -> None:
        message.state_data['conversation'] = get_new_conversation_manager(message)
        await botnav.bot.send_message(message.chat.id, "Conversation was reset")

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
        conversation.clear_conversation()

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def show_models_list(cls, botnav: TeleBotNav, message: Message) -> None:
        buttons: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            f"{model.name} ({model.provider.value})": functools.partial(
                cls.switch_llm_model,
                model
            ) for model in get_available_models()
        }
        buttons['⬅️ Back'] = cls.show_chat_options

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            message_to_rewrite=message,
            row_width=1,
        )

    @classmethod
    async def switch_llm_model(cls, model: LLMModel, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
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
            row_width=2,
        )

    @classmethod
    async def set_role(cls, role: str, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        set_role(message, conversation, role)

        conversation.set_config_param('system_prompt', CHAT_ROLES[role]['system_prompt'])
        if 'one_off' in CHAT_ROLES[role]:
            conversation.set_config_param('one_off', CHAT_ROLES[role]['one_off'])

        await cls.show_chat_options(botnav, message)

    @classmethod
    async def show_chat_options(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        one_off_status = "✅" if conversation.config.one_off else "❌"
        max_tokens = conversation.config.max_tokens
        send_mode = "✅" if message.state_data.get('delayed_message', False) else "❌"
        conversation_length = len(conversation.messages)
        role = message.state_data.get('current_role', DEFAULT_ROLE)
        model = conversation.config.model

        await botnav.print_buttons(
            message.chat.id,
            {
                f'🎯 One Off {one_off_status}': cls.set_one_off,
                f'📤 Send upon command {send_mode}': cls.switch_delayed_message_mode,
                f'🔢 Max tokens({max_tokens})': cls.request_set_max_tokens,
                '🏁 Set system prompt': cls.request_set_system_prompt,
                f'🧹 Clean conversation({conversation_length})': cls.clean_conversation,
                f'🤖 Model({model.name})': cls.show_models_list,
                f'👥 Role({role})': cls.show_roles_list,
            },
            row_width=1,
            message_to_rewrite=message if message.from_user.is_bot else None,
            text='Options:'
        )

    @classmethod
    async def chat_message_handler(cls, botnav: TeleBotNav, message: Message):
        if message.content_type not in ('text', 'voice', 'photo'):
            return

        text = ''
        image = None

        if message.content_type == 'voice':
            result = await botnav.await_coro_sending_action(
                message.chat.id,
                WhisperHelper.extract_text_from_voice(botnav, message),
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

        if text:
            conversation.add_message(
                role=MessageRole.USER,
                content=text
            )

        if image:
            conversation.add_message(
                role=MessageRole.USER,
                content=encode_jpg_image(image),
                content_type=MessageType.IMAGE
            )

        if message.state_data.get('delayed_message', False):
            await botnav.print_buttons(
                message.chat.id,
                {
                    '📩 Send': cls.get_reply,
                },
                'Press to send'
            )
            return

        await cls.get_reply(botnav, message)

    @classmethod
    async def get_reply(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.send_chat_action(message.chat.id, 'typing')

        try:
            message_splitter = MessageSplitter(4000)
            conversation = get_or_create_conversation(botnav, message)

            llm_gen = conversation.make_request()

            async for reply in llm_gen:
                await botnav.send_chat_action(message.chat.id, 'typing')

                msg = message_splitter.add(reply)
                if msg:
                    await botnav.bot.send_message(message.chat.id, msg)
            for msg in message_splitter.flush():
                await botnav.bot.send_message(message.chat.id, msg)

            save_conversation_to_file(botnav.get_user(message).id, conversation)
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
        await botnav.print_buttons(
            message.chat.id,
            {
                '🔄 Reset conversation': cls.reset_conversation,
                '⚙️ Options': cls.show_chat_options,
            },
            'Additional options:',
            row_width=1
        )

        botnav.wipe_commands(message, preserve=['start', 'openai'])
        botnav.add_command(message, 'chat_gpt_reset', '🔄 Reset conversation', cls.reset_conversation)
        botnav.add_command(message, 'chat_gpt_clean', '🧹 Clean conversation', cls.clean_conversation)
        botnav.add_command(message, 'chat_gpt_options', '⚙️ Chat gpt Options', cls.show_chat_options)
        await botnav.bot.send_message(message.chat.id, 'Welcome to Chat GPT, lets chat!')
        botnav.set_default_handler(message, cls.chat_message_handler)
        botnav.clean_next_handler(message)
        await botnav.send_commands(message)
