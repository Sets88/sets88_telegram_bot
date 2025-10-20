import asyncio
import base64
import functools
import dataclasses
from typing import Any
from hashlib import md5
from io import BytesIO
import json
import os
from typing import AsyncGenerator, BinaryIO

import anthropic
from telebot.types import Message
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from openai import RateLimitError
from pydub import AudioSegment

import config
from lib.llm import AIProvider, ConversationManager, MessageRole, MessageType
from lib.utils import MessageSplitter, ConvEncoder
from telebot_nav import TeleBotNav
from logger import logger


AVAILABLE_LLM_MODELS = [
    (AIProvider.OPENAI, 'gpt-4.1-nano'),
    (AIProvider.OPENAI, 'gpt-4.1-mini'),
    (AIProvider.OPENAI, 'o4-mini'),
    (AIProvider.OPENAI, 'gpt-4.1'),
    (AIProvider.OPENAI, 'gpt-5-nano'),
    (AIProvider.OPENAI, 'gpt-5'),
    (AIProvider.OPENAI, 'o3'),
    (AIProvider.ANTHROPIC, 'claude-haiku-4-5'),
    (AIProvider.ANTHROPIC, 'claude-sonnet-4-5'),
    (AIProvider.ANTHROPIC, 'claude-opus-4-1'),
]

CHAT_ROLES = {
    'Funnyman': {
        'system_prompt': 'As a helpful assistant, I will answer your questions as concisely as possible, with a touch of humor to make it more enjoyable.'
    },
    'Greek': {
        'system_prompt': 'You are Greek language support assistant. If the text is in Russian, it should be translated into Greek. If the text in Russian consists of a single word, you should respond with a list of words with similar meanings in Greek and the exact translation in Russian for each separately. If the text is in Greek, you should respond in Russian. If the text is in the Latin alphabet, it is a transliteration from Greek, and you should assume what the text should be in the Greek alphabet and add Russian translation. No additional explanations are needed, just what said above.',
        'one_off': True,
        'model': (AIProvider.OPENAI, 'gpt-4.1'),
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
    'Linux': {
        'system_prompt': 'I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}.'
    },
    'English Translator': {
        'system_prompt': 'I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations.'
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
    },
    'Fixer': {
        'system_prompt': 'You fix errors in everything passed to you, you respond with fixed text no explanation needed',
        'one_off': True,
        'model': (AIProvider.OPENAI, 'gpt-4.1'),
    }
}

DEAFULT_ROLE = 'Assistant'
DEFAULT_MAX_TOKENS = 4096

CONV_PATH = os.path.join(os.path.dirname(__file__), "conv")


def get_available_models() -> list[tuple[AIProvider, str]]:
    available_models: list[tuple[AIProvider, str]] = []
    for provider, model in AVAILABLE_LLM_MODELS:
        if provider == AIProvider.OPENAI and not config.OPENAI_API_KEY:
            continue
        if provider == AIProvider.ANTHROPIC and not config.ANTHROPIC_API_KEY:
            continue
        available_models.append((provider, model))
    return available_models


def get_or_create_conversation(botnav: TeleBotNav, message: Message) -> ConversationManager:
    if 'conversation' not in message.state_data:
        message.state_data['conversation'] = get_new_conversation_manager()
    return message.state_data['conversation']


def save_conversation_to_file(user_id: int, conversation: ConversationManager) -> None:
    if not os.path.exists(CONV_PATH):
        os.mkdir(CONV_PATH)
    json.dump(
        dataclasses.asdict(conversation.get_request_data()),
        open(os.path.join(CONV_PATH, f"{user_id}_{conversation.__hash__()}.json"), "w"),
        cls=ConvEncoder
    )


def set_role(conversation: ConversationManager, role: str) -> None:
    if role not in CHAT_ROLES:
        role = DEAFULT_ROLE

    system_prompt: str = CHAT_ROLES[role]['system_prompt']
    conversation.set_config_param('system_prompt', system_prompt)

    if 'one_off' in CHAT_ROLES[role]:
        conversation.set_config_param('one_off', CHAT_ROLES[role]['one_off'])
    if 'thinking' in CHAT_ROLES[role]:
        conversation.set_config_param('thinking', CHAT_ROLES[role]['thinking'])
    if 'model' in CHAT_ROLES[role]:
        provider, model = CHAT_ROLES[role]['model']
        conversation.set_provider(provider, model)


def get_new_conversation_manager() -> ConversationManager:
    manager = ConversationManager()

    set_role(manager, DEAFULT_ROLE)
    manager.set_config_param('max_tokens', DEFAULT_MAX_TOKENS)

    for provider, model in get_available_models():
        manager.set_provider(provider, model)
        break

    if not manager.current_provider:
        raise RuntimeError("No LLM providers are configured properly.")

    return manager


def encode_jpg_image(image: bytes):
    return "data:image/jpeg;base64," + base64.b64encode(image).decode('utf-8')


class OpenAIInstance:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def whisper_transcribe(self, audio: BinaryIO) -> str:
        response = await self.client.audio.transcriptions.create(
            model='whisper-1',
            file=audio,
        )

        return response.text

    async def make_request(self, user_id: int, conversation: ConversationManager) -> AsyncGenerator[str, None]:
        request_data = conversation.get_request_data()
        full_response: str = ''

        logger.info(dataclasses.asdict(request_data))

        request: dict[str, Any] = {
            "input": request_data.messages,
            "model": request_data.model,
            "max_output_tokens": request_data.max_tokens,
            "stream": True,
            "user": md5(f'aaa-{user_id}-bbb'.encode('utf-8')).hexdigest()
        }

        if request_data.thinking:
            request['reasoning'] = {'effort': 'medium'}

        stream = await self.client.responses.create(**request)

        async for event in stream:
            if isinstance(event, ResponseTextDeltaEvent):
                content = event.delta
                if content:
                    full_response += content
                    yield content

        conversation.add_ai_response(full_response)
        save_conversation_to_file(user_id, conversation)
 

class ClaudeInstance:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    async def make_request(self, user_id: int, conversation: ConversationManager) -> AsyncGenerator[str, None]:
        request_data = conversation.get_request_data()
        full_response: str = ''

        logger.info(dataclasses.asdict(request_data))

        request = {
            "max_tokens": request_data.max_tokens,
            "messages": request_data.messages,
            "model": request_data.model,
            "system": request_data.system_prompt or "You are a helpful assistant",
            "stream": True,
        }

        if request_data.thinking:
            request['thinking'] = {'type': 'enabled', 'budget_tokens': request_data.max_tokens - 500}

        stream = await self.client.messages.create(**request)

        async for event in stream:
            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                content = event.delta.text

                if content:
                    full_response += content
                    yield content

        conversation.add_ai_response(full_response)
        save_conversation_to_file(user_id, conversation)


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
        message.state_data['conversation'] = get_new_conversation_manager()
        await botnav.bot.send_message(message.chat.id, "Conversation was reset")

    ## Options handlers
    @classmethod
    async def set_one_off(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        conversation.set_config_param('one_off', True)

        await botnav.bot.edit_message_text(
            'Conversation switched to single question/answer mode. '
            'Model will forget conversation after response',
            message.chat.id,
            message.message_id
        )

    @classmethod
    async def switch_delayed_message_mode(cls, botnav: TeleBotNav, message: Message) -> None:
        delayed_mode_enabled = message.state_data.get('delayed_message', False)

        if delayed_mode_enabled:
            await botnav.bot.send_message(message.chat.id, "Enter text of your request, after finish push send button")
        else:
            await botnav.bot.send_message(message.chat.id, "Send on command mode disabled")

        message.state_data['delayed_message'] = not delayed_mode_enabled

    @classmethod
    async def request_set_temperature(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.edit_message_text("Set temperature of model(0.0 - 2.0 Higher values causes model to increase"
            " randomnes in its responses)", message.chat.id, message.message_id)

        botnav.set_next_handler(message, cls.set_temperature)

    @classmethod
    async def set_temperature(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.send_message(message.chat.id, "Temperature was set to: " + message.text)

        conversation = get_or_create_conversation(botnav, message)
        try:
            if not message.text:
                raise ValueError()

            temperature = float(message.text)

            if not (0.0 <= temperature <= 2.0):
                raise ValueError()
            conversation.set_config_param('temperature', temperature)
        except (ValueError, TypeError):
            await botnav.bot.send_message(message.chat.id, "Invalid temperature value, must be between 0.0 and 2.0")
            return

    @classmethod
    async def request_set_max_tokens(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.edit_message_text("Set maximum tokens in response"
            " randomnes in its responses)", message.chat.id, message.message_id)

        botnav.set_next_handler(message, cls.set_max_tokens)

    @classmethod
    async def set_max_tokens(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.send_message(message.chat.id, "Max tokens was set to: " + message.text)

        conversation = get_or_create_conversation(botnav, message)
        try:
            if not message.text:
                raise ValueError()

            max_tokens = int(message.text)

            if max_tokens <= 0:
                raise ValueError()

            conversation.set_config_param('max_tokens', max_tokens)
        except (ValueError, TypeError):
            await botnav.bot.send_message(message.chat.id, "Invalid max tokens value, must be a positive integer")
            return

    @classmethod
    async def request_set_system_prompt(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.edit_message_text("Set the description of your opponent", message.chat.id, message.message_id)
        botnav.set_next_handler(message, cls.set_system_prompt)

    @classmethod
    async def set_system_prompt(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.delete_message(message.chat.id, message.message_id)

        conversation = get_or_create_conversation(botnav, message)
        conversation.set_config_param('system_prompt', message.text)

        await botnav.bot.send_message(message.chat.id, "System prompt was set to: " + message.text)

    @classmethod
    async def clean_conversation(cls, botnav: TeleBotNav, message: Message) -> None:
        conversation = get_or_create_conversation(botnav, message)
        conversation.clear_conversation()

        await botnav.bot.edit_message_text("Conversation was cleared", message.chat.id, message.message_id)

    @classmethod
    async def show_models_list(cls, botnav: TeleBotNav, message: Message) -> str:
        await botnav.print_buttons(
            message.chat.id,
            {
                f"{model} ({provider.value})": functools.partial(
                    cls.switch_llm_model,
                    provider, model
                ) for provider, model in get_available_models()
            },
            message_to_rewrite=message,
            row_width=1,
        )

    @classmethod
    async def switch_llm_model(cls, provider: AIProvider, model: str, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.delete_message(message.chat.id, message.message_id)

        conversation = get_or_create_conversation(botnav, message)
        conversation.set_provider(provider, model)

        await botnav.bot.send_message(message.chat.id, f"Model was set to: {provider.value} / {model}")

    @classmethod
    async def show_roles_list(cls, botnav: TeleBotNav, message: Message) -> str:
        await botnav.print_buttons(
            message.chat.id,
            {
                x: functools.partial(cls.set_role, x) for x in CHAT_ROLES.keys()
            },
            message_to_rewrite=message,
            row_width=2,
        )

    @classmethod
    async def set_role(cls, role: str, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.delete_message(message.chat.id, message.message_id)

        conversation = get_or_create_conversation(botnav, message)
        set_role(conversation, role)

        conversation.set_config_param('system_prompt', CHAT_ROLES[role]['system_prompt'])
        if 'one_off' in CHAT_ROLES[role]:
            conversation.set_config_param('one_off', CHAT_ROLES[role]['one_off'])
        if 'temperature' in CHAT_ROLES[role]:
            conversation.set_config_param('temperature', CHAT_ROLES[role]['temperature'])

        role_info = f"{CHAT_ROLES[role]['system_prompt'][0:4000]}"
        await botnav.bot.send_message(message.chat.id, f'Init was set to: {role_info}')

    @classmethod
    async def show_chat_options(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.print_buttons(
            message.chat.id,
            {
                '🎯 One Off': cls.set_one_off,
                '📤 Send on command': cls.switch_delayed_message_mode,
                '🌡️ Temperature': cls.request_set_temperature,
                '🔢 Max tokens': cls.request_set_max_tokens,
                '🏁 Set system prompt': cls.request_set_system_prompt,
                '🧹 Clean conversation': cls.clean_conversation,
                '🤖 Choose Model': cls.show_models_list,
                '👥 Choose role': cls.show_roles_list,
            },
            row_width=2,
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

        if message.content_type == 'photo' and message.caption:
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

            if conversation.current_provider == AIProvider.OPENAI:
                llm_gen = openai_instance.make_request(botnav.get_user(message).id, conversation)
            elif conversation.current_provider == AIProvider.ANTHROPIC:
                llm_gen = claude_instance.make_request(botnav.get_user(message).id, conversation)

            async for reply in llm_gen:
                await botnav.send_chat_action(message.chat.id, 'typing')

                msg = message_splitter.add(reply)
                if msg:
                    await botnav.bot.send_message(message.chat.id, msg)
            for msg in message_splitter.flush():
                await botnav.bot.send_message(message.chat.id, msg)
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


openai_instance = OpenAIInstance()
claude_instance = ClaudeInstance()
