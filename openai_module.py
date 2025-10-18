import io
import asyncio
from io import BytesIO
import functools
from typing import BinaryIO

from openai import AsyncOpenAI
from telebot.types import Message
from pydub import AudioSegment

import config
from telebot_nav import TeleBotNav
from logger import logger
from llm_module import LLMRouter


class OpenAiAdapter():
    def __init__(self) -> None:
        self.conversations = {}
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def dalle_generate_image(self, prompt: str) -> str:
        response = await self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url

    async def whisper_transcribe(self, audio: BinaryIO) -> str:
        response = await self.client.audio.transcriptions.create(
            model='whisper-1',
            file=audio,
        )

        return response.text

    async def tts_generate_audio(self, text: str, voice: str) -> BinaryIO:
        response = await self.client.audio.speech.create(
            model='tts-1',
            input=text,
            voice=voice
        )
        return response


class WhisperRouter:
    @classmethod
    def get_mp3_from_ogg(cls, file_content: BinaryIO) -> BytesIO:
        file = BytesIO(file_content)
        file.seek(0)
        ogg = AudioSegment.from_ogg(file)
        mp3 = BytesIO()
        ogg.export(mp3, format='mp3')
        mp3.seek(0)
        return mp3

    @classmethod
    async def extract_text_from_voice(cls, botnav: TeleBotNav, message: Message) -> str:
        file_info = await botnav.bot.get_file(message.voice.file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        file = await asyncio.to_thread(cls.get_mp3_from_ogg, file_content)
        file.name = 'voice.mp3'
        text = await openai_instance.whisper_transcribe(file)
        return text

    @classmethod
    async def whisper_message_handler(cls, botnav: TeleBotNav, message: Message) -> None:
        if message.content_type != 'voice':
            return

        try:
            text = await botnav.await_coro_sending_action(
                message.chat.id,
                cls.extract_text_from_voice(botnav, message),
                'typing'
            )
            await botnav.bot.send_message(message.chat.id, text)
        except Exception as exc:
            await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
            logger.exception(exc)

    @classmethod
    async def run(cls, botnav: TeleBotNav, message: Message) -> None:
        botnav.wipe_commands(message, preserve=['start', 'openai'])
        await botnav.bot.send_message(message.chat.id, 'Welcome to Whisper, send me voice message to transcribe!')
        botnav.set_default_handler(message, cls.whisper_message_handler)
        botnav.clean_next_handler(message)
        await botnav.send_commands(message)


class DallERouter:
    @classmethod
    async def dalle_message_handler(cls, botnav: TeleBotNav, message: Message) -> None:
        if message.content_type != 'text':
            return

        try:
            url = await botnav.await_coro_sending_action(
                message.chat.id,
                openai_instance.dalle_generate_image(message.text),
                'upload_photo'
            )

            await botnav.bot.send_photo(message.chat.id, url)
        except Exception as exc:
            await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
            logger.exception(exc)

    @classmethod
    async def run(cls, botnav: TeleBotNav, message: Message):
        botnav.wipe_commands(message, preserve=['start', 'openai'])
        await botnav.bot.send_message(message.chat.id, 'Welcome to DALL-E, ask me to draw something!')
        botnav.set_default_handler(message, cls.dalle_message_handler)
        botnav.clean_next_handler(message)
        await botnav.send_commands(message)


class TTSRouter:
    @classmethod
    async def tts_message_handler(cls, botnav: TeleBotNav, message: Message) -> None:
        if message.content_type != 'text':
            return

        if 'openai_params' not in message.state_data:
            message.state_data['openai_params'] = {}

        voice = message.state_data['openai_params'].get('tts_voice', 'alloy')

        try:
            response = await botnav.await_coro_sending_action(
                message.chat.id,
                openai_instance.tts_generate_audio(message.text, voice),
                'upload_audio'
            )

            await botnav.bot.send_voice(message.chat.id, io.BytesIO(response.content))
        except Exception as exc:
            await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
            logger.exception(exc)

    @classmethod
    async def run(cls, botnav: TeleBotNav, message: Message):
        await botnav.print_buttons(
            message.chat.id,
            {
                'Alloy': functools.partial(set_openai_param, 'tts_voice', 'alloy'),
                'Echo': functools.partial(set_openai_param, 'tts_voice', 'echo'),
                'Fable': functools.partial(set_openai_param, 'tts_voice', 'fable'),
                'Onyx': functools.partial(set_openai_param, 'tts_voice', 'onyx'),
                'Nova': functools.partial(set_openai_param, 'tts_voice', 'nova'),
                'shimmer': functools.partial(set_openai_param, 'tts_voice', 'shimmer'),
            },
            'Available voices:',
            row_width=3
        )

        botnav.wipe_commands(message, preserve=['start', 'openai'])
        await botnav.bot.send_message(message.chat.id, 'Welcome to TTS, send me text to speech!')
        botnav.set_default_handler(message, cls.tts_message_handler)
        botnav.clean_next_handler(message)
        await botnav.send_commands(message)


async def set_openai_param(param: str, value: str, botnav: TeleBotNav, message: Message) -> None:
    if 'openai_params' not in message.state_data:
        message.state_data['openai_params'] = {}

    message.state_data['openai_params'][param] = value

    await botnav.bot.send_message(message.chat.id, f'OpenAI param {param} was set to {value}')


async def start_openai(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            '🤖 Chat GPT': LLMRouter.run,
            '🖌️ Dall-E': DallERouter.run,
            '🗣️ Whisper': WhisperRouter.run,
            '💬 TTS': TTSRouter.run
        }, 'Choose',
        row_width=2
    )
    botnav.wipe_commands(message, preserve=['start'])
    botnav.add_command(message, 'openai', '🧠 OpenAI models', start_openai)
    await botnav.send_commands(message)


openai_instance = OpenAiAdapter()
