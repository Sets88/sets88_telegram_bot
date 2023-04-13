import os
import json
import asyncio
from io import BytesIO
import functools
from random import randrange
from time import time
from typing import BinaryIO

import openai
from telebot.types import Message
from pydub import AudioSegment

import config
from telebot_nav import TeleBotNav
from logger import logger


openai.api_key = config.OPENAI_API_KEY


CONV_PATH = os.path.join(os.path.dirname(__file__), "conv")


OPENAI_OPTIONS = dict(
  temperature=0.7,
  max_tokens=1024,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

DEFAULT_GPT_MODEL = 'gpt-3.5-turbo'

AVAILABLE_GPT_MODELS = [
    'gpt-3.5-turbo', 'gpt-4'
]

CHAT_ROLES = {
    'Funnyman': {
        'init': 'As a helpful assistant, I will answer your questions as concisely as possible, with a touch of humor to make it more enjoyable.'
    },
    'Shakespeare': {
        'init': 'You always speak in a very old-fashioned English manner, similar to that of Shakespeare.'
    },
    'IT': {
        'init': 'You are an IT nerd who is so deeply involved in technology that you may only be understood by other IT experts.'
    },
    'Chef': {
        'init': 'You are a helpful cooking expert who answers questions by providing a short explanation and a list of easy-to-follow steps. You list the required ingredients, tools, and instructions.'
    },
    'Sarcastic': {
        'init': 'You are John Galt from the book Atlas Shrugged. You answer questions honestly, but do it in a sarcastic way like Chandler from Friends.'
    },
    'ConspTheory': {
        'init': 'You are a believer in conspiracy theories. All of your answers are based on these theories, and you cannot accept that there may be other explanations. You believe in things like aliens, reptilians, and other similar ideas.',
        'temperature': 1.0
    },    'JW': {
        'init': "You are a member of Jehovah's Witnesses and you do not have any doubts about the existence of God. You are willing to say anything to prove it.",
        'temperature': 1.0
    },
    'Linguist': {
        'init': "You are a helpful lingual assitant.",
        'temperature': 0.0
    },
    'Linux': {
        'init': 'I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}.'
    },
    'English Translator': {
        'temperature': 0.0,
        'init': 'I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations.'
    },
    'Interviewer': {
        'init': 'I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the position position. I want you to only reply as the interviewer. Do not write all the conservation at once. I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers.'
    }
}


class OpenAi():
    def __init__(self) -> None:
        self.conversations = {}

    def chat_load_converstation(self, user_id: int, conversation_id: str) -> None:
        self.conversations[f'{user_id}_{conversation_id}'] = json.load(
            open(os.path.join(CONV_PATH, f"{user_id}_{os.path.basename(conversation_id)}.json"), "r")
        )

    def chat_new_conversation(self, user_id: int) -> None:
        new_conversation_id = f"{int(time())}_{randrange(10000, 99999)}"
        self.conversations[f'{user_id}_{new_conversation_id}'] = {
            'init': 'Assistant is an intelligent chatbot designed to help users answer questions is the main goal of the Assistant.',
            'one_off': False,
            'model': DEFAULT_GPT_MODEL,
            'messages': []
        }
        self.chat_save_conversation(user_id, new_conversation_id)

        return new_conversation_id

    def chat_save_conversation(self, user_id: int, conversation_id: str) -> None:
        if not os.path.exists(CONV_PATH):
            os.mkdir(CONV_PATH)
        json.dump(
            self.conversations[f'{user_id}_{conversation_id}'],
            open(os.path.join(CONV_PATH, f"{user_id}_{conversation_id}.json"), "w")
        )

    def chat_add_message(self, user_id: int, conversation_id: str, message: str) -> None:
        conv_data = self.conversations[f'{user_id}_{conversation_id}']
        if conv_data['one_off']:
            conv_data['messages'] = []

        conv_data['messages'].append(
            {'role': 'user', 'content': message}
        )

        self.chat_save_conversation(user_id, conversation_id)

    async def chat_get_reply(self, user_id: int, conversation_id: str) -> str:
        full_response = ""

        params = OPENAI_OPTIONS.copy()

        for key, val in self.conversations[f'{user_id}_{conversation_id}'].items():
            if key in params:
                params[key] = val

        gen = await openai.ChatCompletion.acreate(
            model=params.get('model', DEFAULT_GPT_MODEL),
            messages=[
                {"role": "system", "content": self.conversations[f'{user_id}_{conversation_id}']['init']},
                *self.conversations[f'{user_id}_{conversation_id}']['messages'],
            ],
            stream=True,
            **params
        )

        async for response in gen:
            if response.choices:
                content = response.choices[0].delta.get('content')
                if content:
                    full_response += content
                    yield content
        self.conversations[f'{user_id}_{conversation_id}']['messages'].append(
            {'role': 'assistant', 'content': full_response}
        )
        self.chat_save_conversation(user_id, conversation_id)

    def chat_set_options(self, user_id: int, conversation_id: str, **kwargs) -> None:
        conv_data = self.conversations[f'{user_id}_{conversation_id}']
        conv_data.update(kwargs)

    async def dalle_generate_image(self, prompt: str) -> str:
        response = await openai.Image.acreate(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        return response['data'][0]['url']

    async def whisper_transcribe(self, audio: BinaryIO) -> str:
        response = await openai.Audio.atranscribe(
            model='whisper-1',
            file=audio,
        )

        return response.text


def get_mp3_from_ogg(file_content: BinaryIO) -> BytesIO:
    file = BytesIO(file_content)
    file.seek(0)
    ogg = AudioSegment.from_ogg(file)
    mp3 = BytesIO()
    ogg.export(mp3, format='mp3')
    mp3.seek(0)
    return mp3


async def extract_text_from_voice(botnav: TeleBotNav, message: Message) -> str:
    file_info = await botnav.bot.get_file(message.voice.file_id)
    file_content = await botnav.bot.download_file(file_info.file_path)
    file = await asyncio.to_thread(get_mp3_from_ogg, file_content)
    file.name = 'voice.mp3'
    text = await openai_instance.whisper_transcribe(file)
    return text


async def chat_gpt_message_handler(botnav: TeleBotNav, message: Message) -> None:
    if message.content_type not in ('text', 'voice'):
        return

    if message.content_type == 'voice':
        text = await extract_text_from_voice(botnav, message)
        await botnav.bot.send_message(message.chat.id, f'You said: "{text}"')

    if message.content_type == 'text':
        text = message.text

    get_or_create_conversation(botnav, message)

    openai_instance.chat_add_message(
        botnav.get_user(message).id,
        message.state_data['conversation_id'],
        text
    )
    parts = []

    await botnav.send_chat_action(message.chat.id, 'typing')

    try:
        async for reply in openai_instance.chat_get_reply(message.from_user.id, message.state_data['conversation_id']):
            await botnav.send_chat_action(message.chat.id, 'typing')
            parts.append(reply)
        await botnav.bot.send_message(message.chat.id, "".join(parts))
    except Exception as exc:
        if getattr(exc, 'code', None) == 'context_length_exceeded':
            await botnav.bot.send_message(message.chat.id, getattr(exc, 'user_message', "Something went wrong, try again later"))
            await chat_clean_conversation(botnav, message)
            return

        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)
        message.state_data.clear()


async def start_dalle(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.send_message(message.chat.id, 'Welcome to DALL-E, ask me to draw something!')
    await botnav.set_default_handler(message, dalle_message_handler)


async def dalle_message_handler(botnav: TeleBotNav, message: Message) -> None:
    if message.content_type != 'text':
        return

    await botnav.bot.send_chat_action(message.chat.id, 'upload_photo')
    url = await openai_instance.dalle_generate_image(message.text)
    await botnav.bot.send_chat_action(message.chat.id, 'upload_photo')
    await botnav.bot.send_photo(message.chat.id, url)


def get_or_create_conversation(botnav: TeleBotNav, message: Message) -> str:
    if 'conversation_id' not in message.state_data:
        message.state_data['conversation_id'] = openai_instance.chat_new_conversation(botnav.get_user(message).id)
    return message.state_data['conversation_id']


async def chat_set_role(role: str, botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.delete_message(message.chat.id, message.message_id)

    openai_instance.chat_set_options(
        message.chat.id,
        get_or_create_conversation(botnav, message),
        **CHAT_ROLES[role]
    )

    await botnav.bot.send_message(message.chat.id, "Init was set to: " + CHAT_ROLES[role]['init'])


async def chat_set_model(model: str, botnav: TeleBotNav, message: Message) -> None:
    if model not in AVAILABLE_GPT_MODELS:
        return

    await botnav.bot.delete_message(message.chat.id, message.message_id)

    openai_instance.chat_set_options(
        message.chat.id,
        get_or_create_conversation(botnav, message),
        model=model
    )

    await botnav.bot.send_message(message.chat.id, "Model was set to: " + model)



async def chat_set_temparature(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.send_message(message.chat.id, "Temperature was set to: " + message.text)

    openai_instance.chat_set_options(
        message.chat.id,
        get_or_create_conversation(botnav, message),
        temperature=float(message.text)
    )


async def chat_set_init(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.delete_message(message.chat.id, message.message_id)

    openai_instance.chat_set_options(
        message.chat.id,
        get_or_create_conversation(botnav, message),
        init=message.text
    )

    await botnav.bot.send_message(message.chat.id, "Init was set to: " + message.text)


async def chat_reset_conversation(botnav: TeleBotNav, message: Message) -> None:
    message.state_data['conversation_id'] = openai_instance.chat_new_conversation(message.chat.id)
    await botnav.bot.edit_message_text("Conversation was reset", message.chat.id, message.message_id)


async def chat_choose_role(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            x: functools.partial(chat_set_role, x) for x in CHAT_ROLES.keys()
        },
        message_to_rewrite=message,
        row_width=2,
    )


async def chat_models_list(botnav: TeleBotNav, message: Message) -> str:
    await botnav.print_buttons(
        message.chat.id,
        {
            x: functools.partial(chat_set_model, x) for x in AVAILABLE_GPT_MODELS
        },
        message_to_rewrite=message,
        row_width=2,
    )


async def chat_one_off(botnav: TeleBotNav, message: Message) -> None:
    openai_instance.chat_set_options(
        message.chat.id,
        get_or_create_conversation(botnav, message),
        one_off=True
    )

    await botnav.bot.edit_message_text(
        'Conversation switched to single question/answer mode. '
        'Model will forget conversation after response',
        message.chat.id,
        message.message_id
    )


async def chat_set_init_request(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.edit_message_text("Set the description of your opponent", message.chat.id, message.message_id)
    await botnav.set_next_handler(message, chat_set_init)


async def chat_temperature(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.edit_message_text("Set temperature of model(0.0 - 2.0 Higher values causes model to increase"
                                " randomnes in its responses)", message.chat.id, message.message_id)

    await botnav.set_next_handler(message, chat_set_temparature)


async def chat_clean_conversation(botnav: TeleBotNav, message: Message) -> None:
    openai_instance.chat_set_options(
        message.chat.id,
        get_or_create_conversation(botnav, message),
        message=[]
    )

    await botnav.bot.edit_message_text("Conversation was reset", message.chat.id, message.message_id)


async def chat_options(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            'One Off': chat_one_off,
            'Temperature': chat_temperature,
            'Set init': chat_set_init_request,
            'Clean conversation': chat_clean_conversation,
            'Choose Model': chat_models_list,
        },
        message_to_rewrite=message,
        row_width=2
    )


async def start_chat_gpt(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            'Reset conversation': chat_reset_conversation,
            'Choose role': chat_choose_role,
            'Options': chat_options,
        },
        'For configurations purpose:',
        row_width=2
    )
    await botnav.bot.send_message(message.chat.id, 'Welcome to Chat GPT, lets chat!')
    await botnav.set_default_handler(message, chat_gpt_message_handler)


openai_instance = OpenAi()
