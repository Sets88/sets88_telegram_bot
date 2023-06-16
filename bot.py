import asyncio

from telebot import ExceptionHandler
from telebot.types import Message
from telebot.asyncio_storage import StateMemoryStorage

from telebot_nav import TeleBotNav
from config import TELEGRAM_TOKEN
from config import ALLOWED_USER_NAMES

import config
import openai_module
import replicate_module
import youtube_dl_module
from logger import logger


class ExceptionH(ExceptionHandler):
    def handle(self, exception: Exception):
        logger.exception(exception)


async def start(botnav: TeleBotNav, message: Message) -> None:
    logger.info(f'{message.from_user.username} {message.chat.id}')

    if (ALLOWED_USER_NAMES and botnav.get_user(message).username.lower() not in ALLOWED_USER_NAMES):
        logger.info(f'{message.from_user.username} {message.chat.id} not allowed')
        await botnav.bot.send_message(message.chat.id, "Build your own bot here is a source code: https://github.com/Sets88/sets88_telegram_bot")
        return

    await botnav.print_buttons(
        message.chat.id,
        {
            'Chat GPT': openai_module.start_chat_gpt if config.OPENAI_API_KEY else None,
            'Dall-E': openai_module.start_dalle if config.OPENAI_API_KEY else None,
            'Whisper': openai_module.start_whisper if config.OPENAI_API_KEY else None,
            'Replicate': replicate_module.start_replicate if config.REPLICATE_API_KEY else None,
            'Youtube-DL': youtube_dl_module.start_youtube_dl,
        }, 'Choose',
        row_width=2
    )
    await botnav.send_commands()


async def main() -> None:
    await botnav.set_command('start', 'Start the bot', start)
    await botnav.set_global_default_handler(start)
    await botnav.bot.polling()


botnav = TeleBotNav(
    TELEGRAM_TOKEN,
    state_storage=StateMemoryStorage(),
    exception_handler=ExceptionH()
)


if __name__ == '__main__':
    asyncio.run(main())