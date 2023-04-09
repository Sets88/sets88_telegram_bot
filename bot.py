import asyncio
import logging

from telebot import ExceptionHandler
from telebot.types import Message
from telebot.asyncio_storage import StateMemoryStorage

from telebot_nav import TeleBotNav
from config import TELEGRAM_TOKEN
from config import ALLOWED_USER_NAMES
import openai_module


logging.basicConfig(level=logging.INFO)


class ExceptionH(ExceptionHandler):
    def handle(self, exception: Exception):
        logging.exception(exception)


async def start(botnav: TeleBotNav, message: Message) -> None:
    logging.info(f'{message.from_user.username} {message.chat.id}')

    if (botnav.get_user(message).username.lower() not in ALLOWED_USER_NAMES):
        return

    await botnav.print_buttons(
        message.chat.id,
        {
            'Chat GPT': openai_module.start_chat_gpt,
            'Dall-E': openai_module.start_dalle,
        }, 'Choose'
    )
    await botnav.send_commands()


async def main() -> None:
    await botnav.set_command('start', 'Start the bot', start)
    await botnav.set_globl_default_handler(start)
    await botnav.bot.polling()


botnav = TeleBotNav(
    TELEGRAM_TOKEN,
    state_storage=StateMemoryStorage(),
    exception_handler=ExceptionH()
)


if __name__ == '__main__':
    asyncio.run(main())