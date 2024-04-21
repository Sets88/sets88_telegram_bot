import asyncio
import time

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
import scheduler_module
import tools_module
import claude_module
from logger import logger


class ExceptionH(ExceptionHandler):
    def handle(self, exception: Exception):
        logger.exception(exception)


async def start(botnav: TeleBotNav, message: Message) -> None:
    botnav.clean_default_handler(message)
    botnav.clean_next_handler(message)
    logger.info(f'{message.from_user.username} {message.chat.id}')
    username = botnav.get_user(message).username

    if (ALLOWED_USER_NAMES and (
            not username or
            username.lower() not in ALLOWED_USER_NAMES
        )
    ):
        logger.info(f'{username} {message.chat.id} not allowed')
        await botnav.bot.send_message(message.chat.id, "Build your own bot, here is a source code: https://github.com/Sets88/sets88_telegram_bot")
        return

    await botnav.print_buttons(
        message.chat.id,
        {
            '🧠 OpenAI': openai_module.start_openai if config.OPENAI_API_KEY else None,
            '💻 Replicate': replicate_module.start_replicate if config.REPLICATE_API_KEY else None,
            '💭 Claude 3': claude_module.start_claude if config.ANTHROPIC_API_KEY else None,
            '📼 Youtube-DL': youtube_dl_module.start_youtube_dl,
            'Tools': tools_module.start_tools,
            'Scheduled scripts': scheduler_module.start_schedules if config.SCHEDULES else None,
        }, 'Choose',
        row_width=2
    )
    botnav.wipe_commands(message)
    botnav.add_command(message, 'start', '🏁 Start the bot', start)
    await botnav.send_commands(message)


async def main() -> None:
    if config.SCHEDULES:
        await scheduler_module.manager.run(botnav)
    await botnav.send_init_commands({'start': '🏁 Start the bot'})
    await botnav.set_global_default_handler(start)
    await botnav.bot.polling(non_stop=True)


botnav = TeleBotNav(
    TELEGRAM_TOKEN,
    state_storage=StateMemoryStorage(),
    exception_handler=ExceptionH()
)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.exception(exc)
        time.sleep(10)
        raise exc