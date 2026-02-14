import asyncio
import time

from telebot import ExceptionHandler
from telebot.types import Message
from telebot.asyncio_storage import StateMemoryStorage

from telebot_nav import TeleBotNav
from config import TELEGRAM_TOKEN
from config import ALLOWED_USER_IDS

from lib.permissions import is_replicate_available
from lib.user_helpers import get_user_display_name
import config
import openai_module
import llm_module
import replicate_module
import youtube_dl_module
import scheduler_module
import tools_module
import greek_learning_module
from logger import logger


class ExceptionH(ExceptionHandler):
    def handle(self, exception: Exception):
        logger.exception(exception)


async def start(botnav: TeleBotNav, message: Message) -> None:
    botnav.clean_default_handler(message)
    botnav.clean_next_handler(message)
    user = botnav.get_user(message)
    user_id = user.id
    user_id_str = str(user_id)
    display_name = get_user_display_name(user_id)

    logger.info(f'{display_name} (ID: {user_id}) {message.chat.id}')

    if ALLOWED_USER_IDS and user_id_str not in ALLOWED_USER_IDS:
        logger.info(f'{display_name} (ID: {user_id}) {message.chat.id} not allowed')
        await botnav.bot.send_message(message.chat.id, "Build your own bot, here is a source code: https://github.com/Sets88/sets88_telegram_bot")
        return

    await botnav.print_buttons(
        message.chat.id,
        {
            '🧠 OpenAI': openai_module.start_openai if config.OPENAI_API_KEY else None,
            '🧠 LLM': llm_module.LLMRouter.run if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY else None,
            '🇬🇷 Greek Learning': greek_learning_module.start_greek if config.GREEK_LEARNING_WEBAPP_URL else None,
            '💻 Replicate': replicate_module.start_replicate if is_replicate_available(botnav, message) else None,
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

    # Start Greek Learning Web App server
    if config.GREEK_LEARNING_WEBAPP_URL:
        asyncio.create_task(greek_learning_module.start_web_app(botnav))

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