import asyncio
import os
import time

from telebot import ExceptionHandler
from telebot.types import Message, WebAppInfo
from telebot.asyncio_storage import StateMemoryStorage
from telebot import types

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
import webapp_server
import webapp_apps_module
from logger import logger


class ExceptionH(ExceptionHandler):
    def handle(self, exception: Exception):
        logger.exception(exception)


async def _open_shared_webapp(botnav: TeleBotNav, message: Message, app_id: str) -> None:
    """Respond to a deep-link that points at a user-generated web app."""
    app_path = os.path.join(
        os.path.dirname(__file__), 'webapp', 'apps', app_id, 'index.html'
    )
    if not os.path.exists(app_path):
        await botnav.bot.send_message(message.chat.id, "Web app not found.")
        return

    app_url = f"{config.WEBAPP_BASE_URL.rstrip('/')}/apps/{app_id}/index.html"
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton(
        text='🌐 Open Web App',
        web_app=WebAppInfo(url=app_url),
    ))
    await botnav.bot.send_message(
        message.chat.id,
        'Click the button to open the shared web app:',
        reply_markup=markup,
    )


async def start(botnav: TeleBotNav, message: Message) -> None:
    botnav.clean_default_handler(message)
    botnav.clean_next_handler(message)
    user = botnav.get_user(message)
    user_id = user.id
    user_id_str = str(user_id)
    display_name = get_user_display_name(user_id)

    logger.info(f'{display_name} (ID: {user_id}) {message.chat.id}')

    # Handle deep-link for shared web apps (any user may open a shared app)
    text = message.text or ''
    args = text.split()
    if len(args) > 1 and args[1].startswith('app_'):
        app_id = args[1][4:]
        await _open_shared_webapp(botnav, message, app_id)
        return

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
            '📱 My Web Apps': webapp_apps_module.start_my_apps if config.WEBAPP_BASE_URL else None,
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

    # Start shared web app server (greek app + user-generated apps)
    if config.GREEK_LEARNING_WEBAPP_URL:
        asyncio.create_task(webapp_server.start_server(botnav))

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