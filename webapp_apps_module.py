import os
import re
import shutil
import functools

from telebot import types
from telebot.types import Message, WebAppInfo

import config
from telebot_nav import TeleBotNav
from logger import logger


def _apps_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'webapp', 'apps'))


def _app_owner(app_id: str) -> str | None:
    parts = app_id.split('_', 1)
    return parts[0] if len(parts) == 2 and parts[0].isdigit() else None


def _read_app_title(index_path: str) -> str | None:
    try:
        with open(index_path, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read(4096)
        m = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip() or None
    except Exception:
        pass
    return None


def _get_user_apps(user_id_str: str) -> list[dict]:
    apps_root = _apps_root()
    result = []
    if not os.path.isdir(apps_root):
        return result
    for app_id in sorted(os.listdir(apps_root)):
        if _app_owner(app_id) != user_id_str:
            continue
        index_path = os.path.join(apps_root, app_id, 'index.html')
        if not os.path.isfile(index_path):
            continue
        title = _read_app_title(index_path) or app_id
        result.append({'app_id': app_id, 'title': title})
    return result


async def start_my_apps(botnav: TeleBotNav, message: Message) -> None:
    user_id_str = str(botnav.get_user(message).id)
    apps = _get_user_apps(user_id_str)

    if not apps:
        await botnav.bot.send_message(message.chat.id, "You don't have any web apps yet")
        return

    base_url = config.WEBAPP_BASE_URL.rstrip('/')
    markup = types.InlineKeyboardMarkup()

    for app in apps:
        app_id = app['app_id']
        title = app['title']
        app_url = f"{base_url}/apps/{app_id}/index.html"

        open_btn = types.InlineKeyboardButton(
            text=f'🌐 {title}',
            web_app=WebAppInfo(url=app_url),
        )

        delete_handler = functools.partial(_confirm_delete, app_id)
        botnav.buttons[str(delete_handler.__hash__())] = delete_handler
        delete_btn = types.InlineKeyboardButton(
            text='🗑',
            callback_data=str(delete_handler.__hash__()),
        )

        markup.row(open_btn, delete_btn)

    await botnav.bot.send_message(
        message.chat.id,
        f"Your web apps ({len(apps)}):",
        reply_markup=markup,
    )


async def _confirm_delete(app_id: str, botnav: TeleBotNav, message: Message) -> None:
    user_id_str = str(botnav.get_user(message).id)
    app_id = os.path.basename(app_id)
    if _app_owner(app_id) != user_id_str:
        await botnav.bot.send_message(message.chat.id, "No access.")
        return

    index_path = os.path.join(_apps_root(), app_id, 'index.html')
    title = _read_app_title(index_path) or app_id

    yes_handler = functools.partial(_do_delete, app_id)
    await botnav.print_buttons(
        message.chat.id,
        {
            f'✅ Yes, delete': yes_handler,
            '❌ Cancel': start_my_apps,
        },
        text=f'Delete "{title}"? This action cannot be undone.',
        row_width=2,
    )


async def _do_delete(app_id: str, botnav: TeleBotNav, message: Message) -> None:
    user_id_str = str(botnav.get_user(message).id)
    app_id = os.path.basename(app_id)

    if _app_owner(app_id) != user_id_str:
        await botnav.bot.send_message(message.chat.id, "No access.")
        return

    app_dir = os.path.join(_apps_root(), app_id)
    try:
        shutil.rmtree(app_dir)
        await botnav.bot.send_message(message.chat.id, "✅ App deleted.")
    except Exception as exc:
        logger.exception(exc)
        await botnav.bot.send_message(message.chat.id, "Failed to delete app.")

    await start_my_apps(botnav, message)
