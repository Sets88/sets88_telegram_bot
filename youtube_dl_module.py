import os
import asyncio
from uuid import uuid4

import yt_dlp
from telebot.types import Message

from telebot_nav import TeleBotNav
from logger import logger


VIDEO_TMP_DIR = '/tmp'


def download_file(url: str) -> None:
    with yt_dlp.YoutubeDL(dict(format='mp4/bestaudio/best')) as ydl:
        filename = f'video_downloaded_{uuid4().hex}'
        ydl.params['outtmpl']['default'] = f'{VIDEO_TMP_DIR}/{filename}.%(ext)s'
        ydl.download(url)
        filename = [x for x in os.listdir(VIDEO_TMP_DIR) if x.startswith(filename)][0]
        return os.path.join(VIDEO_TMP_DIR, filename)


async def youtube_dl_message_handler(botnav: TeleBotNav, message: Message) -> None:
    if message.content_type != 'text':
        return

    try:
        filename = await botnav.await_coro_sending_action(
            message.chat.id,
            asyncio.to_thread(download_file, message.text),
            'upload_video'
        )

        try:
            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_video(message.chat.id, open(filename, 'rb'), supports_streaming=True),
                'upload_video'
            )
        finally:
            os.unlink(filename)
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)


async def start_youtube_dl(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.send_message(message.chat.id, 'Welcome to Youtube download, send me links to download!')
    await botnav.set_default_handler(message, youtube_dl_message_handler)
