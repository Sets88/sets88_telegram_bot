import os
import asyncio
import functools
from uuid import uuid4

import yt_dlp
from telebot.types import Message

from telebot_nav import TeleBotNav
from logger import logger


VIDEO_TMP_DIR = '/tmp'


def download_file(url: str, options: dict) -> None:
    with yt_dlp.YoutubeDL(dict(format=options['format'])) as ydl:
        filename = f'video_downloaded_{uuid4().hex}'
        ydl.params['outtmpl']['default'] = f'{VIDEO_TMP_DIR}/{filename}.%(ext)s'
        ydl.download(url)
        filename = [x for x in os.listdir(VIDEO_TMP_DIR) if x.startswith(filename)][0]
        return os.path.join(VIDEO_TMP_DIR, filename)


async def yt_set_format(format: str, botnav: TeleBotNav, message: Message) -> None:
    message.state_data['options']['format'] = format
    await botnav.bot.send_message(message.chat.id, f'Format set to {format}')


async def yt_format(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            'Audio+Video': functools.partial(yt_set_format, 'mp4/bestaudio/best'),
            'Audio Only': functools.partial(yt_set_format, 'm4a[vcodec=none]'),
        },
        message_to_rewrite=message,
        row_width=2
    )


async def youtube_dl_message_handler(botnav: TeleBotNav, message: Message) -> None:
    if message.content_type != 'text':
        return

    try:
        filename = await botnav.await_coro_sending_action(
            message.chat.id,
            asyncio.to_thread(download_file, message.text, message.state_data['options']),
            'upload_video'
        )

        try:
            ext = os.path.splitext(filename)[-1].strip('.')
            if ext == 'm4a':
                await botnav.await_coro_sending_action(
                    message.chat.id,
                    botnav.bot.send_audio(message.chat.id, open(filename, 'rb')),
                    'upload_audio'
                )
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
    await botnav.print_buttons(
        message.chat.id,
        {
            'Format': yt_format,
        },
        'For configurations purpose:',
        row_width=2
    )

    message.state_data['options'] = {
        'format': 'mp4/bestaudio/best'
    }

    await botnav.bot.send_message(message.chat.id, 'Welcome to Youtube download, send me links to download!')
    await botnav.set_default_handler(message, youtube_dl_message_handler)
