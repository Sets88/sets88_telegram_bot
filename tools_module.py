import json
import asyncio
import functools
from datetime import datetime
from datetime import timezone

from telebot.types import Message

from telebot_nav import TeleBotNav
from logger import logger


def split_by_chunks(text: str, length: int) -> str:
    lines = text.split('\n')
    result = ''
    for line in lines:
        while len(line) > length:
            if len(result) + len(line[:length]) > length:
                yield result
                result = ''

            result += line[:length]

            yield result

            line = line[length:]
            result = ''

        if len(result) + len(line) > length:
            yield result
            result = ''

        result += line + '\n'

    if result:
        yield result


async def run_command_handler(command: str, botnav: TeleBotNav, message: Message) -> None:
    try:
        proc = await botnav.await_coro_sending_action(
            message.chat.id,
            asyncio.create_subprocess_exec(
                command, message.text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            ),
            'typing'
        )

        data = await proc.stdout.read()

        for text in split_by_chunks(data.decode(), 4000):
            await botnav.bot.send_message(
                message.chat.id,
                f'```\n{text}\n```',
                disable_web_page_preview=True,
                parse_mode='MarkdownV2'
            )
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)


async def whois(botnav: TeleBotNav, message: Message) -> None:
    try:
        await botnav.bot.send_message(message.chat.id, 'Send domain or ip address to get whois info')
        botnav.set_default_handler(message, functools.partial(run_command_handler, 'whois'))
        botnav.clean_next_handler(message)
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)


async def message_details_handler(botnav: TeleBotNav, message: Message) -> None:
    try:
        main_msg = json.dumps(message.json, indent=4, ensure_ascii=False)

        for text in split_by_chunks(main_msg, 4000):
            await botnav.bot.send_message(
                message.chat.id,
                f'```json\n{text}\n```',
                disable_web_page_preview=True,
                parse_mode='MarkdownV2'
            )
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)


async def message_details(botnav: TeleBotNav, message: Message) -> None:
    try:
        await botnav.bot.send_message(message.chat.id, 'Forward message to get more details about it')
        botnav.set_default_handler(message, message_details_handler)
        botnav.clean_next_handler(message)
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)



async def unixtime_handler(botnav: TeleBotNav, message: Message) -> None:
    try:
        msg = message.text.strip()
        if msg.replace(".", "").isnumeric():
            dt = datetime.utcfromtimestamp(float(msg))
            await botnav.bot.send_message(
                message.chat.id,
                f'Unixtime ```\n{msg}\n``` UTC datetime```\n{dt}\n```',
                parse_mode='MarkdownV2'
            )
            return

        unixt = int(datetime.strptime(message.text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())

        await botnav.bot.send_message(
            message.chat.id,
            f'UTC datetime```\n{msg}\n``` Unixtime ```\n{unixt}\n```',
            parse_mode='MarkdownV2'
        )
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)


async def unixtime(botnav: TeleBotNav, message: Message) -> None:
    try:
        await botnav.bot.send_message(
            message.chat.id,
            'Pass unixtime to get datetime or datetime \(format: YYYY\-MM\-DD HH:mm:ss\) to get unixtime, \ncurrent UTC datetime' +
            f'```\n{datetime.utcnow().replace(microsecond=0)}\n``` Unixtime' +
            f'```\n{int(datetime.now().timestamp())}\n```',
            parse_mode='MarkdownV2'
        )
        botnav.set_default_handler(message, unixtime_handler)
        botnav.clean_next_handler(message)
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)


async def start_tools(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            'Msg details': message_details,
            'Whois': whois,
            'unixtime': unixtime
        }, 'Choose',
        row_width=2
    )
    botnav.wipe_commands(message, preserve=['start'])
    await botnav.send_commands(message)
