import time
import asyncio
import functools
from typing import Optional
from typing import Callable

from telebot import ExceptionHandler
from telebot.async_telebot import AsyncTeleBot
from telebot.asyncio_storage import StateMemoryStorage
from telebot.asyncio_storage import StateStorageBase
from telebot.types import BotCommand
from telebot.types import CallbackQuery
from telebot.types import InlineKeyboardMarkup
from telebot.types import InlineKeyboardButton
from telebot.types import Message
from telebot.types import User

from logger import logger


def throttle(delay: int):
    @functools.lru_cache(1024)
    def run_task(time_arg: float, func: Callable, *args, **kwargs):
        return asyncio.create_task(func(*args, **kwargs))


    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            time_arg = time.monotonic()
            time_arg = time_arg - (time_arg % delay)

            task = run_task(time_arg, func, *args, **kwargs)

            return await task

        return wrapper

    return decorator


class TeleBotNav:
    def __init__(
        self,
        token: str,
        exception_handler: Optional[ExceptionHandler] = None,
        state_storage: Optional[StateStorageBase] = None
    ):
        self.buttons = {}
        self.commands = {}
        self.handlers = {}
        self.global_default_handler = None

        if not state_storage:
            state_storage = StateMemoryStorage()

        self.bot = AsyncTeleBot(
            token,
            state_storage=state_storage,
            exception_handler=exception_handler
        )

        self.bot.register_callback_query_handler(self.callback_query_handler, func=lambda message: True)
        self.bot.register_message_handler(self.message_handler, content_types=['audio', 'photo', 'voice', 'video', 'document',
            'text', 'location', 'contact', 'sticker'])

    async def set_globl_default_handler(self, func: Callable) -> None:
        self.global_default_handler = func

    async def set_default_handler(self, message: Message, func: Callable) -> None:
        message.state_data['default_handler'] = func

    async def set_next_handler(self, message: Message, func: Callable) -> None:
        message.state_data['next_handler'] = func

    async def set_command(self, command: str, description: str, func: Callable) -> None:
        self.commands[command] = {
            'description': description,
            'func': func,
        }

    async def send_commands(self) -> None:
        await self.bot.set_my_commands([
            BotCommand(x, y['description']) for x, y in self.commands.items()
        ])

    async def callback_query_handler(self, call: CallbackQuery) -> None:
        func_name = 'unknown'
        if call.data in self.buttons:
            if hasattr(self.buttons[call.data], '__name__'):
                func_name = self.buttons[call.data].__name__
            else:
                func_name = str(self.buttons[call.data])

        logger.info(f"{call.from_user.username} pressed: {call.data}({func_name})")

        # To make data appear
        await self.bot.set_state(call.from_user.id, call.message.chat.id, '')

        async with self.bot.retrieve_data(call.from_user.id, call.message.chat.id) as state_data:
            call.message.state_data = state_data
            call.message.user = call.from_user
            if call.data in self.buttons:
                await self.buttons[call.data](self, call.message)
            elif self.global_default_handler:
                await self.bot.set_state(call.message.chat.id, '')
                await self.global_default_handler(self, call.message)


    async def message_handler(self, message: Message) -> None:
        logger.info(f"{self.get_user(message).username} sent: {message.text}")

        # To make data appear
        await self.bot.set_state(message.from_user.id, message.chat.id, '')

        async with self.bot.retrieve_data(message.from_user.id, message.chat.id) as state_data:
            message.state_data = state_data
            if message.content_type == 'text' and message.text.startswith('/') and message.text[1:] in self.commands:
                await self.commands[message.text[1:]]['func'](self, message)
            elif state_data and 'next_handler' in state_data:
                func = state_data.pop('next_handler')
                await func(self, message)
            elif state_data and 'default_handler' in state_data:
                await state_data['default_handler'](self, message)
            elif self.global_default_handler:
                await self.global_default_handler(self, message)

    async def print_buttons(
        self,
        chat_id: int,
        buttons: dict,
        text: str = "",
        message_to_rewrite: Optional[int] = None,
        row_width: int = 1
    ) -> None:
        markup = InlineKeyboardMarkup()
        markup.row_width = row_width

        for x, y in buttons.items():
            self.buttons[str(y.__hash__())] = y

        markup.add(*[InlineKeyboardButton(x, callback_data=f'{y.__hash__()}') for x, y in buttons.items() if y])
        if message_to_rewrite:
            await self.bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_to_rewrite.message_id,
                reply_markup=markup
            )
        else:
            await self.bot.send_message(chat_id, text, reply_markup=markup)

    @throttle(1)
    async def send_chat_action(self, chat_id: int, action: str) -> None:
        await self.bot.send_chat_action(chat_id, action)

    def get_user(self, message: Message) -> User:
        if hasattr(message ,'user'):
            return message.user

        return message.from_user