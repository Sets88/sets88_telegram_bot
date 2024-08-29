import functools
import base64

import anthropic
from telebot.types import Message

from logger import logger
import config
from telebot_nav import TeleBotNav
from lib.utils import MessageSplitter


CLAUDE_MODELS = [
    'claude-3-5-sonnet-20240620',
    'claude-3-haiku-20240307',
    'claude-3-sonnet-20240229',
    'claude-3-opus-20240229'
]


claude_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


def get_default_claude_params() -> dict:
    return {
        'model': CLAUDE_MODELS[0],
        'max_tokens': 4096,
        'temperature': 1.0,
        'system': '',
        'messages': [],
        'one_off': False
    }


async def claude_send_message(botnav: TeleBotNav, message: Message) -> None:
    try:
        if 'claude_params' not in message.state_data:
            message.state_data['claude_params'] = get_default_claude_params()

        if message.state_data['claude_params'].get('one_off', False):
            message.state_data['claude_params']['messages'] = []

        conv_params = message.state_data['claude_params']

        if message.content_type == 'photo':

            file_info = await botnav.bot.get_file(message.photo[-1].file_id)
            image = await botnav.bot.download_file(file_info.file_path)
            image = base64.b64encode(image).decode('utf-8')

            data = {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/jpeg',
                            'data': image
                        }
                    }
                ]
            }
            if message.caption:
                data['content'].append({
                    'type': 'text',
                    'text': message.caption
                })
            conv_params['messages'].append(data)
        if message.content_type == 'text':
            conv_params['messages'].append({
                'role': 'user',
                'content': message.text
            })

        message_parts = []

        if 'put_in_the_mouth' in conv_params:
            conv_params['messages'].append({
                'role': 'assistant',
                'content': conv_params['put_in_the_mouth']
            })
            message_parts.append(conv_params['put_in_the_mouth'] + ' ')
            del conv_params['put_in_the_mouth']

        stream = await claude_client.messages.create(
            max_tokens=conv_params['max_tokens'],
            messages=conv_params['messages'],
            system=conv_params['system'],
            temperature=conv_params['temperature'],
            model=conv_params['model'],
            stream=True,
        )

        message_splitter = MessageSplitter(4000)

        async for event in stream:
            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                msg = message_splitter.add(event.delta.text)

                if msg:
                    message_parts += msg
                    await botnav.bot.send_message(message.chat.id, "".join(msg))

        for msg in message_splitter.flush():
            message_parts += msg
            await botnav.bot.send_message(message.chat.id, "".join(msg))

        conv_params['messages'].append({
            'role': 'assistant',
            'content': "".join(message_parts)
        })
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)
        message.state_data.clear()


async def claude_set_system_meta(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    await botnav.bot.send_message(message.chat.id, f"You set the system meta: {message.text}")
    message.state_data['claude_params']['system'] = message.text


async def claude_put_in_the_mouth(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    message.state_data['claude_params']['put_in_the_mouth'] = message.text
    await botnav.bot.send_message(message.chat.id, f"You put in the Claude's mouth: {message.text}")


async def claude_set_temperature(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    message.state_data['claude_params']['temperature'] = float(message.text)
    await botnav.bot.send_message(message.chat.id, "You set temperature to: " + message.text)


async def claude_set_max_tokens(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    message.state_data['claude_params']['max_tokens'] = int(message.text)
    await botnav.bot.send_message(message.chat.id, f"You set maximum amount of tokens in response to: {message.text}")


async def claude_set_temperature_request(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    await botnav.bot.send_message(
        message.chat.id,
        f'Send text to set temperature of Claude 3, current values is ${message.state_data["claude_params"]["temperature"]}'
    )
    botnav.set_next_handler(message, claude_set_temperature)


async def claude_set_max_tokens_request(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    await botnav.bot.send_message(
        message.chat.id,
        f'Send text to set temperature of Claude 3, current values is ${message.state_data["claude_params"]["max_tokens"]}'
    )
    botnav.set_next_handler(message, claude_set_max_tokens)


async def claude_set_system_meta_request(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.send_message(message.chat.id, "Send text to set system meta of Claude 3")
    botnav.set_next_handler(message, claude_set_system_meta)


async def claude_put_in_the_mouth_request(botnav: TeleBotNav, message: Message) -> None:
    await botnav.bot.send_message(message.chat.id, "Send text to put in the Claude's mouth")
    botnav.set_next_handler(message, claude_put_in_the_mouth)


async def claude_switch_one_off(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    message.state_data['claude_params']['one_off'] = not message.state_data['claude_params'].get('one_off', False)

    await botnav.bot.edit_message_text(
        f'One off mode has been switched to {message.state_data["claude_params"]["one_off"]}',
        message.chat.id,
        message.message_id
    )


async def claude_reset_conversation(botnav: TeleBotNav, message: Message) -> None:
    message.state_data['claude_params'] = get_default_claude_params()
    await botnav.bot.edit_message_text("All conversation parameters have been reset", message.chat.id, message.message_id)


async def claude_clean_conversation(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' in message.state_data:
        message.state_data['claude_params']['messages'] = []
        await botnav.bot.edit_message_text("Conversation messages have been cleared", message.chat.id, message.message_id)
        return

    await claude_reset_conversation(botnav, message)


async def claude_set_model(model: str, botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    message.state_data['claude_params']['model'] = model
    await botnav.bot.edit_message_text(f"Model has been set to {model}", message.chat.id, message.message_id)


async def claude_models_list_menu(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            x: functools.partial(claude_set_model, x) for x in CLAUDE_MODELS
        },
        'Choose a model',
        row_width=1
    )


async def claude_options_menu(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            '🧹 Clean conversation': claude_clean_conversation,
            '🔄 Reset conversation': claude_reset_conversation,
            '👄 Put in the mouth': claude_put_in_the_mouth_request,
            '🗄️ Set system meta data': claude_set_system_meta_request,
            '🤖 Choose Model': claude_models_list_menu,
            '🌡️ Set Temperature': claude_set_temperature_request,
            '🔢 Set Max tokens': claude_set_max_tokens_request,
            '🎯 One Off': claude_switch_one_off
        },
        text='Options:',
        row_width=2
    )


async def claude_message_handler(botnav: TeleBotNav, message: Message) -> None:
    if message.content_type not in ('text', 'photo'):
        return

    await botnav.await_coro_sending_action(
        message.chat.id,
        claude_send_message(botnav, message),
        'typing'
    )


async def start_claude(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            'Options': claude_options_menu,
        },
        'Welcome to Claude 3, send me a message to start a conversation!',
        row_width=2
    )

    botnav.wipe_commands(message, preserve=['start'])
    botnav.add_command(message, 'clean_conversation', '🧹 Clean conversation', claude_clean_conversation)
    botnav.add_command(message, 'reset_conversation', '🔄 Reset conversation', claude_reset_conversation)
    botnav.add_command(message, 'other_options', '⚙️ Options', claude_options_menu)

    await botnav.send_commands(message)
    botnav.set_default_handler(message, claude_message_handler)
    botnav.clean_next_handler(message)
