import functools

import anthropic
from telebot.types import Message

import config
from telebot_nav import TeleBotNav


CLAUDE_MODELS = [
    'claude-3-haiku-20240307',
    'claude-3-sonnet-20240229',
    'claude-3-opus-20240229'
]


claude_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


def get_default_claude_params() -> dict:
    return {
        'model': CLAUDE_MODELS[0],
        'max_tokens': 4096,
        'system': None,
        'messages': [],
    }


async def claude_send_message(botnav: TeleBotNav, message: Message) -> None:
    if 'claude_params' not in message.state_data:
        message.state_data['claude_params'] = get_default_claude_params()

    conv_params = message.state_data['claude_params']

    conv_params['messages'].append({
        'role': 'user',
        'content': message.text
    
    })

    stream = await claude_client.messages.create(
        max_tokens=conv_params['max_tokens'],
        messages=conv_params['messages'],
        model=conv_params['model'],
        stream=True,
    )

    full_message = []

    message_parts = []

    async for event in stream:
        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
            message_parts.append(event.delta.text)
            full_message.append(event)

        if len(message_parts) > 500:
            await botnav.bot.send_message(message.chat.id, "".join(message_parts))
            message_parts = []

    if message_parts:
        await botnav.bot.send_message(message.chat.id, "".join(message_parts))
    
    conv_params['messages'].append({
        'role': 'assistant',
        'content': "".join(message_parts)
    })


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
            '🤖 Choose Model': claude_models_list_menu,
        },
        text='Options:',
        row_width=2
    )


async def claude_message_handler(botnav: TeleBotNav, message: Message) -> None:
    if message.content_type != 'text':
        return

    await claude_send_message(botnav, message)


async def start_claude(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            'Options': claude_options_menu,
        },
        'Welcome to Claude 3, send me a message to start a conversation!',
        row_width=2
    )

    await botnav.set_default_handler(message, claude_message_handler)


