import os
import asyncio
import functools
from io import BytesIO

from telebot.types import Message
import replicate
import aiohttp

import config
from telebot_nav import TeleBotNav
from logger import logger


if config.REPLICATE_API_KEY:
    os.environ['REPLICATE_API_TOKEN'] = config.REPLICATE_API_KEY


REPLICATE_MODELS = {
    'stable-diffusion': {
        'replicate_id': 'stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf',
        'description': 'Stable Diffusion is a GAN-based image super-resolution model trained on real-world images. It can be used to upscale images to 4x the original resolution.',
        'input_type': 'text',
        'output_type': 'photo'
    },
    'real-esrgan': {
        'description': 'Real-ESRGAN is a GAN-based image super-resolution model trained on real-world images. It can be used to upscale images to 4x the original resolution.',
        'replicate_id': 'nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b',
        'input_type': 'photo',
        'output_type': 'file'
    },
    'kandinsky': {
        'description': 'text2img model trained on LAION HighRes and fine-tuned on internal datasets',
        'replicate_id': 'ai-forever/kandinsky-2:601eea49d49003e6ea75a11527209c4f510a93e2112c969d548fbb45b9c4f19f',
        'input_type': 'text',
        'output_type': 'photo'
    },
    'dolly': {
        'description': 'An open source instruction-tuned large language model developed by Databricks',
        'replicate_id': 'replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5',
        'input_type': 'text',
        'output_type': 'text'
    },
    'blip': {
        'description': 'Bootstrapping Language-Image Pre-training, send photo to get caption',
        'replicate_id': 'salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746',
        'input_type': 'photo',
        'output_type': 'text'
    },
    'openjourney': {
        'description': 'Stable Diffusion fine tuned on Midjourney v4 images.',
        'replicate_id': 'prompthero/openjourney:9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb',
        'input_type': 'text',
        'output_type': 'photo'
    }
}


def replicate_execute(replicate_id, input_data):
    output = replicate.run(
        replicate_id,
        input=input_data
    )
    return output


async def replicate_choose_model(model: str, botnav: TeleBotNav, message: Message) -> None:
    if model not in REPLICATE_MODELS:
        return

    message.state_data['replicate_model'] = model

    await botnav.bot.send_message(message.chat.id, "Model was set to: " + REPLICATE_MODELS[model]['description'])


async def download_file(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception('Could not download file')

            file = BytesIO(await resp.read())
            file.name = os.path.basename(url)
            return file


async def send_sending_action(botnav, message, model):
    if model['output_type'] == 'photo':
        await botnav.send_chat_action(message.chat.id, 'upload_photo')
    if model['output_type'] == 'text':
        await botnav.send_chat_action(message.chat.id, 'typing')
    if model['output_type'] == 'file':
        await botnav.send_chat_action(message.chat.id, 'upload_document')


async def await_response(botnav: TeleBotNav, message: Message, model, coro: asyncio.coroutine):
    task = asyncio.create_task(coro)
    while not task.done():
        await asyncio.sleep(0.1)
        await send_sending_action(botnav, message, model)
    return await task


async def replicate_message_handler(botnav: TeleBotNav, message: Message) -> None:
    replicate_model_name = message.state_data.get('replicate_model', None)
    if not replicate_model_name:
        return

    replicate_model = REPLICATE_MODELS[replicate_model_name]
    input_data = {}

    input_data.update(replicate_model.get('default_params', {}))

    if message.content_type == 'text' != replicate_model['input_type']:
        return

    if message.content_type == 'photo' != replicate_model['input_type']:
        return

    if message.content_type == 'text':
        input_data['prompt'] = message.text

    if message.content_type == 'photo':
        file_info = await botnav.bot.get_file(message.photo[-1].file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        input_data['image'] = BytesIO(file_content)

    try:
        result = await await_response(
            botnav,
            message,
            replicate_model,
            asyncio.to_thread(replicate_execute, replicate_model['replicate_id'], input_data)
        )

        if replicate_model['output_type'] == 'photo':
            if isinstance(result, list):
                for photo in result:
                    await botnav.bot.send_photo(message.chat.id, photo)
            elif isinstance(result, str):
                await botnav.bot.send_photo(message.chat.id, result)
        if replicate_model['output_type'] == 'text':
            parts = []
            for part in result:
                await send_sending_action(botnav, message, replicate_model)
                parts.append(part)

                if len(parts) > 500:
                    await botnav.bot.send_message(message.chat.id, "".join(parts))
                    parts = []
            await botnav.bot.send_message(message.chat.id, "".join(parts))
        if replicate_model['output_type'] == 'file':
            if isinstance(result, list):
                for document_url in result:
                    document = await download_file(document_url)
                    await botnav.bot.send_document(message.chat.id, document)
            elif isinstance(result, str):
                document = await download_file(result)
                await botnav.bot.send_document(message.chat.id, document)
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)
        message.state_data.clear()


async def start_replicate(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            x: functools.partial(replicate_choose_model, x) for x in REPLICATE_MODELS.keys()
        },
        message_to_rewrite=message,
        row_width=2,
    )

    await botnav.set_default_handler(message, replicate_message_handler)
