import os
import asyncio
import functools
import json
from io import BytesIO
from copy import copy

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
        'description': 'Stable Diffusion, a latent text-to-image diffusion model capable of generating photo-realistic images given any text input',
        'input_type': 'text',
        'output_type': 'photo'
    },
    'real-esrgan': {
        'description': 'Real-ESRGAN is a GAN-based image super-resolution model trained on real-world images. It can be used to upscale images to 4x the original resolution.',
        'replicate_id': 'nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b',
        'input_type': 'photo',
        'output_type': 'file',
        'available_params': {
            'scale': {
                'type': 'int',
                'default': 4,
                'min': 2,
                'max': 10,
                'description': 'Scale factor'
            },
        }
    },
    'kandinsky': {
        'description': 'Kandinsky 2.1, text2img model trained on LAION HighRes and fine-tuned on internal datasets',
        'replicate_id': 'ai-forever/kandinsky-2:601eea49d49003e6ea75a11527209c4f510a93e2112c969d548fbb45b9c4f19f',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'num_inference_steps': {
                'type': 'int',
                'default': 50,
                'min': 1,
                'max': 1000,
                'description': 'Number of denoising steps'
            },
            'guidance_scale': {
                'type': 'int',
                'default': 4,
                'description': 'Scale for classifier-free guidance'
            },
            'scheduler': {
                'type': 'select',
                'options': ['ddim_sampler', 'p_sampler', 'plms_sampler'],
                'default': 'p_sampler'
            },
            'width': {
                'type': 'int',
                'default': 512,
                'description': 'Width of the image'
            },
            'height': {
                'type': 'int',
                'default': 512,
                'description': 'Height of the image'
            },
        }
    },
    'dolly': {
        'description': 'Dolly, an open source instruction-tuned large language model developed by Databricks',
        'replicate_id': 'replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5',
        'input_type': 'text',
        'output_type': 'text'
    },
    'blip': {
        'description': 'Blip, bootstrapping Language-Image Pre-training, send photo to get caption',
        'replicate_id': 'salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746',
        'input_type': 'photo',
        'output_type': 'text',
        'available_params': {
            'task': {
                'type': 'select',
                'options': ['image_captioning', 'visual_question_answering'],
                'default': 'image_captioning'
            },
            'question': {
                'type': 'str',
                'description': 'Question for VQA'
            },
        }
    },
    'openjourney': {
        'description': 'OpenJourney, Stable Diffusion fine tuned on Midjourney v4 images.',
        'replicate_id': 'prompthero/openjourney:9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb',
        'input_type': 'text',
        'output_type': 'photo'
    },
    'styleclip': {
        'description': 'StyleCLIP, Text-Driven Manipulation of StyleGAN Imagery',
        'replicate_id': 'orpatashnik/styleclip:7af9a66f36f97fee2fece7dcc927551a951f0022cbdd23747b9212f23fc17021',
        'input_type': 'photo',
        'input_field': 'input',
        'output_type': 'photo',
        'available_params': {
            'neutral': {
                'type': 'str',
                'description': 'Neutral image description'
            },
            'target': {
                'type': 'str',
                'description': 'Target image description'
            },
        }
    }
}


def replicate_execute(replicate_id: str, input_data: dict):
    logger.info(input_data)
    output = replicate.run(
        replicate_id,
        input=input_data
    )
    return output


async def replicate_set_select_param(param_name: str, value: str, botnav: TeleBotNav, message: Message):
    await botnav.bot.delete_message(message.chat.id, message.message_id)
    message.state_data['replicate_params'][param_name] = value
    await botnav.bot.send_message(message.chat.id, f"Param {param_name} was set to: {value}")
    await replicate_print_params_buttons(botnav, message)


async def replicate_set_input_param(param_name: str, botnav: TeleBotNav, message: Message):
    param = REPLICATE_MODELS[message.state_data['replicate_model']]['available_params'][param_name]
    value = message.text

    if param['type'] == 'int':
        value = int(value)

    message.state_data['replicate_params'][param_name] = value
    await botnav.bot.send_message(message.chat.id, f"Param {param_name} was set to: {value}")
    await replicate_print_params_buttons(botnav, message)


async def replicate_choose_param(model_name_param_name: str, botnav: TeleBotNav, message: Message):
    model_name, param_name = model_name_param_name.split(':')

    if model_name not in REPLICATE_MODELS:
        return

    model = REPLICATE_MODELS[model_name]

    if param_name not in model['available_params']:
        return

    param = model['available_params'][param_name]
    await botnav.bot.delete_message(message.chat.id, message.message_id)

    if param['type'] == 'int':
        text = "Please enter integer value "
        if param.get('description'):
            text += f"({param['description']}) "
        if param.get('min'):
            text += f"greater than {param['min']} "
        if param.get('max'):
            text += f"less than {param['max']} "
        if param.get('default'):
            text += f"or leave empty for default value ({param['default']})"

        await botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
        await botnav.bot.send_message(message.chat.id, text)
        return

    if param['type'] == 'str':
        text = "Please enter string value "
        if param.get('description'):
            text += f"({param['description']}) "
        if param.get('default'):
            text += f"or leave empty for default value ({param['default']})"

        await botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
        await botnav.bot.send_message(message.chat.id, text)
        return

    if param['type'] == 'select':
        text = "Please choose one of the following options "
        if param.get('description'):
            text += f"({param['description']}) "

        buttons = {
            x: functools.partial(replicate_set_select_param, param_name, x) for x in param['options']
        }

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            text=text,
            row_width=2,
        )


def replicate_get_params_buttons(model_name: str):
    model = REPLICATE_MODELS[model_name]
    buttons = {
        x: functools.partial(replicate_choose_param, f'{model_name}:{x}') for x in model['available_params'].keys()
    }

    return buttons


async def replicate_print_params_buttons(botnav: TeleBotNav, message: Message):
    model = REPLICATE_MODELS[message.state_data['replicate_model']]

    if model.get('available_params'):
        buttons = replicate_get_params_buttons(message.state_data['replicate_model'])

        await botnav.print_buttons(
            message.chat.id,
            buttons,
            text="If you want to set additional params use buttons:",
            row_width=2,
        )


async def replicate_choose_model(model_name: str, botnav: TeleBotNav, message: Message) -> None:
    if model_name not in REPLICATE_MODELS:
        return

    model = REPLICATE_MODELS[model_name]

    default_params = copy(model.get('default_params', {}))

    message.state_data['replicate_model'] = model_name
    message.state_data['replicate_params'] = default_params

    await botnav.bot.send_message(message.chat.id, "Model was set to: " + REPLICATE_MODELS[model_name]['description'])

    if model.get('available_params'):
        await replicate_print_params_buttons(botnav, message)


async def download_file(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception('Could not download file')

            file = BytesIO(await resp.read())
            file.name = os.path.basename(url)
            return file


def get_await_action_type(model):
    if model['output_type'] == 'photo':
        return 'upload_photo'
    if model['output_type'] == 'text':
        return 'typing'
    if model['output_type'] == 'file':
        return 'upload_document'


async def replicate_message_handler(botnav: TeleBotNav, message: Message) -> None:
    replicate_model_name = message.state_data.get('replicate_model', None)
    if not replicate_model_name:
        return

    replicate_model = REPLICATE_MODELS[replicate_model_name]

    input_data = message.state_data.get('replicate_params', {})

    if message.content_type == 'text' != replicate_model['input_type']:
        return

    if message.content_type == 'photo' != replicate_model['input_type']:
        return

    if message.content_type == 'text':
        input_data['prompt'] = message.text

    if message.content_type == 'photo':
        file_info = await botnav.bot.get_file(message.photo[-1].file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        input_data[replicate_model.get('input_field', 'image')] = BytesIO(file_content)

    try:
        result = await botnav.await_coro_sending_action(
            message.chat.id,
            asyncio.to_thread(replicate_execute, replicate_model['replicate_id'], input_data),
            get_await_action_type(replicate_model)
        )

        if replicate_model['output_type'] == 'photo':
            if isinstance(result, list):
                for photo in result:
                    await botnav.await_coro_sending_action(
                        message.chat.id,
                        botnav.bot.send_photo(message.chat.id, photo),
                        'upload_photo'
                    )
            elif isinstance(result, str):
                await botnav.await_coro_sending_action(
                    message.chat.id,
                    botnav.bot.send_photo(message.chat.id, result),
                    'upload_photo'
                )

        if replicate_model['output_type'] == 'text':
            parts = []
            for part in result:
                await botnav.send_chat_action(message.chat.id, 'typing')
                parts.append(part)

                if len(parts) > 500:
                    await botnav.bot.send_message(message.chat.id, "".join(parts))
                    parts = []
            await botnav.bot.send_message(message.chat.id, "".join(parts))

        if replicate_model['output_type'] == 'file':
            if isinstance(result, list):
                for document_url in result:
                    document = botnav.await_coro_sending_action(
                        message.chat.id,
                        download_file(document_url),
                        'upload_document'
                    )

                    await botnav.await_coro_sending_action(
                        message.chat.id,
                        botnav.bot.send_document(message.chat.id, document, timeout=120),
                        'upload_document'
                    )
            elif isinstance(result, str):
                    document = await botnav.await_coro_sending_action(
                        message.chat.id,
                        download_file(result),
                        'upload_document'
                    )

                    await botnav.await_coro_sending_action(
                        message.chat.id,
                        botnav.bot.send_document(message.chat.id, document, timeout=120),
                        'upload_document'
                    )
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
