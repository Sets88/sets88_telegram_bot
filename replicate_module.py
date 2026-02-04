import os
import asyncio
import functools
from io import BytesIO
from copy import copy
from typing import Any
import mimetypes

from telebot.types import Message
import replicate
from replicate.helpers import FileOutput
from replicate.exceptions import ModelError
import aiohttp

from lib.permissions import get_permission, is_permitted
import config
from telebot_nav import TeleBotNav
from logger import logger


if config.REPLICATE_API_KEY:
    os.environ['REPLICATE_API_TOKEN'] = config.REPLICATE_API_KEY


REPLICATE_MODELS = {
    'stable-diffusion': {
        'replicate_id': 'stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316',
        'description': 'Stable Diffusion, a latent text-to-image diffusion model capable of generating photo-realistic images given any text input',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'image': {
                'type': 'photo',
                'description': 'Input image, for image2image task'
            },
            'mask': {
                'type': 'photo',
                'description': 'Mask image, for inpaint task'
            }
        }
    },
    'real-esrgan': {
        'description': 'Real-ESRGAN is a GAN-based image super-resolution model trained on real-world images. It can be used to upscale images to 4x the original resolution.',
        'replicate_id': 'nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa',
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
        'description': 'Kandinsky 2.2, text2img model trained on LAION HighRes and fine-tuned on internal datasets',
        'replicate_id': 'ai-forever/kandinsky-2.2:ea1addaab376f4dc227f5368bbd8eff901820fd1cc14ed8cad63b29249e9d463',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'num_inference_steps': {
                'type': 'int',
                'default': 75,
                'min': 1,
                'max': 1000,
                'description': 'Number of denoising steps'
            },
            'width': {
                'type': 'int',
                'default': 1024,
                'min': 384,
                'max': 2048,
                'description': 'Width of the image'
            },
            'height': {
                'type': 'int',
                'default': 1024,
                'min': 384,
                'max': 2048,
                'description': 'Height of the image'
            },
            'num_outputs': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 10,
                'description': 'Amount of images'
            }
        }
    },
    'nano-banana-pro': {
        'description': 'Google\'s state of the art image generation and editing model',
        'replicate_id': 'google/nano-banana-pro',
        'input_field': 'prompt',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'image_input': {
                'type': 'photo_list',
                'description': 'Input images'
            }
        }
    },
    'qwen-image-edit': {
        'description': 'The latest Qwen-Image’s iteration with improved multi-image editing, single-image consistency, and native support for ControlNet',
        'replicate_id': 'qwen/qwen-image-edit-plus',
        'input_field': 'prompt',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'image': {
                'type': 'photo_list',
                'description': 'Input images'
            }
        }
    },
    'flux-2-pro': {
        'description': 'High-quality image generation and editing with support for eight reference images',
        'replicate_id': 'black-forest-labs/flux-2-pro',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'safety_tolerance': {
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 5,
                'description': 'Safety tolerance, 1 is most strict and 5 is most permissive'
            },
            'input_images': {
                'type': 'photo_list',
                'description': 'Input images'
            }

        }
    },
    'seedream-4.5': {
        'description': 'Unified text-to-image generation and precise single-sentence editing at up to 4K',
        'replicate_id': 'bytedance/seedream-4.5',
        'input_field': 'prompt',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'image_input': {
                'type': 'photo_list',
                'description': 'Input images'
            }
        }
    },
    'veo-3.1': {
        'description': 'New and improved version of Veo 3, with higher-fidelity video, context-aware audio, reference image and last frame support',
        'replicate_id': 'google/veo-3.1',
        'input_type': 'text',
        'input_field': 'prompt',
        'output_type': 'video',
        'available_params': {
            'reference_images': {
                'type': 'photo_list',
                'description': 'Input images'
            },
            'duration': {
                'type': 'int',
                'default': 8,
                'min': 1,
                'max': 10,
                'description': 'Duration of the video in seconds'
            }
        }
    },
    'veo-3.1-fast': {
        'description': 'New and improved version of Veo 3 Fast, with higher-fidelity video, context-aware audio and last frame support',
        'replicate_id': 'google/veo-3.1-fast',
        'input_type': 'text',
        'input_field': 'prompt',
        'output_type': 'video',
        'available_params': {
            'image': {
                'type': 'photo',
                'description': 'Input image'
            },
            'duration': {
                'type': 'int',
                'default': 8,
                'min': 1,
                'max': 10,
                'description': 'Duration of the video in seconds'
            }
        }
    },
    'speech-02-turbo': {
        'description': 'Text-to-Audio (T2A) that offers voice synthesis, emotional expression, and multilingual capabilities. Designed for real-time applications with low latency',
        'replicate_id': 'minimax/speech-02-turbo',
        'input_type': 'text',
        'input_field': 'text',
        'output_type': 'audio'
    }
}


def get_model_params(telebot_nav: TeleBotNav, message: Message, model_name: str) -> dict[str, Any]:
    if model_name not in REPLICATE_MODELS:
        return {}

    if not is_permitted(telebot_nav, message, 'can_use_replicate_models'):
        return {}

    exclude_models = []

    if not get_permission(telebot_nav, message, 'is_admin'):
        exclude_models = get_permission(telebot_nav, message, 'exclude_replicate_models')

    if exclude_models and model_name in exclude_models:
        return {}

    return REPLICATE_MODELS[model_name]


def replicate_execute(replicate_id: str, input_data: dict):
    logger.info(input_data)
    output = replicate.run(
        replicate_id,
        input=input_data
    )
    return output


async def replicate_execute_and_send(botnav: TeleBotNav, message: Message, model_name: str, input_data: dict[str, Any]) -> Any:
    replicate_model = get_model_params(botnav, message, model_name)

    if not replicate_model:
        raise ValueError(f'Unknown model {model_name}')

    result = await botnav.await_coro_sending_action(
        message.chat.id,
        asyncio.to_thread(replicate_execute, replicate_model['replicate_id'], input_data),
        get_await_action_type(replicate_model)
    )

    if isinstance(result, list) and replicate_model['output_type'] == 'video':
        result_videos = []
        for video in result:
            video_to_send = video
            if isinstance(video, FileOutput):
                video_to_send = video.url
            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_video(message.chat.id, video_to_send),
                'upload_video'
            )
            result_videos.append(video)
        return result_videos

    elif isinstance(result, list) and replicate_model['output_type'] == 'photo':
        result_photos = []
        for photo in result:
            image_to_send = photo

            if isinstance(photo, FileOutput):
                image_to_send = photo.url

            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_photo(message.chat.id, image_to_send),
                'upload_photo'
            )
            result_photos.append(photo)
        return result_photos

    elif isinstance(result, str) and replicate_model['output_type'] == 'photo':
        await botnav.await_coro_sending_action(
            message.chat.id,
            botnav.bot.send_photo(message.chat.id, result),
            'upload_photo'
        )
        return result

    elif isinstance(result, str) and replicate_model['output_type'] == 'video':
        await botnav.await_coro_sending_action(
            message.chat.id,
            botnav.bot.send_video(message.chat.id, result),
            'upload_video'
        )
    if replicate_model['output_type'] == 'file':
        if isinstance(result, FileOutput):
            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_document(message.chat.id, result.url, timeout=120),
                'typing'
            )

            document = await botnav.await_coro_sending_action(
                message.chat.id,
                download_file(result.url),
                'upload_document'
            )

            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_document(message.chat.id, document, timeout=120),
                'upload_document'
            )

            return document

        elif isinstance(result, list):
            document_list = []
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
                document_list.append(document)
            return document_list

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
                return document
    elif isinstance(result, FileOutput):
        mime_type, _ = mimetypes.guess_type(result.url)
        if mime_type and mime_type.startswith('video'):
            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_video(message.chat.id, result.url),
                'upload_video'
            )
            return result

        elif mime_type and mime_type.startswith('image'):
            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_photo(message.chat.id, result.url),
                'upload_photo'
            )
            return result
        else:
            document = await botnav.await_coro_sending_action(
                message.chat.id,
                download_file(result.url),
                'upload_document'
            )

            await botnav.await_coro_sending_action(
                message.chat.id,
                botnav.bot.send_document(message.chat.id, document, timeout=120),
                'upload_document'
            )
            return document

    if replicate_model['output_type'] == 'text':
        parts = []
        for part in result:
            await botnav.send_chat_action(message.chat.id, 'typing')
            parts.append(part)

            if len(parts) > 500:
                await botnav.bot.send_message(message.chat.id, "".join(parts))
                parts = []
        await botnav.bot.send_message(message.chat.id, "".join(parts))
        return "".join(parts)


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

    if param['type'] == 'float':
        value = float(value)

    if param['type'] == 'bool':
        value = bool(int(value))

    if param['type'] == 'photo':
        file_info = await botnav.bot.get_file(message.photo[-1].file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        value = BytesIO(file_content)

    if param['type'] == 'photo_list':
        if not message.state_data['replicate_params'].get(param_name):
            message.state_data['replicate_params'][param_name] = []

        value = message.state_data['replicate_params'][param_name]

        file_info = await botnav.bot.get_file(message.photo[-1].file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        value.append(BytesIO(file_content))

    message.state_data['replicate_params'][param_name] = value
    await botnav.bot.send_message(message.chat.id, f"Param {param_name} was set to: {value}")
    await replicate_print_params_buttons(botnav, message)


async def replicate_choose_param(model_name_param_name: str, botnav: TeleBotNav, message: Message):
    model_name, param_name = model_name_param_name.split(':')

    model = get_model_params(botnav, message, model_name)

    if not model:
        return

    if param_name not in model['available_params']:
        return

    param = model['available_params'][param_name]
    await botnav.bot.delete_message(message.chat.id, message.message_id)

    if param['type'] == 'bool':
        text = "Please enter bool value 1 for True or 0 for False"

        botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
        await botnav.bot.send_message(message.chat.id, text)
        return


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

        botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
        await botnav.bot.send_message(message.chat.id, text)
        return

    if param['type'] == 'float':
        text = "Please enter integer value "
        if param.get('description'):
            text += f"({param['description']}) "
        if param.get('min'):
            text += f"greater than {param['min']} "
        if param.get('max'):
            text += f"less than {param['max']} "
        if param.get('default'):
            text += f"or leave empty for default value ({param['default']})"

        botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
        await botnav.bot.send_message(message.chat.id, text)
        return

    if param['type'] == 'str':
        text = "Please enter string value "
        if param.get('description'):
            text += f"({param['description']}) "
        if param.get('default'):
            text += f"or leave empty for default value ({param['default']})"

        botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
        await botnav.bot.send_message(message.chat.id, text)
        return

    if param['type'] == 'photo':
        text = "Please send photo"
        if param.get('description'):
            text += f"({param['description']}) "

        botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
        await botnav.bot.send_message(message.chat.id, text)
        return

    if param['type'] == 'photo_list':
        text = "Please send photos"
        if param.get('description'):
            text += f"({param['description']}) "

        botnav.set_next_handler(message, functools.partial(replicate_set_input_param, param_name))
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
    model = get_model_params(botnav, message, model_name)
    if not model:
        return

    default_params = copy(model.get('default_params', {}))

    message.state_data['replicate_model'] = model_name
    message.state_data['replicate_params'] = default_params

    await botnav.bot.send_message(message.chat.id, "Model has been set to: " + model['description'])

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
    if model['output_type'] == 'video':
        return 'upload_video'
    return 'typing'


async def replicate_message_handler(botnav: TeleBotNav, message: Message) -> None:
    replicate_model_name = message.state_data.get('replicate_model', None)

    replicate_model = get_model_params(botnav, message, replicate_model_name)

    if not replicate_model:
        return

    input_data = message.state_data.get('replicate_params', {})

    if message.content_type == 'text' != replicate_model['input_type']:
        return

    if message.content_type == 'photo' != replicate_model['input_type']:
        return

    if message.content_type == 'text':
        input_field = replicate_model.get('input_field', 'prompt')
        input_data[input_field] = message.text

    if message.content_type == 'photo':
        file_info = await botnav.bot.get_file(message.photo[-1].file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        input_data[replicate_model.get('input_field', 'image')] = BytesIO(file_content)

    try:
        await replicate_execute_and_send(botnav, message, replicate_model_name, input_data)
    except ModelError as exc:
        await botnav.bot.send_message(message.chat.id, f"Model error occurred: {exc}")
    except Exception as exc:
        await botnav.bot.send_message(message.chat.id, "Something went wrong, try again later")
        logger.exception(exc)
        message.state_data.clear()


async def start_replicate(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        message.chat.id,
        {
            x: functools.partial(replicate_choose_model, x) for x in REPLICATE_MODELS.keys() if get_model_params(botnav, message, x)
        },
        row_width=2,
        text='Choose model:'
    )

    botnav.wipe_commands(message, preserve=['start'])
    botnav.add_command(message, 'replicate_models', '🧰 Replicate models', start_replicate)
    await botnav.send_commands(message)
    botnav.set_default_handler(message, replicate_message_handler)
    botnav.clean_next_handler(message)
