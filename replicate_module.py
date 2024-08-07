import os
import asyncio
import functools
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
    'llama-2-70b': {
        'description': 'A 70 billion parameter language model from Meta, fine tuned for chat completions',
        'replicate_id': 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
        'input_field': 'prompt',
        'input_type': 'text',
        'output_type': 'text',
        'available_params': {
            'temperature': {
                'type': 'float',
                'description': 'Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75',
                'default': 1,
                'min': 0.5,
                'max': 1,
            },
            'max_new_tokens': {
                'type': 'int',
                'default': 128,
                'description': 'Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens'
            },
            'system_prompt': {
                'type': 'str',
                'default': 'You are a helpful assistant',
                'description': 'System prompt to send to the model. This is prepended to the prompt and helps guide system behavior'
            }
        }
    },
    'sdxl-controlnet-lora': {
        'description': 'SDXL Canny controlnet with LoRA support.',
        'replicate_id': 'batouresearch/sdxl-controlnet-lora:a65bcd11a0db0f9cd33d6bf2a76925235c45450c71f38c1150b932a72e50a7f9',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'image': {
                'type': 'photo',
                'description': 'Input image'
            }
        }
    },
    'blip-2': {
        'description': 'Blip, bootstrapping Language-Image Pre-training, send photo to get caption or ask question',
        'replicate_id': 'andreasjansson/blip-2:9109553e37d266369f2750e407ab95649c63eb8e13f13b1f3983ff0feb2f9ef7',
        'input_type': 'photo',
        'input_field': 'image',
        'output_type': 'text',
        'available_params': {
            'question': {
                'type': 'str',
                'description': 'Question for VQA, default is "What is this a picture of?"',
                'default': 'What is this a picture of?'
            },
            'temperature': {
                'type': 'float',
                'description': 'Temperature for use with nucleus sampling (minimum: 0.5; maximum: 1) default is 1',
                'default': 1,
                'min': 0.5,
                'max': 1,
            }
        }
    },
    'flux-pro': {
        'description': 'State-of-the-art image generation with top of the line prompt following, visual quality, image detail and output diversity.',
        'replicate_id': 'black-forest-labs/flux-pro',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'safety_tolerance': {
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 5,
                'description': 'Safety tolerance, 1 is most strict and 5 is most permissive'
            }
        }
    },
    'LCM':  {
        'description': 'latent-consistency-model: Synthesizing High-Resolution Images with Few-Step Inference',
        'replicate_id': 'luosiallen/latent-consistency-model:553803fd018b3cf875a8bc774c99da9b33f36647badfd88a6eec90d61c5f62fc',
        'input_field': 'prompt',
        'input_type': 'text',
        'output_type': 'photo',
        'available_params': {
            'num_images': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 5,
                'description': 'Number of images to output'
            }
        }
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
    },
    'controlnet-prompt': {
        'description': 'controlnet 1.1 lineart x realistic-vision-v2.0 (updated to v5), send photo and ask to modify it',
        'replicate_id': 'usamaehsan/controlnet-1.1-x-realistic-vision-v2.0:51778c7522eb99added82c0c52873d7a391eecf5fcc3ac7856613b7e6443f2f7',
        'input_type': 'text',
        'input_field': 'prompt',
        'output_type': 'photo',
        'available_params': {
            'image': {
                'type': 'photo',
                'description': 'Input image'
            },
        }
    },
    'controlnet-scrible': {
        'description': 'ControlNet, generate detailed images from scribbled drawings',
        'replicate_id': 'jagilley/controlnet-scribble:435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117',
        'input_type': 'text',
        'input_field': 'prompt',
        'output_type': 'photo',
        'available_params': {
            'image': {
                'type': 'photo',
                'description': 'Mask image'
            },
        }
    },
    'Kandins-CN': {
        'description': 'Kandinsky Image Generation with ControlNet Conditioning',
        'replicate_id': 'cjwbw/kandinsky-2-2-controlnet-depth:98b54ca0b42be225e927f1dae2d9c506e69fe5b3bce301e13718d662a227a12b',
        'input_type': 'text',
        'input_field': 'prompt',
        'output_type': 'photo',
        'available_params': {
            'image': {
                'type': 'photo',
                'description': 'Input image'
            },
            'task': {
                'type': 'select',
                'options': ['text2img', 'img2img'],
                'default': 'img2img'
            },
            'num_inference_steps': {
                'type': 'int',
                'default': 70,
                'min': 1,
                'max': 500,
                'description': 'Number of inference steps, if you want to get more detailed image increase this number'
            }
        }
    },
    'controlnet-hed': {
        'description': 'ControlNet, modify images using HED maps',
        'replicate_id': 'jagilley/controlnet-hed:cde353130c86f37d0af4060cd757ab3009cac68eb58df216768f907f0d0a0653',
        'input_type': 'text',
        'input_field': 'prompt',
        'output_type': 'photo',
        'available_params': {
            'input_image': {
                'type': 'photo',
                'description': 'Input image'
            },
        }
    },
    'controlnet-normal': {
        'description': 'ControlNet, modify images using normal maps',
        'replicate_id': 'jagilley/controlnet-normal:cc8066f617b6c99fdb134bc1195c5291cf2610875da4985a39de50ee1f46d81c',
        'input_type': 'text',
        'input_field': 'prompt',
        'output_type': 'photo',
        'available_params': {
            'image': {
                'type': 'photo',
                'description': 'Input image'
            },
        }
    },
    'img2prompt': {
        'description': 'Get an approximate text prompt, with style, matching an image. (Optimized for stable-diffusion (clip ViT-L/14))',
        'replicate_id': 'methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5',
        'input_type': 'photo',
        'output_type': 'text'
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

    if param['type'] == 'float':
        value = float(value)

    if param['type'] == 'bool':
        value = bool(int(value))

    if param['type'] == 'photo':
        file_info = await botnav.bot.get_file(message.photo[-1].file_id)
        file_content = await botnav.bot.download_file(file_info.file_path)
        value = BytesIO(file_content)

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
        row_width=2,
        text='Choose model:'
    )

    botnav.wipe_commands(message, preserve=['start'])
    botnav.add_command(message, 'replicate_models', '🧰 Replicate models', start_replicate)
    await botnav.send_commands(message)
    botnav.set_default_handler(message, replicate_message_handler)
    botnav.clean_next_handler(message)
