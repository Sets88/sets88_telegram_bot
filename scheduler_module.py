import asyncio
import os
from typing import Any
from io import StringIO
from time import time
from collections import defaultdict
import json
from functools import partial

from telebot.types import Message
import aiohttp

from logger import logger
from telebot_nav import TeleBotNav


message_throttle_locks = {}


class ScheduleMeta:
    def __init__(self, schedule):
        self.user_id = schedule['user_id']
        self.name = schedule['name']
        self.tasks = schedule['tasks']
        self.interval = schedule['interval']
        self.store = schedule['store']
        self.state = schedule['state']
        self.task = None
        self.vars = {}


class MetaLangExecutor:
    COMMANDS = [
        'http_get', 'json', 'dict_get', 'gt', 'gte', 'lt',
        'eq', 'neq', 'set_var', 'lte', 'format', 'send_message'
    ]
    def __init__(self, botnav: TeleBotNav, schedule_meta: ScheduleMeta):
        self.schedule_meta = schedule_meta
        self.botnav = botnav

    async def http_get(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    raise Exception('Error opening url')

                return await resp.read()

    async def json(self, data: str|bytes) -> dict|list:
        return json.loads(data)

    async def dict_get(self, data: dict, key: str) -> Any:
        return data[key]

    async def gt(self, first: Any, second: Any) -> bool:
        return first > second

    async def gte(self, first: Any, second: Any) -> bool:
        return first >= second

    async def lt(self, first: Any, second: Any) -> bool:
        return first < second

    async def eq(self, first: Any, second: Any) -> bool:
        return first == second

    async def neq(self, first: Any, second: Any) -> bool:
        return first != second

    async def lte(self, first: Any, second: Any) -> bool:
        return first <= second

    async def format(self, format: str, **kwargs) -> bool:
        return format.format(**kwargs)

    async def set_var(self, key: str, value: Any) -> None:
        self.schedule_meta.vars[key] = value

    async def send_message(self, text, throttle: int|None = 10) -> None:
        key = f'{self.schedule_meta.user_id}_{self.schedule_meta.name}'
        if key in message_throttle_locks and message_throttle_locks[key] > time():
            return
        
        message_throttle_locks[key] = time() + throttle
        await self.botnav.bot.send_message(self.schedule_meta.user_id, text)

    def parse_params(self, command: dict) -> tuple[list, dict]:
        args = []
        kwargs = {}

        for key, value in command.get('kwargs', {}).items():
            if key == 'self':
                continue
            if 'value' in value:
                kwargs[key] = value['value']
            elif 'var' in value:
                kwargs[key] = self.schedule_meta.vars.get(value['var'])
            elif 'store' in value:
                kwargs[key] = self.schedule_meta.store.get(value['store'])
        
        for value in command.get('args', []):
            if 'value' in value:
                args.append(value['value'])
            elif 'var' in value:
                args.append(self.schedule_meta.vars.get(value['var']))
            elif 'store' in value:
                args.append(self.schedule_meta.store.get(value['store']))

        return args, kwargs

    async def process_tree(self, tasks: list[dict]) -> None:
        for command in tasks:
            if (
                command.get('action') not in ['call', 'test'] or
                command.get('command') not in self.COMMANDS
            ):
                continue
            elif command['action'] == 'call':
                args, kwargs = self.parse_params(command)
                result = await getattr(self, command['command'])(*args, **kwargs)

                if 'put_into' in command:
                    for key in command['put_into']:
                        self.schedule_meta.vars[key] = result
            elif command['action'] == 'test':
                args, kwargs = self.parse_params(command)
                result = await getattr(self, command['command'])(*args, **kwargs)

                if result:
                    await self.process_tree(command['tasks'])

                if not command.get('pass_on_true'):
                    return
            else:
                raise Exception('Invalid action')

    async def run(self):
        await self.process_tree(self.schedule_meta.tasks)



class SchedulesManager:
    def __init__(self):
        self.schedules_metas = defaultdict(dict)
        self.botnav = None

    async def create_task(self, schedule: ScheduleMeta) -> None:
        while True:
            try:
                try:
                    executor = MetaLangExecutor(self.botnav, schedule)
                    await executor.run()

                    await asyncio.sleep(schedule.interval)
                except Exception as exc:
                    logger.exception(exc)
                    await asyncio.sleep(schedule.interval)
                finally:
                    # Just one run
                    if schedule.interval == 0:
                        schedule.state = 'stopped'
                        self.save_to_file(schedule.user_id)
                        break

            except Exception as exc:
                logger.exception(exc)
                await asyncio.sleep(120)

    async def list_schedules(self, user_id) -> list[ScheduleMeta]:
        return self.schedules_metas.get(user_id, {}).values()

    def save_to_file(self, user_id: int) -> None:
        with open(f'schedules/{user_id}.json', 'w') as f:
            json.dump(
                [dict(
                    user_id=x.user_id,
                    name=x.name,
                    interval=x.interval,
                    tasks=x.tasks,
                    store=x.store,
                    state=x.state
                ) for x in self.schedules_metas[user_id].values()],
                f
            )

    def get_schedule(self, user_id: int, schedule_name: str) -> ScheduleMeta|None:
        return self.schedules_metas.get(user_id, {}).get(schedule_name)

    def delete_schedule(self, user_id, schedule_name):
        schedule = self.schedules_metas[user_id][schedule_name]
        if schedule.task:
            schedule.task.cancel()
        del self.schedules_metas[user_id][schedule_name]
        self.save_to_file(user_id)

    def stop_schedule(self, user_id, schedule_name):
        """Stop execution of schedule"""
        schedule = self.schedules_metas[user_id][schedule_name]

        if schedule.task:
            schedule.task.cancel()

        schedule.state = 'stopped'
        self.save_to_file(user_id)

    def start_schedule(self, user_id, schedule_name):
        schedule = self.schedules_metas[user_id][schedule_name]

        if schedule.task:
            schedule.task.cancel()

        schedule.task = asyncio.create_task(self.create_task(schedule))
        schedule.state = 'running'
        self.save_to_file(user_id)

    def add_schedule(self, user_id: int, schedule_data: dict) -> None:
        schedule = ScheduleMeta(schedule_data)
        self.schedules_metas[user_id][schedule.name] = schedule

        if schedule.state == 'running':
            schedule.task = asyncio.create_task(self.create_task(schedule))

        self.save_to_file(user_id)

    async def run(self, botnav: TeleBotNav) -> None:
        self.botnav = botnav

        directory = os.path.join(os.path.dirname(__file__), 'schedules')
        if not os.path.exists(directory):
            os.mkdir(directory)

        files = [x for x in os.walk(directory)][0][2]

        for filename in files:
            with open(os.path.join(directory, filename)) as f:
                try:
                    schedules = json.load(f)
                    for schedule_data in schedules:
                        schedule = ScheduleMeta(schedule_data)
                        self.schedules_metas[schedule.user_id][schedule.name] = schedule

                        if schedule.state == 'running':
                            schedule.task = asyncio.create_task(self.create_task(schedule))
                        
                except Exception as exc:
                    logger.exception(exc)
                    continue


class ListSchedulesRouter:
    @classmethod
    def format_md_schedule_description(cls, schedule: ScheduleMeta) -> str:
        schedule_params = {
            'name': schedule.name,
            'interval': schedule.interval if schedule.interval else 'Manual run',
            'state': schedule.state
        }
        return cls.format_md_dict(schedule_params)

    @classmethod
    async def list(cls, botnav: TeleBotNav, message: Message) -> None:
        schedules = await manager.list_schedules(botnav.get_user(message).id)
        if not schedules:
            await botnav.bot.send_message(botnav.get_user(message).id, 'No schedules found')
            return

        await botnav.print_buttons(
            botnav.get_user(message).id,
            {
                x.name: partial(cls.select_schedule, x.name) for x in schedules
            },
            'Schedules',
            row_width=1
        )

    @classmethod
    async def select_schedule(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, 'Schedule not found')
            return
        
        buttons = {
            'Delete': partial(cls.delete, schedule_name),
            'Stored Variables': partial(cls.stored_variables, schedule_name),
            'Show Config': partial(cls.show_config, schedule_name)
        }

        if schedule.state == 'running':
            buttons['Stop'] = partial(cls.stop, schedule_name)
        elif schedule.state == 'stopped':
            buttons['Start'] = partial(cls.start, schedule_name)

        await botnav.print_buttons(
            botnav.get_user(message).id,
            buttons,
            cls.format_md_schedule_description(schedule),
            row_width=2,
            parse_mode='MarkdownV2'
        )

    @classmethod
    async def show_config(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, 'Schedule not found')
            return

        config = json.dumps(schedule.tasks, indent=4)

        description = cls.format_md_schedule_description(schedule)

        if len(config) > 3500:
            document = StringIO(config)
            document.filename = 'config.json'

            await botnav.bot.send_message(
                botnav.get_user(message).id,
                description,
                parse_mode='MarkdownV2'
            )

            await botnav.bot.send_document(
                botnav.get_user(message).id,
                document,
                visible_file_name=f'{schedule.name}.json'
            )
            return

        await botnav.bot.send_message(
            botnav.get_user(message).id,
            f"""
            {description}
            ```json\n{json.dumps(schedule.tasks, indent=4)}\n```
            """,
            parse_mode='MarkdownV2'
        )

    @classmethod
    async def delete(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        manager.delete_schedule(botnav.get_user(message).id, schedule_name)
        await botnav.bot.delete_message(botnav.get_user(message).id, message.message_id)
        await botnav.bot.send_message(botnav.get_user(message).id, f'Schedule "{schedule_name}" deleted')

    @classmethod
    async def stop(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, f'Schedule {schedule_name} not found')
            return

        manager.stop_schedule(botnav.get_user(message).id, schedule_name)
        await botnav.bot.delete_message(botnav.get_user(message).id, message.message_id)
        await botnav.bot.send_message(botnav.get_user(message).id, f'Schedule "{schedule_name}" stopped')

    @classmethod
    async def start(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, f'Schedule {schedule_name} not found')
            return

        if schedule.task:
            schedule.task.cancel()
        
        manager.start_schedule(botnav.get_user(message).id, schedule_name)
        await botnav.bot.delete_message(botnav.get_user(message).id, message.message_id)
        await botnav.bot.send_message(botnav.get_user(message).id, f'Schedule "{schedule_name}" started')

    @classmethod
    async def stored_variables(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, f'Schedule {schedule_name} not found')
            return

        variables = cls.format_md_dict(schedule.store)

        await botnav.bot.send_message(
            botnav.get_user(message).id,
            f'Current variables: \n{variables}\nSend name of the variable to set',
            parse_mode='MarkdownV2')
        await botnav.set_next_handler(message, partial(cls.set_stored_variable_name, schedule_name))

    @classmethod
    async def set_stored_variable_name(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, 'Schedule not found')
            return

        message.state_data['update_variable'] = {
            'name': message.text
        }

        await botnav.print_buttons(
            botnav.get_user(message).id,
            {
                'int': partial(cls.coerce_variable, 'int'),
                'str': partial(cls.coerce_variable, 'str'),
                'float': partial(cls.coerce_variable, 'float'),
                'Delete': partial(cls.delete_stored_variable, schedule_name, message.text)
            },
            'Coerce or delete',
            row_width=2
        )
        await botnav.bot.send_message(
            botnav.get_user(message).id,
            'Send value of the variable',
        )
        await botnav.set_next_handler(message, partial(cls.set_stored_variable_value, schedule_name))

    @classmethod
    async def delete_stored_variable(cls, schedule_name: str, varname: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, 'Schedule not found')
            return

        if varname not in schedule.variables:
            await botnav.bot.send_message(botnav.get_user(message).id, 'Variable not found')
            return

        del schedule.variables[varname]

        manager.save_to_file(botnav.get_user(message).id)

        variables = "\n".join([f'{key} = {val}' for key, val in schedule.variables.items()])

        await botnav.bot.send_message(
            botnav.get_user(message).id,
            'Variable deleted:\n```ini\n' + variables + '\n```',
            parse_mode='MarkdownV2'
        )
        botnav.clean_next_handler(message)

    @classmethod
    def coerce_value(cls, coerce_type: str, old_value, new_value) -> Any:
        coerce = str
        if coerce_type == 'int':
            coerce = int
        elif coerce_type == 'float':
            coerce = float
        elif coerce_type == 'str':
            coerce = str
        elif old_value is not None:
            coerce = type(old_value)
        try:
            return coerce(new_value)
        except (TypeError, ValueError):
            return str(new_value)

    @classmethod
    def format_md_dict(cls, variables: dict) -> str:
        lines = []
        for key, val in variables.items():
            if isinstance(val, str):
                lines.append(f'{key} = "{val}"')
            else:
                lines.append(f'{key} = {val}')

        if not lines:
            return 'No variables stored'

        nl = '\n'
        return f'```ini{nl}{nl.join(lines)}{nl}```'

    @classmethod
    async def set_stored_variable_value(cls, schedule_name: str, botnav: TeleBotNav, message: Message) -> None:
        schedule = manager.get_schedule(botnav.get_user(message).id, schedule_name)

        if not schedule:
            await botnav.bot.send_message(botnav.get_user(message).id, 'Schedule not found')
            return

        varname = message.state_data['update_variable']['name']

        value = cls.coerce_value(
            message.state_data['update_variable'].get('coerce'),
            schedule.store.get(varname),
            message.text
        )

        schedule.store[varname] = value

        manager.save_to_file(botnav.get_user(message).id)

        variables = cls.format_md_dict(schedule.store)

        await botnav.bot.send_message(
            botnav.get_user(message).id,
            f'Stored variable updated:\n{variables}',
            parse_mode='MarkdownV2'
        )
        del message.state_data['update_variable']

    @classmethod
    async def coerce_variable(cls, coerce_type: str, botnav: TeleBotNav, message: Message) -> None:
        if coerce_type in ['int', 'str', 'float']:
            message.state_data['update_variable']['coerce'] = coerce_type
            await botnav.bot.send_message(botnav.get_user(message).id, f'Coerce type set to {coerce_type}')
        else:
            await botnav.bot.send_message(botnav.get_user(message).id, 'Invalid coerce type')
            return


class ScheduleAddRouter:
    @classmethod
    async def add(cls, botnav: TeleBotNav, message: Message) -> None:
        await botnav.bot.send_message(botnav.get_user(message).id, 'Send schedule name')
        await botnav.set_next_handler(message, cls.set_name)

    @classmethod
    async def set_name(cls, botnav: TeleBotNav, message: Message) -> None:
        message.state_data['new_schedule'] = {
            'name': message.text
        }

        if manager.get_schedule(botnav.get_user(message).id, message.text):
            await botnav.bot.send_message(botnav.get_user(message).id, 'Schedule with this name already exists')
            return

        await botnav.bot.send_message(
            botnav.get_user(message).id,
            f'Schedule name set to {message.text},'
                '\nSend interval in seconds or 0 for manual run')
        await botnav.set_next_handler(message, cls.set_interval)

    @classmethod
    async def set_interval(cls, botnav: TeleBotNav, message: Message) -> None:
        message.state_data['new_schedule']['interval'] = int(message.text)
        await botnav.bot.send_message(
            botnav.get_user(message).id,
            f'Interval set to {message.text},\nSend configuration JSON'
        )
        await botnav.set_next_handler(message, cls.set_config)

    @classmethod
    async def set_config(cls, botnav: TeleBotNav, message: Message) -> None:
        try:
            text = message.text

            if message.content_type == 'document':
                file_info = await botnav.bot.get_file(message.document.file_id)
                document = await botnav.bot.download_file(file_info.file_path)
                text = document

            message.state_data['new_schedule']['tasks'] = json.loads(text)
        except Exception as exc:
            logger.exception(exc)
            await botnav.bot.send_message(botnav.get_user(message).id, 'Invalid JSON')
            return

        await botnav.bot.send_message(botnav.get_user(message).id, 'Configuration set')
        manager.add_schedule(
            botnav.get_user(message).id, {
                'user_id': botnav.get_user(message).id,
                'name': message.state_data['new_schedule']['name'],
                'interval': message.state_data['new_schedule']['interval'],
                'tasks': message.state_data['new_schedule']['tasks'],
                'state': 'stopped',
                'store': {}
            }
        )
        del message.state_data['new_schedule']


async def help_schedules(botnav: TeleBotNav, message: Message) -> None:
    help = """
    Schedules module allows you to create schedules that will run tasks at specified intervals
    Here is the example of the configuration JSON:
    ```json
[
    {
        "action": "call",
        "command": "http_get",
        "args": [
            {
                "value": "https://example.com"
            }
        ],
        "put_into": ["response"]
    },
    {
        "action": "call",
        "command": "json",
        "args": [
            {
                "var": "response"
            }
        ],
        "put_into": ["data"]
    },
    {
        "action": "call",
        "command": "dict_get",
        "kwargs": {
            "data": {
                "var": "data"
            },
            "key": {
                "value": "value"
            }
        },
        "put_into": ["value"]
    },
    {
        "action": "test",
        "command": "eq",
        "args": [
            {
                "var": "value"
            },
            {
                "value": "example"
            }
        ],
        "tasks": [
            {
                "action": "call",
                "command": "format",
                "kwargs": {
                    "format": {
                        "value": "Value is {res_value}"
                    },
                    "res_value": {
                        "var": "value"
                    }
                },
                "put_into": [
                    "message"
                ]
            },
            {
                "action": "call",
                "command": "send_message",
                "args": [
                    {
                        "var": "message"
                    }
                ]
            }
        ]
    }
]
    ```

    This configuration will:
    1\. Make a GET request to https://example\.com
    2\. Parse JSON response
    3\. Get value from the parsed JSON
    4\. Check if the value is equal to "example"
    5\. If it is, format the message and send it to the user
    """
    await botnav.bot.send_message(botnav.get_user(message).id, help, parse_mode='MarkdownV2')


async def start_schedules(botnav: TeleBotNav, message: Message) -> None:
    await botnav.print_buttons(
        botnav.get_user(message).id,
        {
            'List': ListSchedulesRouter.list,
            'Add': ScheduleAddRouter.add,
            'Help': help_schedules
        },
        'Options',
        row_width=2
    )

    botnav.wipe_commands(message, preserve=['start'])

    await botnav.send_commands(message)

manager = SchedulesManager()