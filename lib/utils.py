from enum import Enum
import json
from datetime import datetime
import dataclasses


class MessageSplitter:
    def __init__(self, hard_limit):
        self.hard_limit = hard_limit
        self.buffer = ''

    def get_message(self):
        if len(self.buffer) > self.hard_limit:
            buf = self.buffer[0:self.hard_limit]
            index = buf.rfind('\n')
            if index == -1:
                index = buf.rfind(' ')
            if index == -1:
                index = self.hard_limit
            buf = buf[0:index]
            self.buffer = self.buffer[index:]
            return buf

    def add(self, message: str):
        self.buffer += message
        return self.get_message()

    def flush(self):
        while message := self.get_message():
            yield message
        yield self.buffer
        self.buffer = ''


class ConvEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        if callable(obj):
            return str(obj)
        if hasattr(obj, 'dump') and callable(obj.dump):
            return obj.dump()
        # Let the base class default method raise the TypeError
        return super().default(obj)
