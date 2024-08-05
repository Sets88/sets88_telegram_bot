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

    def add(self, message):
        self.buffer += message
        return self.get_message()

    def flush(self):
        while message := self.get_message():
            yield message
        yield self.buffer
        self.buffer = ''
