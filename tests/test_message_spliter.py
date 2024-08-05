from random import randint
from lib.utils import MessageSplitter

def test_1():
    parts = [
        7, 16, 8, 5, 11, 20, 10, 18, 12, 14, 13, 5, 5, 7, 13, 10, 18, 20, 17, 16, 19,
        16, 9, 11, 17, 12, 8, 20, 10, 10, 20, 7, 15, 10, 20, 6, 18, 19, 7, 5, 16, 13, 8, 16, 8, 11
    ]
    text = ''
    hard_limit = 15
    splitter = MessageSplitter(hard_limit)

    text_sample = """Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's 
standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type
specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially
unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently
with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."""

    pos = 0

    for i in parts:
        res = splitter.add(text_sample[pos:pos + i])
        if res:
            text = text + res
            assert len(res) <= hard_limit
        pos = pos + i

    for msg in splitter.flush():
        if msg:
            text = text + msg

    assert(text == text_sample)
