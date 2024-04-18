from unittest import IsolatedAsyncioTestCase

from pytest import mark

from lib.metalang_executor import MetaLangExecutor


class MetaLangExecutorTest(MetaLangExecutor):
    COMMANDS = ['return_value']

    async def return_value(self, value, test = None, unused = None) -> str:
        if test:
            return value + test
        return value


@mark.asyncio
class TestMetalangExecutorTest(IsolatedAsyncioTestCase):
    async def test_assign(self):
        dsl_code = """value = return_value("Test!")"""
        executor = MetaLangExecutorTest()
        await executor.run(dsl_code)
        self.assertEqual(executor.vars, {"value": "Test!"})

    async def test_call(self):
        dsl_code = """
            value = return_value("Test", test="!")
            value2 = return_value("Test2", unused="test", test="!")
        """
        executor = MetaLangExecutorTest()
        await executor.run(dsl_code)
        self.assertEqual(executor.vars, {"value": "Test!", "value2": "Test2!"})

    async def test_comp(self):
        dsl_code = """
            gte = 10 >= 5
            gte2 = 10 >= 10
            gte3 = 10 >= 11
            gte3 = 10 >= 11
            gt = 10 > 5
            gt2 = 10 > 9
            gt3 = 10 > 10
            lt = 10 < 11
            lt2 = 10 < 10
            lt3 = 10 < 9
            lte = 10 <= 11
            lte2 = 10 <= 10
            lte3 = 10 <= 9
            e = 10 == 10
            e2 = 10 == 11
            ne = 10 != 11
            ne2 = 10 != 10
        """
        executor = MetaLangExecutorTest()
        await executor.run(dsl_code)
        assert executor.vars['gte'] is True
        assert executor.vars['gte2'] is True
        assert executor.vars['gte3'] is False
        assert executor.vars['gt'] is True
        assert executor.vars['gt2'] is True
        assert executor.vars['gt3'] is False
        assert executor.vars['lt'] is True
        assert executor.vars['lt2'] is False
        assert executor.vars['lt3'] is False
        assert executor.vars['lte'] is True
        assert executor.vars['lte2'] is True
        assert executor.vars['lte3'] is False
        assert executor.vars['e'] is True
        assert executor.vars['e2'] is False
        assert executor.vars['ne'] is True
        assert executor.vars['ne2'] is False


@mark.asyncio
@mark.parametrize('value, field, test_value', [
    (6, 'value1', 'Test_if'),
    (11, 'value1', 'Test_if'),
    (4, 'value2', 'Test_elif'),
    (1, 'value3', 'Test_else')
])
async def test_if(value, field, test_value):
    dsl_code = f"""
        value = {value}
        if value > 5 {{
            value1 = return_value("Test_if")
        }}
        elif value > 3 {{
            value2 = return_value("Test_elif")
        }}
        else {{
            value3 = return_value("Test_else")
        }}
    """
    executor = MetaLangExecutorTest()
    await executor.run(dsl_code)
    assert executor.vars[field] == test_value
    assert sorted(list(executor.vars.keys())) == ['value', field]
