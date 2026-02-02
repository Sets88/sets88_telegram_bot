import os
import sys
import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


if not os.path.exists(CONFIG_PATH):
    print(f'{CONFIG_PATH} not found')
    sys.exit(1)


configdata = json.load(open(CONFIG_PATH, 'r'))

ALLOWED_USER_NAMES = configdata['ALLOWED_USER_NAMES']
OPENAI_API_KEY = configdata.get('OPENAI_API_KEY', None)
OLLAMA_HOST = configdata.get('OLLAMA_HOST', None)

TELEGRAM_TOKEN = configdata['TELEGRAM_TOKEN']

REPLICATE_API_KEY = configdata.get('REPLICATE_API_KEY', None)

ANTHROPIC_API_KEY = configdata.get('ANTHROPIC_API_KEY', None)

DEFAULT_LLM_MODEL = configdata.get('DEFAULT_LLM_MODEL', 'claude-haiku-4-5')

OPENROUTER_API_KEY = configdata.get('OPENROUTER_API_KEY', None)

YT_DL_DIR = configdata.get('YT_DL_DIR', None)
YT_DL_URL = configdata.get('YT_DL_URL', None)
SCHEDULES = configdata.get('SCHEDULES', None)

USER_PERMISSIONS = configdata.get('USER_PERMISSIONS', {})

MCP_FETCH_URL = configdata.get('MCP_FETCH_URL', None)

# Greek Learning Web App configuration
GREEK_LEARNING_WEBAPP_PORT = configdata.get('GREEK_LEARNING_WEBAPP_PORT', 8180)
GREEK_LEARNING_WEBAPP_URL = configdata.get('GREEK_LEARNING_WEBAPP_URL', 'http://localhost:8180/greek/')
