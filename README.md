Personal Telegram bot with a comprehensive AI assistant platform:

## 🤖 AI Model Support
- **OpenAI Models**: GPT-4, GPT-5, O3, O4-mini, DALL-E, Whisper, TTS
- **Anthropic Models**: Claude 3/3.5, Claude Haiku/Sonnet/Opus 4.x series
- **Local Models via Ollama**: Qwen, Gemma, open-source GPT models
- **Replicate Models**: Stable Diffusion, Flux Pro, Kandinsky, Llama, and 50+ others

## 🛠️ Advanced Features
- **Function Calling & Tools**: LLM models can call functions, generate images, search web
- **Memory System**: AI remembers user preferences and information across conversations
- **Video Generation**: Google Veo 3.1 models with context-aware audio
- **Voice Processing**: Speech-to-text with Whisper, multi-modal conversations
- **Image Analysis**: Upload and analyze images with AI
- **Permissions System**: Granular access control per user and feature
- **17 AI Personas**: Specialized roles from IT Expert to Stand-up Comedian
- **Video Downloads**: Support for various video hosting platforms
- **Meta-language Scheduler**: Automated task execution with custom scripting language

# Here are a few simple steps to set it up

1. Clone the repository onto the server
2. Copy the config.json.example file to config.json
3. Register a Telegram bot
    - Send the command /newbot to the @BotFather bot
    - Enter the bot's name
    - Save the received token in config.json -> TELEGRAM_TOKEN
4. Add the usernames of users who will have access to the bot to config.json -> ALLOWED_USER_NAMES

## If you need to use OpenAi(Optional):

5. Get the key from OpenAi:
    - Register at https://platform.openai.com/signup
    - Create a key at https://platform.openai.com/api-keys
    - Save the received key in config.json -> OPENAI_API_KEY

## If you need to use Anthropic Claude models(Optional):

6. Get the key from Anthropic:
    - Register at https://console.anthropic.com/
    - Create a key at https://console.anthropic.com/account/keys
    - Save the received key in config.json -> ANTHROPIC_API_KEY

## If you need to use replicate models(Optional):

7. Get the key from replicate:
    - Register at https://replicate.com/signin
    - Create a key at https://replicate.com/account/api-tokens
    - Save the received key in config.json -> REPLICATE_API_KEY

## If you need to use local Ollama models(Optional):

8. Install and configure Ollama:
    - Download from https://ollama.ai/download
    - Pull desired models: `ollama pull qwen3:32b`
    - Set OLLAMA_HOST in config.json (default: http://localhost:11434)

## Permissions Configuration(Optional):

9. Configure user permissions in config.json -> USER_PERMISSIONS:
    - Set `is_admin`, `can_use_tools`, `can_use_ollama_llm_models` per user
    - Control access to specific features and models

10. Install dependencies, and start the bot by running the command:

```run.sh```

![How it looks](/images/bot_screenshot.png)
![How it looks](/images/bot_screenshot2.png)
