Personal Telegram bot with a wide range of capabilities:
- Access to most OpenAi models, such as GPT, DALL-E, Whisper, TTS
- Access to replicate models, such as Stable diffusion, Kandinsky, Blip, Llama, and others
- Ability to download videos from various video hosting platforms

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

## If you need to use replicate models(Optional):

6. Get the key from replicate:
    - Register at https://replicate.com/signin
    - Create a key at https://replicate.com/account/api-tokens
    - Save the received key in config.json -> REPLICATE_API_KEY


7. Install dependencies, and start the bot by running the command:

```run.sh```

![How it looks](/images/bot_screenshot.png)
