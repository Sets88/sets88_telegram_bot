HELP_CONTENT = """
🤖 **Bot Settings Description**

**🎯 One Off** - One-time message mode
• ✅ Enabled: Each message is processed independently without previous context
• ❌ Disabled: Bot remembers entire conversation history and responds in context

**📤 Send upon command** - Manual send mode  
• ✅ Enabled: Messages are not sent automatically, need to press "Send" button
• ❌ Disabled: Messages are processed immediately after sending

**🔢 Max tokens** - Maximum tokens in response
• Limits AI response length (typically 1 token ≈ 0.75 words)
• Recommended values: 1024-4096 for normal conversation

**🏁 Set system prompt** - System prompt
• Defines AI behavior and role
• You can set custom instructions for the bot

**🧹 Clean conversation** - Clear conversation history
• Removes all message history in current conversation
• Useful when changing topics or when context becomes too long

**💾 Memory** - AI memory management
• AI remembers your preferences, settings, and personal information
• Access via Options → Memory to view or delete stored information
• AI automatically saves relevant details during conversations

**🤖 Model** - AI model selection
• GPT-4.1-mini: Fast and economical model
• GPT-4.1: More powerful model with better response quality
• Claude models: Anthropic models with good context understanding
• Ollama models: Local open-source models

**👥 Role** - Bot roles
• Assistant: Regular helper
• Funnyman: Responds with humor
• IT: Technical specialist
• Chef: Cooking expert
• Greek: Greek language translator
• English Translator: English translator and corrector
• And other specialized roles...

---

📚 **AI Usage Examples**

**1. General conversation:**
"Explain how blockchain works in simple terms"

**2. Programming:**
"Write a Python function to sort a list in descending order"

**3. Code analysis:**
"Find errors in this code: [paste code]"

**4. Creative tasks:**
"Come up with a name for a food delivery startup"

**5. Working with images:**
Send a photo and ask: "What's shown in this picture?"

**6. Image generation:**
"Generate an image: sunset over ocean in impressionist style"

**7. Data analysis:**
"Analyze this sales data and give recommendations"

**8. Translations:**
Use "English Translator" or "Greek" role for quality translations

**9. Learning:**
"Explain quantum physics using Schrödinger's cat example"

**10. Voice messages:**
Send a voice message - bot will recognize speech and respond

**11. Web app creation:**
"Create a web app — a calculator with dark theme"
"Make a todo list app and add sharing"
Bot will generate a full single-page app and open it right in Telegram

---

🌐 **Web Apps**

The bot can create fully functional web applications accessible directly in Telegram:
• Just describe what you want — the bot generates a complete HTML/JS/CSS app
• Apps are hosted automatically and open as Telegram Mini Apps
• You can request edits: "add a dark theme", "add export to CSV"
• Apps support AI features (LLM, image generation) via built-in APIs
• Share your app with other users via a link
• Manage your apps via **📱 My Web Apps** in the main menu

Examples of apps you can create:
– Calculators, converters, timers
– Note-taking, todo lists, habit trackers
– Mini games (quiz, memory, snake)
– Data visualisation (charts, tables)
– AI-powered tools (text summariser, translator, image generator)

---

💡 **Useful Tips:**

• Use "One Off" mode for independent questions
• Clear history when changing discussion topics  
• Choose appropriate role for specific tasks
• Experiment with different models for better results
• Use images for visual analysis
• Voice messages are convenient for quick text input
"""