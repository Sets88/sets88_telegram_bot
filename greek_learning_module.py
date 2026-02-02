"""
Greek Learning Module - Telegram Bot Interface
Provides bot commands and navigation for Greek word learning
"""

from telebot.types import Message, WebAppInfo
from telebot import types

import config
from telebot_nav import TeleBotNav
from logger import logger
from lib.greek_learning import GreekLearningManager, ExerciseDirection


async def start_greek(botnav: TeleBotNav, message: Message) -> None:

    try:
        # Web App URL (will be configured in Phase 4)
        web_app_url = getattr(config, 'GREEK_LEARNING_WEBAPP_URL', 'http://localhost:8080/greek/') + 'index.html'

        # Create inline keyboard with Web App button
        markup = types.InlineKeyboardMarkup()
        web_app_button = types.InlineKeyboardButton(
            text="🇬🇷 Open Greek Learning App",
            web_app=WebAppInfo(url=web_app_url)
        )
        markup.add(web_app_button)

        await botnav.bot.send_message(
            message.chat.id,
            "Click the button below to open the Greek Learning Web App! 🎓",
            reply_markup=markup
        )

    except Exception as exc:
        logger.exception(f"Error opening Web App: {exc}")
        await botnav.bot.send_message(
            message.chat.id,
            "Web App is not available yet. Please try again later."
        )

    botnav.wipe_commands(message, preserve=['start'])
    botnav.add_command(message, 'greek', '🇬🇷 Greek Learning', start_greek)
    await botnav.send_commands(message)


# ========== Web App Server & API ==========

import hmac
import hashlib
import json
import random
from urllib.parse import parse_qs, unquote
from aiohttp import web
import os


def validate_telegram_init_data(init_data: str, bot_token: str) -> dict | None:
    """
    Validate Telegram WebApp initData using HMAC-SHA256
    Returns user data if valid, None otherwise
    """
    try:
        # Parse query string
        parsed = parse_qs(init_data)

        # Get hash from params
        if 'hash' not in parsed:
            logger.error("No hash in initData")
            return None

        received_hash = parsed['hash'][0]

        # Build data_check_string (all params except hash, sorted alphabetically)
        data_check_pairs = []
        for key in sorted(parsed.keys()):
            if key == 'hash':
                continue
            value = parsed[key][0]
            data_check_pairs.append(f"{key}={value}")

        data_check_string = '\n'.join(data_check_pairs)

        # Compute secret key: HMAC_SHA256(bot_token, "WebAppData")
        secret_key = hmac.new(
            b"WebAppData",
            bot_token.encode(),
            hashlib.sha256
        ).digest()

        # Compute expected hash: HMAC_SHA256(data_check_string, secret_key)
        expected_hash = hmac.new(
            secret_key,
            data_check_string.encode(),
            hashlib.sha256
        ).hexdigest()

        # Compare hashes
        if expected_hash != received_hash:
            logger.error(f"Hash mismatch: expected {expected_hash}, got {received_hash}")
            return None

        # Extract and return user data
        if 'user' in parsed:
            user_data = json.loads(unquote(parsed['user'][0]))
            return user_data

        return None

    except Exception as exc:
        logger.exception(f"Error validating Telegram initData: {exc}")
        return None


class GreekWebApp:
    """
    aiohttp Web Application for Greek Learning Mini Web App
    Provides REST API endpoints and serves static files
    """

    def __init__(self, botnav: TeleBotNav):
        self.botnav = botnav
        self.app = web.Application(middlewares=[self._cors_middleware])
        self._setup_routes()
        self._setup_static()

    @web.middleware
    async def _cors_middleware(self, request, handler):
        """CORS middleware for development and Telegram WebApp"""
        if request.method == 'OPTIONS':
            response = web.Response()
        else:
            try:
                response = await handler(request)
            except Exception as exc:
                logger.exception(f"Handler error: {exc}")
                return web.json_response({'error': str(exc)}, status=500)

        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Telegram-Init-Data'
        return response

    def _setup_static(self):
        """Setup static file serving"""
        static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webapp'))
        self.app.router.add_static('/greek/', path=os.path.join(static_path, 'greek'), name='static')
        logger.info(f"Serving static files from {static_path}/greek")

    def _setup_routes(self):
        """Setup API routes"""
        # Health check endpoint (no auth required)
        self.app.router.add_get('/greek/api/health', self.health_check)

        # API endpoints
        self.app.router.add_get('/greek/api/words', self.get_words)
        self.app.router.add_get('/greek/api/learned', self.get_learned)
        self.app.router.add_post('/greek/api/fetch-words', self.fetch_words)
        self.app.router.add_get('/greek/api/exercise', self.get_exercise)
        self.app.router.add_get('/greek/api/exercise/matching', self.get_matching_exercise)
        self.app.router.add_post('/greek/api/validate', self.validate_answer)
        self.app.router.add_post('/greek/api/validate-matching', self.validate_matching_answers)
        self.app.router.add_post('/greek/api/mark-learned', self.mark_learned)
        self.app.router.add_post('/greek/api/move-to-learning', self.move_to_learning)
        self.app.router.add_delete('/greek/api/words/{word_id}', self.delete_word)
        self.app.router.add_get('/greek/api/stats', self.get_stats)

        logger.info("Greek Learning API routes configured")

    async def _authenticate(self, request: web.Request) -> dict | None:
        """Authenticate request using Telegram initData"""
        init_data = request.headers.get('X-Telegram-Init-Data')

        logger.info(f"Authentication attempt for {request.path}")
        logger.info(f"Headers: {dict(request.headers)}")

        if not init_data:
            logger.error("No X-Telegram-Init-Data header")
            return None

        logger.info(f"Validating initData (length: {len(init_data)})")
        user = validate_telegram_init_data(init_data, config.TELEGRAM_TOKEN)

        if user:
            logger.info(f"Authentication successful for user {user.get('id')}")
        else:
            logger.error("Authentication failed: invalid signature")

        return user

    # ========== API Endpoints ==========

    async def health_check(self, request: web.Request) -> web.Response:
        """GET /api/greek/health - Health check endpoint (no auth)"""
        return web.json_response({
            'status': 'ok',
            'service': 'Greek Learning API',
            'version': '1.0.0'
        })

    async def get_words(self, request: web.Request) -> web.Response:
        """GET /api/greek/words - Get current learning words"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            manager = GreekLearningManager(user['id'])
            words = await manager.load_words()

            words_data = [w.to_dict() for w in words]
            return web.json_response({
                'words': words_data,
                'total_count': len(words_data)
            })

        except Exception as exc:
            logger.exception(f"Error getting words: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def get_learned(self, request: web.Request) -> web.Response:
        """GET /api/greek/learned - Get learned words"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            manager = GreekLearningManager(user['id'])
            learned = await manager.load_learned_words()

            learned_data = [w.to_dict() for w in learned]
            return web.json_response({
                'words': learned_data,
                'total_count': len(learned_data)
            })

        except Exception as exc:
            logger.exception(f"Error getting learned words: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def fetch_words(self, request: web.Request) -> web.Response:
        """POST /api/greek/fetch-words - Fetch new words from OpenAI"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            data = await request.json()
            count = data.get('count', 5)

            if count < 1 or count > 600:
                return web.json_response({'error': 'Count must be between 1 and 600'}, status=400)

            manager = GreekLearningManager(user['id'])
            new_words = await manager.fetch_words_from_openai(count)

            if not new_words:
                return web.json_response({'error': 'Failed to fetch words'}, status=500)

            await manager.add_words(new_words)

            # Get updated word list
            all_words = await manager.load_words()

            return web.json_response({
                'new_words': [w.to_dict() for w in new_words],
                'total_count': len(all_words)
            })

        except Exception as exc:
            logger.exception(f"Error fetching words: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def get_exercise(self, request: web.Request) -> web.Response:
        """GET /api/greek/exercise - Generate exercise sentence"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            # Get word_id from query params (optional - random if not provided)
            word_id = request.rel_url.query.get('word_id')
            direction_str = request.rel_url.query.get('direction', 'random')

            # Get word_type filter (optional)
            word_type = request.rel_url.query.get('word_type', '').strip()
            word_types = [word_type] if word_type else None

            manager = GreekLearningManager(user['id'])

            # Select word
            if word_id:
                word = await manager.get_word_by_id(word_id)
                if not word:
                    return web.json_response({'error': 'Word not found'}, status=404)
            else:
                # Use new method with filtering
                word = await manager.get_random_word(word_types=word_types)
                if not word:
                    error_msg = 'No words available'
                    if word_types:
                        error_msg += f' for type: {word_type}'
                    return web.json_response({'error': error_msg}, status=404)

            # Select direction
            if direction_str == 'random':
                direction = random.choice([ExerciseDirection.GREEK_TO_RUSSIAN, ExerciseDirection.RUSSIAN_TO_GREEK])
            else:
                try:
                    direction = ExerciseDirection(direction_str)
                except ValueError:
                    return web.json_response({'error': 'Invalid direction'}, status=400)

            # Generate exercise
            exercise = await manager.generate_exercise(word, direction)

            return web.json_response({
                'word_id': exercise.word_id,
                'direction': exercise.direction,
                'sentence': exercise.sentence,
                'sentence_translation': exercise.sentence_translation,
                'correct_answer': exercise.correct_answer,
                'translated_word': exercise.translated_word,
            })

        except Exception as exc:
            logger.exception(f"Error generating exercise: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def validate_answer(self, request: web.Request) -> web.Response:
        """POST /api/greek/validate - Validate user's answer"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            data = await request.json()
            word_id = data.get('word_id')
            user_answer = data.get('user_answer')
            correct_answer = data.get('correct_answer')  # Sent from client
            direction_str = data.get('direction')

            if not all([word_id, user_answer, correct_answer, direction_str]):
                return web.json_response({'error': 'Missing required fields'}, status=400)

            # Normalize answers for comparison (remove punctuation, lowercase, strip whitespace)
            import re
            def normalize_answer(text):
                # Remove common punctuation
                text = re.sub(r'[.,!?;:»«"""\'()—–\-\[\]]', '', text)
                # Remove extra whitespace and convert to lowercase
                return text.strip().lower()

            user_normalized = normalize_answer(user_answer)
            correct_normalized = normalize_answer(correct_answer)

            # Log for debugging
            logger.info(f"Answer validation: user='{user_answer}' (normalized: '{user_normalized}'), correct='{correct_answer}' (normalized: '{correct_normalized}')")

            # Validate answer
            is_correct = user_normalized == correct_normalized

            # Update statistics
            manager = GreekLearningManager(user['id'])
            direction = ExerciseDirection(direction_str)

            await manager.update_word_stats(word_id, is_correct)
            await manager.update_user_stats(direction, is_correct)

            # Get word for explanation
            word = await manager.get_word_by_id(word_id)
            if word:
                explanation = f"{'Правильно' if is_correct else 'Неправильно'}! {word.greek} означает '{word.russian}'"
            else:
                explanation = "Правильно!" if is_correct else "Неправильно!"

            return web.json_response({
                'correct': is_correct,
                'correct_answer': correct_answer,
                'explanation': explanation
            })

        except Exception as exc:
            logger.exception(f"Error validating answer: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def mark_learned(self, request: web.Request) -> web.Response:
        """POST /api/greek/mark-learned - Mark word as learned"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            data = await request.json()
            word_id = data.get('word_id')

            if not word_id:
                return web.json_response({'error': 'Missing word_id'}, status=400)

            manager = GreekLearningManager(user['id'])
            success = await manager.mark_as_learned(word_id)

            if success:
                return web.json_response({
                    'success': True,
                    'message': 'Word marked as learned'
                })
            else:
                return web.json_response({'error': 'Word not found'}, status=404)

        except Exception as exc:
            logger.exception(f"Error marking learned: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def move_to_learning(self, request: web.Request) -> web.Response:
        """POST /api/greek/move-to-learning - Move word back to learning"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            data = await request.json()
            word_id = data.get('word_id')

            if not word_id:
                return web.json_response({'error': 'Missing word_id'}, status=400)

            manager = GreekLearningManager(user['id'])
            success = await manager.move_to_learning(word_id)

            if success:
                return web.json_response({
                    'success': True,
                    'message': 'Word moved back to learning'
                })
            else:
                return web.json_response({'error': 'Word not found'}, status=404)

        except Exception as exc:
            logger.exception(f"Error moving to learning: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def delete_word(self, request: web.Request) -> web.Response:
        """DELETE /api/greek/words/{word_id} - Delete word"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            word_id = request.match_info['word_id']

            manager = GreekLearningManager(user['id'])
            success = await manager.delete_word(word_id)

            if success:
                return web.json_response({'success': True})
            else:
                return web.json_response({'error': 'Word not found'}, status=404)

        except Exception as exc:
            logger.exception(f"Error deleting word: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def get_stats(self, request: web.Request) -> web.Response:
        """GET /api/greek/stats - Get user statistics"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            manager = GreekLearningManager(user['id'])
            stats = await manager.load_stats()

            return web.json_response(stats.to_dict())

        except Exception as exc:
            logger.exception(f"Error getting stats: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def get_matching_exercise(self, request: web.Request) -> web.Response:
        """GET /api/greek/exercise/matching - Generate matching cards exercise"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            # Get direction from query params (optional - random if not provided)
            direction_str = request.rel_url.query.get('direction', 'greek_to_russian')

            # Get word_type filter (optional)
            word_type = request.rel_url.query.get('word_type', '').strip()
            word_types = [word_type] if word_type else None

            try:
                direction = ExerciseDirection(direction_str)
            except ValueError:
                direction = ExerciseDirection.GREEK_TO_RUSSIAN

            manager = GreekLearningManager(user['id'])
            exercise = await manager.generate_matching_cards_exercise(direction, word_types=word_types)

            if not exercise:
                error_msg = 'No words available'
                if word_types:
                    error_msg += f' for type: {word_type}'
                return web.json_response({'error': error_msg}, status=400)

            return web.json_response(exercise.to_dict())

        except Exception as exc:
            logger.exception(f"Error generating matching exercise: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)

    async def validate_matching_answers(self, request: web.Request) -> web.Response:
        """POST /api/greek/validate-matching - Update statistics for matching exercise"""
        user = await self._authenticate(request)
        if not user:
            return web.json_response({'error': 'Unauthorized'}, status=401)

        try:
            data = await request.json()
            results = data.get('results', [])  # List of {word_id: str, is_correct: bool}
            direction_str = data.get('direction', 'greek_to_russian')

            if not results:
                return web.json_response({'error': 'Missing results'}, status=400)

            try:
                direction = ExerciseDirection(direction_str)
            except ValueError:
                direction = ExerciseDirection.GREEK_TO_RUSSIAN

            manager = GreekLearningManager(user['id'])

            # Update statistics for each word
            for result in results:
                word_id = result.get('word_id')
                is_correct = result.get('is_correct', False)

                if word_id:
                    await manager.update_word_stats(word_id, is_correct)
                    await manager.update_user_stats(direction, is_correct)

            return web.json_response({
                'success': True,
                'message': f'Updated statistics for {len(results)} words'
            })

        except Exception as exc:
            logger.exception(f"Error validating matching answers: {exc}")
            return web.json_response({'error': 'Internal server error'}, status=500)


async def start_web_app(botnav: TeleBotNav) -> None:
    """
    Start aiohttp web server for Greek Learning Web App
    Runs in background as async task
    """
    try:
        webapp = GreekWebApp(botnav)
        runner = web.AppRunner(webapp.app)
        await runner.setup()

        # Get port from config or use default
        port = getattr(config, 'GREEK_LEARNING_WEBAPP_PORT', 8080)

        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()

        logger.info(f"🇬🇷 Greek Learning Web App started on http://0.0.0.0:{port}")
        logger.info(f"   API: http://0.0.0.0:{port}/greek/api/")
        logger.info(f"   Static: http://0.0.0.0:{port}/greek/")

    except Exception as exc:
        logger.exception(f"Failed to start Greek Learning Web App: {exc}")
