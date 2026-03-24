"""
Shared aiohttp web server for all web applications.
Serves Greek learning app (/greek/) and user-generated apps (/apps/).
"""
import os
import hmac
import hashlib
import asyncio
import json as _json
from urllib.parse import parse_qsl
from aiohttp import web

import config
from logger import logger


def _validate_init_data(init_data_raw: str) -> str | None:
    """Validate Telegram WebApp initData HMAC; return user_id str or None."""
    try:
        params = dict(parse_qsl(init_data_raw, keep_blank_values=True))
        received_hash = params.pop('hash', None)
        if not received_hash:
            return None
        data_check_string = '\n'.join(f'{k}={v}' for k, v in sorted(params.items()))
        secret_key = hmac.new(
            b'WebAppData',
            config.TELEGRAM_TOKEN.encode(),
            hashlib.sha256
        ).digest()
        computed = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(computed, received_hash):
            return None
        user = _json.loads(params.get('user', '{}'))
        return str(user['id'])
    except Exception as exc:
        logger.exception(f"Failed to validate init data: {exc}")
        return None


def _authenticate_request(request: web.Request):
    """Returns (user_id_str, None) or (None, error_response)."""
    init_data = request.headers.get('X-Telegram-Init-Data', '')
    if not init_data:
        logger.error(f"Failed to start shared web app server: Missing X-Telegram-Init-Data")
        return None, web.json_response({'error': 'Missing X-Telegram-Init-Data'}, status=401)
    user_id_str = _validate_init_data(init_data)
    if not user_id_str:
        logger.error(f"Failed to start shared web app server: Invalid init data")
        return None, web.json_response({'error': 'Invalid init data'}, status=401)
    if str(user_id_str) not in config.ALLOWED_USER_IDS:
        logger.error(f"Failed to start shared web app server: Forbidden user {user_id_str}")
        return None, web.json_response({'error': 'Forbidden'}, status=403)
    return int(user_id_str), None


async def _llm_complete(botnav, user_id: int, messages: list[dict], system: str, model: str | None = None) -> str:
    from llm_module import conversations, AVAILABLE_LLM_MODELS
    from lib.llm import MessageRole

    base = conversations.get(user_id)
    if base is None:
        raise ValueError('No active conversation for this user')

    sub = base.invoke_subagent()

    if system:
        sub.set_config_param('system_prompt', system)

    if model and model in AVAILABLE_LLM_MODELS:
        sub.set_model(AVAILABLE_LLM_MODELS[model])

    for msg in messages:
        role = MessageRole.USER if msg.get('role') == 'user' else MessageRole.ASSISTANT
        sub.add_message(role, content=msg.get('content', ''))

    text_parts: list[str] = []
    async for chunk in sub.make_request(
        extra_params={
            "user_id": user_id,
            "botnav": botnav,
        }
    ):
        if chunk:
            text_parts.append(chunk)

    return ''.join(text_parts)


class WebAppServer:
    """Central aiohttp application shared by all sub-apps."""

    def __init__(self, botnav) -> None:
        self.botnav = botnav
        self.app = web.Application(middlewares=[self._middleware])
        self._setup_static()
        self._setup_routes()

    @web.middleware
    async def _middleware(self, request: web.Request, handler):
        logger.info(f"{request.method} {request.path} from {request.remote}")
        if request.method == 'OPTIONS':
            response = web.Response()
        else:
            try:
                response = await handler(request)
            except Exception as exc:
                logger.exception(f"Handler error: {exc}")
                return web.json_response({'error': str(exc)}, status=500)
            logger.info(f"{request.method} {request.path} -> {response.status}")

        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Telegram-Init-Data'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://telegram.org; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "media-src 'self' blob: https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'"
        )
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        return response

    def _setup_static(self) -> None:
        webapp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webapp'))

        greek_path = os.path.join(webapp_path, 'greek')
        if os.path.exists(greek_path):
            self.app.router.add_static('/greek/', path=greek_path, name='greek_static')
            logger.info(f"Serving /greek/ from {greek_path}")

        apps_path = os.path.join(webapp_path, 'apps')
        os.makedirs(apps_path, exist_ok=True)
        self.app.router.add_static('/apps/', path=apps_path, name='apps_static')
        logger.info(f"Serving /apps/ from {apps_path}")

    def _setup_routes(self) -> None:
        self.app.router.add_post('/api/llm', self._handle_llm)
        self.app.router.add_post('/api/replicate', self._handle_replicate)
        self.app.router.add_get('/api/models', self._handle_models)

    async def _handle_llm(self, request: web.Request) -> web.Response:
        user_id, err = _authenticate_request(request)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        messages = body.get('messages')
        if not messages or not isinstance(messages, list):
            return web.json_response({'error': 'messages field required'}, status=400)
        system = body.get('system', 'You are a helpful assistant.')
        model = body.get('model')
        try:
            text = await _llm_complete(self.botnav, user_id, messages, system, model)
        except ValueError as exc:
            return web.json_response({'error': str(exc)}, status=503)
        return web.json_response({'text': text})

    async def _handle_replicate(self, request: web.Request) -> web.Response:
        user_id, err = _authenticate_request(request)
        if err:
            return err
        from lib.permissions import is_replicate_available, get_allowed_replicate_models
        if not is_replicate_available(user_id):
            return web.json_response({'error': 'Replicate not available for this user'}, status=403)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        model_name = body.get('model')
        input_data = body.get('input', {})
        if not model_name:
            return web.json_response({'error': 'model field required'}, status=400)
        allowed = get_allowed_replicate_models(user_id)
        if model_name not in allowed:
            return web.json_response({'error': f'Model {model_name!r} not allowed'}, status=403)
        from replicate_module import REPLICATE_MODELS, replicate_execute, build_full_params
        from replicate.helpers import FileOutput
        replicate_model = REPLICATE_MODELS[model_name]
        full_input = build_full_params(replicate_model, input_data)
        output = await asyncio.to_thread(replicate_execute, replicate_model['replicate_id'], full_input)

        def _serialise(v):
            if isinstance(v, FileOutput):
                return v.url
            if isinstance(v, list):
                return [_serialise(i) for i in v]
            return v

        return web.json_response({'output': _serialise(output)})

    async def _handle_models(self, request: web.Request) -> web.Response:
        user_id, err = _authenticate_request(request)
        if err:
            return err
        from lib.permissions import get_allowed_replicate_models
        from llm_module import AVAILABLE_LLM_MODELS, AIProvider
        llm_names = [x for x, y in AVAILABLE_LLM_MODELS.items() if y.provider == AIProvider.OPENROUTER]
        replicate_names = get_allowed_replicate_models(user_id)
        return web.json_response({'llm': llm_names, 'replicate': replicate_names})


_server: WebAppServer | None = None


def get_server(botnav) -> WebAppServer:
    global _server
    if _server is None:
        _server = WebAppServer(botnav)
    return _server


async def start_server(botnav) -> None:
    """Start the shared web server, registering all sub-apps."""
    try:
        from greek_learning_module import GreekWebApp

        server = get_server(botnav)
        GreekWebApp(botnav, server.app)

        runner = web.AppRunner(server.app)
        await runner.setup()

        port = getattr(config, 'WEBAPP_PORT', 8180)
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()

        logger.info(f"Shared web app server started on http://0.0.0.0:{port}")
    except Exception as exc:
        logger.exception(f"Failed to start shared web app server: {exc}")
