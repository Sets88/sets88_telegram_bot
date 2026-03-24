from typing import Any

from telebot.types import Message

from telebot_nav import TeleBotNav
from lib.structs import AIProvider, LLMModel
import config


def is_replicate_available(user_id: int) -> bool:
    if not config.REPLICATE_API_KEY:
        return False

    if not is_permitted(user_id, 'can_use_replicate_models'):
        return False

    return True


def is_llm_model_allowed(user_id: int, model: LLMModel) -> bool:
    if model.provider == AIProvider.OLLAMA:
        if is_permitted(user_id, 'can_use_ollama_llm_models'):
            return True
        return False

    return True


def is_permitted(user_id: int, permission: str) -> bool:
    if get_permission(user_id, 'is_admin'):
        return True
    
    perm = get_permission(user_id, permission)
    return bool(perm)


def get_permission(user_id: int, permission: str) -> Any:
    if not config.USER_PERMISSIONS:
        return None

    user_id_str = str(user_id)

    default_permissions = config.USER_PERMISSIONS.get('default', {})

    user_permissions = config.USER_PERMISSIONS.get(user_id_str, None)

    if not user_permissions:
        user_permissions = default_permissions

    permission_value = user_permissions.get(permission, None)

    if permission_value is None:
        permission_value = default_permissions.get(permission, None)

    return permission_value


def get_permission_for_id(user_id_str: str, permission: str) -> Any:
    if not config.USER_PERMISSIONS:
        return None
    default_permissions = config.USER_PERMISSIONS.get('default', {})
    user_permissions = config.USER_PERMISSIONS.get(user_id_str, None)
    if not user_permissions:
        user_permissions = default_permissions
    permission_value = user_permissions.get(permission, None)
    if permission_value is None:
        permission_value = default_permissions.get(permission, None)
    return permission_value


def get_allowed_replicate_models(user_id: int) -> list[str]:
    from replicate_module import REPLICATE_MODELS  # local import — avoids circular dep
    if not is_replicate_available(user_id):
        return []
    if get_permission(user_id, 'is_admin'):
        return list(REPLICATE_MODELS.keys())
    exclude = get_permission(user_id, 'exclude_replicate_models') or []
    return [m for m in REPLICATE_MODELS if m not in exclude]
