from typing import Any

from telebot.types import Message

from telebot_nav import TeleBotNav
from lib.structs import AIProvider, LLMModel
import config


def is_replicate_available(botnav: TeleBotNav, message: Message) -> bool:
    if not config.REPLICATE_API_KEY:
        return False

    if not is_permitted(botnav, message, 'can_use_replicate_models'):
        return False

    return True


def is_llm_model_allowed(botnav: TeleBotNav, message: Message, model: LLMModel) -> bool:
    if model.provider == AIProvider.OLLAMA:
        if is_permitted(botnav, message, 'can_use_ollama_llm_models'):
            return True
        return False

    return True


def is_permitted(botnav: TeleBotNav, message: Message, permission: str) -> bool:
    if get_permission(botnav, message, 'is_admin'):
        return True
    
    perm = get_permission(botnav, message, permission)

    return bool(perm)


def get_permission(botnav: TeleBotNav, message: Message, permission: str) -> Any:
    if not config.USER_PERMISSIONS:
        return None

    user_id = botnav.get_user(message).id
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


def is_permitted_for_id(user_id_str: str, permission: str) -> bool:
    if get_permission_for_id(user_id_str, 'is_admin'):
        return True
    return bool(get_permission_for_id(user_id_str, permission))


def is_replicate_available_for_id(user_id_str: str) -> bool:
    if not config.REPLICATE_API_KEY:
        return False
    return is_permitted_for_id(user_id_str, 'can_use_replicate_models')


def get_allowed_replicate_models_for_id(user_id_str: str) -> list[str]:
    from replicate_module import REPLICATE_MODELS  # local import — avoids circular dep
    if not is_replicate_available_for_id(user_id_str):
        return []
    if get_permission_for_id(user_id_str, 'is_admin'):
        return list(REPLICATE_MODELS.keys())
    exclude = get_permission_for_id(user_id_str, 'exclude_replicate_models') or []
    return [m for m in REPLICATE_MODELS if m not in exclude]
