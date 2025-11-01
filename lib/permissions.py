from typing import Any

from telebot.types import Message

from telebot_nav import TeleBotNav
from lib.llm import AIProvider, LLMModel
import config


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

    username = botnav.get_user(message).username

    if not username:
        return None

    default_permissions = config.USER_PERMISSIONS.get('default', {})

    user_permissions = config.USER_PERMISSIONS.get(username.lower(), None)

    if not user_permissions:
        user_permissions = default_permissions

    permission_value = user_permissions.get(permission, None)

    if permission_value is None:
        permission_value = default_permissions.get(permission, None)

    return permission_value
