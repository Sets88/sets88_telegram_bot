import config


def get_user_display_name(user_id: int|str) -> str:
    """Get display name from ALLOWED_USER_IDS, fallback to user_id if not found"""
    user_id_str = str(user_id)
    return config.ALLOWED_USER_IDS.get(user_id_str, f"User_{user_id_str}")
