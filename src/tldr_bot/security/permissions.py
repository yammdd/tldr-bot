"""Discord role hierarchy permission checks for sensitive bot configuration commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PermissionCheckResult:
    allowed: bool
    reason: str
    bot_top_role_name: str = "@everyone"


def check_role_above_bot(member: Any, guild: Any) -> PermissionCheckResult:
    """Allow only the Server Owner or members whose top role is strictly above the bot's top role."""
    if guild is None or member is None:
        return PermissionCheckResult(allowed=False, reason="guild_only")

    bot_member = getattr(guild, "me", None)
    bot_top_role = getattr(bot_member, "top_role", None) if bot_member is not None else None
    bot_role_name = getattr(bot_top_role, "name", "@everyone") if bot_top_role is not None else "@everyone"

    owner_id = getattr(guild, "owner_id", None)
    member_id = getattr(member, "id", None)
    if owner_id is not None and member_id == owner_id:
        return PermissionCheckResult(
            allowed=True,
            reason="guild_owner",
            bot_top_role_name=bot_role_name,
        )

    if bot_top_role is None:
        return PermissionCheckResult(
            allowed=False,
            reason="bot_role_unknown",
            bot_top_role_name=bot_role_name,
        )

    member_top_role = getattr(member, "top_role", None)
    if member_top_role is None:
        return PermissionCheckResult(
            allowed=False,
            reason="role_too_low",
            bot_top_role_name=bot_role_name,
        )

    member_pos = int(getattr(member_top_role, "position", 0))
    bot_pos = int(getattr(bot_top_role, "position", 0))
    if member_pos > bot_pos:
        return PermissionCheckResult(
            allowed=True,
            reason="role_above_bot",
            bot_top_role_name=bot_role_name,
        )

    return PermissionCheckResult(
        allowed=False,
        reason="role_too_low",
        bot_top_role_name=bot_role_name,
    )
