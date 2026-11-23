"""Core /tldr slash command and 'TL;DR from here' message context menu."""

from __future__ import annotations

import logging
from datetime import timedelta

import discord
from discord import app_commands
from discord.ext import commands

from tldr_bot.discord_ext.cache import SummaryCache
from tldr_bot.discord_ext.cogs.admin_cog import LANGUAGE_CHOICES
from tldr_bot.discord_ext.concurrency import ConcurrencyLimitError
from tldr_bot.discord_ext.fetcher import FetchResult, fetch_channel_messages, parse_since_duration
from tldr_bot.discord_ext.preprocessor import PreprocessedConversation, preprocess_messages
from tldr_bot.discord_ext.ui_config import format_llm_error
from tldr_bot.discord_ext.ui_summary import (
    SummaryActionView,
    SummarySessionContext,
    build_summary_embeds,
)
from tldr_bot.i18n import t
from tldr_bot.llm.client import LLMError
from tldr_bot.llm.prompts import SUMMARY_STYLES
from tldr_bot.llm.summarizer import SummaryResult
from tldr_bot.security.vault import GuildLLMConfig

logger = logging.getLogger(__name__)

STYLE_CHOICES: list[app_commands.Choice[str]] = [
    app_commands.Choice(name=spec.label, value=spec.id)
    for spec in SUMMARY_STYLES.values()
]


class TldrCog(commands.Cog):
    """Provides /tldr summarization and right-click Message Context Menu summarization."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._ctx_menu = app_commands.ContextMenu(
            name="TL;DR from here",
            callback=self._context_menu_tldr,
        )

    async def cog_load(self) -> None:
        self.bot.tree.add_command(self._ctx_menu)

    async def cog_unload(self) -> None:
        self.bot.tree.remove_command(self._ctx_menu.name, type=self._ctx_menu.type)

    def _format_any_error(self, lang_code: str, exc: Exception) -> str:
        if isinstance(exc, LLMError):
            return format_llm_error(lang_code, exc)
        if isinstance(exc, ConcurrencyLimitError):
            return t(lang_code, exc.scope)
        return t(lang_code, "err_unknown", detail=type(exc).__name__)

    def _has_channel_history_permission(self, interaction: discord.Interaction) -> bool:
        guild = interaction.guild
        channel = interaction.channel
        if guild is None or channel is None:
            return False
        me = guild.me
        if me is None or not hasattr(channel, "permissions_for"):
            return True
        perms = channel.permissions_for(me)
        return bool(perms.view_channel and perms.read_message_history)

    async def _rerun_session(
        self,
        ctx: SummarySessionContext,
        new_style: str,
        bypass_cache: bool,
    ) -> tuple[SummaryResult, bool]:
        creds = self.bot.vault.get_runtime_credentials(ctx.guild_id)
        if creds is None:
            raise LLMError("not_configured")

        cache_key = SummaryCache.make_key(
            guild_id=ctx.guild_id,
            channel_id=0,
            latest_message_id=ctx.latest_message_id,
            scope_tag=ctx.scope_tag,
            style_id=new_style,
            language_code=ctx.lang_code,
            focus_query=ctx.focus_query,
            model_id=creds.model,
        )
        if not bypass_cache:
            cached = self.bot.summary_cache.get(cache_key)
            if cached is not None:
                return cached, True

        async with self.bot.concurrency.acquire(ctx.guild_id):
            result = await self.bot.summarizer.summarize(
                creds=creds,
                channel_name=ctx.channel_name,
                time_range=ctx.conv.time_range_label,
                participants_summary=ctx.conv.participants_summary,
                turns=ctx.conv.turns,
                valid_ref_ids=ctx.conv.valid_ref_ids,
                language_code=ctx.lang_code,
                style_id=new_style,
                focus_query=ctx.focus_query,
            )
        self.bot.summary_cache.set(cache_key, result)
        return result, False

    async def _validate_access(
        self,
        interaction: discord.Interaction,
        *,
        override_language: str | None,
    ) -> tuple[discord.Guild, object, str, GuildLLMConfig] | None:
        guild = interaction.guild
        channel = interaction.channel
        if guild is None or channel is None or not hasattr(channel, "history"):
            await interaction.response.send_message(t("en", "guild_only"), ephemeral=True)
            return None

        lang = override_language or self.bot.prefs.get_language(guild.id)

        creds = self.bot.vault.get_runtime_credentials(guild.id)
        if creds is None:
            await interaction.response.send_message(t(lang, "not_configured"), ephemeral=True)
            return None

        if not self._has_channel_history_permission(interaction):
            await interaction.response.send_message(
                t(lang, "perm_no_channel_history"), ephemeral=True
            )
            return None

        wait_sec = self.bot.cooldowns.check_remaining(guild.id, interaction.user.id)
        if wait_sec > 0:
            await interaction.response.send_message(
                t(lang, "cooldown_wait", seconds=wait_sec), ephemeral=True
            )
            return None

        return guild, channel, lang, creds

    async def _fetch_history(
        self,
        interaction: discord.Interaction,
        channel: object,
        *,
        channel_name: str,
        lang: str,
        count: int | None,
        since_delta: timedelta | None,
        after_message: discord.Message | None,
    ) -> FetchResult | None:
        async def _on_fetch_progress(fetched: int) -> None:
            try:
                await interaction.edit_original_response(
                    content=t(lang, "progress_fetching", channel=channel_name, fetched=fetched)
                )
            except discord.HTTPException:
                pass

        try:
            return await fetch_channel_messages(
                channel,
                count=count,
                since=since_delta,
                after_message=after_message,
                hard_cap=self.bot.settings.max_messages_hard_cap,
                on_progress=_on_fetch_progress,
            )
        except discord.Forbidden:
            await interaction.edit_original_response(content=t(lang, "perm_no_channel_history"))
            return None
        except Exception as exc:
            logger.exception(
                "Error fetching channel history in guild %s",
                getattr(interaction.guild, "id", None),
            )
            await interaction.edit_original_response(
                content=t(lang, "summary_failed", error=self._format_any_error(lang, exc))
            )
            return None

    async def _generate_or_load_summary(
        self,
        interaction: discord.Interaction,
        *,
        guild_id: int,
        creds: GuildLLMConfig,
        conv: PreprocessedConversation,
        channel_name: str,
        full_scope_tag: str,
        latest_message_id: int,
        style_id: str,
        lang: str,
        focus_query: str | None,
    ) -> tuple[SummaryResult, bool] | None:
        cache_key = SummaryCache.make_key(
            guild_id=guild_id,
            channel_id=0,
            latest_message_id=latest_message_id,
            scope_tag=full_scope_tag,
            style_id=style_id,
            language_code=lang,
            focus_query=focus_query,
            model_id=creds.model,
        )
        cached = self.bot.summary_cache.get(cache_key)
        if cached is not None:
            return cached, True

        non_break_count = sum(1 for turn in conv.turns if not turn.is_session_break)

        async def _on_stage(mode: str, est_tokens: int, chunk_count: int) -> None:
            try:
                if mode == "map_reduce":
                    msg = t(
                        lang,
                        "progress_map_reduce",
                        fetched=conv.raw_message_count,
                        tokens=est_tokens,
                        chunks=chunk_count,
                        model=creds.model,
                    )
                else:
                    msg = t(
                        lang,
                        "progress_analyzing",
                        fetched=conv.raw_message_count,
                        lines=non_break_count,
                        tokens=est_tokens,
                        model=creds.model,
                    )
                await interaction.edit_original_response(content=msg)
            except discord.HTTPException:
                pass

        try:
            result = await self.bot.summarizer.summarize(
                creds=creds,
                channel_name=channel_name,
                time_range=conv.time_range_label,
                participants_summary=conv.participants_summary,
                turns=conv.turns,
                valid_ref_ids=conv.valid_ref_ids,
                language_code=lang,
                style_id=style_id,
                focus_query=focus_query,
                on_stage_update=_on_stage,
            )
        except LLMError as exc:
            await interaction.edit_original_response(
                content=t(lang, "summary_failed", error=format_llm_error(lang, exc))
            )
            return None
        except Exception as exc:
            logger.exception("Unexpected summarization failure in guild %s", guild_id)
            await interaction.edit_original_response(
                content=t(lang, "summary_failed", error=self._format_any_error(lang, exc))
            )
            return None

        self.bot.summary_cache.set(cache_key, result)
        return result, False

    async def _execute_tldr(
        self,
        interaction: discord.Interaction,
        *,
        count: int | None = None,
        since_delta: timedelta | None = None,
        after_message: discord.Message | None = None,
        scope_tag: str,
        style_id: str,
        focus_query: str | None,
        override_language: str | None,
    ) -> None:
        validated = await self._validate_access(
            interaction, override_language=override_language
        )
        if validated is None:
            return
        guild, channel, lang, creds = validated

        try:
            async with self.bot.concurrency.acquire(guild.id):
                self.bot.cooldowns.touch(guild.id, interaction.user.id)
                channel_name = getattr(channel, "name", "channel")
                await interaction.response.defer(ephemeral=True, thinking=True)

                fetch_res = await self._fetch_history(
                    interaction,
                    channel,
                    channel_name=channel_name,
                    lang=lang,
                    count=count,
                    since_delta=since_delta,
                    after_message=after_message,
                )
                if fetch_res is None:
                    return

                conv = preprocess_messages(
                    fetch_res.messages,
                    requester_id=interaction.user.id,
                    max_single_message_chars=self.bot.settings.max_single_message_chars,
                )
                if not any(not turn.is_session_break for turn in conv.turns):
                    await interaction.edit_original_response(content=t(lang, "no_messages_found"))
                    return

                full_scope_tag = f"{getattr(channel, 'id', 0)}:{scope_tag}"
                outcome = await self._generate_or_load_summary(
                    interaction,
                    guild_id=guild.id,
                    creds=creds,
                    conv=conv,
                    channel_name=channel_name,
                    full_scope_tag=full_scope_tag,
                    latest_message_id=fetch_res.latest_message_id,
                    style_id=style_id,
                    lang=lang,
                    focus_query=focus_query,
                )
                if outcome is None:
                    return
                result, from_cache = outcome

                requester_display = getattr(interaction.user, "display_name", str(interaction.user))
                pages = build_summary_embeds(
                    result=result,
                    conv=conv,
                    channel_name=channel_name,
                    lang_code=lang,
                    style_id=style_id,
                    hit_fetch_cap=fetch_res.hit_hard_cap,
                    from_cache=from_cache,
                    public_share=False,
                    requester_display=requester_display,
                )
                session_ctx = SummarySessionContext(
                    guild_id=guild.id,
                    channel_name=channel_name,
                    requester_id=interaction.user.id,
                    requester_display=requester_display,
                    conv=conv,
                    result=result,
                    lang_code=lang,
                    style_id=style_id,
                    focus_query=focus_query,
                    scope_tag=full_scope_tag,
                    latest_message_id=fetch_res.latest_message_id,
                    hit_fetch_cap=fetch_res.hit_hard_cap,
                    from_cache=from_cache,
                )
                view = SummaryActionView(
                    ctx=session_ctx,
                    pages=pages,
                    rerun_callback=self._rerun_session,
                    format_error_fn=self._format_any_error,
                )
                await interaction.edit_original_response(
                    content=None,
                    embed=pages[0],
                    view=view,
                )
        except ConcurrencyLimitError as exc:
            await interaction.response.send_message(t(lang, exc.scope), ephemeral=True)

    @app_commands.command(
        name="tldr",
        description="Summarize recent channel messages with clear speaker attribution and decisions",
    )
    @app_commands.guild_only()
    @app_commands.describe(
        count="Number of recent messages to summarize (e.g. 100, 300, 500; default 100)",
        since="Or summarize by time window (e.g. 30m, 2h, 12h, 24h, 3d, today)",
        style="Summary format (default: Standard)",
        focus="Optional: specific topic or question to focus on (e.g. 'dinner plans', 'deadlines')",
        language="Optional: override output language for this summary only",
    )
    @app_commands.choices(style=STYLE_CHOICES, language=LANGUAGE_CHOICES)
    async def tldr_command(
        self,
        interaction: discord.Interaction,
        count: int | None = None,
        since: str | None = None,
        style: app_commands.Choice[str] | None = None,
        focus: str | None = None,
        language: app_commands.Choice[str] | None = None,
    ) -> None:
        guild = interaction.guild
        lang = (
            language.value
            if language is not None
            else self.bot.prefs.get_language(guild.id if guild else None)
        )
        settings = self.bot.settings

        since_delta: timedelta | None = None
        target_count: int | None = None
        if since and since.strip():
            since_delta = parse_since_duration(since, max_days=settings.max_days_since)
            if since_delta is None:
                await interaction.response.send_message(
                    t(lang, "invalid_since", max_days=settings.max_days_since),
                    ephemeral=True,
                )
                return
            scope_tag = f"since:{since.strip().lower()}"
        else:
            target_count = settings.default_message_count if count is None else int(count)
            if (
                target_count < settings.min_message_count
                or target_count > settings.max_messages_hard_cap
            ):
                await interaction.response.send_message(
                    t(
                        lang,
                        "invalid_count",
                        min_count=settings.min_message_count,
                        max_count=settings.max_messages_hard_cap,
                    ),
                    ephemeral=True,
                )
                return
            scope_tag = f"count:{target_count}"

        await self._execute_tldr(
            interaction,
            count=target_count,
            since_delta=since_delta,
            scope_tag=scope_tag,
            style_id=style.value if style else "standard",
            focus_query=focus,
            override_language=language.value if language else None,
        )

    async def _context_menu_tldr(
        self,
        interaction: discord.Interaction,
        message: discord.Message,
    ) -> None:
        await self._execute_tldr(
            interaction,
            after_message=message,
            scope_tag=f"after:{message.id}",
            style_id="standard",
            focus_query=None,
            override_language=None,
        )
