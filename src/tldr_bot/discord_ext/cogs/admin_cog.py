"""Admin & configuration slash commands: /config, /model, /language, /status, /reset_config, /help."""

from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from tldr_bot.discord_ext.ui_config import (
    ModelPickerView,
    ProviderConfigView,
    format_llm_error,
    verify_and_switch_model,
)
from tldr_bot.discord_ext.ui_summary import COLOR_AMBER, COLOR_INK_SLATE, COLOR_SAGE
from tldr_bot.i18n import LANGUAGE_SPECS, get_language_spec, t
from tldr_bot.llm.client import LLMError
from tldr_bot.security.permissions import check_role_above_bot

LANGUAGE_CHOICES: list[app_commands.Choice[str]] = [
    app_commands.Choice(name=spec.label, value=spec.code)
    for spec in LANGUAGE_SPECS.values()
]


class AdminCog(commands.Cog):
    """Handles write-only API key configuration, model switching, summary language, and status."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    async def _require_higher_role(self, interaction: discord.Interaction) -> bool:
        """Verify caller is Owner or holds a role strictly above the bot."""
        perm = check_role_above_bot(interaction.user, interaction.guild)
        if not perm.allowed:
            key = "guild_only" if perm.reason == "guild_only" else "perm_role_too_low"
            await interaction.response.send_message(
                t(key, bot_role=perm.bot_top_role_name),
                ephemeral=True,
            )
            return False
        return True

    @app_commands.command(
        name="config",
        description="Configure API Key, Endpoint, and Model privately (Owner or Role above Bot)",
    )
    @app_commands.guild_only()
    async def config_command(self, interaction: discord.Interaction) -> None:
        if not await self._require_higher_role(interaction):
            return

        embed = discord.Embed(
            title=t("config_panel_title"),
            description=t("config_panel_desc", storage_mode=self.bot.vault.storage_mode),
            color=COLOR_INK_SLATE,
        )
        view = ProviderConfigView(
            vault=self.bot.vault,
            llm_client=self.bot.llm_client,
            cache=self.bot.summary_cache,
        )
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    @app_commands.command(
        name="model",
        description="Switch active LLM model without re-entering the API key (Owner or Role above Bot)",
    )
    @app_commands.guild_only()
    @app_commands.describe(
        model_id="Optional: enter a Model ID directly, or leave blank to list models from your endpoint"
    )
    async def model_command(
        self,
        interaction: discord.Interaction,
        model_id: str | None = None,
    ) -> None:
        if not await self._require_higher_role(interaction):
            return

        assert interaction.guild is not None
        guild_id = interaction.guild.id
        creds = self.bot.vault.get_runtime_credentials(guild_id)
        status = self.bot.vault.get_public_status(guild_id)
        if creds is None or not status.configured:
            await interaction.response.send_message(t("not_configured"), ephemeral=True)
            return

        if model_id and model_id.strip():
            target = model_id.strip()
            await interaction.response.send_message(
                t("model_testing", model=target, provider_name=status.provider_name),
                ephemeral=True,
            )
            _ok, msg = await verify_and_switch_model(
                guild_id=guild_id,
                new_model=target,
                vault=self.bot.vault,
                llm_client=self.bot.llm_client,
                cache=self.bot.summary_cache,
            )
            await interaction.edit_original_response(content=msg)
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        models: list[str] = []
        fetch_error = ""
        try:
            models = await self.bot.llm_client.list_chat_models(creds)
        except LLMError as exc:
            fetch_error = format_llm_error(exc)
        except Exception as exc:
            fetch_error = type(exc).__name__

        if models:
            desc = t(
                "model_panel_desc",
                provider_name=status.provider_name,
                endpoint_host=status.endpoint_host,
                current_model=status.model,
                count=len(models),
            )
        else:
            desc = t(
                "model_panel_no_list",
                provider_name=status.provider_name,
                endpoint_host=status.endpoint_host,
                current_model=status.model,
                reason=fetch_error or "empty list",
            )

        embed = discord.Embed(
            title=t("model_panel_title"),
            description=desc,
            color=COLOR_INK_SLATE,
        )
        view = ModelPickerView(
            models=models,
            current_model=status.model,
            panel_embed=embed,
            vault=self.bot.vault,
            llm_client=self.bot.llm_client,
            cache=self.bot.summary_cache,
        )
        await interaction.edit_original_response(embed=embed, view=view)

    @app_commands.command(
        name="language",
        description="Set the summary output language for this server (Owner or Role above Bot)",
    )
    @app_commands.guild_only()
    @app_commands.describe(language="Language the AI writes summaries in")
    @app_commands.choices(language=LANGUAGE_CHOICES)
    async def language_command(
        self,
        interaction: discord.Interaction,
        language: app_commands.Choice[str],
    ) -> None:
        if not await self._require_higher_role(interaction):
            return

        assert interaction.guild is not None
        spec = self.bot.prefs.set_language(interaction.guild.id, language.value)
        await interaction.response.send_message(
            t("lang_updated", language_label=spec.label),
            ephemeral=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @app_commands.command(
        name="status",
        description="View active bot configuration and limits (never displays the API key or endpoint)",
    )
    @app_commands.guild_only()
    async def status_command(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(t("guild_only"), ephemeral=True)
            return

        summary_lang = get_language_spec(self.bot.prefs.get_language(guild.id))
        status = self.bot.vault.get_public_status(guild.id)
        settings = self.bot.settings

        embed = discord.Embed(
            title=t("status_title"),
            color=COLOR_SAGE if status.configured else COLOR_AMBER,
        )
        embed.add_field(
            name=t("status_field_state"),
            value=t("status_configured_yes" if status.configured else "status_configured_no"),
            inline=False,
        )
        if status.configured:
            embed.add_field(
                name=t("status_field_provider"),
                value=f"**{status.provider_name}**",
                inline=True,
            )
            embed.add_field(
                name=t("status_field_model"),
                value=f"`{status.model}`",
                inline=True,
            )
        embed.add_field(
            name=t("status_field_lang"),
            value=summary_lang.label,
            inline=False,
        )
        embed.add_field(
            name=t("status_field_limits"),
            value=t(
                "status_limits_val",
                max_msgs=settings.max_messages_hard_cap,
                max_days=settings.max_days_since,
                max_tokens=settings.max_total_input_tokens,
                storage_mode=status.storage_mode,
            ),
            inline=False,
        )
        embed.set_footer(text=t("status_footer"))
        await interaction.response.send_message(
            embed=embed,
            ephemeral=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @app_commands.command(
        name="reset_config",
        description="Delete the server's saved API key and LLM configuration (Owner or Role above Bot)",
    )
    @app_commands.guild_only()
    async def reset_config_command(self, interaction: discord.Interaction) -> None:
        if not await self._require_higher_role(interaction):
            return

        assert interaction.guild is not None
        deleted = self.bot.vault.delete_config(interaction.guild.id)
        self.bot.summary_cache.clear_guild(interaction.guild.id)
        key = "config_reset_done" if deleted else "config_reset_empty"
        await interaction.response.send_message(t(key), ephemeral=True)

    @app_commands.command(
        name="help",
        description="Quick setup and usage guide for TL;DR Bot",
    )
    async def help_command(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title=t("help_title"),
            description=t("help_desc"),
            color=COLOR_INK_SLATE,
        )
        embed.add_field(
            name=t("help_setup_header"),
            value=t("help_setup_body"),
            inline=False,
        )
        embed.add_field(
            name=t("help_usage_header"),
            value=t("help_usage_body"),
            inline=False,
        )
        embed.set_footer(text=t("help_footer"))
        await interaction.response.send_message(
            embed=embed,
            ephemeral=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
