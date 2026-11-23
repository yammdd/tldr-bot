"""Private Ephemeral Provider Select + Write-Only API Key Modal with Live Smoke Testing."""

from __future__ import annotations

import logging

import discord

from tldr_bot.discord_ext.cache import SummaryCache
from tldr_bot.i18n import GuildPreferencesStore, t
from tldr_bot.llm.client import LLMError, UnifiedLLMClient
from tldr_bot.llm.presets import PROVIDER_PRESETS, ProviderPreset, get_preset
from tldr_bot.security.permissions import check_role_above_bot
from tldr_bot.security.vault import KeyVault

logger = logging.getLogger(__name__)


def format_llm_error(lang_code: str, error: LLMError) -> str:
    return t(
        lang_code,
        error.code,
        model=error.model,
        detail=error.detail or error.code,
    )


class ApiConfigModal(discord.ui.Modal):
    """Private modal dialog for entering API Key, Endpoint URL, and Model ID."""

    def __init__(
        self,
        *,
        preset: ProviderPreset,
        vault: KeyVault,
        llm_client: UnifiedLLMClient,
        prefs: GuildPreferencesStore,
        cache: SummaryCache,
        lang_code: str,
    ) -> None:
        super().__init__(
            title=t(lang_code, "config_modal_title", provider_name=preset.name)[:45]
        )
        self._preset = preset
        self._vault = vault
        self._llm = llm_client
        self._prefs = prefs
        self._cache = cache

        self.api_key_input = discord.ui.TextInput(
            label=t(lang_code, "config_field_apikey")[:45],
            style=discord.TextStyle.short,
            placeholder="sk-... / AIza... / AQ....",
            required=True,
            min_length=6,
            max_length=300,
        )
        self.endpoint_input = discord.ui.TextInput(
            label=t(lang_code, "config_field_endpoint")[:45],
            style=discord.TextStyle.short,
            default=preset.default_base_url,
            placeholder="https://api.example.com/v1",
            required=True,
            min_length=8,
            max_length=250,
        )
        self.model_input = discord.ui.TextInput(
            label=t(
                lang_code, "config_field_model", default_model=preset.default_model
            )[:45],
            style=discord.TextStyle.short,
            default="" if preset.is_custom else preset.default_model,
            placeholder=preset.default_model,
            required=True,
            min_length=2,
            max_length=120,
        )

        self.add_item(self.api_key_input)
        self.add_item(self.endpoint_input)
        self.add_item(self.model_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        lang = self._prefs.get_language(guild.id if guild else None)
        perm = check_role_above_bot(interaction.user, guild)
        if not perm.allowed:
            key = "guild_only" if perm.reason == "guild_only" else "perm_role_too_low"
            await interaction.response.send_message(
                t(lang, key, bot_role=perm.bot_top_role_name),
                ephemeral=True,
            )
            return

        assert guild is not None
        raw_api_key = str(self.api_key_input.value).strip()
        raw_endpoint = str(self.endpoint_input.value).strip()
        raw_model = str(self.model_input.value).strip()

        await interaction.response.send_message(
            t(
                lang,
                "config_testing",
                provider_name=self._preset.name,
                model=raw_model,
            ),
            ephemeral=True,
        )

        try:
            smoke = await self._llm.verify_credentials(
                base_url=raw_endpoint,
                api_key=raw_api_key,
                model=raw_model,
            )
        except LLMError as exc:
            err_msg = format_llm_error(lang, exc)
            await interaction.edit_original_response(
                content=t(lang, "config_failed_keep_old", error=err_msg)
            )
            return
        except Exception as exc:  # pragma: no cover - defensive catch
            logger.exception("Unexpected error during API key verification for guild %s", guild.id)
            await interaction.edit_original_response(
                content=t(
                    lang,
                    "config_failed_keep_old",
                    error=t(lang, "err_unknown", detail=type(exc).__name__),
                )
            )
            return

        status = self._vault.set_config(
            guild.id,
            provider_id=self._preset.id,
            provider_name=self._preset.name,
            base_url=smoke.normalized_base_url,
            model=raw_model,
            api_key=raw_api_key,
        )
        self._cache.clear_guild(guild.id)

        await interaction.edit_original_response(
            content=t(
                lang,
                "config_success",
                latency_ms=smoke.latency_ms,
                provider_name=status.provider_name,
                endpoint_host=status.endpoint_host,
                model=status.model,
                storage_mode=status.storage_mode,
            )
        )


class ProviderSelect(discord.ui.Select):
    def __init__(
        self,
        *,
        vault: KeyVault,
        llm_client: UnifiedLLMClient,
        prefs: GuildPreferencesStore,
        cache: SummaryCache,
        lang_code: str,
    ) -> None:
        options = [
            discord.SelectOption(
                label=p.name,
                value=p.id,
                description=p.description[:100],
            )
            for p in PROVIDER_PRESETS
        ]
        super().__init__(
            placeholder=t(lang_code, "config_select_placeholder")[:100],
            min_values=1,
            max_values=1,
            options=options,
        )
        self._vault = vault
        self._llm = llm_client
        self._prefs = prefs
        self._cache = cache
        self._lang = lang_code

    async def callback(self, interaction: discord.Interaction) -> None:
        perm = check_role_above_bot(interaction.user, interaction.guild)
        if not perm.allowed:
            lang = self._prefs.get_language(interaction.guild.id if interaction.guild else None)
            key = "guild_only" if perm.reason == "guild_only" else "perm_role_too_low"
            await interaction.response.send_message(
                t(lang, key, bot_role=perm.bot_top_role_name),
                ephemeral=True,
            )
            return

        preset = get_preset(self.values[0])
        modal = ApiConfigModal(
            preset=preset,
            vault=self._vault,
            llm_client=self._llm,
            prefs=self._prefs,
            cache=self._cache,
            lang_code=self._lang,
        )
        await interaction.response.send_modal(modal)


class ProviderConfigView(discord.ui.View):
    """Ephemeral configuration launcher view."""

    def __init__(
        self,
        *,
        vault: KeyVault,
        llm_client: UnifiedLLMClient,
        prefs: GuildPreferencesStore,
        cache: SummaryCache,
        lang_code: str,
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(
            ProviderSelect(
                vault=vault,
                llm_client=llm_client,
                prefs=prefs,
                cache=cache,
                lang_code=lang_code,
            )
        )


async def verify_and_switch_model(
    *,
    guild_id: int,
    new_model: str,
    vault: KeyVault,
    llm_client: UnifiedLLMClient,
    cache: SummaryCache,
    lang_code: str,
) -> tuple[bool, str]:
    """Run a live smoke test with the guild's existing key and switch model only if it passes."""
    creds = vault.get_runtime_credentials(guild_id)
    if creds is None:
        return False, t(lang_code, "not_configured")

    clean_model = new_model.strip()
    try:
        smoke = await llm_client.verify_credentials(
            base_url=creds.base_url,
            api_key=creds.api_key,
            model=clean_model,
        )
    except LLMError as exc:
        return False, t(
            lang_code,
            "model_failed_keep_old",
            new_model=clean_model,
            old_model=creds.model,
            error=format_llm_error(lang_code, exc),
        )
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.exception("Unexpected error during model switch in guild %s", guild_id)
        return False, t(
            lang_code,
            "model_failed_keep_old",
            new_model=clean_model,
            old_model=creds.model,
            error=t(lang_code, "err_unknown", detail=type(exc).__name__),
        )

    status = vault.update_model(guild_id, clean_model)
    cache.clear_guild(guild_id)
    return True, t(
        lang_code,
        "model_switched",
        latency_ms=smoke.latency_ms,
        provider_name=status.provider_name,
        endpoint_host=status.endpoint_host,
        model=status.model,
    )


class ManualModelModal(discord.ui.Modal):
    """Modal allowing admins to type any exact model ID without re-entering the API key."""

    def __init__(
        self,
        *,
        current_model: str,
        vault: KeyVault,
        llm_client: UnifiedLLMClient,
        prefs: GuildPreferencesStore,
        cache: SummaryCache,
        lang_code: str,
    ) -> None:
        super().__init__(title=t(lang_code, "model_modal_title")[:45])
        self._vault = vault
        self._llm = llm_client
        self._prefs = prefs
        self._cache = cache

        self.model_input = discord.ui.TextInput(
            label=t(lang_code, "model_modal_field")[:45],
            style=discord.TextStyle.short,
            default=current_model,
            placeholder=current_model or "gemini-3.8-flash",
            required=True,
            min_length=2,
            max_length=120,
        )
        self.add_item(self.model_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        lang = self._prefs.get_language(guild.id if guild else None)
        perm = check_role_above_bot(interaction.user, guild)
        if not perm.allowed:
            key = "guild_only" if perm.reason == "guild_only" else "perm_role_too_low"
            await interaction.response.send_message(
                t(lang, key, bot_role=perm.bot_top_role_name),
                ephemeral=True,
            )
            return

        assert guild is not None
        status = self._vault.get_public_status(guild.id)
        if not status.configured:
            await interaction.response.send_message(t(lang, "not_configured"), ephemeral=True)
            return

        new_model = str(self.model_input.value).strip()
        await interaction.response.send_message(
            t(lang, "model_testing", model=new_model, provider_name=status.provider_name),
            ephemeral=True,
        )
        _ok, msg = await verify_and_switch_model(
            guild_id=guild.id,
            new_model=new_model,
            vault=self._vault,
            llm_client=self._llm,
            cache=self._cache,
            lang_code=lang,
        )
        await interaction.edit_original_response(content=msg)


class ModelPickerView(discord.ui.View):
    """Paginated dropdown of models discovered from the guild's API + manual entry fallback."""

    _PAGE_SIZE = 25

    def __init__(
        self,
        *,
        models: list[str],
        current_model: str,
        panel_embed: discord.Embed,
        vault: KeyVault,
        llm_client: UnifiedLLMClient,
        prefs: GuildPreferencesStore,
        cache: SummaryCache,
        lang_code: str,
    ) -> None:
        super().__init__(timeout=300)
        self._models = models
        self._current_model = current_model
        self._panel_embed = panel_embed
        self._vault = vault
        self._llm = llm_client
        self._prefs = prefs
        self._cache = cache
        self._lang = lang_code
        self._page = 0
        self._total_pages = max(1, (len(models) + self._PAGE_SIZE - 1) // self._PAGE_SIZE)
        self._rebuild_items()

    def _rebuild_items(self) -> None:
        self.clear_items()
        lang = self._lang

        if self._models:
            start = self._page * self._PAGE_SIZE
            slice_models = self._models[start : start + self._PAGE_SIZE]
            active_tag = t(lang, "model_option_active")
            options = [
                discord.SelectOption(
                    label=mid[:100],
                    value=mid[:100],
                    description=active_tag if mid == self._current_model else None,
                    default=mid == self._current_model,
                )
                for mid in slice_models
            ]
            placeholder = t(
                lang,
                "model_select_placeholder",
                start=start + 1,
                end=start + len(slice_models),
                total=len(self._models),
            )[:100]
            select = discord.ui.Select(
                placeholder=placeholder,
                min_values=1,
                max_values=1,
                options=options,
                row=0,
            )
            select.callback = self._on_select
            self.add_item(select)

        if self._total_pages > 1:
            prev_btn = discord.ui.Button(
                label=t(lang, "btn_prev"),
                style=discord.ButtonStyle.secondary,
                disabled=self._page <= 0,
                row=1,
            )
            prev_btn.callback = self._on_prev
            next_btn = discord.ui.Button(
                label=t(lang, "btn_next"),
                style=discord.ButtonStyle.secondary,
                disabled=self._page >= self._total_pages - 1,
                row=1,
            )
            next_btn.callback = self._on_next
            self.add_item(prev_btn)
            self.add_item(next_btn)

        manual_btn = discord.ui.Button(
            label=t(lang, "model_btn_manual"),
            style=discord.ButtonStyle.secondary,
            row=1,
        )
        manual_btn.callback = self._on_manual
        self.add_item(manual_btn)

    async def _on_prev(self, interaction: discord.Interaction) -> None:
        self._page = max(0, self._page - 1)
        self._rebuild_items()
        await interaction.response.edit_message(embed=self._panel_embed, view=self)

    async def _on_next(self, interaction: discord.Interaction) -> None:
        self._page = min(self._total_pages - 1, self._page + 1)
        self._rebuild_items()
        await interaction.response.edit_message(embed=self._panel_embed, view=self)

    async def _on_manual(self, interaction: discord.Interaction) -> None:
        perm = check_role_above_bot(interaction.user, interaction.guild)
        if not perm.allowed:
            key = "guild_only" if perm.reason == "guild_only" else "perm_role_too_low"
            await interaction.response.send_message(
                t(self._lang, key, bot_role=perm.bot_top_role_name),
                ephemeral=True,
            )
            return

        modal = ManualModelModal(
            current_model=self._current_model,
            vault=self._vault,
            llm_client=self._llm,
            prefs=self._prefs,
            cache=self._cache,
            lang_code=self._lang,
        )
        await interaction.response.send_modal(modal)

    async def _on_select(self, interaction: discord.Interaction) -> None:
        guild = interaction.guild
        lang = self._prefs.get_language(guild.id if guild else None)
        perm = check_role_above_bot(interaction.user, guild)
        if not perm.allowed:
            key = "guild_only" if perm.reason == "guild_only" else "perm_role_too_low"
            await interaction.response.send_message(
                t(lang, key, bot_role=perm.bot_top_role_name),
                ephemeral=True,
            )
            return

        assert guild is not None
        selected = interaction.data["values"][0]  # type: ignore[index]
        status = self._vault.get_public_status(guild.id)
        if not status.configured:
            await interaction.response.edit_message(
                content=t(lang, "not_configured"), embed=None, view=None
            )
            return

        await interaction.response.edit_message(
            content=t(lang, "model_testing", model=selected, provider_name=status.provider_name),
            embed=None,
            view=None,
        )
        ok, msg = await verify_and_switch_model(
            guild_id=guild.id,
            new_model=selected,
            vault=self._vault,
            llm_client=self._llm,
            cache=self._cache,
            lang_code=lang,
        )
        if ok:
            await interaction.edit_original_response(content=msg, embed=None, view=None)
        else:
            # Keep the picker visible on failure so the admin can pick another model right away
            await interaction.edit_original_response(
                content=msg, embed=self._panel_embed, view=self
            )
