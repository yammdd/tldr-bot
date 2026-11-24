"""Main entrypoint for multi-server AutoSharded Discord TL;DR Bot."""

from __future__ import annotations

import logging
import sys

import discord
from discord.ext import commands

from tldr_bot.discord_ext.cache import SummaryCache, UserCooldownTracker
from tldr_bot.discord_ext.cogs.admin_cog import AdminCog
from tldr_bot.discord_ext.cogs.tldr_cog import TldrCog
from tldr_bot.discord_ext.concurrency import ConcurrencyGuard
from tldr_bot.health import HealthServer
from tldr_bot.i18n import GuildPreferencesStore, t
from tldr_bot.llm.client import UnifiedLLMClient
from tldr_bot.llm.summarizer import ConversationSummarizer
from tldr_bot.security.redactor import SecretRedactingFilter
from tldr_bot.security.vault import KeyVault
from tldr_bot.settings import Settings

logger = logging.getLogger(__name__)


def configure_logging(level_name: str = "INFO") -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    redactor = SecretRedactingFilter()
    handler.addFilter(redactor)
    root.handlers.clear()
    root.addHandler(handler)
    root.addFilter(redactor)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


class TldrBot(commands.AutoShardedBot):
    """Multi-server AutoSharded Discord TL;DR Bot."""

    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True

        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="/tldr",
            ),
        )
        self.settings = settings
        self.vault = KeyVault(
            mode=settings.key_storage_mode,
            db_file=settings.db_file_path,
            master_key=settings.vault_master_key,
        )
        self.prefs = GuildPreferencesStore(
            db_file=settings.db_file_path,
            default_language=settings.default_language,
            persist=True,
        )
        self.llm_client = UnifiedLLMClient(
            allow_local_endpoints=settings.allow_local_endpoints,
            request_timeout_seconds=settings.llm_request_timeout_seconds,
            smoke_test_timeout_seconds=settings.smoke_test_timeout_seconds,
        )
        self.summarizer = ConversationSummarizer(self.llm_client, settings)
        self.summary_cache = SummaryCache(
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
        )
        self.cooldowns = UserCooldownTracker(
            cooldown_seconds=settings.user_cooldown_seconds
        )
        self.concurrency = ConcurrencyGuard(
            max_per_guild=settings.max_concurrent_per_guild,
            max_global=settings.max_concurrent_global,
        )
        self._health_server: HealthServer | None = (
            HealthServer(self, port=settings.health_port)
            if settings.health_port > 0
            else None
        )

    async def setup_hook(self) -> None:
        self.tree.on_error = self._on_app_command_error
        await self.add_cog(AdminCog(self))
        await self.add_cog(TldrCog(self))

        if self._health_server is not None:
            await self._health_server.start()

        if self.settings.dev_guild_id:
            dev_guild = discord.Object(id=self.settings.dev_guild_id)
            self.tree.copy_global_to(guild=dev_guild)
            synced = await self.tree.sync(guild=dev_guild)
            logger.info(
                "Synced %d application commands to dev guild %s",
                len(synced),
                self.settings.dev_guild_id,
            )
        else:
            synced = await self.tree.sync()
            logger.info("Synced %d global application commands", len(synced))

    async def _on_app_command_error(
        self,
        interaction: discord.Interaction,
        error: discord.app_commands.AppCommandError,
    ) -> None:
        guild_id = interaction.guild.id if interaction.guild else None
        logger.exception(
            "Unhandled app command error in guild %s: %s", guild_id, type(error).__name__
        )
        msg = t("unexpected_command_error")
        try:
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass

    async def on_ready(self) -> None:
        assert self.user is not None
        logger.info(
            "TL;DR Bot online as %s (ID: %s) | Guilds: %d | Shards: %s | Vault: %s | Default lang: %s",
            self.user,
            self.user.id,
            len(self.guilds),
            self.shard_count,
            self.vault.storage_mode,
            self.settings.default_language,
        )

    async def on_guild_join(self, guild: discord.Guild) -> None:
        logger.info("Joined guild %s (total guilds: %d)", guild.id, len(self.guilds))

    async def on_guild_remove(self, guild: discord.Guild) -> None:
        self.summary_cache.clear_guild(guild.id)
        if self.settings.purge_on_guild_remove:
            self.vault.delete_config(guild.id)
            self.prefs.delete_guild(guild.id)
            logger.info("Removed from guild %s and purged stored data", guild.id)
        else:
            logger.info("Removed from guild %s (cleared RAM cache)", guild.id)

    async def close(self) -> None:
        if self._health_server is not None:
            await self._health_server.stop()
        self.vault.close()
        self.prefs.close()
        # Guard against closing AutoShardedBot before connect() initializes the shard queue
        if hasattr(self, "_AutoShardedClient__queue"):
            await super().close()
        else:
            await self.http.close()


def run() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    if not settings.discord_token or settings.discord_token == "your_discord_bot_token_here":
        logger.error(
            "DISCORD_TOKEN is not configured. Copy .env.example to .env and set DISCORD_TOKEN."
        )
        raise SystemExit(1)

    bot = TldrBot(settings)
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    run()
