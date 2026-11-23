# Privacy Policy - TL;DR Bot

*Last updated: November 2026*

## 1. Data Processed
- **Channel Messages (Transient Only):** When a member runs `/tldr` or the message context menu, the bot reads recent messages in that channel strictly in memory to generate a summary via the server's configured LLM endpoint. **Raw chat messages are never written to disk or stored in any database.**
- **Temporary Summary Cache:** Generated summaries may be cached in volatile RAM for up to 3 minutes to avoid duplicate API requests and are automatically evicted.
- **Server Configuration:** Each server's selected LLM provider metadata, default language (`/language`), and write-only API key (encrypted per server with AES-256-GCM and guild-bound AAD when `KEY_STORAGE_MODE=encrypted`, or RAM-only when `KEY_STORAGE_MODE=memory`).

## 2. Third-Party LLM Providers
When `/tldr` is invoked, preprocessed conversation transcripts are transmitted solely to the LLM endpoint configured by that server's administrators via `/config`. Each server controls which provider receives its transcripts and is subject to that provider's privacy terms.

## 3. Data Deletion & Control
- Server Owners or administrators with a role higher than the bot can permanently delete stored API credentials at any time using `/reset_config`.
- Removing the bot from a server clears its in-memory cache immediately (and purges stored configuration automatically when the host enables `PURGE_ON_GUILD_REMOVE=true`).
