# Security Policy

## Core Security Guarantees

`tldr-bot` is designed as a multi-server bot where every server brings its own LLM API key:

- **Write-only credentials:** API keys are submitted via private Discord Modals and are never shown again in commands, embeds, or logs.
- **Per-guild encryption:** Keys are encrypted at rest in SQLite WAL using AES-256-GCM with per-record random nonces and `guild_id` bound as AAD (or kept RAM-only with `KEY_STORAGE_MODE=memory`).
- **Role hierarchy enforcement:** Only the Server Owner or members with a role strictly above the bot's top role can modify API keys, models, or language settings.
- **SSRF protection:** Custom endpoints must use `https://` and resolve only to public IP addresses.
- **Zero chat log storage:** Discord channel messages are processed in memory and never persisted to disk.

## Reporting a Vulnerability

Please **do not** open a public GitHub Issue for security vulnerabilities, credential leaks, SSRF bypasses, or permission escalations.

Instead, report the issue privately using **GitHub Security Advisories** (Repository **Security** tab > **Report a vulnerability**) or contact the repository maintainer directly.

Include:
- Affected file or command
- Minimal reproduction steps or proof of concept
- Potential impact and any suggested fix

We will acknowledge reports promptly and coordinate a fix before public disclosure.
