# TL;DR Bot

Multi-server Discord bot that distills 5 to 1,500 unread channel messages into structured, scannable summaries. Every summary clearly attributes who participated, who said what, what was decided, and what remains open, with direct message jump links and a private Mentions of you section.

<p align="center">
  <img src="assets/bot-avatar.png" alt="TL;DR Bot icon" width="128" height="128" />
</p>

---

## Architecture & Capabilities

- **5-Stage Multi-Party Summarization Pipeline:**
  - Resolves `@mentions`, `#channels`, colliding display names (`Name (@username)`), and reply chains (`->(Name #id)`) to untangle parallel conversations accurately.
  - Coalesces rapid consecutive turns from the same speaker (within 2 minutes), preserves short confirmations (`ok`, `yes`, `+1`), truncates mega-pastes over 800 characters, and inserts session breaks after 30+ minutes of silence.
  - Estimates tokens with `tiktoken` (calibrated for Latin, Vietnamese, and CJK): switches automatically between **Single-Pass (up to 28K tokens)** and **Session-Aware Parallel Map-Reduce**, bounded by hard limits (**1,500 messages / 100K input tokens**).
  - Enforces structured JSON output with code-level post-validation: verifies every `ref_id` before building jump links, neutralizes `@everyone`/`@here` pings, and isolates personal mentions to private views only.
- **Write-Only API Key Security & Multi-Server Isolation:**
  - Configure via `/config` using a private **Ephemeral Select Menu + Discord Modal** so credentials never touch public chat.
  - **Mandatory Live Smoke Test:** runs a tiny real completion check before saving; on error, reports the exact cause and keeps the previous working configuration intact.
  - **Per-Guild AES-256-GCM in SQLite WAL:** each server has its own encrypted row with a unique 12-byte nonce and guild-bound AAD (`guild_id`) to prevent cross-server ciphertext swapping. Supports RAM-only storage via `KEY_STORAGE_MODE=memory`.
  - **Role Hierarchy Guard:** only the **Server Owner** or members with a role strictly higher than the bot's top role can run `/config`, `/model`, `/language`, or `/reset_config`.
  - **SSRF Protection & Secret Redaction:** enforces `https://`, blocks private/localhost IPs after DNS resolution, and strips API keys from logs and error messages.
- **Multi-Server Production Scale:**
  - `AutoShardedBot` for automatic sharding across many servers.
  - `ConcurrencyGuard` limiting concurrent runs per guild and globally so one busy server never starves others.
  - Built-in `GET /healthz` endpoint for Docker, Railway, Render, Fly.io, and Kubernetes probes.
- **Unified OpenAI-Compatible Support:** built-in presets for **Google Gemini, DeepSeek, OpenAI, Anthropic Claude, OpenRouter, Groq**, and any **Custom Endpoint**, plus **`/model`** to discover and switch models without re-entering the API key.

---

## Local Setup

### 1. Create a Bot on the Discord Developer Portal
1. Create an application at [Discord Developer Portal](https://discord.com/developers/applications) and open the **Bot** tab.
2. Upload the minimalist avatar from `assets/bot-avatar.png` (vector source in `assets/bot-avatar.svg`).
3. Enable **Public Bot** (for multi-server invites) and **Privileged Gateway Intents > `MESSAGE CONTENT INTENT`** *(required to read channel history)*.
4. Open **OAuth2 > URL Generator**:
   - Scopes: `bot`, `applications.commands`
   - Permissions: `View Channels`, `Send Messages`, `Read Message History`, `Embed Links`
5. Invite the bot to your server, then open **Server Settings > Roles** and place the Bot role below your Admin/Mod roles and above regular members.

### 2. Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env
```

Set `DISCORD_TOKEN` and generate a strong `VAULT_MASTER_KEY` in `.env`:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Start the bot:

```powershell
.\.venv\Scripts\python.exe run.py
```

---

## Deploy on Railway (Recommended)

This repository includes `railway.toml` and an automatic volume-permission entrypoint in `Dockerfile`.

1. Sign in at [Railway](https://railway.com) and select **New Project > Deploy from GitHub Repo** (`yammdd/tldr-bot`).
2. Attach a persistent volume: open the service **Volumes > Add Volume** and set **Mount Path** to `/app/data`.
3. Open **Variables** and add:
   - `DISCORD_TOKEN`: your bot token from the Discord Developer Portal
   - `VAULT_MASTER_KEY`: a 32+ character random secret (keep this fixed across redeploys)
   - `DB_FILE_PATH`: `/app/data/tldr_bot.db`
   - `HEALTH_PORT`: `8080`
4. Railway builds the Docker image automatically, verifies `/healthz`, and redeploys on every push to `main`.

---

## Multi-Server Docker Deployment

```bash
cp .env.example .env
# Fill DISCORD_TOKEN and VAULT_MASTER_KEY in .env
docker compose up -d --build
```

Verify container readiness:

```bash
curl http://127.0.0.1:8080/healthz
```

*Production notes:*
- Mount a persistent volume at `/app/data` and keep `VAULT_MASTER_KEY` fixed across redeploys (changing it invalidates decryption of existing rows).
- Set `HEALTH_PORT=8080` so your hosting platform can probe `/healthz`.

---

## Commands

| Command / Action | Permission | Description |
| :--- | :--- | :--- |
| `/config` | Owner or Role > Bot | Pick a provider or custom endpoint, enter API Key privately, and run a live smoke test |
| `/model [model_id]` | Owner or Role > Bot | Fetch chat models from the active endpoint and switch models without re-entering the API key |
| `/language` | Owner or Role > Bot | Set the server default language (`en`, `vi`, `ja`, `ko`, `zh`, `auto`; default `en`) |
| `/status` | Everyone | View active provider, model, default language, and context limits (never displays API key or endpoint URL) |
| `/reset_config` | Owner or Role > Bot | Delete the server's stored API key and LLM configuration |
| `/help` | Everyone | Quick setup and usage guide |
| `/tldr [count] [since] [focus] [private]` | Everyone | Summarize recent messages publicly by default (or set `private:True`) |
| Right-click message > **Apps > TL;DR from here** | Everyone | Summarize everything from the selected message up to the present |

---

## Automated Tests

```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## Legal & Privacy

See [PRIVACY.md](PRIVACY.md) and [TERMS.md](TERMS.md).
