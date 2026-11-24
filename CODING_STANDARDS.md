# Coding Standards & Clean Code Rules

These standards keep `tldr-bot` fast, predictable, secure, and easy for multiple contributors to maintain without accumulating bloat.

---

## 1. Core Principles

- **Simplicity over features:** Every new command, button, option, or config flag adds long-term maintenance cost. Prefer refining existing behavior over adding knobs.
- **Strict module boundaries:** Keep business logic out of Discord cogs so it stays unit-testable offline.
  - `security/` must not import `discord` (except lightweight type hints or `permissions.py` duck-typed guild/member attributes).
  - `llm/` must not import `discord`.
  - `storage/` only owns SQLite connection setup and table DDL.
  - `discord_ext/cogs/` should orchestrate validation, service calls, and responses; heavy logic belongs in `preprocessor.py`, `summarizer.py`, `ui_summary.py`, or `ui_config.py`.
- **Delete dead code:** Do not leave commented-out blocks, unused imports, unused parameters, or backward-compatibility shims for internal functions.

---

## 2. Python Style & Typing

- Target **Python 3.11+** compatibility (CI tests both 3.11 and 3.13).
- Start every module with `from __future__ import annotations`.
- Type-annotate all function signatures, method signatures, and non-trivial attributes using modern built-in generics (`list[str]`, `dict[str, Any]`, `str | None`).
- Use `@dataclass(frozen=True)` for immutable value objects and config records (see `GuildLLMConfig`, `LanguageSpec`, `SummaryResult`).
- Keep functions focused and shallow: prefer early returns over deeply nested `if/else` trees.
- Never swallow exceptions silently with bare `except:` or `except Exception: pass` unless handling benign Discord UI race conditions (such as `discord.HTTPException` on progress edits), and always document why when ignoring an error.
- Keep logging structured and concise; use `logger = logging.getLogger(__name__)` with `%s` formatting (not f-strings inside `logger.*` calls).

---

## 3. Async & Discord.py Rules

- **Never block the event loop:** Do not run synchronous network I/O, `time.sleep`, or heavy blocking operations inside `async def` handlers.
- **Defer early:** Any command that fetches Discord history or calls an LLM must call `await interaction.response.defer(...)` within Discord's 3-second window.
- **Prevent accidental pings:** Always pass `allowed_mentions=discord.AllowedMentions.none()` on public bot messages and sanitize `@everyone` / `@here` in model output.
- **Visibility defaults:**
  - Public by default: `/tldr`, `TL;DR from here`, `/help`, `/status`, `/language` (with optional `private:True` on `/tldr`).
  - Always ephemeral (`ephemeral=True`): `/config`, `/model`, `/reset_config`, permission errors, validation errors, and any response that references credentials or custom endpoint details.
- **Permission checks:** Every admin command or interactive component callback (select menus, modals, buttons that mutate config) must re-check `check_role_above_bot(interaction.user, interaction.guild)` at execution time, not just when rendering the panel.
- **Respect Discord limits:** Truncate embed titles (256 chars), field values (1024 chars), embed totals (under 6000 chars), modal titles (45 chars), and select options (25 per menu). Never truncate inside a markdown `[label](url)` link; truncate body text before appending links.

---

## 4. Security & Privacy Invariants (Non-Negotiable)

Any PR that weakens these invariants will be rejected:

1. **Write-only API keys:** Never add a command, button, log line, or error message that displays or exports a stored API key.
2. **No endpoint leaks in public views:** `/status` must never display the custom endpoint URL or host.
3. **Mandatory live smoke test before saving:** `/config` and `/model` must verify credentials against the target endpoint before overwriting saved state. Failed checks must preserve the existing working config.
4. **SSRF protection on every endpoint:** All user-supplied base URLs must pass through `security/ssrf.py` (`https://` enforcement and DNS resolution check blocking loopback, private, and link-local IPs).
5. **Per-guild cryptographic isolation:** Encrypted keys in SQLite use AES-256-GCM with a unique 12-byte nonce and `guild_id` bound as AAD. Never bypass `KeyVault`.
6. **Zero chat persistence:** Channel messages exist in memory only during summarization and are never written to SQLite, files, or logs.
7. **Secret redaction:** Pass any upstream error detail through `security/redactor.py` before showing it to users or logging it.
8. **No secrets in git:** Never commit `.env`, `.key`, `*.db*`, real API keys, or real Discord tokens.

---

## 5. UI, Language, and Copy Discipline

- **100% English repo and bot UI:** All code, comments, commit messages, documentation, tests, slash command descriptions, embed headers, buttons, and error strings live in plain English.
- **Strict separation of UI and `/language`:**
  - All bot UI strings come from `UI_STRINGS` via `t(key, **kwargs)` in `src/tldr_bot/i18n.py`.
  - `/language` sets the **LLM summary output language only** (passed into prompt instructions in `src/tldr_bot/llm/prompts.py`). Do not wire `prefs.get_language()` back into UI components.
- **Minimalist copy hygiene (enforced in tests):**
  - No exclamation marks (`!`) in `UI_STRINGS`.
  - No em-dashes (`\u2014`) or en-dashes (`\u2013`) in UI copy or docs.
  - No decorative emojis in bot UI labels, buttons, or headings.
  - No references to unrelated third-party AI products in copy or docs.
- **Quiet UI surfaces:** Keep interactive controls minimal (currently only `Regenerate`, plus `Prev`/`Next` when a summary spans multiple pages). Do not add redundant buttons or tone dropdowns.

---

## 6. Testing Standards

- Every bug fix must include a regression test that would have failed before the fix.
- Every new feature or edge case must include unit tests under `tests/`.
- All tests must run **100% offline** with no live Discord or LLM network calls; use `unittest.mock.AsyncMock` / `MagicMock` and `tmp_path` for SQLite tests.
- Keep `test_guild_language_preferences_sqlite_and_copy_hygiene` green and extend it when adding new global copy or security invariants.

---

## 7. Deployment & Infrastructure Rules

- **Railway Dockerfile rules:** Do not add `VOLUME` or `HEALTHCHECK` instructions to `Dockerfile` (Railway's build validator rejects them). Health probing belongs in `railway.toml` and `docker-compose.yml`, served by `src/tldr_bot/health.py` on `HEALTH_PORT` (with automatic `PORT` fallback).
- **Volume permissions:** Keep `docker-entrypoint.sh` responsible for fixing `/app/data` ownership and dropping privileges to the non-root user (`botuser`, UID `10001`).
- **Database migrations:** If you change SQLite tables in `src/tldr_bot/storage/database.py`, make schema updates idempotent (`CREATE TABLE IF NOT EXISTS` or guarded `ALTER TABLE`) so existing persistent Railway volumes upgrade cleanly without data loss.
