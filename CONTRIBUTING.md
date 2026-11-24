# Contributing Guide

This document defines how we plan work, branch, commit, open pull requests, and ship changes to `tldr-bot`. Keep changes small, focused, and easy to review.

> **Production deploy warning:** `main` auto-deploys to Railway on every push or merge. Never push untested code directly to `main`.

---

## 1. Workflow Overview

1. **Align first:** Open or claim a GitHub Issue before writing code for anything larger than a small typo or bug fix. Agree on scope early so work is not duplicated.
2. **Sync `main`:** Always branch from an up-to-date `main`.
3. **Work on a short-lived branch:** Keep each branch focused on one logical change (one feature, one bug fix, or one refactor).
4. **Verify locally:** Run bytecode compilation and the full `pytest` suite before pushing.
5. **Open a Pull Request:** Fill out the PR template, keep the diff focused, and wait for CI (Python 3.11 and 3.13) plus review approval.
6. **Squash and merge:** Use **Squash and merge** so `main` stays linear and easy to revert if needed, then delete the merged branch.

```bash
git checkout main
git pull origin main
git checkout -b feat/short-description
```

---

## 2. Branch Naming

Use lowercase `kebab-case` with a clear prefix:

| Prefix | Use for | Example |
| :--- | :--- | :--- |
| `feat/` | New user-facing or internal capability | `feat/thread-auto-detect` |
| `fix/` | Bug fixes | `fix/claude-stream-fallback` |
| `refactor/` | Internal cleanup without behavior change | `refactor/decouple-ui-lang` |
| `docs/` | Documentation and templates | `docs/contributing-guide` |
| `test/` | Test additions or fixture cleanup | `test/ssrf-ipv6-cases` |
| `chore/` | Dependencies, CI, Docker, or Railway config | `chore/bump-discord-py` |

Rules:
- Never commit directly to `main`.
- Rebase onto or merge `origin/main` before requesting final review if your branch has fallen behind.
- Delete branches after merge.

---

## 3. Commit Messages (Conventional Commits)

Follow [Conventional Commits](https://www.conventionalcommits.org/) in plain English:

```text
<type>(<optional-scope>): <imperative lowercase summary>
```

**Allowed types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

**Common scopes:** `llm`, `ux`, `ui`, `i18n`, `security`, `storage`, `discord`, `deploy`, `ci`

**Good examples:**
- `fix(llm): retry without response_format on empty proxy output`
- `feat(discord): add thread channel support for /tldr`
- `refactor(i18n): decouple UI strings from summary output language`
- `test(security): cover private IPv6 SSRF rejection`

**Avoid:**
- Vague messages (`update`, `fix stuff`, `wip`, `final`)
- Mixing unrelated refactors and feature changes in one commit
- Non-English commit messages or trailing punctuation

---

## 4. Pull Request Rules

### Before Opening a PR
- Keep PRs small (ideally under ~300 changed lines of code; split larger work into sequential PRs).
- Re-read your entire `git diff` yourself first and remove debug prints, unused imports, or stray files.
- Ensure all automated checks pass locally:

```bash
# Linux / macOS
python -m compileall -q src tests run.py
pytest -v
```

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m compileall -q src tests run.py
.\.venv\Scripts\pytest.exe -v
```

- Verify `git status` never includes `.env`, `.key`, `*.db*`, or local scratch files.

### Code Review Expectations
- **Author:** Explain *why* the change exists, list manual Discord checks if UI or LLM behavior changed, and attach screenshots for Discord embed or button changes.
- **Reviewer:** Check correctness, security invariants (see [CODING_STANDARDS.md](CODING_STANDARDS.md)), test coverage, and simplicity. Distinguish blocking issues from optional nits.
- **Merge rule:** At least one approval and green GitHub Actions CI (`CI / test (3.11)` and `CI / test (3.13)`).

---

## 5. Local Development Setup

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\pip.exe install -r requirements.txt
# Linux/macOS: .venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Generate a local master key for `VAULT_MASTER_KEY` in `.env`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

For safe testing in a private Discord server, set `DEV_GUILD_ID` in `.env` so slash commands sync instantly to that test guild instead of waiting for global sync.

---

## 6. Repository Map

| Path | Responsibility |
| :--- | :--- |
| `src/tldr_bot/main.py` | `AutoShardedBot` lifecycle, cog loading, logging setup, global error handler |
| `src/tldr_bot/settings.py` | Environment variable loading and runtime limits |
| `src/tldr_bot/i18n.py` | English UI string catalog (`t(key)`), LLM summary language specs, SQLite guild preferences |
| `src/tldr_bot/health.py` | `GET /healthz` readiness server for Railway and Docker |
| `src/tldr_bot/storage/` | SQLite WAL initialization and schema (`database.py`) |
| `src/tldr_bot/security/` | AES-256-GCM write-only vault (`vault.py`), SSRF guard (`ssrf.py`), role hierarchy (`permissions.py`), log redaction (`redactor.py`) |
| `src/tldr_bot/llm/` | Provider presets (`presets.py`), unified OpenAI-compatible client (`client.py`), token budgeting (`token_budget.py`), prompts (`prompts.py`), single-pass and map-reduce summarizer (`summarizer.py`) |
| `src/tldr_bot/discord_ext/` | Message fetcher (`fetcher.py`), preprocessor (`preprocessor.py`), RAM cache (`cache.py`), concurrency guard (`concurrency.py`), config modals/views (`ui_config.py`), embed renderer (`ui_summary.py`), slash command cogs (`cogs/`) |
| `tests/` | Offline unit tests covering preprocessor, token budget, summarizer, client fallbacks, vault, SSRF, permissions, redaction, and copy hygiene |

Read [CODING_STANDARDS.md](CODING_STANDARDS.md) before writing code and [SECURITY.md](SECURITY.md) when handling credentials or endpoints.
