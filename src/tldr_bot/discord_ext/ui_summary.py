"""Minimalist editorial Discord Embed renderer, safe Jump Links, and quiet single-action controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import discord

from tldr_bot.discord_ext.preprocessor import PreprocessedConversation
from tldr_bot.i18n import t
from tldr_bot.llm.summarizer import SummaryResult

COLOR_INK_SLATE = discord.Color.from_rgb(43, 48, 56)       # #2B3038
COLOR_TERRACOTTA = discord.Color.from_rgb(200, 109, 68)    # #C86D44
COLOR_SAGE = discord.Color.from_rgb(61, 122, 90)           # #3D7A5A
COLOR_AMBER = discord.Color.from_rgb(184, 130, 59)         # #B8823B

_FIELD_LIMIT = 1000
_EMBED_TOTAL_LIMIT = 5200


def _render_jump_links(
    ref_ids: tuple[int, ...],
    ref_to_url: dict[int, str],
    link_label: str,
    *,
    max_links: int = 2,
) -> str:
    valid_urls = [ref_to_url[rid] for rid in ref_ids if rid in ref_to_url and ref_to_url[rid]][
        :max_links
    ]
    if not valid_urls:
        return ""
    if len(valid_urls) == 1:
        return f" ([{link_label}]({valid_urls[0]}))"
    links = [f"[{link_label} {idx + 1}]({url})" for idx, url in enumerate(valid_urls)]
    return f" ({', '.join(links)})"


def _format_line_with_suffix(prefix: str, body: str, suffix: str, *, max_len: int = _FIELD_LIMIT) -> str:
    """Safely truncate body text BEFORE appending markdown jump links so URLs never break."""
    available = max_len - len(prefix) - len(suffix)
    if available < 40:
        trim_body = body[: max(20, max_len - len(prefix) - 3)].rstrip()
        return f"{prefix}{trim_body}..." if len(prefix) + len(body) > max_len else f"{prefix}{body}"
    if len(body) <= available:
        return f"{prefix}{body}{suffix}"
    clipped = body[: available - 3].rstrip()
    return f"{prefix}{clipped}...{suffix}"


def _status_tag(status: str, lang_code: str) -> str:
    if not status or status == "info":
        return ""
    label = t(lang_code, f"topic_status_{status}")
    return f" `[{label}]`" if label else ""


def _chunk_lines_for_fields(lines: list[str], max_len: int = _FIELD_LIMIT) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    cur_len = 0

    for line in lines:
        safe_line = line if len(line) <= max_len else f"{line[: max_len - 3]}..."
        added = len(safe_line) + (1 if current else 0)
        if current and cur_len + added > max_len:
            blocks.append("\n".join(current))
            current = [safe_line]
            cur_len = len(safe_line)
        else:
            current.append(safe_line)
            cur_len += added

    if current:
        blocks.append("\n".join(current))
    return blocks


def _format_participants_line(conv: PreprocessedConversation, lang_code: str) -> str:
    if not conv.participants:
        return ""
    top = conv.participants[:6]
    names = ", ".join(f"**{p.display_name}** ({p.message_count})" for p in top)
    if len(conv.participants) > 6:
        names += f", +{len(conv.participants) - 6}"
    return t(
        lang_code,
        "summary_participants",
        count=len(conv.participants),
        list=names,
    )


def _build_header_description(
    *,
    result: SummaryResult,
    conv: PreprocessedConversation,
    lang_code: str,
    hit_fetch_cap: bool,
) -> str:
    parts: list[str] = []
    if conv.time_range_label:
        parts.append(f"`{conv.time_range_label}`")

    participants_line = _format_participants_line(conv, lang_code)
    if participants_line:
        parts.append(participants_line)

    if hit_fetch_cap or result.was_token_capped:
        parts.append(
            t(
                lang_code,
                "summary_capped_note",
                used=result.kept_raw_messages or conv.kept_message_count,
                fetched=conv.raw_message_count,
            )
        )

    if result.overview:
        parts.append(f"\n{result.overview}")

    if result.raw_fallback_markdown:
        parts.append(f"\n{result.raw_fallback_markdown[:3200]}")

    return "\n".join(parts)[:3900]


def _collect_sections(
    *,
    result: SummaryResult,
    conv: PreprocessedConversation,
    lang_code: str,
    public_share: bool,
) -> list[tuple[str, str]]:
    jump_label = t(lang_code, "summary_jump_link")
    jump_full = t(lang_code, "summary_jump_full")
    sections: list[tuple[str, str]] = []

    def _add_section(header: str, lines: list[str]) -> None:
        if not lines:
            return
        for idx, chunk in enumerate(_chunk_lines_for_fields(lines)):
            name = header if idx == 0 else f"{header} ({idx + 1})"
            sections.append((name, chunk))

    topic_lines = [
        _format_line_with_suffix(
            f"- **{topic.title}**{_status_tag(topic.status, lang_code)}: ",
            topic.summary,
            _render_jump_links(topic.ref_ids, conv.ref_to_jump_url, jump_label, max_links=2),
        )
        for topic in result.topics
    ]
    _add_section(t(lang_code, "summary_topics_header"), topic_lines)

    decision_lines = [f"- {d}" for d in result.decisions]
    for act in result.action_items:
        jump = (
            _render_jump_links((act.ref_id,), conv.ref_to_jump_url, jump_label, max_links=1)
            if act.ref_id
            else ""
        )
        owner = f"**{act.owner}:** " if act.owner else ""
        decision_lines.append(_format_line_with_suffix(f"- {owner}", act.task, jump))
    _add_section(t(lang_code, "summary_decisions_header"), decision_lines)

    _add_section(
        t(lang_code, "summary_open_header"),
        [f"- {q}" for q in result.open_questions],
    )

    hl_lines: list[str] = []
    for hl in result.highlights:
        jump = (
            _render_jump_links((hl.ref_id,), conv.ref_to_jump_url, jump_label, max_links=1)
            if hl.ref_id
            else ""
        )
        by_part = f" (**{hl.shared_by}**)" if hl.shared_by else ""
        hl_lines.append(_format_line_with_suffix("- ", f"{hl.label}{by_part}", jump))
    _add_section(t(lang_code, "summary_highlights_header"), hl_lines)

    if not public_share and conv.requester_mentions:
        mention_lines = [
            _format_line_with_suffix(
                f"- **{hit.author_name}** `{hit.timestamp_hhmm}`: \"",
                hit.snippet,
                f"\" ([{jump_full}]({hit.jump_url}))" if hit.jump_url else "\"",
            )
            for hit in conv.requester_mentions[:5]
        ]
        _add_section(t(lang_code, "summary_mentions_header"), mention_lines)

    return sections


def build_summary_embeds(
    *,
    result: SummaryResult,
    conv: PreprocessedConversation,
    channel_name: str,
    lang_code: str,
    hit_fetch_cap: bool = False,
    from_cache: bool = False,
    public_share: bool = True,
    requester_display: str = "",
) -> list[discord.Embed]:
    """Render a SummaryResult into 1 or more clean editorial Discord Embed pages."""
    title = t(lang_code, "summary_title", channel=channel_name)[:250]
    header_desc = _build_header_description(
        result=result,
        conv=conv,
        lang_code=lang_code,
        hit_fetch_cap=hit_fetch_cap,
    )
    sections = _collect_sections(
        result=result,
        conv=conv,
        lang_code=lang_code,
        public_share=public_share,
    )

    has_personal_mentions = not public_share and bool(conv.requester_mentions)
    accent = COLOR_TERRACOTTA if has_personal_mentions else COLOR_INK_SLATE

    cache_badge = t(lang_code, "summary_cache_badge") if from_cache else ""
    non_break_lines = len([turn for turn in conv.turns if not turn.is_session_break])
    if public_share:
        footer_text = t(
            lang_code,
            "summary_public_footer",
            requester=requester_display or "Member",
            fetched=conv.raw_message_count,
            lines=non_break_lines,
            tokens=result.input_tokens_estimated,
            elapsed=result.elapsed_seconds,
            model=result.model_used,
            cache_badge=cache_badge,
        )
    else:
        footer_text = t(
            lang_code,
            "summary_footer",
            fetched=conv.raw_message_count,
            lines=non_break_lines,
            tokens=result.input_tokens_estimated,
            elapsed=result.elapsed_seconds,
            model=result.model_used,
            cache_badge=cache_badge,
        )

    pages: list[discord.Embed] = []
    current_embed = discord.Embed(title=title, description=header_desc, color=accent)
    current_size = len(title) + len(header_desc) + len(footer_text)
    field_count = 0

    for fname, fval in sections:
        added_size = len(fname) + len(fval)
        if (current_size + added_size > _EMBED_TOTAL_LIMIT or field_count >= 20) and field_count > 0:
            pages.append(current_embed)
            current_embed = discord.Embed(title=title, color=accent)
            current_size = len(title) + len(footer_text)
            field_count = 0
        current_embed.add_field(name=fname[:256], value=fval[:1024], inline=False)
        current_size += added_size
        field_count += 1

    pages.append(current_embed)

    total_pages = len(pages)
    for idx, emb in enumerate(pages):
        page_suffix = f" | {idx + 1}/{total_pages}" if total_pages > 1 else ""
        emb.set_footer(text=f"{footer_text}{page_suffix}"[:2048])

    return pages


@dataclass
class SummarySessionContext:
    """Holds already-fetched conversation data so Regenerate avoids refetching Discord history."""

    guild_id: int
    channel_name: str
    requester_id: int
    requester_display: str
    conv: PreprocessedConversation
    result: SummaryResult
    lang_code: str
    focus_query: str | None
    scope_tag: str
    latest_message_id: int
    hit_fetch_cap: bool
    from_cache: bool
    is_private: bool = False


ReRunCallback = Callable[[SummarySessionContext], Awaitable[SummaryResult]]


class SummaryActionView(discord.ui.View):
    """Minimal controls: pagination (only if multi-page) plus a single quiet Regenerate button."""

    def __init__(
        self,
        *,
        ctx: SummarySessionContext,
        pages: list[discord.Embed],
        rerun_callback: ReRunCallback,
        format_error_fn: Callable[[str, Exception], str],
    ) -> None:
        super().__init__(timeout=900)
        self._ctx = ctx
        self._pages = pages
        self._page_idx = 0
        self._rerun = rerun_callback
        self._format_error = format_error_fn
        self._rebuild_buttons()

    def _rebuild_buttons(self) -> None:
        self.clear_items()
        lang = self._ctx.lang_code

        if len(self._pages) > 1:
            prev_btn = discord.ui.Button(
                label=t(lang, "btn_prev"),
                style=discord.ButtonStyle.secondary,
                disabled=self._page_idx <= 0,
                row=0,
            )
            prev_btn.callback = self._on_prev
            next_btn = discord.ui.Button(
                label=t(lang, "btn_next"),
                style=discord.ButtonStyle.secondary,
                disabled=self._page_idx >= len(self._pages) - 1,
                row=0,
            )
            next_btn.callback = self._on_next
            self.add_item(prev_btn)
            self.add_item(next_btn)

        refresh_btn = discord.ui.Button(
            label=t(lang, "btn_refresh"),
            style=discord.ButtonStyle.secondary,
            row=0,
        )
        refresh_btn.callback = self._on_refresh
        self.add_item(refresh_btn)

    async def _on_prev(self, interaction: discord.Interaction) -> None:
        self._page_idx = max(0, self._page_idx - 1)
        self._rebuild_buttons()
        await interaction.response.edit_message(
            embed=self._pages[self._page_idx], view=self
        )

    async def _on_next(self, interaction: discord.Interaction) -> None:
        self._page_idx = min(len(self._pages) - 1, self._page_idx + 1)
        self._rebuild_buttons()
        await interaction.response.edit_message(
            embed=self._pages[self._page_idx], view=self
        )

    async def _on_refresh(self, interaction: discord.Interaction) -> None:
        await interaction.response.edit_message(
            content=t(self._ctx.lang_code, "btn_working"),
            embed=None,
            view=None,
        )
        try:
            new_result = await self._rerun(self._ctx)
        except Exception as exc:
            err_text = self._format_error(self._ctx.lang_code, exc)
            await interaction.edit_original_response(
                content=t(self._ctx.lang_code, "summary_failed", error=err_text),
                embed=None,
                view=self,
            )
            return

        self._ctx.result = new_result
        self._ctx.from_cache = False
        self._pages = build_summary_embeds(
            result=new_result,
            conv=self._ctx.conv,
            channel_name=self._ctx.channel_name,
            lang_code=self._ctx.lang_code,
            hit_fetch_cap=self._ctx.hit_fetch_cap,
            from_cache=False,
            public_share=not self._ctx.is_private,
            requester_display=self._ctx.requester_display,
        )
        self._page_idx = 0
        self._rebuild_buttons()
        await interaction.edit_original_response(
            content=None,
            embed=self._pages[0],
            view=self,
            allowed_mentions=discord.AllowedMentions.none(),
        )
