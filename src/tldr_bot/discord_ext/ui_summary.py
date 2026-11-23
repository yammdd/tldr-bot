"""Unslopped editorial Discord Embed renderer, Jump Link builder, pagination, and action controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import discord

from tldr_bot.discord_ext.preprocessor import PreprocessedConversation
from tldr_bot.i18n import t
from tldr_bot.llm.summarizer import SummaryResult

# Editorial palette: quiet dark slate neutral + one muted terracotta accent when mentions need attention
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
) -> str:
    links: list[str] = []
    for idx, rid in enumerate(ref_ids):
        url = ref_to_url.get(rid)
        if not url:
            continue
        label = link_label if len(ref_ids) == 1 else f"{link_label} {idx + 1}"
        links.append(f"[{label}]({url})")
    return f" ({', '.join(links)})" if links else ""


def _status_tag(status: str, lang_code: str) -> str:
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
    meta_bits: list[str] = []
    if conv.time_range_label:
        meta_bits.append(f"`{conv.time_range_label}`")
    if meta_bits:
        parts.append(" | ".join(meta_bits))

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
    style_id: str,
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
        f"- **{topic.title}**{_status_tag(topic.status, lang_code)}: "
        f"{topic.summary}{_render_jump_links(topic.ref_ids, conv.ref_to_jump_url, jump_label)}"
        for topic in result.topics
    ]
    person_lines = [
        f"- **{person.speaker}:** {person.summary}"
        f"{_render_jump_links(person.ref_ids, conv.ref_to_jump_url, jump_label)}"
        for person in result.by_person
    ]

    if style_id == "by_person" and person_lines:
        _add_section(t(lang_code, "summary_by_person_header"), person_lines)
        _add_section(t(lang_code, "summary_topics_header"), topic_lines)
    else:
        _add_section(t(lang_code, "summary_topics_header"), topic_lines)
        if not topic_lines and person_lines:
            _add_section(t(lang_code, "summary_by_person_header"), person_lines)

    decision_lines = [f"- {d}" for d in result.decisions]
    for act in result.action_items:
        jump = (
            _render_jump_links((act.ref_id,), conv.ref_to_jump_url, jump_label)
            if act.ref_id
            else ""
        )
        owner = f"**{act.owner}:** " if act.owner else ""
        decision_lines.append(f"- {owner}{act.task}{jump}")
    _add_section(t(lang_code, "summary_decisions_header"), decision_lines)

    _add_section(
        t(lang_code, "summary_open_header"),
        [f"- {q}" for q in result.open_questions],
    )

    hl_lines: list[str] = []
    for hl in result.highlights:
        jump = (
            _render_jump_links((hl.ref_id,), conv.ref_to_jump_url, jump_label)
            if hl.ref_id
            else ""
        )
        by_part = f" (**{hl.shared_by}**)" if hl.shared_by else ""
        hl_lines.append(f"- {hl.label}{by_part}{jump}")
    _add_section(t(lang_code, "summary_highlights_header"), hl_lines)

    # Personal Radar: shown strictly in private view, never broadcast on public share
    if not public_share and conv.requester_mentions:
        mention_lines = [
            f"- **{hit.author_name}** `{hit.timestamp_hhmm}`: \"{hit.snippet}\""
            + (f" ([{jump_full}]({hit.jump_url}))" if hit.jump_url else "")
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
    style_id: str,
    hit_fetch_cap: bool = False,
    from_cache: bool = False,
    public_share: bool = False,
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
        style_id=style_id,
        public_share=public_share,
    )

    # Semantic accent restraint: use muted terracotta ONLY when the user was directly mentioned
    has_personal_mentions = not public_share and bool(conv.requester_mentions)
    accent = COLOR_TERRACOTTA if has_personal_mentions else COLOR_INK_SLATE

    cache_badge = t(lang_code, "summary_cache_badge") if from_cache else ""
    if public_share:
        footer_text = t(
            lang_code,
            "summary_public_footer",
            requester=requester_display or "Member",
            fetched=conv.raw_message_count,
            time_range=conv.time_range_label,
            model=result.model_used,
        )
    else:
        footer_text = t(
            lang_code,
            "summary_footer",
            fetched=conv.raw_message_count,
            lines=len([turn for turn in conv.turns if not turn.is_session_break]),
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
    """Holds already-fetched conversation data so action buttons avoid refetching Discord history."""

    guild_id: int
    channel_name: str
    requester_id: int
    requester_display: str
    conv: PreprocessedConversation
    result: SummaryResult
    lang_code: str
    style_id: str
    focus_query: str | None
    scope_tag: str
    latest_message_id: int
    hit_fetch_cap: bool
    from_cache: bool


ReRunCallback = Callable[
    [SummarySessionContext, str, bool], Awaitable[tuple[SummaryResult, bool]]
]


class SummaryActionView(discord.ui.View):
    """Quiet interactive controls under an ephemeral TL;DR summary."""

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
        self._shared = False
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

        share_btn = discord.ui.Button(
            label=t(lang, "btn_share"),
            style=discord.ButtonStyle.primary,
            disabled=self._shared,
            row=1,
        )
        share_btn.callback = self._on_share
        self.add_item(share_btn)

        if self._ctx.style_id != "by_person":
            person_btn = discord.ui.Button(
                label=t(lang, "btn_by_person"),
                style=discord.ButtonStyle.secondary,
                row=1,
            )
            person_btn.callback = self._on_by_person
            self.add_item(person_btn)

        if self._ctx.style_id != "detailed":
            detail_btn = discord.ui.Button(
                label=t(lang, "btn_detailed"),
                style=discord.ButtonStyle.secondary,
                row=1,
            )
            detail_btn.callback = self._on_detailed
            self.add_item(detail_btn)

        refresh_btn = discord.ui.Button(
            label=t(lang, "btn_refresh"),
            style=discord.ButtonStyle.secondary,
            row=1,
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

    async def _on_share(self, interaction: discord.Interaction) -> None:
        channel = interaction.channel
        if channel is None or not hasattr(channel, "send"):
            await interaction.response.send_message(
                t(self._ctx.lang_code, "perm_no_channel_history"), ephemeral=True
            )
            return

        public_pages = build_summary_embeds(
            result=self._ctx.result,
            conv=self._ctx.conv,
            channel_name=self._ctx.channel_name,
            lang_code=self._ctx.lang_code,
            style_id=self._ctx.style_id,
            hit_fetch_cap=self._ctx.hit_fetch_cap,
            from_cache=False,
            public_share=True,
            requester_display=self._ctx.requester_display,
        )
        try:
            await channel.send(
                embeds=public_pages[:3],
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.Forbidden:
            await interaction.response.send_message(
                t(self._ctx.lang_code, "perm_no_channel_history"), ephemeral=True
            )
            return

        self._shared = True
        self._rebuild_buttons()
        await interaction.response.edit_message(
            embed=self._pages[self._page_idx], view=self
        )
        await interaction.followup.send(
            t(self._ctx.lang_code, "btn_shared_ok"), ephemeral=True
        )

    async def _run_variant(
        self, interaction: discord.Interaction, new_style: str, bypass_cache: bool
    ) -> None:
        await interaction.response.edit_message(
            content=t(self._ctx.lang_code, "btn_working"),
            embed=None,
            view=None,
        )
        try:
            new_result, from_cache = await self._rerun(self._ctx, new_style, bypass_cache)
        except Exception as exc:
            err_text = self._format_error(self._ctx.lang_code, exc)
            await interaction.edit_original_response(
                content=t(self._ctx.lang_code, "summary_failed", error=err_text),
                embed=None,
                view=self,
            )
            return

        self._ctx.result = new_result
        self._ctx.style_id = new_style
        self._ctx.from_cache = from_cache
        self._shared = False
        self._pages = build_summary_embeds(
            result=new_result,
            conv=self._ctx.conv,
            channel_name=self._ctx.channel_name,
            lang_code=self._ctx.lang_code,
            style_id=new_style,
            hit_fetch_cap=self._ctx.hit_fetch_cap,
            from_cache=from_cache,
            public_share=False,
            requester_display=self._ctx.requester_display,
        )
        self._page_idx = 0
        self._rebuild_buttons()
        await interaction.edit_original_response(
            content=None,
            embed=self._pages[0],
            view=self,
        )

    async def _on_by_person(self, interaction: discord.Interaction) -> None:
        await self._run_variant(interaction, "by_person", bypass_cache=False)

    async def _on_detailed(self, interaction: discord.Interaction) -> None:
        await self._run_variant(interaction, "detailed", bypass_cache=False)

    async def _on_refresh(self, interaction: discord.Interaction) -> None:
        await self._run_variant(interaction, self._ctx.style_id, bypass_cache=True)
