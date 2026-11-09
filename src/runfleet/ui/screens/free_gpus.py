"""The free GPUs dialog of the Fleet view (``g``, or a click on the fleet panel's free part): where the free GPUs of
one type are (a click on that type's row), or of every type (the header, ``also free``, ``g``).

One entry per cluster partition (with its nodes that have a free GPU of that type, most free first) or workstation
(with its free GPU indices), counted exactly like the panel (:func:`runfleet.ui.fleet_layout.free_spots`); hosts
whose numbers cannot be vouched for are named apart, never counted. ``enter`` closes the dialog and opens that host:
a cluster on that partition with free-GPU nodes only, a workstation on its first free GPU. It reads the store only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from runfleet.ui import keymap, render, theme
from runfleet.ui.fleet_layout import FreeSpot
from runfleet.ui.screens import dialog_list_room
from runfleet.util import humanize

BOX_W = 100                      # = #free-box width
LIST_ROWS = 24                   # = #free-list max-height (less on a short terminal, dialog_list_room)
HOST_W = 12
PART_W = 16


@dataclass(frozen=True, slots=True)
class FreeTarget:
    """What ``enter`` opens: a host, on a partition (cluster) or a GPU (workstation)."""
    host: str
    partition: str | None = None
    gpu: int | None = None


def spot_text(spot: FreeSpot, width: int, *, host_w: int, part_w: int, host_style: str = "") -> Text:
    """``k2          gpu-preempt      37 free / 76  (2 planned)`` and, under it, the nodes with a free GPU
    (``gpu-a01 4/8 · gpu-b02 2/4 · +3 more``) or the workstation's free GPUs (``GPU 0, 2``)."""
    head = Text()
    head.append(humanize.truncate(spot.host, host_w).ljust(host_w), style=f"bold {host_style}".strip())
    head.append("  ")
    head.append(humanize.truncate(spot.partition or "workstation", part_w).ljust(part_w), style=theme.muted())
    head.append("  ")
    head.append(f"{spot.free:>3} free", style=theme.ok() if spot.free else theme.dim())
    head.append(f" / {spot.total}", style=theme.muted())
    if spot.planned:
        head.append(f"  ({spot.planned} planned for pending jobs)", style=theme.dim())
    if not spot.free:
        return render.fit(head, width)
    indent = " " * (host_w + 2)
    second = Text(indent)
    if spot.partition is None:
        second.append("GPU " + ", ".join(map(str, spot.gpus)), style=theme.muted())
        return Text("\n").join([render.fit(head, width), render.fit(second, width)])
    second.append("nodes ", style=theme.dim())
    sep = Text(" · ", style=theme.dim())
    room = width - second.cell_len
    for i, (node, free, total) in enumerate(spot.nodes):
        item = Text(f"{node} {free}/{total}", style=theme.muted())
        rest = len(spot.nodes) - i - 1
        more = Text(f"+{rest} more", style=theme.dim()) if rest else Text("")
        need = (sep.cell_len if i else 0) + item.cell_len + (sep.cell_len + more.cell_len if rest else 0)
        if need > room and i:
            second.append_text(sep)
            second.append(f"+{rest + 1} more", style=theme.dim())
            break
        if i:
            second.append_text(sep)
        second.append_text(item)
        room -= (sep.cell_len if i else 0) + item.cell_len
    return Text("\n").join([render.fit(head, width), render.fit(second, width)])


def type_header(label: str, spots: Sequence[FreeSpot], *, mine: bool) -> Text:
    free = sum(s.free for s in spots)
    total = sum(s.total for s in spots)
    out = Text(label, style="bold")
    out.append(f"  {free} free of {total}", style=theme.ok() if free else theme.dim())
    if mine:
        out.append("  · a type you run on", style=theme.dim())
    return out


class FreeGpusScreen(ModalScreen[FreeTarget | None]):
    """Where the free GPUs are; dismisses with the host to open (None: closed).

    ``order``: the type keys to list, in the panel's order; ``only``: one type key (every spot of it, 0 free too), or
    None (every type, spots with a free GPU only). ``unknown``: hosts not counted."""

    BINDINGS = keymap.bindings("free_gpus")
    HELP_CONTEXT = "free_gpus"
    DEFAULT_CSS = f"""
    #free-box {{ width: {BOX_W}; }}
    #free-list, #free-list:focus {{
        border: none; background: transparent; padding: 0; height: auto; max-height: {LIST_ROWS};
    }}
    #free-note {{ height: auto; padding-top: 1; }}
    """

    def __init__(self, spots: Sequence[FreeSpot], order: Sequence[str], *, only: str | None = None,
                 mine: set[str] | None = None, unknown: Sequence[str] = (), sinfo_age_s: float | None = None,
                 host_order: Sequence[str] = (), host_colors: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.only = only
        self.mine = set(mine or ())
        self.unknown = tuple(unknown)
        self.sinfo_age_s = sinfo_age_s
        self.host_colors = dict(host_colors or {})
        rank = {h: i for i, h in enumerate(host_order)}
        keys = [only] if only is not None else list(order)
        by_key: dict[str, list[FreeSpot]] = {}
        for s in spots:
            if s.key in keys and (only is not None or s.free) and s.total:
                by_key.setdefault(s.key, []).append(s)
        self.groups: list[tuple[str, list[FreeSpot]]] = [
            (k, sorted(by_key[k], key=lambda s: (-s.free, rank.get(s.host, 99), s.host, s.partition or "")))
            for k in keys if k in by_key]
        self.targets: list[FreeTarget] = []

    def text_width(self) -> int:
        """Cells for one row: the box (at most 96 % of the terminal) less border, padding and the scrollbar."""
        term = self.app.size.width or 120
        return max(40, min(BOX_W, term * 96 // 100) - 4 - 2)

    def compose(self) -> ComposeResult:
        box = Vertical(id="free-box", classes="dialog")
        free = sum(s.free for _k, spots in self.groups for s in spots)
        if self.only is not None:
            label = self.groups[0][1][0].label if self.groups else self.only
            box.border_title = f"Free GPUs · {label} · {free} free"
        else:
            box.border_title = f"Free GPUs · {free} free across the fleet"
        box.border_subtitle = "↵ open host · esc close"
        with box:
            if self.groups:
                width = self.text_width()
                all_spots = [s for _k, spots in self.groups for s in spots]
                host_w = min(HOST_W, max((cell_len(s.host) for s in all_spots), default=4))
                part_w = min(PART_W, max((cell_len(s.partition or "workstation") for s in all_spots), default=4))
                options: list[Option] = []
                for key, spots in self.groups:
                    if self.only is None:
                        options.append(Option(type_header(spots[0].label, spots, mine=spots[0].label in self.mine
                                                          and not key.startswith("MIG")),
                                              id=f"h{len(options)}", disabled=True))
                    for s in spots:
                        options.append(Option(spot_text(s, width, host_w=host_w, part_w=part_w,
                                                         host_style=self.host_colors.get(s.host, "")),
                                              id=str(len(self.targets))))
                        self.targets.append(FreeTarget(s.host, s.partition, s.gpus[0] if s.gpus else None))
                yield OptionList(*options, id="free-list")
            else:
                yield Static(Text("no free GPU anywhere in the fleet right now" if self.only is None
                                  else "no GPU of this type is counted right now", style=theme.dim()))
            yield Static(self._note(), id="free-note")

    def _note(self) -> Text:
        out = Text("free = idle per sinfo / nvidia-smi, not what your QOS or association lets you start",
                   style=theme.dim())
        if self.sinfo_age_s is not None:
            age = self.sinfo_age_s
            out.append(f" · sinfo {'<1m' if age < 60 else f'{int(age // 60)}m'} ago", style=theme.dim())
        if self.unknown:
            out.append("\nnot counted (no fresh reading): ", style=theme.warn())
            out.append(", ".join(self.unknown), style=theme.warn())
        return out

    def on_mount(self) -> None:
        found = self.query(OptionList)
        if found:
            self._fit_list()
            menu = found.first()
            menu.focus()
            first = next((i for i in range(menu.option_count) if not menu.get_option_at_index(i).disabled), None)
            if first is not None:
                menu.highlighted = first

    def on_resize(self, _event: events.Resize) -> None:
        self._fit_list()

    def _fit_list(self) -> None:
        for menu in self.query(OptionList):
            menu.styles.max_height = dialog_list_room(self.app.size.height, LIST_ROWS, 3 if self.unknown else 2)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None and event.option.id.isdigit():
            self.dismiss(self.targets[int(event.option.id)])

    def action_cursor_down(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_down()

    def action_cursor_up(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_up()

    def action_close(self) -> None:
        self.dismiss(None)
