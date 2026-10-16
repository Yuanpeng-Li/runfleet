"""Saved Jobs views (§7.6): named filter + state preset + sort + grouping, kept in ``ui.json`` under ``views``.

Pure helpers: name rules (1-24 characters: ASCII letters, digits, space, ``.``, ``_``, ``-``, starting with a letter
or digit, no space at the end; names are unique ignoring case), at most :data:`MAX_VIEWS` views, and a tolerant
reader. An entry with a bad name or a bad value (unknown preset / sort / grouping, a non-string filter, a non-bool
reverse) is ignored, never an error; a missing field takes the Jobs view default. Writes keep the entries this version
cannot read (a newer RunFleet may have written them) and only add, replace or remove the one entry asked for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from runfleet.ui.query import PRESETS, SORT_LABELS, Preset, SortColumn, sort_arrow

NAME_MAX = 24
MAX_VIEWS = 20
GROUPS: tuple[str, ...] = ("none", "host", "array")       # Jobs view grouping (`z`), single source
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


@dataclass(frozen=True, slots=True)
class SavedView:
    filter: str = ""
    preset: Preset = "fresh"
    sort: SortColumn = "host"
    reverse: bool = False
    group: str = "none"

    def to_json(self) -> dict[str, Any]:
        return {"filter": self.filter, "preset": self.preset, "sort": self.sort, "reverse": self.reverse,
                "group": self.group}


def name_problem(name: str) -> str | None:
    """Why ``name`` (already stripped) cannot name a view, None when it can."""
    if not name:
        return "type a name"
    if len(name) > NAME_MAX:
        return f"at most {NAME_MAX} characters"
    if not _NAME.fullmatch(name):
        return "letters, digits, space, . _ - only, starting with a letter or digit"
    if "  " in name:
        return "no double spaces"
    if name.endswith(" "):              # the prompt strips; a hand-written "gpu " would sit next to "gpu" unseen
        return "no space at the end"
    return None


def parse_view(raw: Any) -> SavedView | None:
    """One stored entry → SavedView, None when any present field is unusable."""
    if not isinstance(raw, Mapping):
        return None
    flt = raw.get("filter", "")
    preset = raw.get("preset", "fresh")
    sort = raw.get("sort", "host")
    reverse = raw.get("reverse", False)
    group = raw.get("group", "none")
    if not isinstance(flt, str) or "\n" in flt or preset not in PRESETS or sort not in SORT_LABELS \
            or not isinstance(reverse, bool) or group not in GROUPS:
        return None
    return SavedView(flt, cast(Preset, preset), cast(SortColumn, sort), reverse, group)


def load_views(ui_state: Mapping[str, Any]) -> dict[str, SavedView]:
    """The usable views of ``ui_state["views"]``, sorted by name (ignoring case), at most :data:`MAX_VIEWS`."""
    raw = ui_state.get("views") if isinstance(ui_state, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, SavedView] = {}
    seen: set[str] = set()
    for name in sorted((n for n in raw if isinstance(n, str)), key=lambda n: (n.casefold(), n)):
        view = parse_view(raw[name])
        if view is None or name_problem(name) is not None or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        out[name] = view
        if len(out) >= MAX_VIEWS:
            break
    return out


def find(views: Mapping[str, SavedView], name: str) -> str | None:
    """The stored spelling of ``name`` (names are unique ignoring case)."""
    folded = name.casefold()
    return next((n for n in views if n.casefold() == folded), None)


def save_problem(views: Mapping[str, SavedView], name: str) -> str | None:
    """Why ``name`` cannot be saved now: a bad name, or :data:`MAX_VIEWS` views and ``name`` is not one of them."""
    problem = name_problem(name)
    if problem is None and find(views, name) is None and len(views) >= MAX_VIEWS:
        problem = f"{MAX_VIEWS} views saved already — delete one first (palette › Delete view: …)"
    return problem


def with_view(ui_state: Mapping[str, Any], name: str, view: SavedView) -> dict[str, Any]:
    """The new ``views`` value: ``name`` stored (replacing a view of the same name in any case), the rest kept."""
    raw = ui_state.get("views") if isinstance(ui_state, Mapping) else None
    out = {k: v for k, v in raw.items() if not (isinstance(k, str) and k.casefold() == name.casefold())} \
        if isinstance(raw, Mapping) else {}
    out[name] = view.to_json()
    return out


def without_view(ui_state: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The new ``views`` value with ``name`` (any case) removed."""
    raw = ui_state.get("views") if isinstance(ui_state, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    return {k: v for k, v in raw.items() if not (isinstance(k, str) and k.casefold() == name.casefold())}


def describe(view: SavedView) -> str:
    """One line for the palette and the save prompt: ``failed · sort submit ▼ · group host · /st:f``."""
    parts = [view.preset, f"sort {SORT_LABELS[view.sort]} {sort_arrow(view.sort, view.reverse)}"]
    if view.group != "none":
        parts.append(f"group {view.group}")
    if view.filter:
        parts.append(f"/{view.filter}")
    return " · ".join(parts)
