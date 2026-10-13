"""[WP3] Single source of key bindings and help text (§7.8).

Every key the UI reacts to is declared here, per context. Screens and widgets take their Textual ``BINDINGS`` from
:func:`bindings`; the help overlay (:func:`help_entries`), the hint bar (:func:`hints`) and ``docs/USER_GUIDE.md``
(:func:`markdown`) are generated from the same table, so documentation cannot drift from behaviour.

Actions are Textual action strings resolved on the node that owns the context (``app.`` / ``screen.`` prefixes
where the handler lives elsewhere). Rules: single keys only, no prefix chords.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from textual.binding import Binding, BindingType

Group = Literal["nav", "act", "view"]

GROUP_TITLES: dict[Group, str] = {"nav": "NAVIGATION", "act": "SELECTION & ACTIONS", "view": "VIEW"}


@dataclass(frozen=True, slots=True)
class Key:
    keys: tuple[str, ...]          # Textual key names; the first one is shown in the hint bar
    action: str                    # Textual action string
    description: str               # help overlay / user guide text
    group: Group = "view"
    display: str | None = None     # key column text in help ("j/k ↑/↓"); default derived from keys
    hint: str | None = None        # short hint-bar label; None = not in the hint bar
    mutating: bool = False         # hidden from the hint bar in read-only mode (§6.1)
    priority: bool = False         # Textual priority binding (ctrl+c / ctrl+z)
    bind: bool = True              # False: documented only (handled by a built-in widget, e.g. DataTable arrows)
    in_help: bool = True           # False: summarised by the previous entry's display ("k" under "j/k")
    hint_key: str | None = None    # hint-bar key text when the hint covers a pair ("[/]", "g/G", "o/e")
    summary: str | None = None     # help text when the entry summarises a pair ("all hosts / host N")

    @property
    def label(self) -> str:
        return self.display or "/".join(display_key(k) for k in self.keys)


_KEY_NAMES = {
    "question_mark": "?", "colon": ":", "slash": "/", "plus": "+", "minus": "-", "left_square_bracket": "[",
    "right_square_bracket": "]", "space": "space", "enter": "↵", "escape": "esc", "up": "↑", "down": "↓",
    "left": "←", "right": "→", "pageup": "pgup", "pagedown": "pgdn", "shift+tab": "S-tab", "comma": ",",
    "shift+up": "S-↑", "shift+down": "S-↓",
}


def display_key(key: str) -> str:
    """Textual key name -> what users read (``ctrl+k`` -> ``^k``, ``question_mark`` -> ``?``)."""
    if key in _KEY_NAMES:
        return _KEY_NAMES[key]
    if key.startswith("ctrl+"):
        return "^" + key[5:]
    return key


_DIGITS = tuple(str(i) for i in range(1, 10))
COPY_MENU = ("copy menu: o stdout · e stderr path · i id · h host:id · w workdir · s shell command · b support bundle "
             "· Y again = stdout path")

KEYMAP: dict[str, tuple[Key, ...]] = {
    "global": (
        Key(("question_mark", "f1"), "app.help", "help (this view)", "view", "?/F1", hint="help"),
        Key(("ctrl+p", "colon"), "app.command_palette", "command palette", "view", "^p :", hint="cmd"),
        Key(("comma",), "app.settings", "settings (hosts, defaults)", "view", ","),
        Key(("q",), "app.back_or_quit", "back · quit on the main screen", "nav", hint="quit"),
        Key(("ctrl+c",), "app.quit", "quit now", "nav", priority=True),
        Key(("escape",), "app.escape", "close / clear filter, marks / back", "nav", "esc"),
        Key(("0",), "app.scope_all", "all hosts (1-9: host N)", "nav", "0 / 1-9", summary="all hosts / host N"),
        *(Key((d,), f"app.scope_index({d})", f"host {d}", "nav") for d in _DIGITS),
        Key(("slash",), "app.filter", "filter jobs", "view", hint="filter"),
        Key(("r",), "app.refresh_scope", "refresh hosts in scope (>= 3 s apart)", "view", "r ^r",
            summary="refresh scope / all"),
        Key(("ctrl+r",), "app.refresh_all", "refresh all hosts", "view", in_help=False),
        Key(("plus",), "app.rate(0.5)", "poll faster (down to 0.5×)", "view", "+/-",
            summary="poll faster / slower"),
        Key(("minus",), "app.rate(2.0)", "poll slower (up to 4×)", "view", in_help=False),
        Key(("ctrl+z",), "app.suspend_process", "suspend to the shell", "nav", priority=True),
        Key(("D",), "app.digest", "since you were away: what ended, failures first", "view"),
    ),
    "main": (
        Key(("tab",), "screen.next_view", "next view (Fleet→Jobs→GPUs)", "nav", hint="view"),
        Key(("shift+tab",), "screen.prev_view", "previous view", "nav", in_help=False),
    ),
    "fleet": (
        Key(("up", "k"), "move('up')", "host above", "nav", "arrows hjkl", summary="move between hosts"),
        Key(("down", "j"), "move('down')", "host below", "nav", in_help=False),
        Key(("left", "h"), "move('left')", "previous host", "nav", in_help=False),
        Key(("right", "l"), "move('right')", "next host", "nav", in_help=False),
        Key(("enter",), "open_host", "host detail", "nav", "↵ / double-click", hint="host detail"),
        Key(("space",), "scope_host", "scope to this host", "act", "space", hint="scope"),
        Key(("c",), "app.connect_host", "connect (log in: Duo / password prompts)", "act", hint="connect"),
        Key(("C",), "app.login_all", "log in to every host needing auth (one terminal session)", "act"),
        Key(("s",), "app.shell_host", "shell on the host", "act", hint="shell"),
        Key(("t",), "app.toggle_host", "enable / disable polling", "act", hint="on/off"),
        Key(("e",), "app.host_error", "error details", "act", hint="info"),
    ),
    "jobs": (
        Key(("j",), "cursor_down", "move down", "nav", "j/k ↑/↓", summary="move"),
        Key(("k",), "cursor_up", "move up", "nav", in_help=False),
        Key(("g", "home"), "first", "first row", "nav", "g/G", summary="top / bottom"),
        Key(("G", "end"), "last", "last row", "nav", in_help=False),
        Key(("ctrl+d", "pagedown"), "half_page(1)", "half page down", "nav", "^d/^u", summary="half page"),
        Key(("ctrl+u", "pageup"), "half_page(-1)", "half page up", "nav", in_help=False),
        Key(("enter",), "open", "job detail", "nav", "↵", hint="detail"),
        Key(("l",), "app.log_viewer", "log viewer", "nav", hint="logs"),
        Key(("p",), "app.pager", "log in $PAGER", "nav"),
        Key(("x",), "app.action_menu", "action menu", "act", hint="actions"),
        Key(("ctrl+k",), "app.quick_kill", "cancel job / terminate task", "act", hint="cancel",
            mutating=True),
        Key(("space",), "mark", "mark / unmark", "act", hint="mark"),
        Key(("V",), "mark_all", "mark all / clear marks", "act"),
        Key(("A",), "mark_array", "mark the whole array of this row (cancel / hold act on all of it)", "act"),
        Key(("y",), "app.copy_id", "copy id", "act", "y / Y", summary="copy id / menu (Y Y: stdout)"),
        Key(("Y",), "app.copy_menu", COPY_MENU, "act", in_help=False),
        Key(("s",), "app.shell_job", "shell", "act"),
        Key(("P",), "app.forward_port", "forward a port (TensorBoard, Jupyter …) to localhost", "act"),
        Key(("c",), "app.connect_host", "connect host", "act"),
        Key(("slash",), "app.filter", "filter jobs", "view", "/", hint="filter"),
        Key(("f",), "next_preset", "next state preset", "view", hint="state"),
        Key(("o",), "next_sort", "next sort column", "view", hint="sort"),
        Key(("O",), "reverse_sort", "reverse sort", "view"),
        Key(("z",), "next_group", "group by", "view", hint="group"),
        Key(("u",), "toggle_users", "other users", "view"),
        Key(("w",), "app.toggle_watch", "watch / unwatch (notify on start, end, GPU idle; marked rows too)", "act"),
    ),
    "gpus": (
        Key(("j",), "cursor_down", "move down", "nav", "j/k ↑/↓", summary="move"),
        Key(("k",), "cursor_up", "move up", "nav", in_help=False),
        Key(("g", "home"), "first", "first row", "nav", "g/G", summary="top / bottom"),
        Key(("G", "end"), "last", "last row", "nav", in_help=False),
        Key(("enter",), "open", "owning job / task (or host)", "nav", "↵", hint="open"),
        Key(("x",), "app.action_menu", "actions of the owning task", "act", hint="actions"),
        Key(("ctrl+k",), "app.quick_kill", "terminate the owning task", "act", hint="terminate", mutating=True),
        Key(("c",), "app.connect_host", "connect host", "act"),
        Key(("a",), "app.free_gpus", "copy CUDA_VISIBLE_DEVICES for N free GPUs of this workstation", "act"),
        Key(("o",), "next_sort", "next sort column", "view", hint="sort"),
        Key(("O",), "reverse_sort", "reverse sort", "view"),
        Key(("slash",), "app.filter", "filter jobs (Jobs view)", "view", "/", hint="filter"),
    ),
    "job_detail": (
        Key(("left_square_bracket",), "prev_tab", "previous tab", "nav", "[ ]", hint="tab",
            summary="previous / next tab", hint_key="[/]"),
        Key(("right_square_bracket",), "next_tab", "next tab", "nav", in_help=False),
        Key(("o",), "stream('stdout')", "stdout", "view", "o/e", hint="stream", summary="stdout / stderr",
            hint_key="o/e"),
        Key(("e",), "stream('stderr')", "stderr", "view", in_help=False),
        Key(("F",), "toggle_follow", "follow on / off", "view", hint="follow"),
        Key(("l",), "app.log_viewer", "full log viewer", "nav", hint="full log"),
        Key(("L",), "open_cause", "log viewer at the failure line (cause chip)", "nav"),
        Key(("slash",), "search_log", "search the log (opens the log viewer)", "nav", "/", hint="search"),
        Key(("p",), "app.pager", "log in $PAGER", "nav", hint="pager"),
        Key(("r",), "reload", "refresh detail", "view"),
        Key(("x",), "app.action_menu", "action menu", "act", hint="actions"),
        Key(("ctrl+k",), "app.quick_kill", "cancel / terminate", "act", hint="cancel", mutating=True),
        Key(("s",), "app.shell_job", "shell", "act", hint="shell"),
        Key(("c",), "app.connect_host", "connect host", "act"),
        Key(("y",), "app.copy_id", "copy id", "act", "y / Y", hint="copy id", summary="copy id / menu (Y Y: stdout)"),
        Key(("Y",), "app.copy_menu", COPY_MENU, "act", in_help=False),
        Key(("P",), "app.forward_port", "forward a port (TensorBoard, Jupyter …) to localhost", "act"),
        Key(("w",), "app.toggle_watch", "watch / unwatch (notify on start, end, GPU idle)", "act"),
        Key(("a",), "attach_auto", "live readings: auto-refresh on / off (GPU / Procs tab)", "view"),
        Key(("S",), "attach_srun", "live readings: read this job's own GPUs with srun (asks first; one permanent "
            "sacct row)", "act"),
    ),
    # the GPU / Procs tab of a running Slurm job (Attach, §7.16): listed first there, so `r` reads the node now;
    # the bindings themselves live in "job_detail" (one Textual binding per key)
    "job_live": (
        Key(("r",), "reload", "read the node now (srun asks first) and refresh the detail", "view", hint="read now",
            bind=False),
        Key(("a",), "attach_auto", "auto-refresh on / off (ssh: every attach_refresh while the tab is open)", "view",
            hint="auto", bind=False),
        Key(("S",), "attach_srun", "read this job's own GPUs with srun when ssh landed in another job (asks first)",
            "act", bind=False),
    ),
    "host_detail": (
        Key(("ctrl+k",), "app.quick_kill", "terminate the selected task", "act", hint="terminate", mutating=True),
        Key(("x",), "app.action_menu", "actions", "act", hint="actions"),
        Key(("s",), "app.shell_job", "shell", "act", hint="shell"),
        Key(("c",), "app.connect_host", "connect", "act"),
        Key(("t",), "app.toggle_host", "enable / disable polling", "act"),
        Key(("e",), "app.host_error", "error details", "act"),
    ),
    "host_ws": (
        Key(("enter",), "open", "open the process' task", "nav", "↵", hint="task detail"),
        Key(("l",), "app.log_viewer", "log of the selected task", "nav", hint="log"),
        Key(("u",), "toggle_users", "other users' processes", "view", hint="others"),
        Key(("left_square_bracket",), "partition(-1)", "previous GPU: only its processes, all users (click a GPU too)",
            "view", "[ ]", hint="GPU", summary="previous / next GPU (then all processes again)", hint_key="[/]",
            bind=False),
        Key(("right_square_bracket",), "partition(1)", "next GPU", "view", in_help=False, bind=False),
        Key(("a",), "app.free_gpus", "copy CUDA_VISIBLE_DEVICES for N free GPUs", "act"),
        Key(("o",), "next_sort", "next sort column", "view", hint="sort"),
        Key(("O",), "reverse_sort", "reverse sort", "view"),
    ),
    "host_cluster": (
        Key(("enter",), "open", "my job on the selected node", "nav", "↵", hint="job"),
        Key(("f",), "toggle_free", "nodes with free GPUs only", "view", hint="free GPUs"),
        Key(("left_square_bracket",), "partition(-1)", "previous partition", "nav", "[ ]", hint="partition",
            summary="previous / next partition", hint_key="[/]"),
        Key(("right_square_bracket",), "partition(1)", "next partition", "nav", in_help=False),
    ),
    "log": (
        Key(("F",), "toggle_follow", "follow on / off", "view", hint="follow"),
        Key(("G",), "bottom", "bottom + follow", "nav", "g/G", hint="top/bottom",
            summary="top / bottom + follow", hint_key="g/G"),
        Key(("g",), "top", "top of the buffer", "nav", in_help=False),
        Key(("o",), "stream('stdout')", "stdout", "view", "o/e", hint="stream", summary="stdout / stderr",
            hint_key="o/e"),
        Key(("e",), "stream('stderr')", "stderr", "view", in_help=False),
        Key(("slash",), "search", "search the buffer", "nav", "/", hint="search"),
        Key(("n",), "search_next", "next match", "nav", "n/N", summary="next / previous match"),
        Key(("N",), "search_prev", "previous match", "nav", in_help=False),
        Key(("left_square_bracket",), "prev_error", "previous error / warning line", "nav", "[ ]", hint="error",
            summary="previous / next error or warning line", hint_key="[/]"),
        Key(("right_square_bracket",), "next_error", "next error / warning line", "nav", in_help=False),
        Key(("p",), "app.pager", "log in $PAGER", "nav", hint="pager"),
        Key(("R",), "reconnect", "reopen the log stream", "view"),
        Key(("y",), "copy_path", "copy the path of the shown stream (stdout / stderr)", "act", hint="copy path"),
        Key(("c",), "app.connect_host", "connect host", "act"),
    ),
    "attach_confirm": (
        Key(("y",), "once", "read once (one srun step, one permanent sacct row)", "act", "y", hint="once"),
        Key(("a",), "allow_job", "allow srun reads for this job until RunFleet exits (each read still adds a row)",
            "act", hint="allow for this job"),
        Key(("n", "escape"), "cancel", "cancel", "act", "n/esc", hint="cancel"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "confirm": (
        Key(("y",), "confirm", "confirm", "act", "y", hint="confirm"),
        Key(("n", "escape"), "cancel", "cancel", "act", "n/esc", hint="cancel"),
        Key(("tab",), "noop", "move between params and buttons", "nav", bind=False),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "action_menu": (
        Key(("j",), "cursor_down", "move down", "nav", "j/k", summary="move"),
        Key(("k",), "cursor_up", "move up", "nav", in_help=False),
        Key(("enter",), "noop", "run the action", "act", "↵", bind=False),
        Key(("letter",), "noop", "pick the entry by its key (copy menu: Y again = stdout path)", "act", "a-z",
            bind=False),
        Key(("escape", "q"), "close", "close", "nav", "esc"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "help": (
        Key(("escape",), "close", "close", "nav", "esc"),
        Key(("question_mark",), "noop", "close", "nav", "?", bind=False),
        Key(("f1",), "close", "close", "nav", "F1"),
    ),
    "host_error": (
        Key(("c",), "connect", "connect (log in)", "act", hint="connect"),
        Key(("r",), "retry", "retry now", "act", hint="retry"),
        Key(("t",), "disable", "enable / disable the host", "act", hint="on/off"),
        Key(("escape", "q"), "close", "close", "nav", "esc", hint="close"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "notifications": (
        Key(("j",), "cursor_down", "move down", "nav", "j/k", summary="move"),
        Key(("k",), "cursor_up", "move up", "nav", in_help=False),
        Key(("enter",), "noop", "open the job", "nav", "↵", bind=False),
        Key(("escape", "q"), "close", "close", "nav", "esc"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "digest": (
        Key(("j",), "cursor_down", "move down", "nav", "j/k", summary="move"),
        Key(("k",), "cursor_up", "move up", "nav", in_help=False),
        Key(("enter",), "noop", "open the failed job", "nav", "↵", bind=False),
        Key(("f",), "filter", "show everything that ended since then in the Jobs view", "view"),
        Key(("escape", "q"), "close", "close", "nav", "esc"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "forwards": (
        Key(("j",), "cursor_down", "move down", "nav", "j/k", summary="move"),
        Key(("k",), "cursor_up", "move up", "nav", in_help=False),
        Key(("enter",), "noop", "copy the local URL again", "act", "↵", bind=False),
        Key(("ctrl+k",), "cancel_forward", "cancel the forward (ssh -O cancel, local)", "act"),
        Key(("escape", "q"), "close", "close", "nav", "esc"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "prompt": (
        Key(("enter",), "noop", "use this value", "act", "↵", bind=False),
        Key(("escape",), "close", "cancel", "nav", "esc"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "modal": (
        Key(("escape", "q"), "close", "close", "nav", "esc"),
        Key(("f1",), "app.help", "help (this dialog)", "view", "F1"),
    ),
    "settings": (
        Key(("tab", "right"), "section(1)", "next section (Hosts → General → About)", "nav", "tab/→",
            summary="next / previous section"),
        Key(("shift+tab", "left"), "section(-1)", "previous section", "nav", in_help=False),
        Key(("down", "j"), "cursor(1)", "move down", "nav", "j/k ↑/↓", summary="move"),
        Key(("up", "k"), "cursor(-1)", "move up", "nav", in_help=False),
        Key(("space", "enter"), "activate", "toggle host (save + poll / remove) · change value · open", "act",
            "space/↵"),
        Key(("t",), "test", "test the host now (read-only probe)", "act"),
        Key(("a",), "add", "add an ssh target that is not in ~/.ssh/config", "act"),
        Key(("p",), "poll", "poll a saved host this run left out (its saved settings, file unchanged)", "act"),
        Key(("J", "shift+down"), "move_host(1)", "move host down (card order, number keys)", "act", "J/K S-↓/↑",
            summary="reorder saved hosts"),
        Key(("K", "shift+up"), "move_host(-1)", "move host up", "act", in_help=False),
        Key(("slash",), "filter", "filter hosts", "view", "/"),
        Key(("e",), "edit_config", "open config.toml in $EDITOR", "act"),
        Key(("question_mark", "f1"), "app.help", "these keys", "view", "?/F1"),
        Key(("escape", "q", "comma"), "close", "close", "nav", "esc"),
    ),
}

CONTEXT_TITLES: dict[str, str] = {
    "global": "Global", "main": "Main screen", "fleet": "Fleet view", "jobs": "Jobs view", "gpus": "GPUs view",
    "job_detail": "Job detail", "job_live": "Job detail · GPU / Procs tab (running Slurm job)",
    "host_detail": "Host detail", "host_ws": "Host detail (workstation)",
    "host_cluster": "Host detail (cluster)", "log": "Log viewer", "confirm": "Confirm dialog",
    "attach_confirm": "srun confirmation (live readings)",
    "action_menu": "Action menu", "help": "Help", "host_error": "Host error", "modal": "Dialogs",
    "settings": "Settings", "notifications": "Notifications", "digest": "Since you were away",
    "forwards": "Port forwards", "prompt": "Prompt",
}


def keys_of(context: str) -> tuple[Key, ...]:
    return KEYMAP[context]


def bindings(context: str) -> list[BindingType]:
    """Textual bindings of one context (documented-only entries are skipped)."""
    out: list[BindingType] = []
    for k in KEYMAP[context]:
        if not k.bind:
            continue
        for key in k.keys:
            out.append(Binding(key, k.action, k.description, show=False, priority=k.priority))
    return out


def find(context: str, action: str) -> Key | None:
    for k in KEYMAP[context]:
        if k.action == action:
            return k
    return None


def hints(contexts: Sequence[str], *, readonly: bool = False) -> list[tuple[str, str, str]]:
    """Hint-bar entries ``(key label, text, action)`` for the contexts, most specific first, globals last."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for ctx in contexts:
        for k in KEYMAP[ctx]:
            if k.hint is None or (readonly and k.mutating):
                continue
            key = k.hint_key or display_key(k.keys[0])
            if key in seen:
                continue
            seen.add(key)
            out.append((key, k.hint, k.action))
    return out


def help_entries(contexts: Sequence[str]) -> dict[Group, list[tuple[str, str]]]:
    """Help overlay rows grouped into columns; a key shown by a more specific context hides the global one."""
    out: dict[Group, list[tuple[str, str]]] = {"nav": [], "act": [], "view": []}
    taken: set[str] = set()
    for ctx in contexts:
        for k in KEYMAP[ctx]:
            if k.action.startswith("app.scope_index"):
                continue                 # summarised by the "0 / 1-9" row
            new_keys = [key for key in k.keys if key not in taken]
            if not new_keys:
                continue
            taken.update(k.keys)
            if not k.in_help:
                continue
            out[k.group].append((k.label, k.summary or k.description))
    return out


def all_contexts() -> Iterable[str]:
    return KEYMAP.keys()


def markdown() -> str:
    """Key reference for ``docs/USER_GUIDE.md`` (WP4 pastes / generates it from here)."""
    lines = ["| Context | Keys |", "|---|---|"]
    for ctx, keys in KEYMAP.items():
        cells = []
        for k in keys:
            if k.action.startswith("app.scope_index"):
                continue
            cells.append(f"`{'`/`'.join(display_key(x) for x in k.keys)}` {k.description}")
        lines.append(f"| {CONTEXT_TITLES[ctx]} | {' · '.join(cells)} |")
    return "\n".join(lines) + "\n"
