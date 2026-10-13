"""[WP3] Host detail: GPU panels + processes (workstation) or nodes/partitions (cluster) (mockups G, H).

Both layouts are composed once and toggled by the host's profile, so a host whose first probe arrives while the
screen is open switches layout without re-mounting. Section-level problems are listed in the Connection box with
their stderr excerpt.
"""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Static

from runfleet.core.store import HostState
from runfleet.models import Job, JobKey, StateGroup, Visibility
from runfleet.ui import keymap, render, theme
from runfleet.ui.messages import FleetChanged
from runfleet.ui.screens import FleetScreen
from runfleet.ui.screens.job_detail import stale_banner
from runfleet.ui.widgets.gpu_panel import GPUPanel
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.host_card import busy_note, is_cluster
from runfleet.ui.widgets.job_table import ClickTable
from runfleet.ui.widgets.node_table import NodeTable, PartitionTable, occupant_lines
from runfleet.ui.widgets.proc_table import SORT_LABELS, SORTS, ProcTable
from runfleet.util import humanize


# ---------------------------------------------------------------- pure renderers
def host_metrics_line(state: HostState, width: int, mem_t: tuple[float, float]) -> Text:
    snap = state.snapshot
    m = snap.metrics if snap is not None else None
    if m is None:
        return Text("cpu / memory: not sampled yet", style=theme.dim())
    bar = max(6, min(20, (width - 84) // 2))
    mem_pct = (m.mem_total - m.mem_available) / m.mem_total * 100 if m.mem_total and m.mem_available is not None \
        else None
    out = Text("CPU ", style="bold")
    out.append_text(render.gpu_bar(m.cpu_pct, bar + 2, theme.DEFAULT_UTIL_THRESHOLDS))
    out.append(humanize.pct(m.cpu_pct, width=5))
    out.append(f"  of {m.ncpu or '?'}", style=theme.muted())
    out.append("    MEM ", style="bold")
    out.append_text(render.gpu_bar(mem_pct, bar + 2, mem_t))
    out.append(humanize.pct(mem_pct, width=5))
    if m.mem_total and m.mem_available is not None:
        out.append(f"  {humanize.size(m.mem_total - m.mem_available)[:-1]}/{humanize.size(m.mem_total)}",
                   style=theme.muted())
    if m.load is not None:
        out.append("    load ", style=theme.muted())
        out.append(" ".join(f"{x:.2f}" for x in m.load))
    if m.uptime_s is not None:
        out.append(f"   up {humanize.age(m.uptime_s)}", style=theme.muted())
    if m.swap_total:
        used = m.swap_total - (m.swap_free or 0)
        out.append(f"   swap {humanize.size(used)}/{humanize.size(m.swap_total)}", style=theme.muted())
    return render.fit(out, width)


def connection_lines(state: HostState, now: datetime, width: int) -> list[Text]:
    st = state.status
    snap = state.snapshot
    caps = snap.caps if snap is not None else None
    ok = sum(1 for s in st.sections.values() if s.ok)
    first = Text(f"ssh {state.config.ssh}", style="bold")
    first.append(f" · mux: {st.mux}", style=theme.muted())
    if st.latency_s is not None:
        first.append(f" · probe {st.latency_s:.2f}s", style=theme.muted())
    if st.sections:
        first.append(f" · sections ok {ok}/{len(st.sections)}", style=theme.ok() if ok == len(st.sections)
                     else theme.warn())
    if caps is not None and caps.profile:
        first.append(" · " + "+".join(caps.profile), style=theme.muted())
    if st.tick_interval:
        first.append(f" · every {humanize.age(st.tick_interval)}", style=theme.muted())
    if st.next_poll_at is not None:
        first.append(f" · next poll {humanize.age(max(0.0, (st.next_poll_at - now).total_seconds()))}",
                     style=theme.muted())
    busy = busy_note(st)
    if busy:
        first.append(f" · {busy}", style=theme.warn())
    lines = [render.fit(first, width)]
    if st.error is not None:
        err = Text(f"✖ {st.error.kind.value}: {st.error.summary}", style=theme.conn_style(st.state))
        lines.append(render.fit(err, width))
        if st.error.detail:
            lines.append(render.fit(Text("  " + st.error.detail.splitlines()[0], style=theme.muted()), width))
        lines.append(Text("  e: details · c: connect · r: retry now · t: disable", style=theme.dim()))
    for sec in st.sections.values():
        if not sec.ok:
            msg = sec.error.summary if sec.error is not None else sec.outcome
            line = Text(f"! {sec.name}: {msg}", style=theme.warn())
            if sec.last_ok is not None:
                line.append(f" (last ok {humanize.ago(sec.last_ok, now)})", style=theme.muted())
            lines.append(render.fit(line, width))
    if caps is not None and abs(caps.clock_skew.total_seconds()) > 60:
        lines.append(Text(f"clock skew {caps.clock_skew.total_seconds():+.0f}s", style=theme.warn()))
    if caps is not None and caps.tools and "timeout" not in caps.tools and not is_cluster(state):
        lines.append(Text("remote `timeout` missing — commands are bounded by the local probe timeout",
                          style=theme.dim()))
    return lines


def capability_lines(state: HostState, width: int) -> list[Text]:
    snap = state.snapshot
    caps = snap.caps if snap is not None else None
    if caps is None:
        return [Text("capabilities unknown until the first successful probe", style=theme.dim())]
    tools = Text("tools ", style=theme.label())
    tools.append(" ".join(sorted(caps.tools)) or "none", style=theme.muted())
    if caps.gpu_tres is True:
        tools.append(" · GPU accounting ✓ (sstat gres/gpuutil)", style=theme.ok())
    elif caps.gpu_tres is False:
        tools.append(" · GPU accounting ✗ (job GPU util unavailable)", style=theme.warn())
    tools.append(" · nvidia-smi " + ("✓" if "nvidia-smi" in caps.tools else "✗ (login node)"),
                 style=theme.muted())
    lines = [render.fit(tools, width)]
    cfg = caps.slurm_config
    second = Text()
    if cfg.get("AccountingStorageType"):
        second.append("accounting ✓ · ", style=theme.muted())
    for key in ("PreemptMode", "MinJobAge"):
        if key in cfg:
            second.append(f"{key} {cfg[key]} · ", style=theme.muted())
    if caps.associations:
        protected = set(state.config.slurm.protected_qos) | set(state.config.slurm.protected_partitions)
        accounts = sorted({a[1] for a in caps.associations})
        qos = [a[3] + ("*" if a[3] in protected else "") for a in caps.associations]
        second.append(f"assoc {', '.join(accounts)} ({', '.join(qos)})", style=theme.muted())
        if any(q.endswith("*") for q in qos):
            second.append(" *protected", style=theme.warn())
    if second:
        lines.append(render.fit(second, width))
    return lines


def my_jobs_line(jobs: list[Job], width: int) -> Text:
    out = Text()
    shown = [j for j in jobs if j.is_mine and not j.state.is_terminal]
    for j in shown:
        if out:
            out.append(" · ", style=theme.dim())
        out.append(j.key.label)
        out.append(" ")
        out.append_text(render.state_badge(j.state, j.held))
        if j.state.group is StateGroup.ACTIVE and j.nodelist:
            out.append(f" {j.nodelist}", style=theme.muted())
        elif j.reason:
            out.append(" (")
            out.append_text(render.reason_text(j.reason))
            out.append(")")
    if not shown:
        out.append("no active or queued jobs", style=theme.dim())
    return out


def is_unknown(state: HostState) -> bool:
    """Never connected and no backend configured: neither the workstation nor the cluster layout applies yet."""
    snap = state.snapshot
    return (snap is None or snap.caps is None) and state.config.backend == "auto" and not is_cluster(state)


# ---------------------------------------------------------------- screen
class HostDetailScreen(FleetScreen):
    BINDINGS = keymap.bindings("host_detail") + keymap.bindings("host_ws") + keymap.bindings("host_cluster")
    HELP_CONTEXTS = ("host_ws", "host_detail", "global")
    HELP_TITLE = "Host detail"

    def __init__(self, host: str) -> None:
        super().__init__()
        self.host = host
        self.show_others = False
        self.gpu: int | None = None             # workstation: the GPU whose processes the table shows (all users)
        self.sort = "gpu_mem"
        self.reverse = False
        self.free_only = False
        self.partition: str | None = None
        self.cluster = False
        self.unknown = False

    @property
    def state(self) -> HostState:
        return self.rf.store.hosts[self.host]

    def _extra(self) -> str:
        snap = self.state.snapshot
        caps = snap.caps if snap is not None else None
        if caps is None:
            return ""
        if is_cluster(self.state):
            bits = [f"Slurm {caps.slurm_version}" if caps.slurm_version else "Slurm"]
            if caps.cluster_name:
                bits.append(f"cluster {caps.cluster_name}")
            bits.append(f"login {caps.login_host}")
            return " · ".join(bits)
        return f"driver {caps.driver_version}" if caps.driver_version else caps.login_host

    def compose(self) -> ComposeResult:
        yield FleetHeader(crumbs=["Hosts", self.host], host=self.host, extra=self._extra, id="header")
        yield Static("", id="banner", classes="title-line")
        # scrolling layouts: near the minimum terminal size the panels below the tables stay reachable
        with VerticalScroll(id="ws-layout"):
            yield GPUPanel("", id="gpu-panel", classes="box")
            yield Static("", id="host-box", classes="box")
            with Vertical(id="procs-box", classes="box"):
                yield ProcTable(id="procs")
            yield Static("", id="ws-conn", classes="box")
        with VerticalScroll(id="cluster-layout"):
            with Vertical(id="parts-box", classes="box"):
                yield PartitionTable(id="parts")
            with Vertical(id="nodes-box", classes="box"):
                yield NodeTable(id="nodes")
            yield Static("", id="onnode-box", classes="box")
            yield Static("", id="myjobs-box", classes="box")
            yield Static("", id="caps-box", classes="box")
        yield HintBar(self.help_contexts(), id="hints")

    def on_mount(self) -> None:
        for box_id, title in (("#gpu-panel", "GPUs"), ("#host-box", "Host"), ("#ws-conn", "Connection"),
                              ("#parts-box", "Partitions (assoc, recent jobs, GPU partitions open to you)"),
                              ("#caps-box", "Capabilities / connection"), ("#myjobs-box", f"My jobs on {self.host}")):
            self.query_one(box_id).border_title = title
        self.refresh_data(None)
        self._focus_table()

    def _focus_table(self) -> None:
        if self.cluster:
            self.query_one(NodeTable).focus()
        else:
            self.query_one(ProcTable).focus()

    # ------------------------------------------------------------------ data
    def refresh_data(self, changes: FleetChanged | None) -> None:
        if not self.is_mounted:
            return
        if changes is not None and self.host not in changes.hosts:
            return
        state = self.state
        cluster = is_cluster(state)
        unknown = is_unknown(state)
        if cluster != self.cluster or changes is None or unknown != self.unknown:
            self.cluster, self.unknown = cluster, unknown
            self.query_one("#ws-layout").display = not cluster
            self.query_one("#cluster-layout").display = cluster
            for box in ("#gpu-panel", "#host-box", "#procs-box"):   # never connected: only the Connection panel
                self.query_one(box).display = not unknown
            hints = self.query_one(HintBar)
            hints.set_contexts(self.help_contexts())
            if changes is not None and (self.focused is None or isinstance(self.focused, DataTable)):
                self.call_after_refresh(self._focus_table)     # the focused table may sit in the layout just hidden
        self.query_one(FleetHeader).refresh_content()
        now = self.rf.now()
        banner = stale_banner(state, now)
        widget = self.query_one("#banner", Static)
        widget.display = banner is not None
        widget.update(banner or "")
        if cluster:
            self._render_cluster(state, now)
        else:
            self._render_ws(state, now)

    def refresh_time(self) -> None:
        self.query_one(FleetHeader).refresh_content()

    def _width(self) -> int:
        return (self.size.width or self.app.size.width) - 4

    def _render_ws(self, state: HostState, now: datetime) -> None:
        snap = state.snapshot
        ui = self.rf.service.config.ui
        me = snap.caps.remote_user if snap is not None and snap.caps is not None else None
        width = self._width()
        gpus = snap.gpus if snap else ()
        if self.gpu is not None and self.gpu not in {g.index for g in gpus}:
            self.gpu = None                              # the GPU left the sample (driver reset, nvidia-smi hung)
        panel = self.query_one(GPUPanel)
        panel.show(gpus, ui.gpu_util_thresholds, ui.gpu_mem_thresholds, show_others=True, me=me, selected=self.gpu)
        panel.border_subtitle = ("click it again or [ ] for all processes" if self.gpu is not None
                                 else "click a GPU or [ ]: its processes" if gpus else "")
        self.query_one("#host-box", Static).update(host_metrics_line(state, width, ui.gpu_mem_thresholds))
        box = self.query_one("#procs-box")
        sort = f"sort: {SORT_LABELS[self.sort]} {'▲' if self.reverse else '▼'}"
        if self.gpu is not None:
            box.border_title = f"Processes on GPU {self.gpu} (all users)"
            box.border_subtitle = sort
        else:
            box.border_title = "Processes (GPU: all users · tasks: mine)"
            box.border_subtitle = f"u: {'others shown' if self.show_others else 'others hidden'} · {sort}"
        labels = {j.key: j.key.label for j in snap.jobs.values()} if snap else {}
        self.query_one(ProcTable).show(snap.procs if snap else (), gpus, me=me, show_others=self.show_others,
                                       sort=self.sort, reverse=self.reverse, task_labels=labels, gpu=self.gpu)
        self.query_one("#ws-conn", Static).update(Text("\n").join(connection_lines(state, now, width)))

    def _render_cluster(self, state: HostState, now: datetime) -> None:
        snap = state.snapshot
        width = self._width()
        parts = list(snap.partitions) if snap else []
        names = [p.name for p in parts]
        if self.partition not in names:
            self.partition = names[0] if names else None
        ptable = self.query_one(PartitionTable)
        ptable.show(parts, selected=self.partition)
        ptable.styles.height = max(2, len(parts) + 1)
        nodes = [n for n in (snap.nodes if snap else ()) if self.partition is None or self.partition in n.partitions]
        jobs = list(snap.jobs.values()) if snap else []
        shown = self.query_one(NodeTable).show(nodes, jobs, free_only=self.free_only, now=now)
        box = self.query_one("#nodes-box")
        box.border_title = f"Nodes · {self.partition or 'all'} ({shown})"
        box.border_subtitle = f"f: {'free GPUs only' if self.free_only else 'all nodes'} · [ ]: partition"
        if not nodes:
            box.border_title += " — no node data yet (refreshed every 2 min)"
        elif not shown and self.free_only:
            box.border_title += f" — no node with free GPUs in {self.partition or 'scope'} (f shows all)"
        self._render_onnode()
        self.query_one("#myjobs-box", Static).update(my_jobs_line(jobs, width))
        lines = capability_lines(state, width) + connection_lines(state, now, width)
        self.query_one("#caps-box", Static).update(Text("\n").join(lines))

    def _render_onnode(self) -> None:
        """The jobs on the node under the cursor, every user's (§5.2.9)."""
        snap = self.state.snapshot
        name = self.query_one(NodeTable).cursor_key
        node = next((n for n in (snap.nodes if snap else ()) if n.name == name), None)
        caps = snap.caps if snap is not None else None
        hidden = caps is not None and "jobs" in caps.slurm_config.get("PrivateData", "").lower()
        lines, sub = occupant_lines(node, self._width(), self.rf.now(), hidden=hidden)
        box = self.query_one("#onnode-box", Static)
        box.border_title = f"Jobs on {node.name}" if node is not None else "Jobs on the node"
        box.border_subtitle = sub
        box.update(Text("\n").join(lines))

    # ------------------------------------------------------------------ FleetScreen
    def help_contexts(self) -> list[str]:
        if self.unknown:
            return ["host_detail", "global"]          # c / e / r / t: nothing to sort or open yet
        return ["host_cluster" if self.cluster else "host_ws", "host_detail", "global"]

    def current_host(self) -> str | None:
        return self.host

    def current_job(self) -> Job | None:
        if self.cluster:
            if isinstance(self.focused, PartitionTable):   # a partition is no job target: not the node cursor's job
                return None
            key = self.query_one(NodeTable).cursor_key
            snap = self.state.snapshot
            if key is None or snap is None:
                return None
            for j in snap.jobs.values():
                if key in j.nodes and j.state.group is StateGroup.ACTIVE and j.is_mine:
                    return j
            return None
        proc = self.query_one(ProcTable).current_proc()
        if proc is None or proc.task is None:
            return None
        return self.rf.store.job(proc.task)

    def action_targets(self) -> list[JobKey]:
        job = self.current_job()
        return [job.key] if job is not None else []

    def visibility(self) -> dict[str, Visibility]:
        return {self.host: Visibility.FOCUSED}

    # ------------------------------------------------------------------ events / actions
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if isinstance(event.data_table, NodeTable) and self.cluster:
            self._render_onnode()
        if isinstance(event.data_table, PartitionTable) and event.row_key.value is not None:
            name = str(event.row_key.value)
            if name != self.partition:
                self.partition = name
                self.refresh_data(None)

    def on_click_table_activated(self, event: ClickTable.Activated) -> None:
        self.action_open()

    def on_click_table_context_menu(self, event: ClickTable.ContextMenu) -> None:
        if isinstance(self.focused, PartitionTable):
            return
        self.rf.action_action_menu()

    def action_open(self) -> None:
        if self.cluster and isinstance(self.focused, PartitionTable):   # ↵ on a partition: on to its nodes
            self.query_one(NodeTable).focus()
            return
        job = self.current_job()
        if job is not None:
            self.rf.open_job(job.key)
        elif not self.cluster:
            self.rf.toast("this process belongs to no task", severity="warning")
        else:
            node = self.query_one(NodeTable).cursor_key
            self.rf.toast(f"no job of yours on {node} (its jobs are listed below)" if node else "no node selected",
                          severity="warning")

    def action_toggle_users(self) -> None:
        self.show_others = not self.show_others
        self.refresh_data(None)

    def action_next_sort(self) -> None:
        self.sort = SORTS[(SORTS.index(self.sort) + 1) % len(SORTS)]
        self.reverse = False
        self.refresh_data(None)

    def action_reverse_sort(self) -> None:
        self.reverse = not self.reverse
        self.refresh_data(None)

    def on_gpupanel_picked(self, event: GPUPanel.Picked) -> None:
        """Click on a GPU row: the process table shows that GPU's processes (all users); the same GPU again: all."""
        if self.cluster:
            return
        self.gpu = None if self.gpu == event.index else event.index
        self.refresh_data(None)
        self.query_one(ProcTable).focus()

    def _step_gpu(self, step: int) -> None:
        """``[`` / ``]`` on a workstation: all processes -> GPU 0 -> GPU 1 … -> all again."""
        snap = self.state.snapshot
        stops: list[int | None] = [None, *sorted(g.index for g in (snap.gpus if snap else ()))]
        if len(stops) == 1:
            return
        idx = stops.index(self.gpu) if self.gpu in stops else 0
        self.gpu = stops[(idx + step) % len(stops)]
        self.refresh_data(None)

    def action_toggle_free(self) -> None:
        if not self.cluster:
            return
        self.free_only = not self.free_only
        self.refresh_data(None)

    def action_partition(self, step: int) -> None:
        """``[`` / ``]``: the previous / next partition on a cluster, the previous / next GPU on a workstation."""
        if not self.cluster:
            self._step_gpu(step)
            return
        snap = self.state.snapshot
        names = [p.name for p in snap.partitions] if snap else []
        if not names:
            return
        idx = names.index(self.partition) if self.partition in names else 0
        self.partition = names[(idx + step) % len(names)]
        self.query_one(PartitionTable).select(self.partition)      # no RowHighlighted: refresh once, here
        self.refresh_data(None)

