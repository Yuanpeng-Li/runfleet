"""GPU availability watches (ui/gpu_watch.py, DESIGN.md §7.14): the pure evaluator (per node, freshness, readings in a
row, cooldown and re-arming, MIG, planned GPUs, idle pause), the lenient prompt, ui.json, and the app flow (palette
prompt → fires → toast, terminal, hook, Notifications, fleet panel; the Watches list)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import Any

from textual.widgets import Input, OptionList

from runfleet.config import NotifySettings
from runfleet.core.clock import ManualClock
from runfleet.models import ConnState, GresCount, Node, PartitionSummary, SectionStatus
from runfleet.ui import fleet_layout as fl
from runfleet.ui import gpu_watch as gw
from runfleet.ui.app import RunFleetApp
from runfleet.ui.gpu_watch import Evaluator, GpuWatch, Reading
from runfleet.ui.notify import watch_hook_env
from runfleet.ui.screens.gpu_watch import WatchesScreen, WatchPromptScreen
from runfleet.ui.screens.host_detail import HostDetailScreen
from runfleet.ui.screens.notifications import NotificationsScreen
from runfleet.ui.widgets.fleet_panel import FleetPanel, panel_lines
from tests.ui.fakes import T0, FakeHostState, FakeService, build_world

MIN = timedelta(minutes=1)
H200 = "nvidia_h200_nvl"
MIG = "nvidia_a100_80gb_pcie_3g.40gb"


# ---------------------------------------------------------------- builders
def _states() -> dict[str, Any]:
    _cfg, states = build_world()
    return {s.config.name: s for s in states}


def node(name: str, gres: tuple[tuple[str, int, int], ...], *, parts: tuple[str, ...] = ("gpu",), flags: str = "",
         state: str = "mixed", idle_cpus: int = 8) -> Node:
    """A usable node; ``gres``: (type, total, used)."""
    return Node(name, parts, state, frozenset(flags), True, 8, idle_cpus, 0, 16, None, None, None,
                tuple(GresCount("gpu", t, total, used, None) for t, total, used in gres), None)


def cluster(nodes: tuple[Node, ...], at: datetime, *, parts: tuple[str, ...] = ("gpu",), name: str = "k2",
            sinfo_ok: bool = True, state: ConnState = ConnState.OK) -> FakeHostState:
    """A fresh cluster whose last sinfo sample (and poll) is ``at``."""
    st = _states()[name]
    st.snapshot = dataclasses.replace(st.snapshot, nodes=nodes,
                                      partitions=tuple(PartitionSummary(p, 1, 1, 0, 0, {}, {}) for p in parts))
    sec = SectionStatus("slurm.sinfo", "ok" if sinfo_ok else "error", sinfo_ok, 0, 100,
                        at if sinfo_ok else at - timedelta(hours=3))
    st.status = dataclasses.replace(st.status, state=state, sections={"slurm.sinfo": sec}, last_ok=at)
    return st


def ws(at: datetime, *, gpu_ok: bool = True) -> FakeHostState:
    """labstation (4 A6000: 0 and 2 free) with its nvidia-smi sample taken at ``at``."""
    st = _states()["labstation"]
    st.snapshot = dataclasses.replace(st.snapshot, gpus=tuple(dataclasses.replace(g, sampled_at=at)
                                                              for g in st.snapshot.gpus))
    sec = SectionStatus("nvidia.gpu", "ok" if gpu_ok else "timeout", gpu_ok, 0, 100, at if gpu_ok else None)
    st.status = dataclasses.replace(st.status, sections={**st.status.sections, "nvidia.gpu": sec}, last_ok=at)
    return st


def rd(at: datetime | None, *nodes: gw.NodeGPUs, host: str = "k2", parts: tuple[str, ...] = ("gpu",)) -> Reading:
    return Reading(host, at, "" if at else "no fresh reading", tuple(nodes), parts, cluster=True)


def ng(name: str, *types: tuple[str, int, int], parts: tuple[str, ...] = ("gpu",)) -> gw.NodeGPUs:
    return gw.NodeGPUs(name, parts, types)


# ---------------------------------------------------------------- matching: per node, MIG, planned
def test_a_watch_matches_per_node_never_a_fleet_sum() -> None:
    w = GpuWatch(3, "H200")
    two_and_two = rd(T0, ng("n1", (H200, 2, 0)), ng("n2", (H200, 2, 0)))
    assert gw.matches(w, two_and_two) == []                           # 4 free, but no node has 3
    three = rd(T0, ng("n1", (H200, 2, 0)), ng("n2", (H200, 3, 0)))
    assert [(m.node, m.free) for m in gw.matches(w, three)] == [("n2", 3)]
    # a model is one request: the best matching type of a node, never two types added up
    mixed = rd(T0, ng("n1", ("nvidia_a100-sxm4-80gb", 2, 0), ("nvidia_a100-sxm4-40gb", 2, 0)))
    assert gw.matches(GpuWatch(3, "A100"), mixed) == []
    assert gw.matches(GpuWatch(2, "A100-80G"), mixed)[0].free == 2
    # "any GPU" is an untyped request: every whole-GPU type of the node adds up (slices do not)
    assert gw.matches(GpuWatch(4), rd(T0, ng("n1", ("nvidia_a100-sxm4-80gb", 2, 0), ("nvidia_a100-sxm4-40gb", 2, 0),
                                              (MIG, 7, 0))))[0].free == 4
    assert gw.matches(GpuWatch(1, "H100"), rd(T0, ng("n1", ("nvidia_gh200_480gb", 4, 0)))) == []   # not GH200
    # partition: only nodes in it
    w = GpuWatch(1, "H200", partition="gpu-preempt")
    split = rd(T0, ng("n1", (H200, 4, 0), parts=("gpu",)), ng("n2", (H200, 1, 0), parts=("gpu-preempt",)),
               parts=("gpu", "gpu-preempt"))
    assert [m.node for m in gw.matches(w, split)] == ["n2"]
    assert gw.matches(GpuWatch(1, "H200"), rd(None, ng("n1", (H200, 4, 0)))) == []   # no fresh reading: nothing


def test_mig_slices_and_whole_gpus_never_mix() -> None:
    slices = rd(T0, ng("n1", (MIG, 5, 0), ("nvidia_a100_80gb_pcie_1g.10gb", 7, 0)))
    assert gw.matches(GpuWatch(1, "A100"), slices) == []               # a free slice is not a free GPU
    assert gw.matches(GpuWatch(1), slices) == []
    assert gw.matches(GpuWatch(6, mig=""), slices)[0].free == 7        # one profile per request: the best one
    assert gw.matches(GpuWatch(6, mig="3g.40gb"), slices) == []
    assert gw.matches(GpuWatch(5, "A100", mig="3g.40gb"), slices)[0].free == 5
    assert gw.matches(GpuWatch(1, "H100", mig=""), slices) == []
    whole = rd(T0, ng("n1", ("nvidia_a100_80gb_pcie", 4, 0)))
    assert gw.matches(GpuWatch(1, mig=""), whole) == [] and gw.matches(GpuWatch(4, "A100"), whole)


def test_planned_gpus_count_only_when_the_watch_says_so() -> None:
    """slurmfmt.node_free: a node backfill planned for a pending job ("-" flag / state planned) has its free GPUs as
    planned; a watch leaves them out unless it says ``planned``, and says what it left out."""
    st = cluster((node("n1", ((H200, 4, 1),), flags="-"), node("n2", ((H200, 4, 3),))), T0)
    rds = gw.readings({"k2": st}, T0)
    assert rds["k2"].nodes[0].types == ((H200, 3, 3),)
    assert [(m.node, m.free) for m in gw.matches(GpuWatch(1, "H200"), rds["k2"])] == [("n2", 1)]
    assert gw.matches(GpuWatch(2, "H200"), rds["k2"]) == []
    assert [(m.node, m.free) for m in gw.matches(GpuWatch(2, "H200", planned=True), rds["k2"])] == [("n1", 3)]
    text = gw.now_text(GpuWatch(2, "H200"), rds)
    assert text.startswith("best now: k2 n2 1 free") and "k2 n1 0 free (+3 planned) only with planned GPUs" in text


# ---------------------------------------------------------------- readings: only fresh, successful ones
def test_readings_follow_the_fleet_panels_freshness_rule() -> None:
    nodes = (node("n1", ((H200, 4, 0),)),)
    now = T0
    fresh = gw.readings({"k2": cluster(nodes, now)}, now)["k2"]
    assert fresh.at == now and fresh.cluster and gw.matches(GpuWatch(4, "H200"), fresh)
    failed = gw.readings({"k2": cluster(nodes, now, sinfo_ok=False)}, now)["k2"]
    assert failed.at is None and failed.why == "no fresh reading"       # "k2 ?" in the panel
    down = gw.readings({"k2": cluster(nodes, now, state=ConnState.UNREACHABLE)}, now)["k2"]
    assert down.at is None
    st = cluster(nodes, now)
    old = now - timedelta(seconds=fl.sinfo_bound_s(st) + 1)
    st.status = dataclasses.replace(st.status, sections={"slurm.sinfo": dataclasses.replace(
        st.status.sections["slurm.sinfo"], last_ok=old)})
    assert gw.readings({"k2": st}, now)["k2"].at is None                # sinfo older than its cadence allows
    disabled = cluster(nodes, now, state=ConnState.DISABLED)
    assert gw.readings({"k2": disabled}, now)["k2"].why == "disabled"
    # the shared rule: what the panel counts, the watch counts
    for state in (cluster(nodes, now), cluster(nodes, now, sinfo_ok=False), st):
        assert (fl.free_source(state, now)[0] == "cluster") == (gw.readings({"k2": state}, now)["k2"].at is not None)
    # a workstation: GPUs neither in use nor an old reading, and a successful nvidia-smi sample
    lab = gw.readings({"labstation": ws(now)}, now)["labstation"]
    assert lab.at == now and gw.matches(GpuWatch(2, "A6000"), lab)[0].text() == "labstation 2 free"
    assert gw.matches(GpuWatch(3, "A6000"), lab) == []
    assert gw.readings({"labstation": ws(now, gpu_ok=False)}, now)["labstation"].at is None
    aged = gw.readings({"labstation": ws(now)}, now, old=lambda _s, g: g.index == 0)["labstation"]
    assert gw.matches(GpuWatch(2, "A6000"), aged) == []                 # GPU 0 missed its refresh: not free


def test_readings_memo_rescans_only_on_a_new_sample() -> None:
    st = cluster((node("n1", ((H200, 4, 0),)),), T0)
    memo: dict[str, tuple[Any, ...]] = {}
    first = gw.readings({"k2": st}, T0, memo=memo)["k2"]
    assert gw.readings({"k2": st}, T0, memo=memo)["k2"].nodes is first.nodes
    later = cluster((node("n1", ((H200, 4, 4),)),), T0 + MIN)
    assert gw.readings({"k2": later}, T0 + MIN, memo=memo)["k2"].nodes[0].types == ((H200, 0, 0),)
    gw.readings({}, T0, memo=memo)
    assert memo == {}


# ---------------------------------------------------------------- evaluation: in a row, cooldown, re-arm, pause
def test_fires_only_after_two_readings_in_a_row() -> None:
    ev = Evaluator()
    w = GpuWatch(2, "H200")
    free = ng("n1", (H200, 2, 0))
    assert ev.update([w], {"k2": rd(T0, free)}, T0) == []               # 1 of 2
    assert ev.update([w], {"k2": rd(T0, free)}, T0 + MIN) == []         # the same reading again: not a new one
    assert "1 of 2 readings" in ev.status(w, {"k2": rd(T0, free)}, T0 + MIN)[0]
    fired = ev.update([w], {"k2": rd(T0 + MIN, free)}, T0 + MIN)
    assert [(f.watch, [m.node for m in f.found]) for f in fired] == [(w, ["n1"])]
    e = gw.fired_event(fired[0])
    assert (e.kind, e.host, e.text()) == ("gpu_free", "k2", "free GPUs: 2× H200 anywhere — k2 n1 2 free")

    ev = Evaluator()
    ev.update([w], {"k2": rd(T0, free)}, T0)
    ev.update([w], {"k2": rd(None)}, T0 + MIN)                           # no fresh reading in between: the run breaks
    assert ev.update([w], {"k2": rd(T0 + 2 * MIN, free)}, T0 + 2 * MIN) == []
    assert ev.update([w], {"k2": rd(T0 + 3 * MIN, free)}, T0 + 3 * MIN)

    ev = Evaluator()
    ev.update([w], {"k2": rd(T0, free)}, T0)
    ev.update([w], {"k2": rd(T0 + MIN, ng("n1", (H200, 2, 1)))}, T0 + MIN)   # one GPU taken: the run breaks
    assert ev.update([w], {"k2": rd(T0 + 2 * MIN, free)}, T0 + 2 * MIN) == []


def test_runs_are_counted_per_host() -> None:
    ev = Evaluator()
    w = GpuWatch(1, "H200")
    k2 = lambda t: rd(t, ng("n1", (H200, 1, 0)))                                    # noqa: E731
    vega = lambda t: rd(t, ng("v1", (H200, 1, 0)), host="vega")                     # noqa: E731
    assert ev.update([w], {"k2": k2(T0), "vega": rd(None, host="vega")}, T0) == []
    assert ev.update([w], {"k2": rd(None), "vega": vega(T0 + MIN)}, T0 + MIN) == []  # one each, not in a row
    fired = ev.update([w], {"k2": rd(None), "vega": vega(T0 + 2 * MIN)}, T0 + 2 * MIN)
    assert [m.host for m in fired[0].found] == ["vega"]
    assert ev.update([GpuWatch(1, "H200", host="k2")], {"k2": k2(T0), "vega": vega(T0)}, T0) == []


def test_cooldown_and_rearm_only_after_it_stopped_holding() -> None:
    ev = Evaluator()
    w = GpuWatch(1, "H200")
    free = ng("n1", (H200, 1, 0))
    busy = ng("n1", (H200, 1, 1))

    def step(minute: int, *nodes: gw.NodeGPUs, at: bool = True) -> list[gw.Fired]:
        t = T0 + minute * MIN
        return ev.update([w], {"k2": rd(t if at else None, *nodes)}, t)

    assert not step(0, free) and step(1, free)                          # fires at 1
    assert not step(2, free) and not step(3, free)                      # still holds: quiet, not re-armed
    text, level = ev.status(w, {"k2": rd(T0 + 3 * MIN, free)}, T0 + 3 * MIN)
    assert text.startswith("fired ") and "re-arms once it no longer holds" in text and level == "accent"
    assert not step(4, free, at=False)                                  # unknown is not "stopped holding"
    assert not ev.states[w].armed
    assert not step(5, busy)                                            # stopped: re-armed
    assert ev.states[w].armed
    assert not step(6, free) and not step(7, free)                      # holds again, but within 15 min of 1
    assert "quiet until" in ev.status(w, {"k2": rd(T0 + 7 * MIN, free)}, T0 + 7 * MIN)[0]
    assert not step(15, free)                                           # 14 min after: still quiet
    assert step(16, free)                                               # cooldown over, still holding: fires


def test_an_idle_pause_breaks_the_run_and_says_so() -> None:
    ev = Evaluator()
    w = GpuWatch(1, "H200")
    free = ng("n1", (H200, 1, 0))
    ev.update([w], {"k2": rd(T0, free)}, T0)
    assert ev.update([w], {"k2": rd(T0 + MIN, free)}, T0 + MIN, paused=True) == []
    text, level = ev.status(w, {"k2": rd(T0 + MIN, free)}, T0 + MIN, paused=True)
    assert text.startswith("paused: polling is idle-paused") and level == "warn"
    assert ev.update([w], {"k2": rd(T0 + 3 * MIN, free)}, T0 + 3 * MIN) == []   # a pause sits between: 1 of 2
    assert ev.update([w], {"k2": rd(T0 + 4 * MIN, free)}, T0 + 4 * MIN)


def test_status_lines() -> None:
    ev = Evaluator()
    w = GpuWatch(2, "H200", host="k2")
    rds = {"k2": rd(T0, ng("n1", (H200, 1, 0))), "lyra": rd(None, host="lyra")}
    assert ev.status(w, rds, T0) == ("waiting: best now: k2 n1 1 free", "dim")
    assert ev.status(GpuWatch(1, "H200"), rds, T0)[0].endswith("· not counted: lyra ?")
    assert ev.status(GpuWatch(1, host="gone"), rds, T0) == ("gone is not in the fleet any more", "warn")
    assert ev.status(GpuWatch(1, partition="h200"), rds, T0)[0] == "no host lists partition h200 now"
    assert ev.status(GpuWatch(1, host="lyra"), rds, T0) == ("no fresh reading to count · not counted: lyra ?", "warn")
    ev.update([w], rds, T0)
    ev.update([], rds, T0)
    assert ev.states == {}                                              # a removed watch forgets its state


# ---------------------------------------------------------------- the prompt
def test_prompt_reads_leniently() -> None:
    fleet = gw.fleet_of(_states())

    def watch(text: str) -> GpuWatch | None:
        return gw.parse(text, fleet).watch

    assert watch("2 h200 k2") == GpuWatch(2, "H200", host="k2")
    assert watch("2x H200 on K2") == watch("2xh200 k2") == watch("x2 h200 @k2") == GpuWatch(2, "H200", host="k2")
    assert watch("4 a100 k2/gpu-preempt planned") == GpuWatch(4, "A100", host="k2", partition="gpu-preempt",
                                                               planned=True)
    assert watch("free gpus on k2 partition=labpi") == GpuWatch(1, host="k2", partition="labpi")
    assert watch("2 3090") == watch("3090 2") == GpuWatch(2, "3090")         # a model that is a number
    assert watch("a100-80gb") == GpuWatch(1, "A100-80G")
    assert watch("1 mig k2") == GpuWatch(1, mig="", host="k2")
    assert watch("2 a100-3g.40gb") == watch("2 a100 3g.40gb") == GpuWatch(2, "A100", mig="3g.40gb")
    assert watch("h200 gpu-preempt") == GpuWatch(1, "H200", partition="gpu-preempt")   # a partition some host lists
    parsed = gw.parse("2 h300", fleet)
    assert parsed.watch == GpuWatch(2, "H300") and parsed.notes == ("no H300 GPUs seen in the fleet yet",)
    for text, problem in (("", "type what to wait for"), ("foo", "“foo” is not a host, partition or GPU type"),
                          ("2 h200 nowhere/gpu", "no host “nowhere”"), ("2 4 h200", "one count per watch"),
                          ("h200 a6000", "one GPU type per watch"), ("k2 vega", "one host per watch"),
                          ("k2/h200", "k2 lists no partition “h200” (it lists gpu-preempt, labpi)"),
                          ("labstation/gpu", "labstation has no partitions"), ("100 h200", "a count from 1 to 64"),
                          ("partition=nope", "no host lists a partition “nope”"), ("0 h200", "a count from 1")):
        got = gw.parse(text, fleet)
        assert got.watch is None and got.problem is not None and got.problem.startswith(problem), (text, got)
    assert gw.parse("1 mig labstation", fleet).notes[-1].startswith("a workstation's nvidia-smi lists whole GPUs")


def test_describe_and_limits() -> None:
    assert GpuWatch().describe() == "1× GPU anywhere"
    assert GpuWatch(2, "H200", host="k2", partition="gpu").describe() == "2× H200 on k2/gpu"
    assert GpuWatch(2, "A100", mig="3g.40gb", partition="gpu", planned=True).describe() == \
        "2× A100 MIG 3g.40gb in partition gpu · planned too"
    watches = [GpuWatch(i + 1) for i in range(gw.MAX_WATCHES)]
    assert gw.add_problem(watches[:3], GpuWatch(2)) == "you watch exactly that already"
    assert gw.add_problem(watches, GpuWatch(64)).startswith(f"at most {gw.MAX_WATCHES} watches")
    assert gw.add_problem(watches[:3], GpuWatch(64)) is None


def test_ui_json_round_trip_and_bad_entries() -> None:
    good = GpuWatch(2, "H200", None, "k2", "gpu", True)
    raw = [good.to_json(), {"count": 0}, {"count": True}, {"gpu": "a b"}, {"mig": "big"}, {"planned": "yes"},
           "2 h200", {"gpu": "a100"}, good.to_json(), {"future": 1}]
    state = {"watches": raw}
    assert gw.load_watches(state) == [good, GpuWatch(1, "A100"), GpuWatch()]
    assert gw.load_watches({"watches": {"x": 1}}) == [] and gw.load_watches({}) == []
    added = gw.with_watch(state, GpuWatch(4))
    assert added[:len(raw)] == raw and added[-1] == GpuWatch(4).to_json()         # unreadable entries are kept
    assert gw.with_watch(state, good) == raw
    removed = gw.without_watch(state, good)
    assert good.to_json() not in removed and "2 h200" in removed and {"count": 0} in removed


# ---------------------------------------------------------------- the app
FLUSH = 0.3


def _app(**notify: Any) -> tuple[RunFleetApp, FakeService]:
    svc = FakeService(ManualClock(T0))
    if notify:
        svc.config = dataclasses.replace(svc.config, notify=dataclasses.replace(NotifySettings(), **notify))
    return RunFleetApp(svc, clock=svc.clock, animations=False), svc   # type: ignore[arg-type]


def new_reading(svc: FakeService, host: str = "labstation", *, advance: float = 10.0, busy: tuple[int, ...] = ()
                ) -> None:
    """A fresh nvidia-smi sample of a workstation ``advance`` seconds later (GPUs in ``busy`` now in use)."""
    svc.clock.advance(advance)
    now = svc.clock.now()
    st = svc.store.hosts[host]
    gpus = tuple(dataclasses.replace(g, sampled_at=now, util_pct=90.0 if g.index in busy else g.util_pct)
                 for g in st.snapshot.gpus)
    svc.store.replace_snapshot(host, gpus=gpus)
    sections = {**st.status.sections, "nvidia.gpu": dataclasses.replace(st.status.sections["nvidia.gpu"],
                                                                          last_ok=now)}
    svc.store.set_status(host, dataclasses.replace(st.status, sections=sections, last_ok=now))


def toasts(app: RunFleetApp) -> list[str]:
    return [n.message for n in app._notifications]


async def test_watch_from_the_palette_fires_everywhere(monkeypatch: Any) -> None:
    monkeypatch.delenv("TMUX", raising=False)
    app, svc = _app(terminal="osc9", command="true")
    sent: list[str] = []
    hooks: list[tuple[Any, ...]] = []
    async with app.run_test(size=(172, 62)) as pilot:
        app.alerts.write_terminal = sent.append                         # type: ignore[method-assign]
        app.alerts._start_hook = lambda *a: hooks.append(a)             # type: ignore[method-assign]
        await pilot.pause(FLUSH)
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press(*"Watch free GPUs")
        await pilot.pause(0.5)
        await pilot.press("enter")
        await pilot.pause(0.3)
        prompt = app.screen
        assert isinstance(prompt, WatchPromptScreen)
        await pilot.press(*"3 a6000 labstation")
        await pilot.pause()
        assert "watch for 3× A6000 on labstation" in str(prompt.query_one("#gw-what").render())
        assert "best now: labstation 2 free" in str(prompt.query_one("#gw-now").render())
        prompt.query_one(Input).value = "2x a6000 nowhere"
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is prompt and "“nowhere” is not a host" in str(prompt.query_one("#gw-note").render())
        prompt.query_one(Input).value = "2x a6000 labstation"
        await pilot.pause()
        assert "matches now: labstation 2 free" in str(prompt.query_one("#gw-now").render())
        await pilot.press("enter")
        await pilot.pause(FLUSH)
        watch = GpuWatch(2, "A6000", host="labstation")
        assert app.alerts.gpu_watches == [watch] and app.ui_state["watches"] == [watch.to_json()]
        assert any("watching for 2× A6000 on labstation" in m for m in toasts(app))
        assert not app.alerts.watch_events                              # 1 of 2 readings
        new_reading(svc)
        await pilot.pause(FLUSH)
        text = "free GPUs: 2× A6000 on labstation — labstation 2 free"
        assert [e.text() for e in app.alerts.watch_events] == [text]
        assert text in toasts(app)
        assert sent == [f"\x1b]9;RunFleet · labstation: {text}\x07"]
        assert len(hooks) == 1 and hooks[0][1] == watch_hook_env(app.alerts.watch_events[0])
        assert hooks[0][1]["RUNFLEET_EVENT"] == "gpu_free" and hooks[0][1]["RUNFLEET_NAME"] == "2× A6000 on labstation"
        new_reading(svc)
        await pilot.pause(FLUSH)
        assert len(app.alerts.watch_events) == 1                        # still holds: no second notification
        panel = app.screen.query_one(FleetPanel)
        await pilot.pause(1.1)
        assert any("◎ 2× A6000 on labstation" in line.plain and "labstation 2 free" in line.plain
                   for line in panel.lines)
        app.show_notifications()
        await pilot.pause(0.2)
        assert isinstance(app.screen, NotificationsScreen)
        assert app.screen.events[0] == app.alerts.watch_events[0]
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, HostDetailScreen) and app.screen.host == "labstation"


async def test_watches_list_shows_state_removes_and_says_paused() -> None:
    watch = GpuWatch(2, "A6000", host="labstation")
    svc = FakeService(ManualClock(T0))
    app = RunFleetApp(svc, clock=svc.clock, animations=False,   # type: ignore[arg-type]
                      initial_state={"watches": [watch.to_json(), {"count": "two"}, GpuWatch(1, "H200").to_json()]})
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        assert app.alerts.gpu_watches == [watch, GpuWatch(1, "H200")]     # the bad entry is skipped
        app.show_gpu_watches()
        await pilot.pause(0.2)
        screen = app.screen
        assert isinstance(screen, WatchesScreen)
        menu = screen.query_one(OptionList)
        first = menu.get_option_at_index(0).prompt
        assert "2× A6000 on labstation" in str(first) and "holds on labstation (1 of 2 readings" in str(first)
        assert "1× H200 anywhere" in str(menu.get_option_at_index(1).prompt)
        new_reading(svc, busy=(0,))                                     # GPU 0 taken: only 1 free
        await pilot.pause(FLUSH)
        assert "waiting: best now: labstation 1 free" in str(menu.get_option_at_index(0).prompt)
        svc.store.set_idle_paused(True)
        await pilot.pause(FLUSH)
        assert "paused: polling is idle-paused" in str(menu.get_option_at_index(0).prompt)
        paused = [e for e in app.alerts.watch_events if e.kind == "paused"]
        assert len(paused) == 1 and "2 watches cannot fire" in paused[0].text()
        assert any(m.startswith("GPU watches paused: polling stopped after 2h") for m in toasts(app))
        app.alerts.scan()
        assert len([e for e in app.alerts.watch_events if e.kind == "paused"]) == 1   # said once per pause
        svc.store.set_idle_paused(False)
        await pilot.pause(FLUSH)
        await pilot.press("d")
        await pilot.pause(0.2)
        assert app.alerts.gpu_watches == [GpuWatch(1, "H200")]
        assert app.ui_state["watches"] == [{"count": "two"}, GpuWatch(1, "H200").to_json()]
        assert "stopped watching for 2× A6000 on labstation" in toasts(app)
        await pilot.press("n")
        await pilot.pause(0.2)
        assert isinstance(app.screen, WatchPromptScreen)


async def test_disabled_notify_keeps_the_history_only() -> None:
    app, svc = _app(enabled=False)
    app.alerts.gpu_watches.append(GpuWatch(1, "A6000"))
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        new_reading(svc)
        await pilot.pause(FLUSH)
        assert [e.kind for e in app.alerts.watch_events] == ["gpu_free"]
        assert not any(m.startswith("free GPUs") for m in toasts(app))


def test_fleet_panel_lists_a_fired_watch_of_a_present_host() -> None:
    states = _states()
    hosts = {"k2": states["k2"], "labstation": states["labstation"]}
    fired = gw.WatchEvent("gpu_free", T0 - 5 * MIN, "2× A6000 on labstation", "labstation", "labstation 2 free")
    gone = dataclasses.replace(fired, host="removed", watch="1× H200 on removed")
    paused = gw.WatchEvent("paused", T0, "", "", "polling stopped")
    lines = [x.plain for x in panel_lines(hosts, T0, rows=19, width=81, wide=True, window=86400.0,
                                          watch_events=[fired, gone, paused])]
    rows = [x for x in lines if "◎" in x]
    assert len(rows) == 1 and "labstation ◎ 2× A6000 on labstation" in rows[0] and "labstation 2 free" in rows[0]


async def test_real_stack_watch_fires_on_polled_readings() -> None:
    """On DemoWorld -> collectors -> FleetService: the readings come from the real `slurm.sinfo` / `nvidia.gpu`
    sections, and a watch for what is free fires only after the next poll confirms it."""
    from tests.ui.test_demo_stack import _settle, _stack

    app, service, _world, clock = _stack()
    async with app.run_test(size=(120, 35)) as pilot:
        await _settle(pilot, service)
        rds = app.alerts.gpu_readings()
        assert rds["labstation"].at is not None and rds["k2"].cluster
        assert rds["lynx"].at is None                                   # needs auth: never counted
        parsed = gw.parse("1 a6000 labstation", gw.fleet_of(app.store.hosts))
        assert parsed.watch is not None and app.alerts.add_gpu_watch(parsed.watch) is None
        assert not app.alerts.watch_events
        for _ in range(40):
            clock.advance(5)
            await pilot.pause(0.05)
            if app.alerts.watch_events:
                break
        assert [e.host for e in app.alerts.watch_events] == ["labstation"]


async def test_watches_list_fits_a_short_terminal() -> None:
    """Review fix: ten watches (two lines each) and the foot at 80×24 — the list stays inside the dialog and scrolls,
    so the highlighted watch is never clipped below the box."""
    svc = FakeService(ManualClock(T0))
    watches = [GpuWatch(n, "A6000").to_json() for n in range(1, 11)]
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False,   # type: ignore[arg-type]
                      initial_state={"watches": watches})
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(FLUSH)
        app.show_gpu_watches()
        await pilot.pause(0.3)
        screen = app.screen
        assert isinstance(screen, WatchesScreen)
        menu = screen.query_one(OptionList)
        foot = screen.query_one("#gw-foot")
        box = screen.query_one("#gw-box")
        for _ in range(9):
            await pilot.press("j")
        await pilot.pause(0.2)
        assert menu.highlighted == 9
        assert foot.region.bottom <= box.region.bottom - 1 and menu.region.bottom <= foot.region.y
        assert menu.scroll_y > 0                                             # scrolled to the last watch


async def test_a_watch_of_a_removed_host_says_so_in_notifications() -> None:
    """Review fix: enter on a firing whose host was removed since did nothing at all; now a toast says why."""
    app, _svc = _app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        app.alerts.watch_events.append(gw.WatchEvent("gpu_free", T0, "1× H200 on gone", "gone", "gone n1 1 free"))
        app.show_notifications()
        await pilot.pause(0.2)
        assert isinstance(app.screen, NotificationsScreen)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert not isinstance(app.screen, NotificationsScreen | HostDetailScreen)
        assert "gone is no longer in the fleet" in toasts(app)
