"""[WP2] Attach fixtures (§7.16.8, §10.2): field outputs of the attach probes reformatted into the node sections.

One directory per case:

- `case.txt` — `# …` provenance lines, then `key=value`: `host`, `mode` (ssh|srun), `job` (raw id), `nodes`
  (space-separated, = AttachPlan.nodes), `jobs.<node>` (raw ids node.cg/node.pids read, target first) and optional
  `sluids.<node>` (parallel to jobs, `-` = unknown);
- `_meta.txt` — the login-side sections (`slurm.attach.node<i>`, `slurm.attach.audit`): rc, ms, stderr, in the
  `_meta*.txt` format of tests/fixtures/samples (tests.conftest parses both);
- `node<i>/<section>.txt` + `node<i>/_meta.txt` — stdout and rc/ms/stderr of every nested node section of node i
  (no directory: the hop failed before the node ran anything; the outer stdout is then empty);
- `audit.txt` — stdout of `slurm.attach.audit`.

`node.self.txt` is `@@done` in every field case: the field audits (`pgrep -x nvidia-smi | wc -l` = 0) found no
nvidia-smi left, which the current node.self (`ps` rows of every own nvidia-smi, then `@@done`) prints as no rows.

`results_for(case, plan)` frames each node directory with the plan's own inner nonce (exactly what the node prints)
and returns the SectionResults core would hand to `ingest_attach`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from runfleet.backends.base import AttachPlan
from runfleet.probe.section import SectionResult
from tests.backends.factories import result
from tests.conftest import _parse_meta_file, frame

CASES_DIR = Path(__file__).resolve().parent
CASES = ("k2_three_jobs", "vega_idle_gpu", "lyra_srun_sluid", "k2_gres_none", "vega_adopt_deny",
         "lyra_publickey_deny")


@dataclass
class Case:
    name: str
    host: str
    mode: str
    job: str
    nodes: tuple[str, ...]
    jobs: dict[str, list[str]] = field(default_factory=dict)
    sluids: dict[str, list[str]] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return CASES_DIR / self.name

    def jobs_by_node(self) -> dict[str, list[tuple[str, str]]]:
        """The (raw id, SLUID) lists SlurmCollector would pass to attach.build_plan."""
        out: dict[str, list[tuple[str, str]]] = {}
        for node in self.nodes:
            jobs = self.jobs.get(node, [self.job])
            sluids = self.sluids.get(node, [""] * len(jobs))
            out[node] = list(zip(jobs, sluids, strict=True))
        return out


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def _strip_one(text: str) -> str:
    """The framing parser removes exactly one trailing newline of a section's output."""
    return text[:-1] if text.endswith("\n") else text


def load(name: str) -> Case:
    kv: dict[str, str] = {}
    for line in _read(CASES_DIR / name / "case.txt").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        kv[key.strip()] = value.strip()
    case = Case(name=name, host=kv["host"], mode=kv["mode"], job=kv["job"], nodes=tuple(kv["nodes"].split()))
    for key, value in kv.items():
        if key.startswith("jobs."):
            case.jobs[key[5:]] = value.split()
        elif key.startswith("sluids."):
            case.sluids[key[7:]] = ["" if v == "-" else v for v in value.split()]
    return case


def node_sections(case: Case, i: int) -> dict[str, tuple[str, int, str]] | None:
    """{nested section: (stdout, rc, stderr)} of node i (None: the node never ran the script)."""
    d = case.dir / f"node{i}"
    if not d.is_dir():
        return None
    metas = _parse_meta_file(_read(d / "_meta.txt"))
    out: dict[str, tuple[str, int, str]] = {}
    for name, m in metas.items():
        path = d / f"{name}.txt"
        stdout = _strip_one(_read(path)) if path.is_file() else ""
        out[name] = (stdout, int(m["rc"]), m["stderr"])
    return out


def results_for(case: Case, plan: AttachPlan) -> dict[str, SectionResult]:
    metas = _parse_meta_file(_read(case.dir / "_meta.txt"))
    out: dict[str, SectionResult] = {}
    for i, _node in enumerate(plan.nodes):
        name = f"slurm.attach.node{i}"
        m = metas.get(name, {"rc": 0, "ms": 0, "stderr": ""})
        inner = node_sections(case, i)
        stdout = ""
        if inner is not None:
            order = [s.name for s in plan.inner[name].sections if not s.late] + \
                    [s.name for s in plan.inner[name].sections if s.late]
            stdout = _strip_one(frame({s: inner[s] for s in order if s in inner}, plan.inner[name].nonce))
        out[name] = result(name, stdout, int(m["rc"]), m["stderr"], ms=m["ms"])
    m = metas.get("slurm.attach.audit")
    if m is not None:
        audit = case.dir / "audit.txt"
        out["slurm.attach.audit"] = result("slurm.attach.audit", _strip_one(_read(audit)) if audit.is_file() else "",
                                           int(m["rc"]), m["stderr"], ms=m["ms"])
    return out
