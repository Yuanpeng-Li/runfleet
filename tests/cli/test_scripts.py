"""scripts/: the real-host smoke script's ssh accounting (hermetic: fake ssh, fake runfleet) and the docs generator."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "scripts" / "smoke_real.sh"


def _fake_bin(tmp_path: Path, net_calls: int, doctor_extra: tuple[str, ...] = ()) -> Path:
    """A PATH dir with a no-op `ssh` and a fake `runfleet` that starts `ssh` like doctor/snapshot would
    (`doctor_extra`: more ssh command lines run by `doctor` only, e.g. a cold master)."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    calls = "\n".join(["ssh -G \"$h\" >/dev/null", "ssh -O check \"$h\" >/dev/null"]
                      + ["ssh -T \"$h\" 'exec sh -s' </dev/null"] * net_calls)
    (bin_dir / "runfleet").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo "runfleet 0.2.0"; exit 0; fi\n'
        'for h; do :; done\n'                        # last argument = host
        f'if [ "$1" = doctor ]; then :; {"; ".join(doctor_extra)}\nfi\n'
        f"{calls}\n"
        'case "$1" in doctor) echo "  ✓ probe ok";; '
        'snapshot) echo "runfleet: --debug: raw probe dumps in /x/state/runfleet/debug" >&2; echo "{}";; esac\n',
        encoding="utf-8")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir


def _run_smoke(tmp_path: Path, bin_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", RUNFLEET=str(bin_dir / "runfleet"),
               TMPDIR=str(tmp_path))
    assert shutil.which("ssh", path=env["PATH"]) == str(bin_dir / "ssh")    # never the real client
    return subprocess.run(["bash", str(SMOKE), *args], capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_smoke_script_requires_explicit_approval(tmp_path: Path) -> None:
    out = _run_smoke(tmp_path, _fake_bin(tmp_path, 1))
    assert out.returncode == 2 and "--yes" in out.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_smoke_script_counts_network_sessions_per_host(tmp_path: Path) -> None:
    bin_dir = _fake_bin(tmp_path, 1)                 # 1 per command -> 2 per host (doctor + snapshot)
    out = _run_smoke(tmp_path, bin_dir, "--yes", "--out", str(tmp_path / "o"), "k2", "lynx")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ssh sessions 2 (budget 3) · master logins 0 (+0 hop, budget 1) · ssh -G 2 · ssh -O 2" in out.stdout
    assert out.stdout.count("· ok") == 2
    assert "--debug raw dumps: /x/state/runfleet/debug" in out.stdout and "~/.local" not in out.stdout
    log = (tmp_path / "o" / "ssh-calls.tsv").read_text().splitlines()
    assert sum(1 for line in log if line.startswith("k2\tnet\t")) == 2


COLD_MASTER = ("ssh -fN -o ControlMaster=yes -o 'ProxyCommand=ssh -T -W [%h]:%p jumpvm' \"$h\"",
               "ssh -T -o BatchMode=yes -W '[lyra.example]:22' jumpvm </dev/null")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_smoke_script_budgets_a_cold_managed_master_separately(tmp_path: Path) -> None:
    """m3 F4 residual: a cold lyra = master -fN + its -W hop + 3 sessions; that is within budget."""
    out = _run_smoke(tmp_path, _fake_bin(tmp_path, 1, COLD_MASTER + ("ssh -T \"$h\" 'exec sh -s' </dev/null",)),
                     "--yes", "--out", str(tmp_path / "o"), "lyra")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ssh sessions 3 (budget 3) · master logins 1 (+1 hop, budget 1)" in out.stdout
    kinds = [line.split("\t")[1] for line in (tmp_path / "o" / "ssh-calls.tsv").read_text().splitlines()]
    assert kinds.count("master") == 1 and kinds.count("hop") == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_smoke_script_fails_on_a_second_master_or_a_stray_hop(tmp_path: Path) -> None:
    out = _run_smoke(tmp_path, _fake_bin(tmp_path, 1, COLD_MASTER + COLD_MASTER[:1]), "--yes", "lyra")
    assert out.returncode == 1 and "TOO MANY master logins (> 1)" in out.stdout
    (tmp_path / "fakebin").rename(tmp_path / "fakebin1")
    out = _run_smoke(tmp_path, _fake_bin(tmp_path, 1, COLD_MASTER[1:]), "--yes", "lyra")
    assert out.returncode == 1 and "jump hops without a master login" in out.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_smoke_script_fails_over_budget(tmp_path: Path) -> None:
    out = _run_smoke(tmp_path, _fake_bin(tmp_path, 2), "--yes", "--max-calls", "3", "k2")
    assert out.returncode == 1 and "TOO MANY ssh sessions" in out.stdout


def test_user_guide_key_table_is_generated_from_the_keymap() -> None:
    try:
        from runfleet.ui import keymap
        keymap.markdown()
    except Exception as exc:                          # pragma: no cover - before WP3's keymap lands
        pytest.skip(f"keymap not available: {exc!r}")
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "gen_docs.py"), "--check"], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
