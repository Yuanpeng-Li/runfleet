"""[WP2] GPU type -> memory MiB table.

Slurm reports a job's GPU type (`nvidia_h200_nvl`, `rtx6000`, MIG profiles such as `nvidia_a100_80gb_pcie_3g.40gb`)
but not the GPU's memory; sstat gives used memory only. `mem_mib` supplies the total for bars (§5.2.3).
"""

from __future__ import annotations

import re

# Exact entries (MiB as nvidia-smi reports memory.total for that board).
EXACT_MIB: dict[str, float] = {
    "nvidia_h200_nvl": 143771,
    "nvidia_h100_80gb_hbm3": 81559,
    "nvidia_h100_nvl": 95830,
    "nvidia_geforce_rtx_3090": 24576,
    "nvidia_a30": 24576,
    "a30": 24576,
    "nvidia_rtx_pro_6000_blackwell_server_edition": 97887,
}

# Names Slurm sites use for boards of different sizes: a bare `rtx6000` is a 24 GB Quadro RTX 6000 on old nodes, while
# some sites label an NVIDIA RTX PRO 6000 Blackwell (96 GB) with the same gres name. Unknown is honest: the UI then
# shows memory used only, never a wrong total, until an Attach sample reports the model (gpu_type_source
# "nvidia-smi").
AMBIGUOUS = frozenset({"rtx6000", "rtx_6000", "quadro_rtx_6000"})

_MIG = re.compile(r"(\d+)g\.(\d+)gb$")    # any separator (`…_pcie-1g.10gb` too), like util.humanize.is_mig
_GB = re.compile(r"(\d+)gb")


def mem_mib(gpu_type: str | None) -> float | None:
    """Total memory of one GPU of this type in MiB: exact table, then MIG `<n>g.<m>gb` → m GiB, then a generic
    `<n>gb` → n GiB; unknown (`v100`) or ambiguous (`rtx6000`: 24 GB Quadro or 96 GB RTX PRO 6000 Blackwell) → None."""
    if not gpu_type:
        return None
    t = gpu_type.strip().lower()
    if t in AMBIGUOUS:
        return None
    if t in EXACT_MIB:
        return EXACT_MIB[t]
    m = _MIG.search(t)
    if m:
        return float(int(m.group(2)) * 1024)
    m = _GB.search(t)
    if m:
        return float(int(m.group(1)) * 1024)
    return None
