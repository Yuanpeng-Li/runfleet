"""The demo's `scontrol show partition -o` prints each partition's PreemptMode, as Slurm does, and it reaches
`PartitionCharge.preempt_mode` (the cluster's mode unless `partition_fields` sets one)."""

from __future__ import annotations

import dataclasses

from runfleet.backends.slurm.collector import partition_charges
from runfleet.demo import render
from runfleet.demo.world import DemoWorld, GresDef, SimNode
from runfleet.util.logscan import preempt_mode_for
from tests.backends.factories import caps
from tests.demo.test_render_fixtures import k2_cluster


def test_demo_partitions_print_preempt_mode(world: DemoWorld) -> None:
    for host, mode in (("k2", "REQUEUE"), ("lyra", "CANCEL")):
        charges = partition_charges(render.scontrol_partitions(world.cluster(host)).stdout)
        assert charges and {p.preempt_mode for p in charges} == {mode}
    lyra = partition_charges(render.scontrol_partitions(world.cluster("lyra")).stdout)
    assert preempt_mode_for(dataclasses.replace(caps("lyra"), partition_charges=lyra, slurm_config={}),
                            lyra[0].partition) == "CANCEL"
    c = k2_cluster()
    c.config = {"PreemptMode": "REQUEUE"}
    c.nodes = [SimNode("x1", ("gpu", "lab"), 32, 1000, (GresDef("A30", 4),))]
    c.partition_fields = {"lab": {"PreemptMode": "OFF"}}
    by_name = {p.partition: p.preempt_mode for p in partition_charges(render.scontrol_partitions(c).stdout)}
    assert by_name == {"gpu": "REQUEUE", "lab": "OFF"}
