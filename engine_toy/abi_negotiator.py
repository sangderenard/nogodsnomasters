"""THE ABI NEGOTIATOR: who can share a compiled assembly, and on what terms.

A batch is one topology, because the lane stride is a compile-time
constant -- that is the whole reason `state[lane*STRIDE + slot]` works.
So before anything is lowered, something has to decide which instances
can ride in one assembly together and which need their own. That is this
file. It negotiates; it does not compile, and it does not know what an
engine is beyond the ABI it was handed.

WHAT IS NEGOTIATED

  topology   Instances with different strides cannot share, full stop.
             Bucketed by signature, one assembly per bucket.
  parameters Lanes VARY parameters -- that is what a batch is for -- but
             they must agree on the parameter KEY SET, because the
             parameter block is a stride too. Same keys, different
             floats, is exactly right. Different keys is a different
             assembly.
  substeps   One K for the whole batch, since the lane loop runs in
             lockstep. Chosen so that every lane stays inside its OWN
             published stability limit:
                 K >= max(window_lane / dt_limit_lane)
             Nobody's step is coarsened to fit a budget. A lane that
             needs finer steps gets a shorter window instead -- less
             world time per frame, not a worse integration.

WHY IT REPORTS RATHER THAN COERCES

An instance that cannot join a batch is not an error and must not be
silently reshaped to fit; it gets its own assembly and a stated reason.
Quietly padding mismatched topologies to a common stride would buy
uniformity with wasted width and hide the fact that two things are not
the same machine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Assembly:
    """One compiled unit: a topology, and the lanes that ride in it."""

    topology: str
    members: tuple[str, ...]
    state_stride: int
    parameter_keys: tuple[str, ...]

    @property
    def lanes(self) -> int:
        return len(self.members)

    @property
    def state_scalar_count(self) -> int:
        return self.state_stride * self.lanes


@dataclass(frozen=True)
class Rejection:
    """Why one instance could not join the assembly it looked like it
    belonged to. Stated, never smoothed over."""

    member: str
    wanted: str
    reason: str


@dataclass
class Negotiation:
    assemblies: tuple[Assembly, ...] = ()
    rejections: tuple[Rejection, ...] = ()

    def receipt(self) -> dict[str, Any]:
        return {
            "assemblies": [
                {"topology": a.topology, "lanes": a.lanes,
                 "stride": a.state_stride, "scalars": a.state_scalar_count,
                 "members": list(a.members)}
                for a in self.assemblies
            ],
            "rejections": [(r.member, r.wanted, r.reason) for r in self.rejections],
            "total_lanes": sum(a.lanes for a in self.assemblies),
            "total_assemblies": len(self.assemblies),
        }


def negotiate(named_abis: Mapping[str, Any]) -> Negotiation:
    """Bucket instances into assemblies they can actually share.

    `named_abis` maps an instance name to anything exposing `topology`,
    `state_stride` and `parameters` -- so this works for an engine ABI, a
    member bank, or a tire, without knowing which it has.
    """
    buckets: dict[str, list[tuple[str, Any]]] = {}
    for name, abi in named_abis.items():
        buckets.setdefault(str(abi.topology), []).append((name, abi))

    assemblies: list[Assembly] = []
    rejections: list[Rejection] = []

    for topology, members in buckets.items():
        # the first member sets the terms; the rest must match them
        lead_name, lead = members[0]
        lead_keys = tuple(sorted(lead.parameters))
        lead_stride = int(lead.state_stride)
        joined = [lead_name]
        for name, abi in members[1:]:
            if int(abi.state_stride) != lead_stride:
                rejections.append(Rejection(
                    name, topology,
                    f"stride {int(abi.state_stride)} != {lead_stride} despite matching "
                    f"topology -- the signature is missing something that moves storage"))
                continue
            keys = tuple(sorted(abi.parameters))
            if keys != lead_keys:
                missing = sorted(set(lead_keys) ^ set(keys))
                rejections.append(Rejection(
                    name, topology,
                    f"parameter keys differ ({len(missing)}: {', '.join(missing[:3])}"
                    f"{'...' if len(missing) > 3 else ''})"))
                continue
            joined.append(name)
        assemblies.append(Assembly(
            topology=topology, members=tuple(joined),
            state_stride=lead_stride, parameter_keys=lead_keys))

    # rejected members still have to run -- each becomes its own assembly
    for rej in rejections:
        abi = named_abis[rej.member]
        assemblies.append(Assembly(
            topology=f"{abi.topology}#solo:{rej.member}",
            members=(rej.member,),
            state_stride=int(abi.state_stride),
            parameter_keys=tuple(sorted(abi.parameters))))

    assemblies.sort(key=lambda a: (-a.lanes, a.topology))
    return Negotiation(tuple(assemblies), tuple(rejections))


@dataclass(frozen=True)
class LaneSchedule:
    """The terms the whole batch runs on for one frame."""

    substeps: int
    windows_s: tuple[float, ...]
    #: the step each lane will actually take -- window/K, per lane
    steps_s: tuple[float, ...]
    limited_by: str

    def respects(self, dt_limits_s: Sequence[float]) -> bool:
        """Every lane inside its own published stability floor."""
        return all(step <= limit + 1e-15
                   for step, limit in zip(self.steps_s, dt_limits_s))


def negotiate_substeps(windows_s: Sequence[float],
                       dt_limits_s: Sequence[float],
                       *, max_substeps: int = 4096) -> LaneSchedule:
    """Pick the ONE substep count the batch runs at.

    Lockstep is what a batch needs; stability is what each lane needs.
    Both are satisfiable at once because they constrain different things:
    K is shared, the window is per lane. K is the smallest count that
    keeps every lane's own step inside its own limit.

    If that would exceed `max_substeps`, K is capped and the WINDOWS are
    shortened to compensate -- the lanes advance less world time rather
    than taking a step they cannot survive. Which lane ends up slowest is
    then a fact about the machine, not a policy decision made here.
    """
    if not windows_s:
        return LaneSchedule(0, (), (), "empty")
    if len(windows_s) != len(dt_limits_s):
        raise ValueError("one dt_limit per window is required")

    need = 1
    for window, limit in zip(windows_s, dt_limits_s):
        if limit <= 0.0:
            continue
        need = max(need, int(math.ceil(float(window) / float(limit) - 1e-12)))

    limited_by = "stability"
    k = need
    windows = tuple(float(w) for w in windows_s)
    if k > max_substeps:
        k = max_substeps
        limited_by = "substep-cap"
        # shorten every lane's window so its step still fits its limit
        windows = tuple(
            min(w, (limit * k) if limit > 0.0 else w)
            for w, limit in zip(windows, dt_limits_s)
        )
    steps = tuple(w / k for w in windows)
    return LaneSchedule(substeps=k, windows_s=windows, steps_s=steps,
                        limited_by=limited_by)


if __name__ == "__main__":
    import engines
    from engine_abi import engine_graph_abi

    # two of the same engine plus three different ones: the pair should
    # share an assembly, the rest should each get their own
    fleet = {
        "bay_1": engine_graph_abi(engines.get("alfa-busso-v6-3000-12v")),
        "bay_2": engine_graph_abi(engines.get("alfa-busso-v6-3000-12v")),
        "bay_3": engine_graph_abi(engines.get("vw-vr6-2800-12v")),
        "pump": engine_graph_abi(engines.get("buick-231-oddfire-v6-1975")),
        "genset": engine_graph_abi(engines.get("amc-258-jeep-i6")),
    }
    result = negotiate(fleet)
    for a in result.assemblies:
        print(f"{a.lanes} lane(s)  stride {a.state_stride:5d}  {a.topology}")
        print(f"            members: {', '.join(a.members)}")
    for r in result.rejections:
        print(f"  rejected {r.member}: {r.reason}")

    print()
    sched = negotiate_substeps(windows_s=[1 / 60.0] * 3,
                               dt_limits_s=[1 / 500.0, 1 / 2000.0, 1 / 800.0])
    print(f"K={sched.substeps} ({sched.limited_by}); "
          f"steps {[f'{s*1e6:.0f}us' for s in sched.steps_s]}")
    print("respects every lane's own limit:",
          sched.respects([1 / 500.0, 1 / 2000.0, 1 / 800.0]))
