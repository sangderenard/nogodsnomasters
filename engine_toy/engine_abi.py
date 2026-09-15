"""THE ENGINE LOOP'S ABI: what crosses the boundary, as flat spans.

This is a CONTRACT, not a compiler and not a build. It states the storage
that a compiled engine loop reads and writes, in the shape the native
lane actually needs, so that the lowering has something declared to
lower against instead of a graph of Python objects.

WHY THIS FILE EXISTS. `compile_contract.py` already says it: most of
what looked like the engine sim being uncompilable was the absence of a
declared boundary. It also names the remaining blocker exactly --
`EngineCycleSim` is not declared because "its state is a graph of Python
objects rather than typed spans". This file is the declaration of those
spans. Nothing here asserts a layout the classes do not have; it states
the layout the NATIVE side will own, which is a different thing and the
reason it can be written at all.

THE SHAPE, copied from turing's `vehicle_balloon_tire_native.py` and
`structure_native.py` rather than invented:

  * A Python loop around a compiled kernel is not a compiled program.
    The ballistics lane measured 42 us of marshalling around a 2 us
    kernel. So the loops go INSIDE -- over substeps AND over lanes --
    and arrays are the interface.
  * `state` is a flat tuple of NAMED SCALARS. `parameters` is a flat
    dict of floats. Nothing else crosses. That is what "supply inputs
    and receive outputs in Python" means concretely.
  * The lane (engine) index is the OUTER extent, with a fixed stride:
        state[lane * STATE_STRIDE + slot]
    exactly as the balloon tire indexes `state + w*TIRE_STATE_STRIDE`.

A BATCH IS ONE TOPOLOGY. The stride is a compile-time constant, which is
why the indexing above works at all. A 60-degree V6 and a 14-cylinder
marine two-stroke differ in cylinder count AND circuit count, so they
cannot share a stride. `topology_signature()` is what buckets them: one
compiled assembly per distinct signature, N lanes batched inside it. The
balloon tire has the same constraint and gets it for free, because its
four wheels are the same mesh.

TWO SCALES, AND THEY ARE NOT THE SAME BOUNDARY.

  INTERIOR   what the machine is doing inside itself: crank, cylinders,
             fluid circuits, valve timing, the torque solver's shafts.
             Stiff, phase-locked, and NECESSARILY ONE TIME ZONE -- a
             timing drive carries phase, not just rate, so no adaptor
             can sit inside it at any stiffness. This is also what a
             compiled assembly privately owns and what batches by
             topology, because it is the part whose stride is fixed.

  EXTERIOR   what the machine presents to everything else: its output
             shaft, its ports, what it is bolted to. Small, and the ONLY
             place couplings live -- so the only place a time gradient
             can exist at all. The adaptors (differential, clutch,
             converter) sit here by definition.

The split is not cosmetic. It says which state crosses between machines
(exterior) and which is private to one compiled unit (interior); it says
what the time field applies to (exterior gradients, interiors atomic);
and it says what a runner needs in order to take a machine off your
hands (the interior span plus the exterior inputs, nothing else) -- which
is exactly why offloading is possible: an interior is a pure function of
its own state and its exterior boundary.

AN ENGINE IS ONE KIND OF MACHINE. `machines.py` makes the general case
-- a power pack, a scissor lift, a turret build the same node and edge
documents, so everything downstream already works on them. The interior
spans below are therefore derived from the GRAPH wherever possible, and
only the genuinely engine-specific ones (cylinders, crank) are gated on
an architecture being present.

WHAT THIS DOES NOT DECLARE: TIME. Time velocity is not an engine
property and appears nowhere below. The engine is told a window and a
step count, advances, and publishes what it cost (`proc_ns`) and what
its own stability limit is (`dt_limit_s`). It has no opinion about why
its window is the size it is. Deciding that belongs to the dt system,
which is universal and serves every sim -- not to this ABI, which serves
one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpanDecl:
    """One declared span of the native state vector.

    `count` is per LANE. `slot` is the offset within a lane's stride.
    """

    name: str
    count: int
    slot: int
    unit: str
    note: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        """The flat scalar names this span contributes, in storage order."""
        if self.count == 1:
            return (self.name,)
        return tuple(f"{self.name}_{i}" for i in range(self.count))


@dataclass(frozen=True)
class EngineGraphABI:
    """The complete boundary for one compiled assembly, at both scales."""

    topology: str
    lanes: int
    spans: tuple[SpanDecl, ...]           # INTERIOR: private, batched
    parameters: dict[str, float]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    #: EXTERIOR: the state that crosses to other machines. Small by
    #: design -- if this is large, something interior has leaked out.
    exterior: tuple[SpanDecl, ...] = ()
    kind: str = "engine"

    @property
    def exterior_stride(self) -> int:
        return sum(s.count for s in self.exterior)

    @property
    def exterior_names(self) -> tuple[str, ...]:
        out: list[str] = []
        for span in self.exterior:
            out.extend(span.names)
        return tuple(out)

    @property
    def state_stride(self) -> int:
        """Scalars per lane -- the compile-time constant the native side
        indexes with (`state[lane * stride + slot]`)."""
        return sum(s.count for s in self.spans)

    @property
    def state_scalar_count(self) -> int:
        return self.state_stride * self.lanes

    @property
    def state_names(self) -> tuple[str, ...]:
        """Every state scalar of lane 0, in storage order."""
        out: list[str] = []
        for span in self.spans:
            out.extend(span.names)
        return tuple(out)

    def slot_of(self, name: str) -> int:
        return self.state_names.index(name)

    def receipt(self) -> dict[str, Any]:
        """What this boundary is, flat enough to print or diff."""
        return {
            "topology": self.topology,
            "lanes": self.lanes,
            "state_stride": self.state_stride,
            "state_scalar_count": self.state_scalar_count,
            "spans": [(s.name, s.count, s.slot, s.unit) for s in self.spans],
            "parameters": len(self.parameters),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "kind": self.kind,
            "exterior_stride": self.exterior_stride,
            "exterior": [(s.name, s.count, s.unit) for s in self.exterior],
        }


def topology_signature(engine, graph=None) -> str:
    """What makes two engines shareable in one compiled assembly.

    Everything that changes a STRIDE has to be in here, and nothing that
    does not. Cylinder count, circuit count and node count change the
    state layout; bore, bmep and firing order do not -- those are
    parameters, and parameters are exactly what a batch varies.
    """
    if graph is None:
        graph = _graph_for(engine)
    nodes = graph["nodes"] if isinstance(graph, dict) else graph[0]
    circuits = _circuits_for(engine, graph)
    arch = getattr(engine, "architecture", None)
    if arch is None:
        # a machine: its shape is its graph, which is all a stride needs
        return f"machine.circ{len(circuits)}.node{len(nodes)}"
    return (
        f"cyl{arch.cylinders}"
        f".banks{arch.banks}"
        f".{'2' if arch.two_stroke else '4'}stroke"
        f".{'rotary' if arch.rotary else 'recip'}"
        f".cam{max(1, int(getattr(arch, 'camshaft_count', 1) or 1))}"
        f".circ{len(circuits)}"
        f".node{len(nodes)}"
    )


def _graph_for(subject):
    """The node/edge documents this subject builds. An engine goes
    through drivetrain_graph; a machine builds its own -- and they are
    the same documents, which is the whole reason this works for both."""
    if hasattr(subject, "build_graph"):
        return subject.build_graph()
    import drivetrain_graph as dg

    return dg.build_drivetrain_graph(subject)


def _circuits_for(subject, graph=None):
    import drivetrain_graph as dg

    return dg._discover_fluid_circuits(graph if graph is not None else _graph_for(subject))


def engine_graph_abi(engine, *, lanes: int = 1) -> EngineGraphABI:
    """Declare the complete per-lane state and parameter ABI for one
    engine topology.

    Mirrors `balloon_tire_graph_abi(config)`: topology is baked in, the
    state is named scalars, the parameters are floats, and the lane index
    is the outer extent.
    """
    graph = _graph_for(engine)
    nodes = graph["nodes"] if isinstance(graph, dict) else graph[0]
    circuits = _circuits_for(engine, graph)
    arch = getattr(engine, "architecture", None)
    n_cyl = max(int(arch.cylinders), 0) if arch is not None else 0
    n_circ = len(circuits)
    n_node = len(nodes)

    spans: list[SpanDecl] = []
    slot = 0

    def add(name: str, count: int, unit: str, note: str = "") -> None:
        nonlocal slot
        if count <= 0:
            return
        spans.append(SpanDecl(name=name, count=count, slot=slot, unit=unit, note=note))
        slot += count

    # ---- the crank itself: engine-only, absent on other machines -----
    if arch is not None:
        add("crank_angle_deg", 1, "deg", "total crank angle, not wrapped")
        add("crank_omega", 1, "rad/s")

    # ---- per cylinder -----------------------------------------------
    add("cyl_last_fire_deg", n_cyl, "deg", "total crank deg at last firing")
    add("cyl_last_strength", n_cyl, "frac")
    add("cyl_knock_accum", n_cyl, "frac")
    add("cyl_burned_frac", n_cyl, "frac")
    add("cyl_valve_factor", n_cyl, "frac", "valve/carbon flow derate")
    add("cyl_wall_temp_k", n_cyl, "K")

    # ---- per fluid circuit ------------------------------------------
    add("circuit_pressure_pa", n_circ, "Pa")
    add("circuit_temp_k", n_circ, "K")
    add("circuit_fill_frac", n_circ, "frac")

    # ---- per graph node ---------------------------------------------
    add("node_omega", n_node, "rad/s", "the torque solver's own shaft speeds")

    # ---- lumped subsystems ------------------------------------------
    add("crankcase_pressure_pa", 1, "Pa")
    add("crankcase_blowby_kg_s", 1, "kg/s")
    add("catalyst_temp_k", 1, "K")
    add("oil_temp_k", 1, "K")
    add("coolant_temp_k", 1, "K")

    # ---- EXTERIOR: the whole of what crosses to other machines -------
    # Deliberately tiny. Everything above is private to this assembly;
    # only these cross a coupling, and only these can sit either side of
    # a time gradient.
    ext: list[SpanDecl] = []
    eslot = 0

    def add_ext(name: str, count: int, unit: str, note: str = "") -> None:
        nonlocal eslot
        ext.append(SpanDecl(name=name, count=count, slot=eslot, unit=unit, note=note))
        eslot += count

    add_ext("shaft_omega", 1, "rad/s", "what the coupling sees turning")
    add_ext("shaft_torque_nm", 1, "Nm", "what it delivers through the coupling")
    add_ext("load_omega", 1, "rad/s", "the other side of the coupling")
    add_ext("port_pressure_pa", 1, "Pa", "supply/return boundary")
    add_ext("port_flow_kg_s", 1, "kg/s")
    add_ext("shell_temp_k", 1, "K", "what the bay feels")

    parameters = _parameters_for(engine, n_cyl=n_cyl, n_circ=n_circ)

    inputs = (
        "dt_window_s",
        "substep_count",
        "throttle",
        "clutch_frac",
        "brake_load_nm",
        "ambient_k",
    )
    outputs = (
        "rpm",
        "torque_nm",
        "advanced_s",           # world time this lane actually covered
        "proc_ns",              # what it cost -- published, never interpreted here
        "dt_limit_s",           # this lane's own stability floor, published
                                # so a scheduler can choose a step count
    )

    return EngineGraphABI(
        topology=topology_signature(engine, graph),
        lanes=int(lanes),
        spans=tuple(spans),
        parameters=parameters,
        inputs=inputs,
        outputs=outputs,
        exterior=tuple(ext),
        kind="engine" if arch is not None else "machine",
    )


def _parameters_for(engine, *, n_cyl: int, n_circ: int) -> dict[str, float]:
    """Runtime floats. These are what a batch VARIES between lanes; they
    never change a stride, which is what keeps them out of the topology
    signature."""
    arch = getattr(engine, "architecture", None)
    if arch is None:
        return {"mass_kg": float(getattr(engine, "mass_kg", 0.0) or 0.0)}
    params: dict[str, float] = {
        "displacement_m3": float(engine.displacement_l) / 1000.0,
        "bmep_pa": float(engine.bmep_pa),
        "braking_bmep_pa": float(engine.braking_bmep_pa),
        "inertia_kg_m2": float(engine.inertia_kg_m2),
        "idle_rpm": float(engine.idle_rpm),
        "redline_rpm": float(engine.redline_rpm),
        "torque_peak_rpm": float(engine.torque_peak_rpm),
        "power_peak_rpm": float(engine.power_peak_rpm),
        "combustion_efficiency": float(engine.combustion_efficiency),
        "coupling_efficiency": float(engine.coupling_efficiency),
        "bore_m": float(arch.bore_m),
        "stroke_m": float(arch.stroke_m),
        "rod_length_m": float(arch.rod_length_m),
        "compression_ratio": float(arch.compression_ratio),
        "cycle_degrees": float(arch.cycle_degrees),
    }
    # The firing schedule is a parameter, not a topology fact: two
    # engines with the same cylinder count can fire on different
    # schedules and still share one compiled assembly. An odd-fire crank
    # is exactly that case.
    for i, angle in enumerate(arch.slot_angles_deg()):
        params[f"firing_angle_{i}_deg"] = float(angle)
    return params


if __name__ == "__main__":
    import engines

    for identity in ("alfa-busso-v6-3000-12v", "vw-vr6-2800-12v",
                     "buick-231-oddfire-v6-1975", "amc-258-jeep-i6"):
        abi = engine_graph_abi(engines.get(identity), lanes=4)
        r = abi.receipt()
        print(f"{identity}")
        print(f"   topology   {r['topology']}")
        print(f"   stride     {r['state_stride']} scalars/lane"
              f"   x{r['lanes']} lanes = {r['state_scalar_count']}")
        print(f"   parameters {r['parameters']}")
        print(f"   inputs     {', '.join(r['inputs'])}")
