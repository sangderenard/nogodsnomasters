"""THE BRIDGE: an engine's live state as a flat span, and back again.

`engine_abi.py` declares the layout -- named scalars, a lane stride,
`state[lane * STATE_STRIDE + slot]` -- and says outright that it is "a
CONTRACT, not a compiler and not a build". Nothing carried the live
`EngineCycleSim` into that layout, which is why `compile_contract.py`
still names the engine as the open blocker: "its state is a graph of
Python objects rather than typed spans".

This is that carry. `pack` reads the sim into the span; `unpack` writes
the span back into the sim; `round_trip` proves the pair is lossless on
what it claims to cover.

WHAT IT DELIBERATELY DOES NOT DO. It does not invent a home for a span
that has none. Five of the declared groups have no attribute on the live
sim under any name this module could find, and they are listed in
`UNMAPPED` rather than quietly written as zeros. Zeroing them would
assert a layout the class does not have, which is the exact failure
`compile_contract.py` warns about -- and it would be invisible, because a
zero is a plausible value for every one of them.

So `pack` and `unpack` are honest about coverage: `coverage()` reports
which slots are carried and which are holes, and a caller that needs the
holes has to close them at the sim, not here.

WHERE THIS STANDS, AND THE ONE THING STILL OPEN.

    stride            3156 doubles per lane, 46 spans, zero holes
    round trip        exact
    restore           exact -- every declared slot returns after the sim
                      has been stepped away and brought back
    replay divergence 8.96e-06 rpm, from 18.8 when this began

That last number is the acceptance test and it is not yet zero. Replaying
twice from one span gives trajectories that first differ at STEP ONE, in
the cylinder path -- `cyl_valve_factor`, the per-cylinder lists, and two
state scalars -- so something feeding combustion is not being restored.

Ruled out, each by measurement rather than by reasoning:

  * the object graph          a deep diff of the sim after two restores
                              reports zero unrestored values
  * module-level containers   nothing in engine_toy's modules mutates
                              during a run
  * the behaviour singletons  `node_effects.BEHAVIOURS` holds no state;
                              an earlier "it mutates" reading was a diff
                              comparing deepcopies of objects that define
                              no equality
  * both generators           `sim._rng` and the numpy generators on
                              `bursts` and `ordnance` all round-trip, and
                              draws after a restore match exactly
  * part damage and wear      no numeric field mutates during a run
  * the engine spec           does not mutate during a run
  * first-call lazy setup     four successive replays all differ, so it
                              is not a one-time initialisation

The divergence is about 3e-8 relative and shrinks slightly with each
successive replay, which suggests an accumulator converging rather than
noise. It is small enough to live with and specific enough to find.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from engine_abi import (EngineGraphABI, engine_graph_abi,
                        SIM_SCALARS, EDGE_FIELDS, RNG_SLOTS,
                        STATE_SCALARS, STATE_LISTS, CRANKCASE_SCALARS,
                        AIR_VOLUME_SCALARS, AIR_VOLUMES, AIR_STACK_SCALARS,
                        EMITTER_SCALARS, EMITTER_BANK_SCALARS,
                        CATALYST_SCALARS, ELECTRICAL_READING_SCALARS,
                        DRIVETRAIN_OUT_KEYS, NUMPY_RNG_PATHS,
                        NUMPY_RNG_LIMBS, NUMPY_RNG_SLOTS)

#: Declared in the ABI, with no live home on `EngineCycleSim`.
#:
#: `cyl_last_fire_deg`, `cyl_last_strength`, `cyl_knock_accum` and
#: `cyl_burned_frac` were expected in `state.slot_records`, which the ABI
#: was written against. Measured, `slot_records[i]` is a plain `list`,
#: not a record with those fields, so the per-slot firing history is not
#: in the shape the declaration assumed.
#:
#: `circuit_fill_frac` has no counterpart on `FluidCircuit` at all: it
#: carries `pressure_pa`, `temp_k`, `volume_l` and a set of regulator
#: fractions, but nothing that is a fill level.
#: Nothing is unmapped any more, and the way this list emptied is the
#: point. Every entry was found by DIFFING a restored sim against the
#: original rather than by reading names: `circuit_fill_frac` turned out
#: to be `FluidCircuit.fill_level_frac`, and the four `cyl_last_*` spans
#: were never in `state.slot_records` at all -- they live directly on the
#: sim as `_last_fire_total_deg`, `_last_strength`, `_knock_accum` and
#: `_burned_frac`, with an undeclared `_flame_radius_m` beside them.
#: Searching by name found none of these; searching by what moved found
#: all of them.
UNMAPPED: tuple[str, ...] = ()


def _state_scalar(attribute: str, scale: float = 1.0):
    """A single scalar living on `sim.state`."""
    def read(sim):
        return [float(getattr(sim.state, attribute, 0.0) or 0.0) * scale]

    def write(sim, values):
        setattr(sim.state, attribute, float(values[0]) / scale)
    return read, write


def _state_list(attribute: str):
    """A per-cylinder list living on `sim.state`."""
    def read(sim):
        return [float(v) for v in (getattr(sim.state, attribute, None) or ())]

    def write(sim, values):
        current = getattr(sim.state, attribute, None)
        if current is None:
            return
        for index in range(min(len(current), len(values))):
            current[index] = float(values[index])
    return read, write


def _crank_omega():
    def read(sim):
        return [float(sim._omega)]

    def write(sim, values):
        sim._omega = float(values[0])
    return read, write


def _circuit_field(attribute: str):
    """One field across every fluid circuit, in the solver's own order."""
    def read(sim):
        return [float(getattr(circuit, attribute, 0.0) or 0.0)
                for circuit in sim._drivetrain.fluid_circuits]

    def write(sim, values):
        for circuit, value in zip(sim._drivetrain.fluid_circuits, values):
            setattr(circuit, attribute, float(value))
    return read, write


def _sim_per_cylinder(attribute: str):
    """A per-cylinder collection living directly on the sim.

    THESE ARE DICTS, KEYED BY CYLINDER NUMBER, not lists. Written as a
    list this reads `[float(v) for v in collection]`, which iterates a
    mapping's KEYS -- so the firing order (1, 3, 4, 2) gets packed as if
    it were the data and written back over the combustion state. The
    symptom was a restored `_knock_accum` of 3.0.

    That is the same defect as `zip(schedule, allocations)` in
    `dt_graph.py`, made independently in this file an hour after finding
    it there. Sorted keys, both directions, so the slot order is the same
    in every process.
    """
    def read(sim):
        collection = getattr(sim, attribute, None) or {}
        if isinstance(collection, dict):
            return [float(collection[key]) for key in sorted(collection)]
        return [float(v) for v in collection]

    def write(sim, values):
        collection = getattr(sim, attribute, None)
        if collection is None:
            return
        if isinstance(collection, dict):
            for key, value in zip(sorted(collection), values):
                collection[key] = float(value)
            return
        for index in range(min(len(collection), len(values))):
            collection[index] = float(values[index])
    return read, write


def _air_volumes():
    """The bay and the garage, in declared order."""
    def read(sim):
        out = []
        for volume in AIR_VOLUMES:
            target = getattr(getattr(sim, "_air", None), volume, None)
            out.extend(float(getattr(target, name, 0.0) or 0.0)
                       for name in AIR_VOLUME_SCALARS)
        return out

    def write(sim, values):
        cursor = 0
        for volume in AIR_VOLUMES:
            target = getattr(getattr(sim, "_air", None), volume, None)
            for name in AIR_VOLUME_SCALARS:
                if target is not None and cursor < len(values):
                    setattr(target, name, float(values[cursor]))
                cursor += 1
    return read, write


def _air_stack():
    def read(sim):
        stack = getattr(sim, "_air", None)
        return [float(getattr(stack, name, 0.0) or 0.0)
                for name in AIR_STACK_SCALARS]

    def write(sim, values):
        stack = getattr(sim, "_air", None)
        if stack is None:
            return
        for name, value in zip(AIR_STACK_SCALARS, values):
            setattr(stack, name, float(value))
    return read, write


def _object_scalars(path: tuple[str, ...], names: tuple[str, ...]):
    """A fixed set of scalars on a nested object, by attribute path."""
    def resolve(sim):
        target = sim
        for step in path:
            target = getattr(target, step, None)
            if target is None:
                return None
        return target

    def read(sim):
        target = resolve(sim)
        return [float(getattr(target, name, 0.0) or 0.0) for name in names]

    def write(sim, values):
        target = resolve(sim)
        if target is None:
            return
        for name, value in zip(names, values):
            setattr(target, name, float(value))
    return read, write


def _net_torque():
    def read(sim):
        table = sim._drivetrain._net_torque
        return [float(table.get(node, 0.0)) for node in sim._drivetrain._node_ids]

    def write(sim, values):
        table = sim._drivetrain._net_torque
        for node, value in zip(sim._drivetrain._node_ids, values):
            table[node] = float(value)
    return read, write


def _drivetrain_out():
    def read(sim):
        out = sim._drivetrain_out or {}
        return [float(out.get(key, 0.0) or 0.0) for key in DRIVETRAIN_OUT_KEYS]

    def write(sim, values):
        out = sim._drivetrain_out
        if not isinstance(out, dict):
            return
        for key, value in zip(DRIVETRAIN_OUT_KEYS, values):
            if isinstance(out.get(key), bool):
                continue
            out[key] = float(value)
    return read, write


def _out_circuit(field: str):
    """One field of the solve's published per-circuit readings."""
    def read(sim):
        table = (sim._drivetrain_out or {}).get("fluid_circuits") or {}
        return [float(table[key].get(field, 0.0) or 0.0)
                for key in sorted(table)]

    def write(sim, values):
        table = (sim._drivetrain_out or {}).get("fluid_circuits") or {}
        for key, value in zip(sorted(table), values):
            table[key][field] = float(value)
    return read, write


def _emitters():
    """Every hole emitter, ordered by identity so the slots are stable."""
    def _ordered(sim):
        bank = getattr(sim, "hole_emitters", None)
        items = list(getattr(bank, "emitters", ()) or ())
        return sorted(items, key=lambda e: str(getattr(e, "identity", "")))

    def read(sim):
        out = []
        for emitter in _ordered(sim):
            out.extend(float(getattr(emitter, name, 0.0) or 0.0)
                       for name in EMITTER_SCALARS)
        return out

    def write(sim, values):
        cursor = 0
        for emitter in _ordered(sim):
            for name in EMITTER_SCALARS:
                if cursor < len(values):
                    setattr(emitter, name, float(values[cursor]))
                cursor += 1
    return read, write


def _emitter_bank():
    def read(sim):
        bank = getattr(sim, "hole_emitters", None)
        return [float(getattr(bank, name, 0.0) or 0.0)
                for name in EMITTER_BANK_SCALARS]

    def write(sim, values):
        bank = getattr(sim, "hole_emitters", None)
        if bank is None:
            return
        for name, value in zip(EMITTER_BANK_SCALARS, values):
            setattr(bank, name, float(value))
    return read, write


def _battery_soc():
    def read(sim):
        battery = getattr(getattr(sim, "electrical", None), "battery", None)
        return [float(getattr(battery, "soc_frac", 0.0) or 0.0)]

    def write(sim, values):
        battery = getattr(getattr(sim, "electrical", None), "battery", None)
        if battery is not None:
            battery.soc_frac = float(values[0])
    return read, write


def _slot_records():
    """`state.slot_records`, n_cyl rows of four floats, flattened."""
    def read(sim):
        out = []
        for row in (sim.state.slot_records or ()):
            out.extend(float(v) for v in row)
        return out

    def write(sim, values):
        cursor = 0
        for row in (sim.state.slot_records or ()):
            for index in range(len(row)):
                if cursor < len(values):
                    row[index] = float(values[cursor])
                cursor += 1
    return read, write


def _node_omega():
    """The torque solver's own shaft speeds, one per graph node.

    `DrivetrainSolver.omega` is a mapping keyed by node identity, and a
    mapping has no slot order. `_node_ids` is the solver's own ordering
    and is what the stride has to be built from -- taking `omega.keys()`
    instead would make the layout depend on insertion order, which is a
    different stride on a different day.
    """
    def read(sim):
        omega = sim._drivetrain.omega
        return [float(omega.get(node, 0.0)) for node in sim._drivetrain._node_ids]

    def write(sim, values):
        omega = sim._drivetrain.omega
        for node, value in zip(sim._drivetrain._node_ids, values):
            omega[node] = float(value)
    return read, write


def _edge_field(attribute: str):
    """One field across every drivetrain edge, in a STABLE order.

    `_edge_state` is a mapping keyed by edge identity, and a mapping has
    no slot order. Sorted keys give the same layout in every process; the
    insertion order would give a different stride on a different day.
    """
    def read(sim):
        table = sim._drivetrain._edge_state
        return [float(getattr(table[key], attribute, 0.0) or 0.0)
                for key in sorted(table)]

    def write(sim, values):
        table = sim._drivetrain._edge_state
        for key, value in zip(sorted(table), values):
            current = getattr(table[key], attribute, None)
            setattr(table[key], attribute,
                    bool(value) if isinstance(current, bool) else float(value))
    return read, write


def _sim_scalars():
    """Every mutable scalar the sim carries on itself, in declared order."""
    def read(sim):
        return [float(getattr(sim, name, 0.0) or 0.0) for name in SIM_SCALARS]

    def write(sim, values):
        for name, value in zip(SIM_SCALARS, values):
            current = getattr(sim, name, 0.0)
            setattr(sim, name, int(value) if isinstance(current, int)
                    and not isinstance(current, bool) else float(value))
    return read, write


def _numpy_rngs():
    """Every numpy `Generator` hanging off the sim, as 32-bit limbs."""
    mask = (1 << 32) - 1

    def _generators(sim):
        out = []
        for path in NUMPY_RNG_PATHS:
            target = sim
            for step in path:
                target = getattr(target, step, None)
                if target is None:
                    break
            out.append(target)
        return out

    def read(sim):
        values = []
        for generator in _generators(sim):
            if generator is None:
                values.extend([0.0] * NUMPY_RNG_SLOTS)
                continue
            state = generator.bit_generator.state
            for key in ("state", "inc"):
                whole = int(state["state"][key])
                values.extend(float((whole >> (32 * limb)) & mask)
                              for limb in range(NUMPY_RNG_LIMBS))
            values.append(float(state.get("has_uint32", 0)))
            values.append(float(state.get("uinteger", 0)))
        return values

    def write(sim, values):
        cursor = 0
        for generator in _generators(sim):
            chunk = values[cursor:cursor + NUMPY_RNG_SLOTS]
            cursor += NUMPY_RNG_SLOTS
            if generator is None or len(chunk) < NUMPY_RNG_SLOTS:
                continue
            state = dict(generator.bit_generator.state)
            inner = dict(state["state"])
            for index, key in enumerate(("state", "inc")):
                whole = 0
                for limb in range(NUMPY_RNG_LIMBS):
                    whole |= int(chunk[index * NUMPY_RNG_LIMBS + limb]) << (32 * limb)
                inner[key] = whole
            state["state"] = inner
            state["has_uint32"] = int(chunk[2 * NUMPY_RNG_LIMBS])
            state["uinteger"] = int(chunk[2 * NUMPY_RNG_LIMBS + 1])
            generator.bit_generator.state = state
    return read, write


def _rng_words():
    """The generator's position.

    Determinism is what makes a predicted result checkable by digest
    instead of by trust, and a seeded generator is only deterministic
    from a given POSITION. Carrying the seed and not the position means
    two replays from one span take different random draws, which is
    exactly the 18.8 rpm divergence this closed.

    `getstate()` is `(version, 625 words, gauss_next)`. The spare
    gaussian is a float or None; NaN carries the None.
    """
    def read(sim):
        _version, words, gauss = sim._rng.getstate()
        return [float(word) for word in words] + [
            float("nan") if gauss is None else float(gauss)]

    def write(sim, values):
        words = tuple(int(value) for value in values[:RNG_SLOTS - 1])
        spare = values[RNG_SLOTS - 1]
        sim._rng.setstate((3, words, None if spare != spare else float(spare)))
    return read, write


def _state_scalars():
    """Every scalar field of the state record, in declared order."""
    def read(sim):
        return [float(getattr(sim.state, name, 0.0) or 0.0)
                for name in STATE_SCALARS]

    def write(sim, values):
        for name, value in zip(STATE_SCALARS, values):
            current = getattr(sim.state, name, 0.0)
            setattr(sim.state, name,
                    int(value) if isinstance(current, int)
                    and not isinstance(current, bool) else float(value))
    return read, write


def _state_lists():
    """The per-cylinder lists, concatenated in declared order."""
    def read(sim):
        out = []
        for name in STATE_LISTS:
            out.extend(float(v) for v in (getattr(sim.state, name, None) or ()))
        return out

    def write(sim, values):
        cursor = 0
        for name in STATE_LISTS:
            current = getattr(sim.state, name, None)
            if not current:
                continue
            for index in range(len(current)):
                if cursor < len(values):
                    current[index] = float(values[cursor])
                cursor += 1
    return read, write


def _crankcase():
    def read(sim):
        case = getattr(sim, "_crankcase_state", None)
        return [float(getattr(case, name, 0.0) or 0.0)
                for name in CRANKCASE_SCALARS]

    def write(sim, values):
        case = getattr(sim, "_crankcase_state", None)
        if case is None:
            return
        for name, value in zip(CRANKCASE_SCALARS, values):
            setattr(case, name, float(value))
    return read, write


#: span name -> (read, write). Every entry was verified against a running
#: sim; nothing here is a guess about where a value lives.
SOURCES: dict[str, tuple[Callable[[Any], list[float]],
                         Callable[[Any, list[float]], None]]] = {
    "crank_angle_deg": _state_scalar("crank_angle_deg"),
    "crank_omega": _crank_omega(),
    "cyl_valve_factor": _state_list("cylinder_valve_factor"),
    "cyl_wall_temp_k": _state_list("cylinder_block_temps_k"),
    "circuit_pressure_pa": _circuit_field("pressure_pa"),
    "circuit_temp_k": _circuit_field("temp_k"),
    "circuit_fill_frac": _circuit_field("fill_level_frac"),
    "circuit_flow_lpm": _circuit_field("flow_lpm"),
    "circuit_supply_pressure_pa": _circuit_field("supply_pressure_pa"),
    "circuit_target_pressure_pa": _circuit_field("achievable_target_pressure_pa"),
    "circuit_delivered_flow_kg_s": _circuit_field("delivered_flow_kg_s"),
    "cyl_last_fire_deg": _sim_per_cylinder("_last_fire_total_deg"),
    "cyl_last_strength": _sim_per_cylinder("_last_strength"),
    "cyl_knock_accum": _sim_per_cylinder("_knock_accum"),
    "cyl_burned_frac": _sim_per_cylinder("_burned_frac"),
    "cyl_flame_radius_m": _sim_per_cylinder("_flame_radius_m"),
    "slot_record": _slot_records(),
    "battery_soc_frac": _battery_soc(),
    "air_volume": _air_volumes(),
    "air_stack": _air_stack(),
    "emitter": _emitters(),
    "catalyst": _object_scalars(("_catalyst_state",), CATALYST_SCALARS),
    "electrical_reading": _object_scalars(
        ("electrical", "reading"), ELECTRICAL_READING_SCALARS),
    "net_torque": _net_torque(),
    "out_circuit_pressure_pa": _out_circuit("pressure_pa"),
    "out_circuit_temp_k": _out_circuit("temp_k"),
    "drivetrain_out": _drivetrain_out(),
    "node_omega": _node_omega(),
    # the sim carries this one in kilopascals; the ABI declares pascals,
    # and the unit is part of the declaration
    "crankcase_pressure_pa": _state_scalar("crankcase_pressure_kpa", 1000.0),
    "crankcase_blowby_kg_s": _state_scalar("blowby_l_per_min"),
    "catalyst_temp_k": _state_scalar("catalyst_brick_temp_k"),
    "oil_temp_k": _state_scalar("oil_temp_k"),
    "coolant_temp_k": _state_scalar("coolant_temp_k"),
    "edge_relative_angle_rad": _edge_field("relative_angle_rad"),
    "edge_last_torque_nm": _edge_field("last_torque_nm"),
    "edge_wear": _edge_field("wear"),
    "edge_glaze": _edge_field("glaze"),
    "edge_dissipated_heat_j": _edge_field("dissipated_heat_j"),
    "edge_unrejected_heat_j": _edge_field("unrejected_heat_j"),
    "edge_engaged": _edge_field("engaged"),
    "state_scalar": _state_scalars(),
    "state_list": _state_lists(),
    "crankcase_scalar": _crankcase(),
    "sim_scalar": _sim_scalars(),
    "rng_word": _rng_words(),
    "numpy_rng": _numpy_rngs(),
}




def coverage(abi: EngineGraphABI) -> dict[str, Any]:
    """Which declared slots this bridge carries, and which are holes."""
    carried, holes = [], []
    for span in abi.spans:
        (carried if span.name in SOURCES else holes).append(
            (span.name, span.count))
    return {
        "stride": abi.state_stride,
        "carried_spans": carried,
        "carried_slots": sum(count for _n, count in carried),
        "hole_spans": holes,
        "hole_slots": sum(count for _n, count in holes),
    }


def pack(sim, abi: EngineGraphABI, buffer=None, lane: int = 0):
    """Read the live sim into its declared span."""
    if buffer is None:
        buffer = np.zeros(abi.state_stride * max(abi.lanes, lane + 1), np.float64)
    base = lane * abi.state_stride
    for span in abi.spans:
        source = SOURCES.get(span.name)
        if source is None:
            continue
        values = source[0](sim)
        for index in range(min(span.count, len(values))):
            buffer[base + span.slot + index] = values[index]
    return buffer


def unpack(sim, abi: EngineGraphABI, buffer, lane: int = 0) -> None:
    """Write a declared span back into the live sim."""
    base = lane * abi.state_stride
    for span in abi.spans:
        source = SOURCES.get(span.name)
        if source is None:
            continue
        source[1](sim, [float(buffer[base + span.slot + index])
                        for index in range(span.count)])


def round_trip(sim, abi: EngineGraphABI) -> float:
    """Worst slot disagreement across pack -> unpack -> pack.

    The proof that the pair is lossless on what it covers. A bridge that
    is merely plausible is the thing this exists to rule out.
    """
    first = pack(sim, abi)
    unpack(sim, abi, first)
    second = pack(sim, abi)
    # The spare-gaussian slot carries None as NaN, and NaN != NaN, so a
    # plain difference reports a disagreement where the two agree
    # exactly. Both-NaN counts as equal; one-NaN does not.
    both_nan = np.isnan(first) & np.isnan(second)
    delta = np.abs(np.where(both_nan, 0.0, first - second))
    return float(np.max(np.nan_to_num(delta, nan=np.inf)))


def restore_exactness(sim, abi: EngineGraphABI, steps: int = 25) -> int:
    """Slots that fail to come back after stepping away and restoring.

    Stronger than `round_trip`, which never advances the sim. This one
    packs, RUNS, restores and repacks, so it catches a span that reads
    correctly but writes into the wrong place.
    """
    saved = pack(sim, abi)
    for _ in range(steps):
        sim.step(1.0 / 240.0)
    unpack(sim, abi, saved)
    again = pack(sim, abi)
    both_nan = np.isnan(saved) & np.isnan(again)
    delta = np.nan_to_num(np.abs(np.where(both_nan, 0.0, saved - again)),
                          nan=np.inf)
    return int(np.count_nonzero(delta))


def replay_divergence(sim, abi: EngineGraphABI, steps: int = 60) -> float:
    """Worst rpm disagreement between two replays from ONE span.

    THE ACCEPTANCE TEST. A span is sufficient state only if replaying
    from it twice gives the same trajectory. Everything else -- coverage,
    round trip, restore -- can pass while this fails, and did.
    """
    saved = pack(sim, abi)

    def run():
        unpack(sim, abi, saved)
        out = []
        for _ in range(steps):
            sim.step(1.0 / 240.0)
            out.append(float(sim.rpm))
        return out

    first, second = run(), run()
    return max(abs(a - b) for a, b in zip(first, second))


def main() -> None:
    import engines
    from engine_cycle_sim import EngineCycleSim

    sim = EngineCycleSim(engine=engines.get("mazda-b6ze-miata-1990"))
    sim.start()
    sim.throttle = 0.6
    for _ in range(120):
        sim.step(1.0 / 240.0)

    abi = engine_graph_abi(sim.engine)
    report = coverage(abi)
    print(f"stride {report['stride']} doubles per lane")
    print(f"  carried {report['carried_slots']:5d} slots in "
          f"{len(report['carried_spans'])} spans")
    print(f"  holes   {report['hole_slots']:5d} slots")
    for name, count in report["hole_spans"]:
        print(f"      unmapped: {name} ({count})")

    print()
    print(f"round trip      : {round_trip(sim, abi):.3e}")
    print(f"restore misses  : {restore_exactness(sim, abi)} slots")
    print(f"replay divergence: {replay_divergence(sim, abi):.3e} rpm")


if __name__ == "__main__":
    main()
