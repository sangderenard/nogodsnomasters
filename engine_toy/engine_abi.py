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


#: EVERY MUTABLE SCALAR THE SIM CARRIES ON ITSELF. Measured, not guessed:
#: 53 of them, and leaving them out is why a span could round-trip
#: perfectly and still fail to reproduce a trajectory. Accumulators
#: (`_physics_accum_s`), rate estimators (`_rpm_rate_per_s`), timers
#: (`_surge_timer`) and the driver's own controls all live here.
SIM_SCALARS = (
    '_antilag_cooldown', '_auxiliary_injection_cooling_kw', '_cascade_depth',
    '_charge_energy_factor', '_crank_assist_nm', '_cylinder_volume_m3',
    '_dyno_drum_radius_m', '_dyno_inertia_kg_m2', '_dyno_mass_kg',
    '_dyno_pull_power_ema', '_dyno_pull_timer_s', '_dyno_pull_torque_ema',
    '_effects_tick', '_electric_fan_cmd', '_exhaust_demand_kg_s',
    '_flashback_accum', '_fuel_cooling_kw', '_fuel_demand_kg_s',
    '_fuel_rack_frac', '_fuel_starvation_frac', '_intake_demand_kg_s',
    '_intake_source_temp_k', '_intake_supply_pressure_pa',
    '_last_dyno_torque_nm', '_load_omega', '_map_frac_na', '_omega',
    '_physics_accum_s', '_pneumatic_idle_assist_area_m2',
    '_pneumatic_idle_assist_trip_rpm', '_power_ms', '_prev_throttle',
    '_quick_shift_target_gear', '_quick_shift_timer_s', '_ring_escalation',
    '_ring_escalation_ticks', '_rpm_rate_per_s', '_surge_timer',
    '_time_since_last_ignition_s', '_time_since_start_s', '_torque_ms',
    '_total_crank_deg', '_waste_heat_kw', 'brake_load_nm', 'cell_fan_m3_s',
    'clutch_frac', 'electrical_load_frac', 'external_crank_torque_nm',
    'gear_index', 'hydraulic_flow_frac', 'hydraulic_load_frac',
    'known_accessory_shaft_load_w', 'throttle',
)

#: The per-edge torsional state of the drivetrain solver. `relative_angle_rad`
#: is where a torsion spring's ENERGY lives, so a declaration that carries
#: node speeds and not edge angles has declared the driveline's momentum and
#: thrown away its potential.
EDGE_FIELDS = (
    ("edge_relative_angle_rad", "relative_angle_rad", "rad"),
    ("edge_last_torque_nm", "last_torque_nm", "Nm"),
    ("edge_wear", "wear", "frac"),
    ("edge_glaze", "glaze", "frac"),
    ("edge_dissipated_heat_j", "dissipated_heat_j", "J"),
    ("edge_unrejected_heat_j", "unrejected_heat_j", "J"),
    ("edge_engaged", "engaged", "flag"),
)

#: Mersenne Twister: 624 words of state, an index, and the spare gaussian.
#: The sim is seeded per identity so that the same inputs give bit-identical
#: output -- which is what makes a predicted result checkable by digest
#: rather than by trust. That guarantee only holds if the generator's
#: POSITION is part of the state too.
RNG_SLOTS = 626

#: THE OTHER GENERATORS. `bursts` and `ordnance` each carry their own
#: numpy `Generator`, and neither was declared. Carrying `sim._rng` alone
#: leaves those two free-running, which is the residual noise a replay
#: test cannot get below however much else is carried.
#:
#: A PCG64 state is two 128-bit integers plus two small flags. A span
#: holds doubles, and a double carries 53 bits exactly, so each integer is
#: split into four 32-bit limbs rather than stored whole -- storing a
#: 128-bit value in a float64 silently rounds it, and a rounded generator
#: state is a different generator.
NUMPY_RNG_PATHS = (("bursts", "rng"), ("ordnance", "rng"))
NUMPY_RNG_LIMBS = 4
NUMPY_RNG_SLOTS = 2 * NUMPY_RNG_LIMBS + 2


#: EVERY scalar field the state dataclass carries. Enumerated from a
#: running sim rather than curated: hand-picking "the ones that matter"
#: is how a checkpoint ends up almost right, and almost right is the
#: failure that looks like noise. Derived readings ride along harmlessly
#: because a step overwrites them anyway.
STATE_SCALARS = (
    'ac_compressor_load_w', 'accessory_drag_nm', 'alternator_current_a',
    'alternator_output_frac', 'battery_soc_frac', 'battery_voltage',
    'bay_air_temp_k', 'bay_co_ppm', 'bay_o2_frac', 'blowby_l_per_min',
    'boiler_water_frac', 'boost_frac', 'cam_phase_lag_deg',
    'catalyst_brick_temp_k', 'catalyst_co_efficiency',
    'catalyst_nox_efficiency', 'charge_equivalence_ratio',
    'charge_gas_volume_fraction', 'co_emitted_indoors_kg',
    'co_engine_out_g_s', 'co_tailpipe_g_s',
    'compression_brake_torque_nm', 'coolant_flow_lpm', 'coolant_lost_l',
    'coolant_temp_k', 'coupling_capacity_nm', 'coupling_heat_w',
    'crank_angle_deg', 'crankcase_pressure_kpa', 'current_torque_nm',
    'dyno_absorbed_kw', 'dyno_kinetic_energy_j',
    'dyno_pull_peak_power_kw', 'dyno_pull_peak_power_rpm',
    'dyno_pull_peak_torque_nm', 'dyno_pull_peak_torque_rpm', 'dyno_rpm',
    'dyno_torque_nm', 'electrical_load_w', 'exhaust_flow_demand_frac',
    'exhaust_open_frac', 'exhaust_pressure_frac',
    'exhaust_tailpipe_temp_k', 'exhaust_temp_k',
    'expander_chest_pressure_pa', 'expander_cylinder_temp_k',
    'expander_ice_frac', 'expander_mep_pa', 'fire_event_count',
    'fire_heat_release_w', 'fires_burning', 'fuel_availability_frac',
    'fuel_fill_frac', 'fuel_supply_pressure_pa', 'fuel_temp_k',
    'gasholder_composition_frac', 'hc_tailpipe_g_s',
    'hydraulic_oil_temp_k', 'ignition_timing_deg', 'injector_duty_frac',
    'intake_charge_temp_k', 'intake_flashback_risk',
    'intake_flow_demand_frac', 'intake_o2_factor',
    'intake_runner_temp_k', 'junction_ring_events', 'junction_substeps',
    'knock_intensity', 'manifold_pressure_frac', 'mixture_phi',
    'mounts_lost', 'nitrous_boost_frac', 'nitrous_fill_frac',
    'nox_tailpipe_g_s', 'occupant_cohb_pct', 'oil_consumption_ml_per_h',
    'oil_flow_lpm', 'oil_fuel_dilution_frac', 'oil_pickup_air_frac',
    'oil_pressure_pa', 'oil_temp_k', 'plant_dewpoint_k',
    'plant_gunk_kg', 'pneumatic_idle_assist_boost_frac',
    'pneumatic_idle_assist_delivered_kg_s', 'power_kw', 'power_rms_kw',
    'preignition_count', 'preignition_intensity', 'real_fire_hz',
    'room_co_ppm', 'room_o2_frac', 'room_temp_k', 'rpm', 'smoke_factor',
    'spark_energy_frac', 'starter_current_a', 'sump_oil_l',
    'supercharger_drag_nm', 'throttle_plate_angle_deg', 'torque_rms_nm',
    'turbo_spool_frac', 'valve_float_risk',
)

#: The per-cylinder lists, each `n_cyl` long.
STATE_LISTS = (
    'cylinder_block_temps_k', 'cylinder_breathing_frac', 'cylinder_carbon_frac', 'cylinder_float_rpm', 'cylinder_hotspot_risk', 'cylinder_oil_film_mg', 'cylinder_valve_factor', 'slot_angles_deg',
)

#: The crankcase's own two masses, the only sub-object scalars measured
#: to move during a run.
CRANKCASE_SCALARS = ('oil_kg', 'case_gas_kg')

#: The bay and the room the machine sits in. Air is state: it holds heat,
#: oxygen and carbon monoxide between steps, and a restore that leaves it
#: behind restarts in a different atmosphere than it stopped in.
AIR_VOLUME_SCALARS = (
    'co_ppm', 'draw_m3_s', 'exchange_m3_s', 'exhaust_co_kg_s', 'exhaust_m3_s',
    'exhaust_o2_frac', 'exhaust_temp_k', 'heat_in_w', 'o2_frac', 'temp_k',
    'volume_m3',
)
AIR_VOLUMES = ('bay', 'garage')
AIR_STACK_SCALARS = ('extractor_capture_frac', 'cell_fan_m3_s',
                     'vehicle_speed_mps')

#: Hole emitters. An undamaged engine carries one splash emitter per
#: cylinder (`cylinder_N.bore_bottom.splash.dipper`), so the baseline
#: count is `n_cyl`. DAMAGE CREATES MORE, and that is a topology change
#: rather than a parameter change -- a holed engine has a different
#: stride and therefore cannot share an assembly with an intact one,
#: which is the ABI's own rule applied to damage.
EMITTER_SCALARS = (
    'contained_head_m', 'contained_l', 'deposit_kg_s', 'deposited_kg',
    'dip_depth_m', 'drip_phase', 'drip_rate_hz', 'drop_mass_kg', 'flow_m3_s',
    'gas_temp_k', 'height_above_low_m', 'ingest_kg_s', 'ingested_kg',
    'jet_speed_m_s', 'mass_flow_kg_s', 'radius_m', 'spilled_kg',
    'throw_radius_m',
)
#: NOT declared. `HoleEmitters.lost_l`, `.ingest_kg_s` and `.mix` are
#: dicts keyed by fluid circuit, empty on an intact engine and created
#: entry by entry as fluid is actually lost. They are damage
#: accumulators, so like the emitter count itself they are a topology
#: matter rather than a slot in a fixed stride. Declaring them as two
#: scalars -- which was tried -- writes a float over a mapping and the
#: next step dies in `hole_emitters.step` on `lost_l.get`.
EMITTER_BANK_SCALARS = ()

#: The catalyst brick: a real accumulator, not a reading.
CATALYST_SCALARS = ('brick_temp_k', 'co_efficiency', 'converted_co_kg',
                    'nox_efficiency')

#: The electrical reading LOOKS derived and is not: the next step consumes
#: `reading.alternator_shaft_load_w` when it solves the drivetrain, so a
#: stale reading changes the trajectory.
ELECTRICAL_READING_SCALARS = (
    'alternator_current_a', 'alternator_shaft_load_w', 'battery_current_a',
    'battery_soc_frac', 'load_current_a', 'load_w', 'voltage_v',
)

#: The solver's last published readings. Numeric entries only: the
#: mapping also carries booleans and nested dicts, and a span holds
#: doubles. Also consumed by the following step, not merely reported.
DRIVETRAIN_OUT_KEYS = (
    'ac_compressor_load_w', 'alternator_delivered_w', 'auxiliary_injection_delivered_kg_s', 'auxiliary_injection_fill_frac', 'cam_phase_lag_rad', 'cam_timing_slack_frac', 'coolant_flow_lpm', 'coolant_temp_k', 'cooling_fan_flow_m3_s', 'crank_reaction_torque_nm', 'dyno_absorber_omega', 'exhaust_backpressure_frac', 'exhaust_flow_capacity_kg_s', 'exhaust_temp_k', 'friction_heat_w', 'fuel_availability_frac', 'fuel_composition_frac', 'fuel_delivered_kg_s', 'fuel_fill_frac', 'fuel_pump_flow_capacity_kg_s', 'fuel_temp_k', 'intake_charge_temp_k', 'intake_flow_capacity_kg_s', 'intake_manifold_pressure_pa', 'nitrous_delivered_kg_s', 'nitrous_fill_frac', 'oil_flow_lpm', 'oil_pressure_pa', 'oil_temp_k', 'pneumatic_compressor_delivered_w', 'pneumatic_compressor_running_w', 'pneumatic_idle_assist_delivered_kg_s', 'pneumatic_reserve_fill_frac', 'pneumatic_reserve_pressure_pa', 'supercharger_belt_slack_frac',
)


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
    # `state.slot_records` is n_cyl rows of four floats. The four
    # `cyl_last_*` spans above were written expecting named fields on a
    # record; there is no record, so the block is carried WHOLE rather
    # than split into names whose meaning this file cannot verify.
    add("cyl_flame_radius_m", n_cyl, "m", "the kernel front, mid-burn")
    add("slot_record", n_cyl * 4, "mixed", "state.slot_records, flattened")
    add("battery_soc_frac", 1, "frac")
    add("air_volume", len(AIR_VOLUMES) * len(AIR_VOLUME_SCALARS), "mixed",
        "bay then garage, in AIR_VOLUME_SCALARS order")
    add("air_stack", len(AIR_STACK_SCALARS), "mixed")
    add("emitter", n_cyl * len(EMITTER_SCALARS), "mixed",
        "one splash emitter per cylinder, in EMITTER_SCALARS order")
    add("catalyst", len(CATALYST_SCALARS), "mixed")
    add("electrical_reading", len(ELECTRICAL_READING_SCALARS), "mixed")
    add("net_torque", n_node, "Nm", "the solver's last net torque per node")
    add("drivetrain_out", len(DRIVETRAIN_OUT_KEYS), "mixed",
        "the solve's published readings, in DRIVETRAIN_OUT_KEYS order")
    # `_drivetrain_out["fluid_circuits"]` is a NESTED mapping, one entry
    # per circuit, and it was skipped when the outer mapping was filtered
    # to numeric values. The next step reads it. Leaving it out was worth
    # 8.96e-06 rpm of replay divergence and nothing else was: every other
    # span could be exact while this one quietly carried the solve's last
    # circuit pressures forward from whatever ran most recently.
    add("out_circuit_pressure_pa", n_circ, "Pa")
    add("out_circuit_temp_k", n_circ, "K")

    # ---- per fluid circuit ------------------------------------------
    # Every field of `FluidCircuit` measured to move during a run. The
    # first three were declared from the start; the rest were found by
    # diffing a restored sim against the original, which is the only way
    # to be sure a checkpoint is complete rather than merely plausible.
    add("circuit_pressure_pa", n_circ, "Pa")
    add("circuit_temp_k", n_circ, "K")
    add("circuit_fill_frac", n_circ, "frac")
    add("circuit_flow_lpm", n_circ, "L/min")
    add("circuit_supply_pressure_pa", n_circ, "Pa")
    add("circuit_target_pressure_pa", n_circ, "Pa")
    add("circuit_delivered_flow_kg_s", n_circ, "kg/s")

    # ---- per graph node ---------------------------------------------
    add("node_omega", n_node, "rad/s", "the torque solver's own shaft speeds")

    # ---- per drivetrain edge ----------------------------------------
    edges = graph["edges"] if isinstance(graph, dict) else graph[1]
    n_edge = len(edges)
    for span_name, _attribute, unit in EDGE_FIELDS:
        add(span_name, n_edge, unit)

    # ---- the whole state record, and the crankcase -------------------
    add("state_scalar", len(STATE_SCALARS), "mixed",
        "every scalar field of the state record, in STATE_SCALARS order")
    add("state_list", len(STATE_LISTS) * n_cyl, "mixed",
        "the per-cylinder lists, in STATE_LISTS order")
    add("crankcase_scalar", len(CRANKCASE_SCALARS), "kg")

    # ---- the sim's own scalars, and its generator --------------------
    add("sim_scalar", len(SIM_SCALARS), "mixed",
        "every mutable scalar the sim carries, in SIM_SCALARS order")
    add("rng_word", RNG_SLOTS, "word", "the generator's position")
    add("numpy_rng", len(NUMPY_RNG_PATHS) * NUMPY_RNG_SLOTS, "word",
        "PCG64 state for each generator in NUMPY_RNG_PATHS")

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
