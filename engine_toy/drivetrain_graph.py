"""A drivetrain-only mechanical graph built from the REAL production
function, not a port of it.

turing/src/compiler/abstract_ui_vehicles.py's `_vehicle_powertrain_graph`
was extracted from inside the real game's whole-vehicle graph builder
specifically so it could be called as a standalone subunit: crank,
camshaft (+ bearings + real accessory-drive-belt timing drive),
alternator (+ real accessory-drive-belt takeoff, `use_belt_accessories=
True` -- NOT the default vehicle's no-belt direct-shaft-plus-CVT),
engine->clutch->transmission->transfer-case torque-shaft chain, the real
dog-clutch direct-drive bypass, and structural mounts -- with
`include_wheel_output=False` so it needs no chassis/suspension/tire
config, no VehicleConfiguration, nothing beyond an engine position and a
few real drivetrain/electrical numbers. This module imports that
function directly and calls it; it does not reimplement its mechanisms.

Only two things here have no real-vehicle counterpart at all and stay
genuinely this toy's own: the dyno absorber (test equipment, not a
vehicle part -- no real vehicle would ever have one) and the
supercharger rotor (the real engine preset system has no forced-
induction concept anywhere to extend).

DrivetrainSolver is the real internal solver for this toy: it doesn't
hand-roll physics per mechanism, it walks the graph's own edges (the
real ones from _vehicle_powertrain_graph included) and dispatches on
each edge's `constraint` string -- the graph document IS the program.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import sys
from pathlib import Path

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))
from src.compiler.abstract_ui_vehicles import (  # noqa: E402
    _vehicle_powertrain_graph, FLUID_LINE_MATERIALS, FLUID_MEDIA,
    LINE_INNER_FILM_W_M2K, LINE_OUTER_FILM_W_M2K, BATTERY_MODULES)
from electrical_network import Battery  # noqa: E402

# Real fluid-line constraint kinds (abstract_ui_vehicles.py's own
# `routed_line` set) grouped by what physics actually governs them --
# not per-fluid bespoke classes, one dispatch per physical category:
#   thermal-liquid       -- coolant, oil: incompressible, carries heat
#   compressible-gas      -- pneumatic/air lines: pressure fills/empties
#                            with a real time constant tied to line
#                            volume, same reasoning as MAP's own fill lag
#   incompressible-hydraulic -- brake/suspension hydraulics: pressure
#                            follows its source almost immediately (high
#                            bulk modulus), restricted mainly by flow area
#   high-pressure-liquid-supply -- a real depletable bottle (nitrous) OR
#                            tank (fuel): valve/pump-gated flow out of a
#                            finite reservoir, not just a pressure state
THERMAL_LIQUID_KINDS = {"coolant-line", "oil-line"}
COMPRESSIBLE_GAS_KINDS = {"low-pressure-air-line", "pressure-rated-air-line",
                          "flexible-air-line", "rigid-pneumatic-hard-line",
                          "flexible-pneumatic-hose", "exhaust-flow-path"}
INCOMPRESSIBLE_HYDRAULIC_KINDS = {"pressure-rated-hydraulic-line", "flexible-hydraulic-hose"}
HIGH_PRESSURE_LIQUID_SUPPLY_KINDS = {"nitrous-delivery-line", "fuel-supply-line", "compressed-air-line"}
FUEL_PASSIVE_TAU_S = 120.0  # real, slow passive tank/line heat exchange with ambient (uninsulated)
FUEL_COOLER_TAU_S = 8.0     # real, much faster active ice-box/chiller extraction
# Real isothermal compression work per kg of air, R_specific_air*T*ln(P2/P1),
# evaluated at a typical real reserve-tank target (~120 psi / 827 kPa) from
# atmospheric -- not a per-tank-configurable pressure this pass carries.
PNEUMATIC_SPECIFIC_COMPRESSION_WORK_J_PER_KG = 190_000.0
PNEUMATIC_RESERVE_PRESSURE_PA = 827_000.0   # the same ~120 psi target that work figure is evaluated at
AIR_DENSITY_AT_1_ATM_KG_M3 = 1.2
FLUID_LINE_KINDS = (THERMAL_LIQUID_KINDS | COMPRESSIBLE_GAS_KINDS
                    | INCOMPRESSIBLE_HYDRAULIC_KINDS | HIGH_PRESSURE_LIQUID_SUPPLY_KINDS)

# engine.mass_kg (and, until now, "powertrain.engine"'s own node mass)
# is documented (block_dynamics.py's own build_block_network docstring)
# as "the bare block+crank+heads casting" -- deliberately excluding
# accessories (alternator, camshaft, etc. already carry their own real,
# separate mass_kg elsewhere in this same graph) and the oil pan (its
# own separate node). Real, disclosed, order-of-magnitude split of
# THAT bucket across its three real named parts -- not tuned to fit
# anything, and not claiming finer precision than "bare casting mass
# roughly divides this way" for a typical cast-iron block/head/crank
# combination:
#   block casting (crankcase + cylinder walls): the single largest chunk
#   cylinder heads (with valvetrain hardware bolted on): a close second
#   crankshaft + flywheel: smaller, but real and genuinely off-centerline once flywheel is at the rear
ENGINE_BLOCK_MASS_FRACTION = 0.50
ENGINE_HEAD_MASS_FRACTION = 0.30
ENGINE_CRANK_FLYWHEEL_MASS_FRACTION = 0.20


def parametric_volume_pressure_exchange(p_a: float, v_a: float, p_b: float, v_b: float) -> float:
    """Real two-volume equilibrium pressure when a real valve opens and
    connects them (isothermal ideal-gas mixing: moles, hence P*V, are
    conserved across the union of the two volumes). This is THE real
    mechanism a cylinder filling from a plenum (or dumping into a
    header) actually is -- not two independent pressures glued
    together, one shared equilibrium weighted by how big each volume
    actually is. A tiny cylinder against a much larger plenum barely
    moves the plenum's own pressure and mostly just takes on the
    plenum's value; two similar-sized volumes genuinely average."""
    total_v = max(v_a + v_b, 1e-12)
    return (p_a * v_a + p_b * v_b) / total_v


def step_depletable_reservoir(fill_level_frac: float, bottle_capacity_kg: float, dt: float,
                               demand_kg_s: float = 0.0, production_kg_s: float = 0.0,
                               flow_capacity_kg_s: float = 0.0,
                               valve_open: bool = True) -> tuple[float, float]:
    """THE one real convention for mass transfer through a depletable
    reservoir (a tank, bottle, or gasholder) -- used identically
    whether the reservoir holds fuel, compressed air, or generator-fed
    coal gas, because the real mass-balance physics genuinely IS the
    same regardless of which fluid it is: fill_level_frac rises with
    whatever's being produced into it and falls with whatever's
    actually delivered out, over dt, scaled by the reservoir's own
    real capacity. Every one of HIGH_PRESSURE_LIQUID_SUPPLY_KINDS'
    real circuits (fuel tank, pneumatic reserve, nitrous bottle, and
    now an on-site gas generator's gasholder) is this SAME primitive,
    not a bespoke one per fluid -- what genuinely differs between them
    (a compressor's own pressure-switch hysteresis, a regulator's flow
    ramp, a pump's rated ceiling) is real upstream/downstream gating
    that belongs to that specific real device, computed by the caller
    and handed in here as demand_kg_s/production_kg_s/flow_capacity_
    kg_s -- not duplicated into a parallel copy of this same math.

    Returns (new_fill_level_frac, delivered_kg_s) -- delivered_kg_s is
    the REAL amount actually delivered this tick, capped by both
    flow_capacity_kg_s (0.0 = uncapped, matching every existing caller's
    own convention) and by whatever the reservoir can actually still
    supply, exactly like a real pump can't deliver what an empty tank
    doesn't have."""
    capacity = max(bottle_capacity_kg, 0.01)
    if not valve_open:
        delivered_kg_s = 0.0
    else:
        ceiling_kg_s = flow_capacity_kg_s if flow_capacity_kg_s > 0.0 else demand_kg_s
        # ...and never more than the vessel actually holds plus what is
        # arriving this same tick -- a nearly-empty dome or receiver
        # sags to whatever production can keep up with, continuously,
        # instead of alternating between "full flow" and "nothing" on
        # successive ticks around the empty mark (which is what a bare
        # fill <= 0 gate did once a consumer's draw outran production)
        stock_kg_s = (max(0.0, fill_level_frac) * capacity) / max(dt, 1e-9) + max(0.0, production_kg_s)
        delivered_kg_s = min(demand_kg_s, ceiling_kg_s, stock_kg_s)
    net_kg = (production_kg_s - delivered_kg_s) * dt
    new_fill_level_frac = max(0.0, min(1.0, fill_level_frac + net_kg / capacity))
    return new_fill_level_frac, delivered_kg_s


def step_reservoir_composition(composition_frac: float, fill_level_frac_after: float,
                               bottle_capacity_kg: float, dt: float,
                               production_kg_s: float = 0.0,
                               production_composition_frac: float = 1.0) -> float:
    """THE one real convention for what a depletable reservoir's
    contents actually ARE, alongside step_depletable_reservoir's "how
    much": composition_frac is the mass fraction of the reservoir's
    contents that is its nominal working fluid (fuel gas in a
    gasholder; the balance is air/inert) -- 1.0 = pure. Real well-
    mixed-volume (CSTR) balance, the same law gas_works.GasQualityBath
    uses for a generator's own outlet plumbing, at the reservoir's
    own real mass: only INFLOW moves the composition (toward whatever
    is being pumped in, at a rate set by inflow over contents), while
    drawing mixture OUT at the current composition leaves what
    remains unchanged -- the actual physical reason a gasholder
    first-filled with air takes real time to purge to combustible
    gas, and why an emptied-and-refilled one snaps straight to
    whatever's coming in (nothing left to dilute it). Call AFTER
    step_depletable_reservoir with the fill level it returned, so the
    inflow is diluted into the contents it's actually joining."""
    inflow_kg = max(0.0, production_kg_s) * dt
    if inflow_kg <= 0.0:
        return composition_frac
    contents_kg = max(fill_level_frac_after * max(bottle_capacity_kg, 0.01), inflow_kg)
    new = composition_frac + inflow_kg * (production_composition_frac - composition_frac) / contents_kg
    return max(0.0, min(1.0, new))


def _new_builder() -> tuple[list[dict], list[dict], Any, Any]:
    nodes: list[dict] = []
    edges: list[dict] = []

    def node(identity: str, position: tuple[float, float, float], kind: str, **attributes: Any) -> None:
        nodes.append({"identity": identity, "kind": kind, "reference_position": list(position),
                      **attributes})

    def edge(identity: str, a: str, b: str, constraint: str, **attributes: Any) -> None:
        edges.append({"identity": identity, "a": a, "b": b, "constraint": constraint, **attributes})

    return nodes, edges, node, edge


REFERENCE_COMPRESSOR_OMEGA_RAD_S = 150.0  # a real, typical belt-driven-compressor operating speed

AIR_DENSITY_KG_M3 = 1.225
# The airflow-benefit reference speed a cooling fan's own declared
# fan_airflow_m3_s_per_rad_s attribute is judged against -- unchanged
# from the flat 250.0 this used to be hard-coded as inline, just now
# named and shared so every fan (mechanical or electric) is combined on
# the same real basis unless it declares its own fan_rated_omega_rad_s.
FAN_REFERENCE_OMEGA_RAD_S = 250.0
# Disclosed, real approximate class dimensions -- a typical automotive
# engine-cooling fan blade sweep (~32cm diameter), not a per-engine
# tuned value.
COOLING_FAN_DISK_AREA_M2 = 0.08
# A real small BLDC automotive electric cooling fan motor: its own
# rated speed and airflow coefficient, genuinely independent of engine
# rpm (the entire point of an electric fan over a belt-driven one).
ELECTRIC_FAN_RATED_OMEGA_RAD_S = 300.0
ELECTRIC_FAN_FLOW_COEFF_M3_S_PER_RAD_S = 0.03
ALTERNATOR_TORQUE_OMEGA_FLOOR_RAD_S = 105.0   # electrical_network's real claw-pole cut-in speed
ELECTRIC_FAN_ON_TEMP_K = 358.15   # real ~85 degC thermostatic switch-on point
ELECTRIC_FAN_BAND_K = 10.0
# Engines whose output shaft drives a real fixed-pitch propeller
# directly (no shiftable gearbox to decouple engine speed from load
# speed) -- the real "propeller law" (torque ~ omega^2, calibrated to
# the rated point) is their genuine baseline self-load, not an optional
# test brake. fairbanks-morse/curved-dash/trimmer are also direct-drive
# but turn a line shaft or a centrifugal-clutch cutting head, not a
# fluid-pushing rotor, so they're deliberately excluded here.
PROPELLER_COUPLED_ENGINES = {"wartsila-rta96c-14cyl-marine-diesel", "pw-r1340-wasp"}


def rotor_aero_load_torque_nm(omega_rad_s: float, flow_coeff_m3_s_per_rad_s: float,
                                disk_area_m2: float,
                                fluid_density_kg_m3: float = AIR_DENSITY_KG_M3) -> float:
    """The general real aerodynamic/hydrodynamic drag torque ANY rotor
    pushing a fluid develops resisting its own rotation -- a marine
    propeller, a crank-driven cooling fan, an electric fan's own motor
    shaft, or any future rotor of this kind, the same physics regardless
    of what (if anything) drives it or what it's cooling. Real
    actuator-disk momentum theory: volumetric flow through the disk
    grows linearly with the rotor's own declared flow coefficient
    (Q = flow_coeff * omega -- literally the fan_airflow_m3_s_per_rad_s
    the production compiler already declares on a real fan node), the
    disk's own induced exit velocity is that flow divided by its swept
    area, and the shaft power needed to sustain the resulting momentum
    flux is 0.5*rho*Q*v^2 -- dividing by omega leaves a clean
    torque ~ omega^2 law, the real "propeller law" shape, derived here
    instead of assumed."""
    if omega_rad_s <= 0.0 or flow_coeff_m3_s_per_rad_s <= 0.0 or disk_area_m2 <= 0.0:
        return 0.0
    flow_m3_s = flow_coeff_m3_s_per_rad_s * omega_rad_s
    exit_velocity_m_s = flow_m3_s / disk_area_m2
    power_w = 0.5 * fluid_density_kg_m3 * flow_m3_s * exit_velocity_m_s * exit_velocity_m_s
    return power_w / omega_rad_s


def rotor_load_torque_from_rated_nm(omega_rad_s: float, rated_omega_rad_s: float,
                                      rated_torque_nm: float) -> float:
    """The same quadratic propeller/fan law calibrated instead to one
    real rated operating point, for a rotor whose own flow coefficient
    and disk area aren't separately known -- standard marine/aviation
    practice ('propeller law'): absorbed torque scales with the square
    of shaft speed, self-normalizing so 100% rated speed absorbs exactly
    100% rated torque."""
    if rated_omega_rad_s <= 0.0:
        return 0.0
    ratio = max(0.0, omega_rad_s) / rated_omega_rad_s
    return rated_torque_nm * ratio * ratio


def _add_belt_driven_compressor(node, edge, identity: str, position: tuple[float, float, float],
                                  rated_w: float, mass_kg: float = 4.5, inertia_kg_m2: float = 0.006,
                                  crank_identity: str = "powertrain.engine") -> float:
    """The generic real mechanical PORT any belt-driven compressor plugs
    into the crank through -- a real magnetic friction clutch (the same
    saturating relative-speed mechanism the dyno junction and the
    driver's transmission clutch already use), not a rigid belt-spring
    coupling that can transiently overshoot and shock-stall the engine
    (found the hard way building the AC compressor first). Deliberately
    knows nothing about what the compressor's OWN output actually does
    -- reject heat via a refrigerant loop (AC), charge a reserve air
    tank (pneumatics), anything else with a real rotating compressor
    stage -- that's the caller's own real payload, plugged in on the
    other side. Returns the real max_torque_nm this clutch was sized to,
    in case the caller's own payload math needs it (a compressor's real
    delivered power is capped by this same rating)."""
    max_torque_nm = rated_w / REFERENCE_COMPRESSOR_OMEGA_RAD_S
    node(identity, position, "rotating-mass", mass_kg=mass_kg, inertia_kg_m2=inertia_kg_m2)
    edge(f"{crank_identity.split('.')[-1]}_to_{identity}_clutch", crank_identity, identity, "friction-clutch-shaft",
         stiffness_nm_per_rad_s=max_torque_nm * 8.0, max_torque_nm=max_torque_nm)
    return max_torque_nm


def build_drivetrain_graph(engine) -> dict[str, Any]:
    """Lay out exactly one engine's own drivetrain via the real game's
    own _vehicle_powertrain_graph subunit (camshaft, belt-driven
    alternator, engine/clutch/transmission chain, dog-clutch bypass,
    mounts), plus this toy's own dyno absorber and (if present)
    supercharger -- the two things with no real-vehicle equivalent."""
    import engine_geometry
    from engines import IntakeSystem, ExhaustSystem, FUEL_DENSITY_KG_M3, RPM_TO_RAD_S

    nodes, edges, node, edge = _new_builder()
    front = engine_geometry.accessory_front_point(engine)

    # The real subunit: crank (as "powertrain.engine" -- the real node
    # identity, kept as-is rather than aliased), camshaft with real
    # bearings and a real belt timing drive, alternator on a real belt,
    # the engine/clutch/transmission torque-shaft chain, the real
    # dog-clutch direct-drive bypass, and structural mounts. No wheels,
    # no VehicleConfiguration -- just this engine.
    _vehicle_powertrain_graph(
        node, edge, nodes,
        engine_position=[0.0, 0.0, 0.0],
        # left/right structural mounts (mount.engine_left/right, mount.
        # transmission_left/right, mount.transfer_case_left/right) are
        # spread laterally by this same half_width -- left at its 0.0
        # default they all collapse onto the crank centerline (z=0),
        # which is not where a real engine mount ever sits. block_half_
        # yz_m is the toy's own ONE real place this is computed
        # (engine_geometry.py's own docstring), so mounts spread to just
        # outside the block's real skirt width instead of a separate
        # invented constant.
        half_width=engine_geometry.block_half_yz_m(engine),
        # "powertrain.engine" is the crank's own real node (kept below as
        # the physics reference point) -- it now carries only the crank/
        # flywheel's own real SHARE of engine.mass_kg, not the whole
        # bare-casting figure. The block and head shares land on their
        # own real, separately-positioned nodes further down (block
        # segments already existed; head nodes are new), so a real,
        # non-degenerate center of gravity and inertia distribution can
        # be computed directly off the graph's own node masses/positions
        # instead of one lumped point mass at the crank centerline.
        component_masses={"engine": engine.mass_kg * ENGINE_CRANK_FLYWHEEL_MASS_FRACTION},
        include_wheel_output=False,
        use_belt_accessories=True,
        peak_torque_nm=engine.peak_torque_nm,
        # coolant_pump (electric, EV/servo-only per Accessories' own
        # docstring) doesn't belong on a belt at all -- only the real
        # mechanical water_pump does
        has_water_pump=engine.accessories.water_pump,
        # a two-stroke mixes oil into the fuel (total-loss lubrication) --
        # genuinely no wet sump to have a pan or a port on
        has_oil_pan=not engine.architecture.two_stroke,
        has_mechanical_fan=engine.accessories.mechanical_fan,
        include_cooling_stack=engine.accessories.water_pump,
        has_nitrous=engine.has_nitrous,
        has_auxiliary_injection=engine.has_auxiliary_injection,
        # a real turbocharger, oil-fed off the same real gallery the
        # engine itself uses (unlike a supercharger, which gets no
        # oil-line -- see _vehicle_powertrain_graph's own reasoning)
        # the old "and not two_stroke" gate here was wrong: a large marine
        # two-stroke (the Wärtsilä) cannot run at all without its turbos
        # -- the scavenge air comes from them. What a two-stroke lacks is
        # a wet sump, and the production subunit already gates the turbo
        # OIL lines on has_oil_pan separately.
        has_turbo=engine.forced_induction.kind == "turbo",
        # boost-independent flow-capacity calibration -- see
        # _vehicle_powertrain_graph's own reasoning at
        # powertrain.intake_plenum_port
        displacement_l=engine.displacement_l,
        redline_rpm=engine.redline_rpm,
        # real per-cylinder positions (this toy's own geometry, already
        # used everywhere else for acoustic/vibration siting) -- gives
        # each cylinder its own real intake/exhaust port gathering at
        # the plenum and exhaust manifold, instead of the whole engine
        # breathing through one opaque point
        cylinder_positions=tuple(site.position for site in engine_geometry.cylinder_sites(engine)),
        # what production sizes the heat exchanger from (UA = coolant share
        # of rated waste heat / design delta-T, the same waste-heat relation
        # engine_cycle_sim itself uses) and what it rejects to
        rated_power_w=engine.peak_power_kw * 1000.0 if engine.kind == "combustion" else 0.0,
        combustion_efficiency=engine.combustion_efficiency,
        cooling_medium=engine.cooling_medium,
        starting_system=engine.starting_system,
        # this toy is engine-only (no VehicleConfiguration/chassis of
        # its own), so it opts into the real minimal firewall/dash this
        # subunit can build directly instead of getting one from a
        # whole-vehicle builder that doesn't exist here
        include_firewall=True,
    )

    # The crankshaft itself, drawn as an actual shaft along local X --
    # the one axis every other real position in this graph already
    # agrees is the crank's own axis: cylinder_sites spaces cylinders
    # along X, and the production driveline chain continues straight out
    # along X past the crank (clutch +.15, transmission +.29, ...) --
    # that chain could only be colinear with the crank it bolts to if X
    # is that axis. The real inconsistency (found and fixed above) was
    # never X -- it was Y vs Z for "which way is up off the crank
    # centerline", disagreeing between cylinder_sites, mount_points, and
    # _radial_cylinder_sites. Purely visual -- "crank-shaft-reference" is
    # never dispatched by DrivetrainSolver.step() and carries no physics.
    crank_sites = engine_geometry.cylinder_sites(engine)
    crank_x_min, crank_x_max = engine_geometry.crank_extent(crank_sites)
    node("powertrain.crank_shaft.front", [crank_x_min, 0.0, 0.0], "crank-shaft-endpoint")
    node("powertrain.crank_shaft.rear", [crank_x_max, 0.0, 0.0], "crank-shaft-endpoint")
    edge("powertrain.crank_shaft", "powertrain.crank_shaft.front", "powertrain.crank_shaft.rear",
         "crank-shaft-reference", radius=0.02)

    # A real engine block casting -- the piece that was missing this
    # whole time: with no block volume drawn at all, the only thing
    # standing in for "the block" was powertrain.engine's own mass-
    # scaled box (0.2m across for this engine), while the real cylinder
    # bank the ports/runners actually span is 0.59m -- a 3x mismatch
    # between the pipe network's own implied scale and the only solid
    # shown, which is what made every correctly-positioned pipe still
    # read as floating disconnected from a too-small nub in the middle.
    # Sized off the SAME real crank_extent/bank-radius every other
    # position in this graph now agrees on, centered on the crank's own
    # true axis (Y=0, not offset up the way the first, rejected attempt
    # at this was) and kept short enough in Y/Z to stay clear of the
    # real port/valve-cover region above it (ports sit at ~bank_radius*
    # 0.57 here; half_y below tops out at bank_radius*0.5) --
    # mass_in_total=False since engine.mass_kg is already carried on
    # "powertrain.engine" itself (component_masses={"engine": ...}
    # above); this is bounding geometry only.
    block_half_x = max(0.03, (crank_x_max - crank_x_min) / 2.0)
    block_half_y = block_half_z = engine_geometry.block_half_yz_m(engine)
    # Split into one real segment per cylinder bay when that's a
    # meaningful real axial position at all (inline/V -- crank_sites
    # actually varies each cylinder's own X): each segment can then
    # carry its OWN real block-metal temperature
    # (EngineCycleSim.state.cylinder_block_temps_k, a real per-cylinder
    # heat balance, not a fabricated gradient), instead of one
    # monolithic casting with no internal state at all. A radial engine
    # has no real per-cylinder AXIAL position to segment by (every
    # cylinder in a row shares one crank throw, differing in angle, not
    # x) -- and neither does an electric/0-cylinder unit -- so both
    # keep the single block this always was.
    # Block-casting and cylinder-head mass now lands for real on these
    # segments/head nodes (mass_in_total=True) instead of "powertrain.
    # engine" carrying the whole bare-casting figure at one point -- see
    # ENGINE_BLOCK_MASS_FRACTION/ENGINE_HEAD_MASS_FRACTION above. Head
    # nodes are new (no real per-cylinder head mass existed anywhere
    # before this), placed at each cylinder's own real 3D site position
    # (engine_geometry.cylinder_sites already carries the real bank-
    # angle lateral offset a V/opposed engine's heads genuinely sit at
    # -- reused directly, not re-derived) rather than collapsed onto
    # the crank centerline the way the block segments deliberately are
    # (their body geometry, not their real mass distribution).
    block_mass_kg = engine.mass_kg * ENGINE_BLOCK_MASS_FRACTION
    head_mass_kg = engine.mass_kg * ENGINE_HEAD_MASS_FRACTION
    if engine.architecture.cylinders and not engine.architecture.radial:
        n_cyl = engine.architecture.cylinders
        seg_half_x = max(0.01, block_half_x / n_cyl)
        for site in crank_sites:
            node(f"powertrain.engine_block_body.cylinder_{site.number}",
                 [site.position[0], 0.0, 0.0], "engine-block-component", mass_in_total=True,
                 mass_kg=block_mass_kg / n_cyl,
                 body_half_extent_m=[seg_half_x, block_half_y, block_half_z])
            node(f"powertrain.cylinder_head.cylinder_{site.number}",
                 list(site.position), "engine-head-component", mass_in_total=True,
                 mass_kg=head_mass_kg / n_cyl)
    else:
        # radial/electric: no real per-cylinder axial spread to hang
        # separate head nodes off of (radial cylinders share one crank
        # station, differing in angle only -- see _radial_cylinder_
        # sites' own docstring); the head share stays folded into the
        # one block node rather than fabricating a position for it.
        node("powertrain.engine_block_body", [0.0, 0.0, 0.0],
             "engine-block-component", mass_in_total=True, mass_kg=block_mass_kg + head_mass_kg,
             body_half_extent_m=[block_half_x, block_half_y, block_half_z])

    # "powertrain.engine" (the crank's own real node -- kept, it's the
    # actual physics reference point) used to render its OWN generic
    # mass-based box on top of this: for this engine, 0.098m half-extent
    # -- bigger than a block segment's own 0.09m, poking past the real
    # crankcase surface, sitting dead center with no thermal group at
    # all (grey and non-participating, exactly the "strange box that
    # doesn't match the new sizing" this looked like). The crank
    # genuinely sits INSIDE the block/segments now that those are real
    # geometry; a small explicit size keeps it there instead of
    # poking through.
    crank_node = next((n for n in nodes if n["identity"] == "powertrain.engine"), None)
    if crank_node is not None:
        crank_visual_half = min(block_half_y, block_half_z) * 0.3
        crank_node["body_half_extent_m"] = [crank_visual_half, crank_visual_half, crank_visual_half]

    # A real oil pan spans most of the crankcase's own length and width
    # (it bolts to the crankcase's whole bottom face) -- left at the
    # generic mass-based box every small accessory gets by default
    # (_body_half_extent, ~0.067m for this engine's real 8.8kg pan),
    # against a now-correctly-0.71m-long block it read as a tiny stub,
    # not a pan spanning the block's own underside. Sized the same real
    # way the block itself is, hanging below the block's own bottom
    # face rather than overlapping it.
    oil_pan_node = next((n for n in nodes if n["identity"] == "powertrain.oil_pan"), None)
    if oil_pan_node is not None:
        oil_pan_half_x = block_half_x * 0.8
        oil_pan_half_y = engine_geometry.block_half_yz_m(engine) * 0.35
        oil_pan_half_z = engine_geometry.block_half_yz_m(engine) * 0.85
        oil_pan_node["body_half_extent_m"] = [oil_pan_half_x, oil_pan_half_y, oil_pan_half_z]
        oil_pan_node["reference_position"][1] = -(block_half_y + oil_pan_half_y)

    # The production subunit places the clutch/flywheel/transmission/
    # transfer-case chain at small FIXED offsets (pre_clutch_flywheel_
    # wrench +.09, clutch +.15, transmission +.29, ...) sized for
    # whatever reference engine that subunit was authored against --
    # never scaled by THIS engine's own real cylinder count/spacing. For
    # a 6-cylinder engine, the cylinder bank alone now real-scales out to
    # +-0.294m (engine_geometry.cylinder_sites), so a clutch at +.15
    # lands INSIDE the cylinder bank instead of behind it at the crank's
    # real rear face -- physically impossible; a clutch bolts on beyond
    # the last cylinder, never through the middle of the block. Shifted
    # here so the nearest member of that chain (pre_clutch_flywheel_
    # wrench) sits just past the crank's own real x_max, preserving
    # every node's original spacing/lateral offset relative to each
    # other -- only the whole chain's start point moves.
    # A "twin turbo" is two real compressor/turbine assemblies, each with
    # its own bearing oil feed/drain and its own exhaust take-off. The
    # production subunit emits exactly one; every additional unit is a
    # clone of that real one, mirrored across the bank (z) so a V engine's
    # second turbo sits on its other bank, with its own oil lines and its
    # own share of the exhaust heat path.
    fi_ = engine.forced_induction
    base_turbo = next((n for n in nodes if n["identity"] == "powertrain.turbocharger"), None)
    if fi_.kind == "turbo" and base_turbo is not None and fi_.turbo_count > 1:
        turbo_nodes = [n for n in nodes if n["identity"].startswith("powertrain.turbocharger")]
        turbo_edges = [e for e in edges if e["a"].startswith("powertrain.turbocharger")
                       or e["b"].startswith("powertrain.turbocharger")]
        for k in range(2, fi_.turbo_count + 1):
            suffix = f"_{k}"
            z_sign = -1.0 if k % 2 == 0 else 1.0
            for n in turbo_nodes:
                clone = {kk: (list(v) if isinstance(v, list) else v) for kk, v in n.items()}
                clone["identity"] = n["identity"].replace("powertrain.turbocharger", f"powertrain.turbocharger{suffix}", 1)
                clone["reference_position"][2] = abs(clone["reference_position"][2]) * z_sign
                nodes.append(clone)
            for e in turbo_edges:
                ce = dict(e)
                ce["identity"] = f"{e['identity']}{suffix}"
                for end in ("a", "b"):
                    if ce[end].startswith("powertrain.turbocharger"):
                        ce[end] = ce[end].replace("powertrain.turbocharger", f"powertrain.turbocharger{suffix}", 1)
                edges.append(ce)
        # the one real exhaust heat path is now split across N turbines
        for e in edges:
            if e["identity"].startswith("thermal.exhaust_to_turbine") and "heat_share_frac" in e:
                e["heat_share_frac"] = e["heat_share_frac"] / fi_.turbo_count

    driveline_chain_identities = (
        "powertrain.pre_clutch_flywheel_wrench", "powertrain.clutch", "powertrain.transmission",
        "powertrain.transfer_case", "powertrain.direct_drive_bypass",
        "mount.transmission_left", "mount.transmission_right",
        "mount.transfer_case_left", "mount.transfer_case_right",
    )
    driveline_nodes = {n["identity"]: n for n in nodes if n["identity"] in driveline_chain_identities}
    driveline_shift_x = 0.0
    if driveline_nodes:
        rear_clearance_m = 0.02
        original_min_x = min(n["reference_position"][0] for n in driveline_nodes.values())
        driveline_shift_x = (crank_x_max + rear_clearance_m) - original_min_x
        for n in driveline_nodes.values():
            n["reference_position"][0] += driveline_shift_x

    # mount.engine_left/right as the production subunit declares them
    # sit at ONE real axial station (its own engine_position[0]) -- a
    # real reference vehicle's own reasonable choice, but never
    # rederived for whatever real length THIS engine's own block
    # actually turns out to be. A 2-point mount at a single station can
    # never be a real, stable support regardless (two points are a
    # line, not a polygon -- see engine_mounts.check_stability's own
    # reasoning), and for anything longer than one cylinder the block's
    # own real mass genuinely extends fore/aft of that one station, so
    # the assembly's real center of gravity routinely sits outside it.
    #
    # Rederived here as a real 4-point set spanning the crank's own
    # real front/rear extent instead -- crank_shaft.front/.rear (the
    # timing-cover end vs the flywheel end) are already this engine's
    # own INTRINSIC axial references, independent of how a vehicle
    # later orients the whole crate (a transverse install just rotates
    # this same real geometry about the vertical axis before bolting
    # it in; the engine's own casting attachment points don't change).
    # Nothing external currently dictates a fixed mount spacing this
    # has to match (no real supplied cage exists yet -- see this
    # module's own docstring / ENGINE_TOY_ARCHITECTURE_NOTES.md's "no
    # real cage source" note), so there's no reason to keep the
    # production subunit's own reference-vehicle spacing over a real
    # one derived from THIS engine's own geometry.
    engine_left = next((n for n in nodes if n["identity"] == "mount.engine_left"), None)
    engine_right = next((n for n in nodes if n["identity"] == "mount.engine_right"), None)
    if engine_left is not None and engine_right is not None:
        y = engine_left["reference_position"][1]
        z_left = engine_left["reference_position"][2]
        z_right = engine_right["reference_position"][2]
        # real inset so the mounts land ON the casting, not right at
        # its own edge; sorted so a degenerate zero-length block
        # (single cylinder) still produces two distinct, correctly
        # ordered real stations rather than a reversed pair.
        inset = max(0.02, (crank_x_max - crank_x_min) * 0.08)
        x_front, x_rear = sorted((crank_x_min + inset, crank_x_max - inset))
        nodes.remove(engine_left)
        nodes.remove(engine_right)
        # the production subunit's own mount edges point at the two nodes
        # just removed -- carry their real attributes (constraint kind,
        # transfer, radius) onto one edge per new real mount point instead
        # of leaving two edges dangling at nothing
        stale = [e for e in edges if e["b"] in ("mount.engine_left", "mount.engine_right")]
        template = dict(stale[0]) if stale else {"constraint": "six-axis-compliant-mount", "radius": 0.012,
                                                 "palette": "drivetrain-black",
                                                 "transfer": "force-and-moment-to-chassis"}
        for e in stale:
            edges.remove(e)
        attrs = {k: v for k, v in template.items() if k not in ("identity", "a", "b", "constraint")}
        for suffix, x in (("front", x_front), ("rear", x_rear)):
            for side, z in (("left", z_left), ("right", z_right)):
                mid = f"mount.engine_{suffix}_{side}"
                node(mid, [x, y, z], "powertrain-mount", fixed_to="chassis")
                edge(f"mount.engine.engine_{suffix}_{side}", "powertrain.engine", mid, template["constraint"], **attrs)

    # drivetrain.engine_to_clutch and drivetrain.direct_drive_bypass are
    # both production-authored torque-path edges from "powertrain.engine"
    # directly (the crank's own CENTER reference, x=0) -- meaningless for
    # a real flywheel/clutch, which bolts to the crank's REAR FACE, not
    # its center. Left alone, both edges span the crank's entire half-
    # length regardless of how tight the actual clutch/bypass gap behind
    # the block is. Repointed to originate from powertrain.crank_shaft.
    # rear instead -- the real crank-rear reference already built above,
    # not a new synthetic node (that's the mistake the crank_pulley
    # attempt made). Both kinds ("torque-shaft",
    # "synchronized-positive-dog-clutch-bypass") are never dispatched by
    # DrivetrainSolver.step() -- real clutch/gear physics lives in
    # engine_cycle_sim.py directly -- so this is purely a mesh-accuracy
    # fix with zero physics risk.
    for shaft_edge_id in ("drivetrain.engine_to_clutch", "drivetrain.direct_drive_bypass"):
        shaft_edge = next((e for e in edges if e["identity"] == shaft_edge_id), None)
        if shaft_edge is not None and shaft_edge["a"] == "powertrain.engine":
            shaft_edge["a"] = "powertrain.crank_shaft.rear"

    if not engine.architecture.has_poppet_valves or engine.architecture.rotary:
        # the real subunit always includes a camshaft (every real engine
        # has one) -- this one genuinely doesn't (port-scavenged two-
        # stroke or rotary), so its camshaft node/edges are removed
        # rather than left in place pretending it exists
        nodes[:] = [n for n in nodes if n["identity"] != "powertrain.camshaft"]
        edges[:] = [e for e in edges if "camshaft" not in e["identity"]]
    # The production subunit is a PISTON-engine subunit: it always builds a
    # camshaft, a piston intake plenum, a PCV port, and a wet-sump oil pan/
    # pump. None of those exist on an electric or servo drive unit, a gas
    # turbine, a free-piston atmospheric engine, or a steam/air expander --
    # a dyno-side drive unit with a camshaft and a PCV valve is not a
    # simplification, it is wrong hardware. Purged by kind (the audit's
    # own finding); the driveline chain (clutch/transmission/transfer
    # case) stays because it is this toy's real dyno coupling for every
    # kind. A turbine keeps its oil pump/pan (a real APU has a pressure
    # oil system); the rest have no wet sump at all.
    _PISTON_ONLY = ("powertrain.camshaft", "powertrain.intake_plenum", "powertrain.engine_block_port.pcv")
    _WET_SUMP_ONLY = ("powertrain.oil_pan", "powertrain.oil_pump", "powertrain.engine_block_port.oil_pan")
    purge: tuple[str, ...] = ()
    if engine.kind in ("electric", "servo-electric", "atmospheric", "expander"):
        purge = _PISTON_ONLY + _WET_SUMP_ONLY
    elif engine.kind == "turbine":
        purge = _PISTON_ONLY
    if purge:
        gone = set(purge)
        nodes[:] = [n for n in nodes if n["identity"] not in gone]
        edges[:] = [e for e in edges if e["a"] not in gone and e["b"] not in gone]
    if not engine.accessories.alternator:
        nodes[:] = [n for n in nodes if n["identity"] != "electrical.alternator"]
        edges[:] = [e for e in edges if "alternator" not in e["identity"]]
    if not engine.accessories.mechanical_fan:
        # The production subunit builds powertrain.cooling_fan
        # unconditionally whenever a cooling stack exists at all --
        # only its own drive-belt EDGE is gated by has_mechanical_fan
        # (see _vehicle_powertrain_graph). Left in place, an engine
        # with no mechanical fan (ram-air-cooled aircraft engines,
        # pump-only marine cooling) still carried a real node with
        # nothing ever driving it, sitting at a small fixed offset that
        # this graph's own now-correctly-scaled block can swallow
        # whole for a large engine. Same real removal the alternator
        # gets above, for the same reason: no belt, no real component.
        nodes[:] = [n for n in nodes if n["identity"] != "powertrain.cooling_fan"]
        edges[:] = [e for e in edges if "fan_belt" not in e["identity"]]

    # Auto-assemble the real chosen intake/exhaust hardware onto the
    # production subunit's generic stub geometry -- the same real
    # IntakeSystem/ExhaustSystem dataclasses that already drive breathing
    # physics (engines.derive_gross_bmep_pa, the live resonance/scavenging
    # terms in engine_cycle_sim.py) now drive what gets DRAWN too, instead
    # of the mesh silently staying at the production template's generic
    # 8mm/9mm stub radii regardless of what was actually built.
    #  - runner/primary tube RADIUS comes straight from the real declared
    #    part diameter.
    #  - the shared plenum/manifold node's own standoff distance from the
    #    block scales with real runner/pipe LENGTH relative to a default
    #    build -- an honest stand-in for true per-cylinder equal-length
    #    curved routing (which this toy's straight-tube primitives can't
    #    draw), so a genuinely longer-runner or longer-pipe build visibly
    #    sits farther out, not just a fatter/thinner straight stub.
    intake = engine.intake_system
    exhaust = engine.exhaust_system
    intake_radius_m = max(0.003, intake.runner_diameter_mm / 2000.0)
    exhaust_radius_m = max(0.003, exhaust.primary_diameter_mm / 2000.0)
    for e in edges:
        if e["identity"].endswith(".intake_runner"):
            e["radius"] = intake_radius_m
        elif e["identity"].endswith(".exhaust_primary"):
            e["radius"] = exhaust_radius_m

    # The production subunit places each cylinder's intake/exhaust port
    # at a FIXED +-0.02m either side of the cylinder center, regardless
    # of the real tube radius that ends up running from it -- for any
    # engine whose real intake/exhaust tube diameter exceeds that 0.04m
    # total gap (a completely ordinary case once runner_diameter_mm is
    # real and not a generic stub), the two ports' own tube volumes
    # overlap right at the block face, not "two distinct terminals" at
    # all. Widened here so the gap always clears both real tube radii
    # with a real disclosed margin, scaled to whatever this engine's
    # own intake/exhaust diameters actually are -- done before the
    # riser/sweep waypoint routing below, since those are built directly
    # off these port positions.
    # Offset relative to the CYLINDER's own real z (not the port's
    # current absolute z, which sign-guessing off of would break for any
    # V-bank cylinder whose own z is already negative) -- matched by
    # cylinder index straight from the same cylinder_sites list already
    # used to build cylinder_positions above.
    port_half_gap = max(0.02, (intake_radius_m + exhaust_radius_m) * 0.75)
    cylinder_z_by_index = {i: site.position[2] for i, site in enumerate(engine_geometry.cylinder_sites(engine))}
    # Same real bug as the Z-gap above, on the Y axis: the production
    # subunit's own port Y (cylinder.y + a flat +.03) never scales
    # either, so for any engine whose real bank_radius/block clearance
    # ends up bigger than a few cm (an entirely ordinary case once
    # BANK_RADIUS_M is properly per-engine scaled), the port sits
    # BELOW the block's own real surface -- a port literally inside the
    # crankcase instead of on the head, above it. Floored at the same
    # block_clearance principle the accessory arc already uses (never
    # lowered, only raised, so a small engine's already-clear port
    # position is untouched).
    min_port_y = engine_geometry.block_half_yz_m(engine) + max(intake_radius_m, exhaust_radius_m) * 1.2
    for n in nodes:
        if n["identity"].endswith(".intake_port") or n["identity"].endswith(".exhaust_port"):
            cyl_index = int(n["identity"].split("_")[1].split(".")[0])
            cyl_z = cylinder_z_by_index.get(cyl_index, 0.0)
            is_intake = n["identity"].endswith(".intake_port")
            n["reference_position"][2] = cyl_z + (-port_half_gap if is_intake else port_half_gap)
            n["reference_position"][1] = max(n["reference_position"][1], min_port_y)

    # Every real port of every cylinder, by cylinder KIND (cylinder_
    # ports.py): the production graph already authors a piston engine's
    # intake/exhaust valve ports (the runner/primary endpoints above),
    # so those keep their nodes and the layout is aligned onto them;
    # everything else a real head carries -- plug, injector, glow plug
    # bosses; an Otto-Langen slide valve's gas/air/flame/exhaust
    # openings; an expander's admission/exhaust passages, drain cocks,
    # rod gland, lubricator -- is emitted here as the same "engine-
    # block-port" node kind, tagged with its real port_kind, outward
    # direction and hole radius. The whole layout also rides on the
    # graph document so vehicle_mesh can build the cylinder bodies and
    # port stubs from the graph alone.
    from cylinder_ports import cylinder_port_layout, serialize_layout, PortSpec
    existing_ports = {n["identity"]: n for n in nodes if n["kind"] == "engine-block-port"
                      or n["identity"].endswith((".intake_port", ".exhaust_port"))}
    # Renames are collected into one map and applied to every edge in a
    # single pass AFTER the whole layout is walked (below), rather than
    # rewriting matching edges inline as each port is renamed: the
    # legacy (0-based) and real (1-based) identity ranges overlap for
    # every cylinder but the last, so an inline rewrite of "any edge
    # currently named cylinder_K" would also catch an edge THIS SAME
    # loop had already retargeted to cylinder_K one or two iterations
    # earlier, cascading it forward onto the wrong, later cylinder.
    identity_renames: dict[str, str] = {}
    layout = []
    for geom, ports in cylinder_port_layout(engine):
        aligned = []
        for port in ports:
            ident = f"powertrain.cylinder_{geom.number}.{port.name}"
            # _vehicle_powertrain_graph's own per-cylinder intake/exhaust
            # ports (above) are numbered 0-based straight off the same
            # cylinder_sites() tuple this toy's own geom.number (1-based)
            # is numbered from, so the production node THIS cylinder
            # actually needs sits at geom.number - 1, not geom.number.
            # Looking it up under the 1-based identity always missed --
            # every engine kept an orphaned, un-repointed production
            # "cylinder_0" (a phantom extra cylinder wired straight from
            # plenum/manifold with none of this layout's real geometry)
            # while its genuinely-first real cylinder got a brand new
            # node instead of reusing (and inheriting the flow-capacity-
            # carrying edges of) the one already there for it.
            legacy_ident = f"powertrain.cylinder_{geom.number - 1}.{port.name}"
            authored = existing_ports.get(legacy_ident)
            if authored is not None:
                if legacy_ident != ident:
                    identity_renames[legacy_ident] = ident
                    authored["identity"] = ident
                # the authored valve-port node moves onto the real head
                # position the layout derives from bore/stroke/rod (the
                # runner and primary still start from it -- same node,
                # now sitting in the head rather than at a block-face
                # guess that the taller liner geometry would bury)
                authored["reference_position"] = [float(x) for x in port.position]
                authored.setdefault("port_kind", port.port_kind)
                authored["port_direction"] = list(port.direction)
                authored["port_radius_m"] = port.radius_m
                authored["fluid_role"] = port.fluid_role
            else:
                node(ident, port.position, "engine-block-port", port_kind=port.port_kind,
                     port_direction=list(port.direction), port_radius_m=port.radius_m, fluid_role=port.fluid_role,
                     cylinder=geom.number)
            aligned.append(port)
        layout.append((geom, aligned))
    if identity_renames:
        for e in edges:
            if e["a"] in identity_renames:
                e["a"] = identity_renames[e["a"]]
            if e["b"] in identity_renames:
                e["b"] = identity_renames[e["b"]]
    cylinder_layout = serialize_layout(layout)
    # Every casting's ports (assembly_ports.py): each head's FILL on
    # top plus its deck-face oil feed/return and coolant holes, the
    # crankcase's matching deck holes, main gallery, breather, dipstick
    # and pump pickup, the pan's rim and pickup and drain plug -- then
    # the port-to-port mating solver seals whatever lines up across a
    # joint face (head on block, pan on case) as zero-length gasket
    # edges in the fluid's own circuit. Line ports (fill, breather,
    # dipstick, drain plug, main gallery) stay unconnected until a
    # hose, cap or plug part is put on them.
    from assembly_ports import part_ports, mate_ports, emit_ports_graph
    # a total-loss two-stroke has no wet sump: no pan/drain/dipstick/
    # gallery/fill ports to declare, only the crankcase breather
    casting_ports = part_ports(layout, wet_sump=not engine.architecture.two_stroke)
    mating_result = mate_ports(casting_ports)
    emit_ports_graph(casting_ports, mating_result, node, edge)
    # the dressing (dressing.py): lubrication, filters, rail, ignition
    # wiring, and the intake side -- plenum log with routed runners,
    # throttle body / carburetor / individual throttle bodies
    from dressing import derive_dressing, emit_dressing_graph
    dressing_spec = derive_dressing(engine)
    dressing_report = emit_dressing_graph(engine, layout, dressing_spec, nodes, edges, node, edge, intake_radius_m)
    # Manifolds sit relative to the REAL heads now, not to the block-
    # face guess: the plenum above the intake side, the exhaust
    # manifold beside the exhaust side just under head height.
    #
    # Two real, genuinely different intake configurations both live in
    # this graph on purpose, not by accident: dressing.py's own N-to-M
    # distributor (multiple real inlet chambers -- a dual-quad, individual
    # throttle bodies, ...) AND this single, compact ON-BLOCK plenum
    # positioned relative to the real heads (the layout a compact
    # supercharger's own low-profile manifold actually needs). Both are
    # real and worth keeping -- the bug was letting them fight over the
    # SAME node identity ("powertrain.intake_plenum") when dressing had
    # already built more than one real chamber: this block would drag
    # chamber 0's plenum to a THIRD position near the engine's own
    # geometric center, while its own throttle_body (untouched here)
    # stayed at dressing's real chamber-0 position and the per-cylinder
    # runners (which follow "powertrain.intake_plenum" by identity)
    # followed the plenum to the new, wrong spot -- three real pieces
    # of the same barrel disagreeing about where it is. Resolved by an
    # OWNERSHIP rule, not a chamber-count guess: dressing.py now places
    # every intake configuration it builds -- valley, inboard, piped, any
    # unit/plane count -- and marks the chamber it placed (`plenum_style`).
    # This legacy repositioner only ever applies to a plenum dressing did
    # NOT place (a graph with no dressable intake ports at all), so the
    # two can never fight over the same node again.
    multi_chamber_intake = any(n["identity"] == "powertrain.intake_plenum" and "plenum_style" in n for n in nodes)
    piston_geoms = [geom for geom, _ in layout if geom.kind in ("spark-piston", "compression-piston")]
    if piston_geoms:
        import numpy as _np
        head_top = max(float((_np.array(geom.base) + _np.array(geom.axis) * geom.length_m)[1]) for geom in piston_geoms)
        bore_ref = max(geom.bore_m for geom in piston_geoms)
        x_mid = sum(float(geom.base[0]) for geom in piston_geoms) / len(piston_geoms)
        for n in nodes:
            if n["identity"] == "powertrain.intake_plenum" and not multi_chamber_intake:
                n["reference_position"] = [x_mid - 0.05, head_top + bore_ref * 0.9, -bore_ref * 1.1]
            elif n["identity"] == "powertrain.exhaust_manifold":
                n["reference_position"] = [x_mid, head_top - bore_ref * 0.15, bore_ref * 1.25]
    engine_origin = next((n["reference_position"] for n in nodes if n["identity"] == "powertrain.engine"),
                          [0.0, 0.0, 0.0])
    length_scale = intake.runner_length_m / max(IntakeSystem().runner_length_m, 1e-6)
    pipe_scale = exhaust.total_length_m / max(ExhaustSystem().total_length_m, 1e-6)
    for n in nodes:
        if n["identity"] == "powertrain.intake_plenum" and not multi_chamber_intake:
            n["reference_position"] = [
                engine_origin[i] + (n["reference_position"][i] - engine_origin[i]) * length_scale
                for i in range(3)
            ]
        elif n["identity"] == "powertrain.exhaust_manifold":
            n["reference_position"] = [
                engine_origin[i] + (n["reference_position"][i] - engine_origin[i]) * pipe_scale
                for i in range(3)
            ]

    # Real runner routing instead of one straight tube per cylinder
    # aimed at a single shared point (the "braid" every cylinder's own
    # line crossed every other one in). Ports stay pinned exactly where
    # they are -- only a waypoint is inserted between each port and the
    # shared plenum/manifold, splitting the ONE real physics edge
    # (flow_capacity_kg_s, circuit_identity preserved on both new
    # segments, so the fluid circuit discovery sees the same one real
    # connected circuit either way) into two:
    #   intake:  port -> riser (straight up, clears valve-cover height)
    #            -> plenum tower (raised to real valve-cover height,
    #            engine_geometry.mount_points' own top_r)
    #   exhaust: port -> sweep (down and back immediately, a real
    #            header's own first bend) -> collector
    # Each riser/sweep is also fanned sideways (a real lateral stagger,
    # scaled to the tube's own radius) so cylinders converging on the
    # same shared point don't route directly through each other's real
    # tube volume -- the closest this toy's straight-segment primitives
    # get to relaxing overlapping tubes apart.
    plenum_node = next((n for n in nodes if n["identity"] == "powertrain.intake_plenum"), None)
    manifold_node = next((n for n in nodes if n["identity"] == "powertrain.exhaust_manifold"), None)
    intake_ports = sorted((n for n in nodes if n["identity"].endswith(".intake_port")),
                          key=lambda n: n["reference_position"][0])
    exhaust_ports = sorted((n for n in nodes if n["identity"].endswith(".exhaust_port")),
                           key=lambda n: n["reference_position"][0])
    intake_edges_by_port = {e["b"]: e for e in edges if e["identity"].endswith(".intake_runner")}
    exhaust_edges_by_port = {e["a"]: e for e in edges if e["identity"].endswith(".exhaust_primary")}
    if plenum_node is not None and intake_ports:
        tower_clear_y = engine_geometry._engine_bank_radius_m(engine) * 1.5
        plenum_node["reference_position"][1] = max(plenum_node["reference_position"][1], tower_clear_y)
        n_riser = len(intake_ports)
        for idx, port in enumerate(intake_ports):
            runner_edge = intake_edges_by_port.get(port["identity"])
            if runner_edge is None:
                continue
            fan_z = intake_radius_m * 2.4 * (idx - (n_riser - 1) / 2.0)
            riser_id = f"{port['identity']}.riser"
            riser_pos = [port["reference_position"][0], tower_clear_y, port["reference_position"][2] + fan_z]
            node(riser_id, riser_pos, "engine-block-port", port_kind="intake-riser-waypoint")
            # runner_edge was (a=plenum, b=port) -- repointed to
            # (a=riser, b=port), i.e. riser<->port, keeping the real
            # .intake_runner identity/radius on the port-adjacent
            # segment; the new intake_runner_to_tower edge below covers
            # riser<->plenum. (Changing "b" here instead of "a" was the
            # bug: that silently orphaned the port node entirely, with
            # both segments ending up as duplicate plenum<->riser edges.)
            runner_edge["a"] = riser_id
            cyl_index = port["identity"].split("_")[1].split(".")[0]
            # flow_capacity_kg_s deliberately NOT duplicated here --
            # _discover_fluid_circuits sums it across every edge in the
            # circuit, treating each as an independent parallel path;
            # this segment and the port->riser one above are the SAME
            # physical tube in series (just bent), not two separate
            # chokepoints, so only one of them should carry the real
            # per-cylinder capacity or the circuit's total reads 2x real
            edge(f"powertrain.cylinder_{cyl_index}.intake_runner_to_tower", riser_id, "powertrain.intake_plenum",
                 "low-pressure-air-line", radius=intake_radius_m, circuit_identity="intake-air",
                 medium_rate_state="intake-air-flow-and-temperature")
    header_plan = None
    if manifold_node is None and layout:
        # no downpipe junction authored for this kind: give the planner
        # one to hand off to, placed above the engine
        node("powertrain.exhaust_manifold", [0.0, 0.6, 0.3], "exhaust-manifold", mass_kg=3.0)
        manifold_node = nodes[-1]
    if manifold_node is not None and layout:
        # Procedural header routing (exhaust_header.py): every primary
        # leaves its port a short way along the port's own direction,
        # bends, runs along a rail and joins its group's collector --
        # a real tubular header (grouped by the typical rule) or a log
        # manifold for header_type "stock-manifold". Replaces the old
        # single straight sweep per port whenever the cylinder layout
        # has real exhaust ports to route from.
        from exhaust_header import plan_exhaust_header, emit_header_graph
        header_plan = plan_exhaust_header(layout, header_type=exhaust.header_type, primary_radius_m=exhaust_radius_m)
        if header_plan.groups:
            emit_header_graph(header_plan, node, edge, "powertrain.exhaust_manifold", exhaust_radius_m,
                              exhaust_edges_by_port)
        else:
            header_plan = None
    if manifold_node is not None and exhaust_ports and header_plan is None:
        n_sweep = len(exhaust_ports)
        sweep_toward_x = manifold_node["reference_position"][0]
        for idx, port in enumerate(exhaust_ports):
            primary_edge = exhaust_edges_by_port.get(port["identity"])
            if primary_edge is None:
                continue
            fan_z = exhaust_radius_m * 2.4 * (idx - (n_sweep - 1) / 2.0)
            sweep_id = f"{port['identity']}.sweep"
            port_x = port["reference_position"][0]
            port_y = port["reference_position"][1]
            port_z = port["reference_position"][2]
            sweep_x = port_x + max(-0.03, min(0.03, sweep_toward_x - port_x))
            # Moves OUTWARD from the block (further along the port's own
            # already-clear Z side, real Y unchanged) rather than down
            # -- a fixed -0.03 Y drop used to route the pipe straight
            # back through the block's own surface (the port's Y
            # deliberately clears block_half_yz_m; dropping by more than
            # that margin put the sweep waypoint back inside the block's
            # own box). A real header sweeps back and clear of the
            # block, never back into it.
            outward_z = 0.03 if port_z >= 0.0 else -0.03
            sweep_pos = [sweep_x, port_y, port_z + outward_z + fan_z]
            node(sweep_id, sweep_pos, "engine-block-port", port_kind="exhaust-sweep-waypoint")
            primary_edge["a"] = sweep_id  # sweep -> manifold keeps the real .exhaust_primary identity/radius
            cyl_index = port["identity"].split("_")[1].split(".")[0]
            # flow_capacity_kg_s not duplicated -- see the matching
            # intake comment above; this segment and exhaust_primary
            # (port.sweep -> manifold) are one physical tube in series.
            edge(f"powertrain.cylinder_{cyl_index}.exhaust_port_to_sweep", port["identity"], sweep_id,
                 "exhaust-flow-path", radius=exhaust_radius_m, circuit_identity="exhaust",
                 medium_rate_state="exhaust-pulse-pressure-and-temperature")

    # Real fuel delivery: tank -> pump -> line -> rail (EFI) or float
    # bowl (carbureted) -- the standalone _vehicle_powertrain_graph
    # subunit this toy calls has no fuel-tank concept at all (the real
    # production compiler's own tank/rail nodes live in the FULL
    # whole-vehicle graph builder, unreachable without a
    # VehicleConfiguration this toy deliberately doesn't build -- see
    # module docstring). So this is a toy-own addition, built the same
    # real way the production compiler models nitrous (a depletable
    # "high-pressure-canister" bottle, HIGH_PRESSURE_LIQUID_SUPPLY_KINDS'
    # own generic pump/valve-gated-flow-from-a-finite-reservoir physics,
    # not reinvented) rather than an unmetered infinite fuel assumption.
    # A real single-shaft gas turbine ALSO has a genuine onboard liquid-
    # fuel tank/pump/rail -- the same real hardware class a piston
    # engine's EFI rail is (a pressurized line, a finite tank), not a
    # different mechanism just because the combustor isn't a piston.
    # The Otto-Langen atmospheric engine gets no tank/pump -- its real
    # historical fuel source is a piped coal-gas utility main (a
    # gasworks feed), not an onboard consumable tank at all. But "no
    # tank" is not "no connection": a real gas engine still has a real
    # physical fitting where that main's own flexible line attaches
    # (see the "atmospheric" branch below) -- no tank/pump to model
    # doesn't mean no port to declare.
    if getattr(engine, "fuel_network", None) is not None:
        # A declared fuel NETWORK (fuel_network.py) replaces the built-in
        # fuel branches below wholesale: the same "fuel" circuit
        # identity, the same capacity_kg / flow_capacity_kg_s attribute
        # conventions _discover_fluid_circuits already reads, so every
        # downstream consumer (the depletable-reservoir step, the
        # composition step, fuel_fill_frac/availability outputs) sees
        # one more ordinary fuel circuit -- which is the whole point.
        from fuel_network import emit_fuel_network_graph
        if engine.kind in ("combustion", "turbine"):
            terminal = ("powertrain.fuel_bowl" if engine.kind == "combustion" and engine.carburetor.is_carbureted
                        else "powertrain.fuel_rail")
            node(terminal, (front[0] + 0.05, front[1] + 0.05, front[2]),
                 "float-bowl" if terminal.endswith("fuel_bowl") else "fuel-rail")
        else:
            terminal = "powertrain.engine"
        emit_fuel_network_graph(engine.fuel_network, node, edge, (front[0], front[1], front[2]), terminal,
                                engine.kind == "combustion" and engine.carburetor.is_carbureted)
    elif engine.kind in ("combustion", "turbine"):
        fd = engine.fuel_delivery
        fuel_density = FUEL_DENSITY_KG_M3.get(engine.preferred_fuel_profile, 745.0)
        tank_capacity_kg = fd.tank_capacity_l * fuel_density / 1000.0
        # a turbine has no carburetor float bowl -- its real fuel
        # delivery point is a pressurized rail feeding the combustor's
        # fuel nozzles, the same "fuel-rail" real hardware class an EFI
        # piston engine's injectors are fed from
        rail_identity = ("powertrain.fuel_bowl" if engine.kind == "combustion" and engine.carburetor.is_carbureted
                          else "powertrain.fuel_rail")
        # The tank (and the pump feeding off it) are chassis-mounted --
        # a real fuel tank is never bolted to the engine itself, it sits
        # elsewhere in the vehicle and reaches the engine by a real
        # flexible fuel line. Placed in a genuinely separate area of the
        # scene (well below and off to the side) instead of tight
        # against the block, with the existing fuel.tank_to_pump/
        # fuel.pump_to_rail edges below now actually reading as that
        # real flexible line spanning the distance. The rail/bowl itself
        # IS engine-mounted (bolted to the carb/injectors) and stays
        # close, at `front`.
        chassis_remote = (front[0] - 0.35, -0.30, 0.40)
        tank_pos = (chassis_remote[0], chassis_remote[1], chassis_remote[2])
        pump_pos = (chassis_remote[0] + 0.06, chassis_remote[1] + 0.03, chassis_remote[2] - 0.03)
        rail_pos = (front[0] + 0.03, front[1] + 0.02, front[2])
        node("fuel.tank", tank_pos, "high-pressure-canister",
             mass_kg=2.0 + tank_capacity_kg * 0.05, capacity_kg=tank_capacity_kg)
        node("fuel.pump", pump_pos,
             "electro-mechanical-pump" if fd.pump_kind == "electric" else "mechanical-diaphragm-pump")
        # dressing.emit_dressing_graph (already run, above) creates this
        # exact rail/bowl identity itself whenever it finds real
        # injector/float-bowl bosses to hang it off of -- creating it
        # again here unconditionally, as this used to, left two nodes
        # sharing one identity (the same "which one does a lookup by
        # identity keep" ambiguity the throttle-body duplicate above
        # had), instead of just feeding this real pump/tank plumbing
        # into the one dressing already placed and sized.
        if not any(n["identity"] == rail_identity for n in nodes):
            node(rail_identity, rail_pos, "fuel-rail" if rail_identity.endswith("fuel_rail") else "float-bowl")
        edge("fuel.tank_to_pump", "fuel.tank", "fuel.pump", "fuel-supply-line",
             radius=max(0.002, fd.line_diameter_mm / 2000.0), circuit_identity="fuel",
             medium_rate_state="fuel-flow-and-pressure")
        edge("fuel.pump_to_rail", "fuel.pump", rail_identity, "fuel-supply-line",
             radius=max(0.002, fd.line_diameter_mm / 2000.0), circuit_identity="fuel",
             flow_capacity_kg_s=fd.pump_flow_capacity_kg_s, medium_rate_state="fuel-flow-and-pressure")
    elif engine.kind == "atmospheric":
        if engine.atmospheric_supply_tank_capacity_kg > 0.0:
            # A real on-site generator (gas_works.py) feeds a real
            # gasholder tank instead of an unlimited piped main -- the
            # SAME real depletable/fillable-reservoir mechanics
            # HIGH_PRESSURE_LIQUID_SUPPLY_KINDS already gives a
            # combustion engine's own fuel.tank above (capacity_kg on
            # the node is read into FluidCircuit.bottle_capacity_kg by
            # _discover_fluid_circuits automatically), not a parallel
            # mechanism invented for this engine kind. See
            # DrivetrainSolver._step_fluid_circuits' fuel branch for
            # the real production term layered onto that existing
            # consumption-only logic.
            main_pos = (front[0] - 0.15, front[1], front[2] - 0.05)
            node("fuel.gasholder", main_pos, "high-pressure-canister",
                 capacity_kg=engine.atmospheric_supply_tank_capacity_kg)
            edge("fuel.gasholder_to_engine", "fuel.gasholder", "powertrain.engine",
                 "fuel-supply-line", radius=0.02, circuit_identity="fuel")
        else:
            # No tank, no pump -- the real supply is an unlimited off-
            # engine utility main, not a depletable reservoir, so there's
            # no real fluid-circuit physics to solve here (fuel_fill_frac
            # stays 1.0 -- see engine_cycle_sim._step_atmospheric's own
            # comment). But the real physical connection point still
            # exists: a gas cock/fitting where the main's own flexible
            # line attaches to the engine, off-engine infrastructure by
            # definition (a municipal gas main is never part of the
            # engine itself, same real "supplied-elsewhere" reasoning a
            # car's fuel tank gets, just for a different real reason).
            main_pos = (front[0] - 0.15, front[1], front[2] - 0.05)
            node("fuel.gas_main_connection", main_pos, "gas-supply-fitting")
            edge("fuel.gas_main_to_engine", "fuel.gas_main_connection", "powertrain.engine",
                 "fuel-supply-line", radius=0.006)

    # A real throttle body/butterfly plate, sitting between atmosphere
    # and the plenum -- the standalone subunit has no such node either
    # (the real cable/lever/plate assembly lives in the whole-vehicle
    # graph's controls.throttle.* chain, unreachable here, same as the
    # fuel rail above); this is the toy-own stand-in, positioned along
    # the real line from the engine block to the plenum node (itself
    # already real, from the production subunit) so it sits where a
    # throttle body actually is -- physically between atmosphere/filter
    # and the plenum it feeds. The real plate ANGLE (engine_cycle_sim's
    # throttle_plate_angle_deg, driven by the real cable-linkage-travel
    # + circular-orifice-area physics, not a linear pedal mapping) is
    # exposed on EngineCycleState for a future renderer to actually
    # animate the plate; this pass only places the housing geometry.
    if engine.kind == "combustion" and engine.architecture.cylinders:
        plenum_node = next((n for n in nodes if n["identity"] == "powertrain.intake_plenum"), None)
        # dressing.emit_dressing_graph (above) already builds a real
        # throttle body/carburetor/inlet-elbow under this exact identity
        # whenever the cylinder layout has intake ports at all -- this
        # stand-in is only for the layouts that reach here without one
        # (no discrete intake-valve ports for dressing to hang a
        # throttle body off of, e.g. two-stroke transfer/scavenge
        # ports). Creating a second node under the SAME identity
        # unconditionally, as this used to, didn't replace dressing's
        # real one: both stayed in the graph, so every edge naming
        # "powertrain.throttle_body" (dressing's own air-filter feed
        # included) resolved to whichever of the two a given consumer's
        # identity lookup happened to keep -- usually this one, dropped
        # down near the crank rather than up at the real inlet -- while
        # dressing's correctly-placed one was left with no edges at all.
        if plenum_node is not None and not any(n["identity"] == "powertrain.throttle_body" for n in nodes):
            plenum_pos = plenum_node["reference_position"]
            throttle_body_pos = [
                engine_origin[i] + (plenum_pos[i] - engine_origin[i]) * 0.4 for i in range(3)
            ]
            node("powertrain.throttle_body", throttle_body_pos, "engine-block-component")
            edge("powertrain.throttle_body_to_plenum", "powertrain.throttle_body", "powertrain.intake_plenum",
                 "low-pressure-air-line", radius=intake_radius_m * 1.4, circuit_identity="intake-air")

    # Real front-of-block accessory mounting: every belt-driven accessory
    # sits at its own real, individually-named position immediately off
    # the block's own front face (engine_geometry.accessory_mount_points
    # -- front[0]-0.04, each accessory offset from that by a few real
    # centimeters, the same scale a bolt boss/bracket standoff actually
    # is), not floating out on a separate arc plane disconnected from
    # the block entirely. accessory_mount_points is the one place these
    # positions are defined, so the belt-driven-compressor accessories
    # below (AC, pneumatics) get their own slots there too instead of a
    # second, separate placement scheme.
    fi = engine.forced_induction
    # accessory_mount_points now anchors the whole arc at the crank's
    # own real FRONT FACE (crank_x_min) -- the real timing-cover end,
    # not the crank's geometric center. The belt-drive edges below
    # (production-authored, from "powertrain.engine" directly) are
    # repointed to originate from "powertrain.crank_shaft.rear"'s own
    # sibling, powertrain.crank_shaft.front -- the real front-of-crank
    # reference already built above, not a new synthetic node -- so
    # they stay coplanar with the accessories instead of cutting
    # diagonally across the whole block's length to reach x=0.
    mount_points = engine_geometry.accessory_mount_points(engine)
    node_by_identity = {n["identity"]: n for n in nodes}
    for belt_edge_id in ("drivetrain.engine_to_alternator_belt", "drivetrain.engine_to_water_pump_belt",
                          "drivetrain.engine_to_fan_belt"):
        belt_edge = next((e for e in edges if e["identity"] == belt_edge_id), None)
        if belt_edge is not None and belt_edge["a"] == "powertrain.engine":
            belt_edge["a"] = "powertrain.crank_shaft.front"

    if "alternator" in mount_points:
        target = node_by_identity.get("electrical.alternator")
        if target is not None:
            target["reference_position"] = list(mount_points["alternator"])
    if "water_pump" in mount_points:
        target = node_by_identity.get("powertrain.water_pump")
        if target is not None:
            target["reference_position"] = list(mount_points["water_pump"])
    if "fan" in mount_points:
        target = node_by_identity.get("powertrain.cooling_fan")
        if target is not None:
            new_fan_pos = mount_points["fan"]
            target["reference_position"] = list(new_fan_pos)
            # The fan is one physical stack with the radiator/condenser
            # it pulls air through -- it cannot move onto the accessory
            # arc while they stay behind at the production subunit's own
            # fixed offset (radiator -.10 behind the fan, condenser -.15
            # behind it, the same real relative stacking every build
            # uses); that would tear the cooling stack in half with
            # nothing physically connecting the pieces. Carried along
            # with the fan to its new position, preserving that same
            # real relative offset.
            for identity, relative_offset in (
                ("powertrain.radiator", (-0.10, 0.0, 0.0)),
                ("powertrain.condenser", (-0.15, 0.0, 0.0)),
                ("powertrain.engine_block_port.condenser_refrigerant", (-0.15, 0.01, 0.02)),
            ):
                stack_target = node_by_identity.get(identity)
                if stack_target is not None:
                    stack_target["reference_position"] = [
                        new_fan_pos[i] + relative_offset[i] for i in range(3)
                    ]
    # The two real electrical boxes the sim's control programs live in
    # (ecu.EngineControlUnit / ignition_driver.IgnitionDriver), carried
    # under the production vehicle graph's own identities and kinds
    # (abstract_ui_vehicles.py's electrical_points): the ignition driver
    # rides on the engine at production's own engine-relative offset
    # (+.02 X, +.10 Y above the block, -.12 Z), the ECU is chassis-fixed
    # on the firewall side behind the block (production places it at a
    # chassis-absolute [.18, .26, -.18]; here that offset is taken from
    # the block's rear face since the engine sits at the origin). The
    # "crank-cam-timed-ignition" wire between them is production's own
    # ignition_command edge kind -- physics-inert to DrivetrainSolver,
    # a routing/topology edge only, like cosmetic-belt-wrap.
    node("electrical.ignition_driver",
         [(crank_x_min + crank_x_max) / 2.0 + 0.02, block_half_y + 0.10, -(block_half_z + 0.12)],
         "electrical-junction", fixed_to="engine", circuit_role="ignition_driver",
         mass_kg=0.0, mass_in_total=False, body_half_extent_m=[0.05, 0.02, 0.07])
    node("electrical.ecu", [crank_x_max + 0.18, 0.26, -0.18], "vehicle-computer",
         fixed_to="chassis", circuit_role="ecu",
         mass_kg=0.0, mass_in_total=False, body_half_extent_m=[0.02, 0.08, 0.10])
    edge("ignition_command", "electrical.ecu", "electrical.ignition_driver",
         "crank-cam-timed-ignition", wire_gauge_mm2=8.0)
    # The electrical network as real hardware, production identities and
    # circuits throughout (abstract_ui_vehicles.py's electrical_points /
    # wire_routes): the battery and fusebox are free-mounted (chassis
    # side, here in the front-corner pocket and beside the ECU), the
    # ground is a real single-point strap lug on the block rear face
    # (electrical.engine_ground -- the heaviest return current in the
    # vehicle, starter and ignition, lands on the block). Every feed
    # wire is production's insulated-copper-wire kind with its own real
    # circuit name and fuse rating; the ground straps carry the new
    # chassis-ground-return circuit. Physics-inert to the torque solver
    # -- the bus itself is solved in electrical_network.py.
    # The battery BANK as real modules (electrical_network.Battery picks
    # the production BATTERY_MODULES type and the series x parallel
    # count for this engine's system voltage and cold-crank current):
    # each module is its own placed node with its real case size and
    # mass, laid in a row in the tray in the front-corner pocket, wired
    # module-to-module (series strings then parallel ties) into one
    # terminal node electrical.battery that the rest of the harness
    # already lands on. A trimmer gets one 7 Ah SLA; a 24 V Cat C18 gets
    # two 8Ds in series -- never one imaginary battery.
    bank = Battery.for_engine(engine)
    module_half = [float(v) for v in BATTERY_MODULES[bank.module_kind]["half_extent_m"]]
    tray = [crank_x_min - 0.12, block_half_y * 0.5, -(block_half_z + 0.28)]
    pitch_z = 2.0 * module_half[2] + 0.02
    module_ids = []
    for s in range(bank.series_count):
        for p in range(bank.parallel_count):
            idx = s * bank.parallel_count + p
            mid = f"electrical.battery.module_{idx + 1}"
            module_ids.append(mid)
            node(mid, [tray[0] - p * (2.0 * module_half[0] + 0.02), tray[1],
                       tray[2] - idx * pitch_z + (bank.module_count - 1) * pitch_z / 2.0],
                 "lead-acid-battery-module", fixed_to="chassis", circuit_role="battery",
                 module_kind=bank.module_kind, series_position=s + 1, parallel_position=p + 1,
                 mass_kg=bank.module_mass_kg, mass_in_total=True, body_half_extent_m=module_half,
                 material="pressed-steel")
    node("electrical.battery", [tray[0] + 0.02, tray[1] + module_half[1] + 0.03, tray[2]],
         "electrical-junction", fixed_to="chassis", circuit_role="battery",
         mass_kg=0.0, mass_in_total=False, body_half_extent_m=[0.03, 0.015, 0.03],
         bank_module_kind=bank.module_kind, bank_series=bank.series_count, bank_parallel=bank.parallel_count,
         bank_capacity_ah=bank.capacity_ah, bank_nominal_v=bank.nominal_voltage_v)
    # series strings: module i -> i+1 within a string; string ends and
    # parallel ties land on the terminal
    for s in range(bank.series_count):
        for p in range(bank.parallel_count):
            idx = s * bank.parallel_count + p
            a = module_ids[idx]
            b = module_ids[(s + 1) * bank.parallel_count + p] if s + 1 < bank.series_count else "electrical.battery"
            edge(f"electrical.wire.battery_link_{idx + 1}", a, b, "insulated-copper-wire", radius=.006,
                 palette="active", circuit="battery-interconnect", maximum_current_a=400.0,
                 electrical_authority="vehicle-computer-fusebox-relay-dispatch",
                 routing="relaxed-multi-segment-harness", bundle_kind="battery-cable")
    node("electrical.fusebox", [crank_x_max + 0.18, 0.26, 0.18], "electrical-junction",
         fixed_to="chassis", circuit_role="fusebox",
         mass_kg=0.0, mass_in_total=False, body_half_extent_m=[0.03, 0.06, 0.08])
    node("electrical.engine_ground", [crank_x_max - 0.02, -block_half_y * 0.6, -(block_half_z + 0.01)],
         "electrical-junction", fixed_to="engine", circuit_role="engine_ground",
         mass_kg=0.0, mass_in_total=False, body_half_extent_m=[0.012, 0.012, 0.006])
    present = {n["identity"] for n in nodes}
    for name, a, b, circuit, maximum_current in (
        ("battery_feed", "electrical.battery", "electrical.fusebox", "battery-main", 80.0),
        ("ecu_feed", "electrical.fusebox", "electrical.ecu", "computer-and-ignition", 12.0),
        ("alternator_charge", "electrical.alternator", "electrical.battery", "alternator-charge", 95.0),
        ("battery_ground_strap", "electrical.battery", "electrical.engine_ground", "chassis-ground-return", 180.0),
        ("ac_clutch_feed", "electrical.fusebox", "powertrain.ac_compressor", "computer-and-ignition", 10.0),
    ):
        if a not in present or b not in present:
            continue   # no alternator on a total-loss build, no AC compressor fitted, ...
        edge(f"electrical.wire.{name}", a, b, "insulated-copper-wire", radius=.0035,
             palette="active", circuit=circuit, maximum_current_a=maximum_current,
             electrical_authority="vehicle-computer-fusebox-relay-dispatch",
             routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")
    # Real natural shapes instead of the generic mass-scaled CUBE every
    # small part defaults to -- unconditional (not only when the fan
    # exists to carry them along above), since a radiator/condenser can
    # be present with no mechanical fan at all (water_pump alone).
    radiator_half_yz = engine_geometry.block_half_yz_m(engine) * 1.6
    for identity in ("powertrain.radiator", "powertrain.condenser"):
        stack_target = node_by_identity.get(identity)
        if stack_target is not None:
            # a real core: thin in the airflow direction (X, this toy's
            # own crank/forward axis), tall and wide in Y/Z -- not a cube
            stack_target["body_half_extent_m"] = [0.018, radiator_half_yz, radiator_half_yz * 0.85]
    fan_target = node_by_identity.get("powertrain.cooling_fan")
    if fan_target is not None:
        # a real circular fan blade sweep -- rendered as a disc (vehicle_
        # mesh.py's build_drivetrain_solid_parts special-cases any node
        # declaring fan_disk_radius_m into a real short/wide tube instead
        # of the generic cuboid every other node gets), not a small cube
        fan_target["fan_disk_radius_m"] = radiator_half_yz * 0.8
    if "ac_compressor" in mount_points:
        # a real belt-driven AC compressor, through the generic real
        # mechanical port every belt-driven compressor plugs into the
        # crank through (_add_belt_driven_compressor) -- what it does
        # with that real rotating power (reject heat via a refrigerant
        # loop) is this accessory's own concern, not the port's.
        ac_rated_w = max(300.0, min(4000.0, engine.peak_torque_nm * 10.0))
        _add_belt_driven_compressor(node, edge, "ac_compressor", mount_points["ac_compressor"], ac_rated_w,
                                     crank_identity="powertrain.crank_shaft.front")
    if engine.pneumatics.compressor_fitted:
        # Typed compressed-air hardware (engines.PneumaticSystem /
        # production's PNEUMATIC_COMPONENTS): the compressor is either the
        # SAME belt-clutch mechanical port the AC compressor uses (a real
        # crank-belt piston compressor), or a motor-driven unit -- on the
        # 12/24 V bus (electrical.pneumatic_compressor, production's own
        # identity and compressor_feed circuit), or, for a ship, on the
        # generator switchboard outside this subgraph. A real pressure-
        # switch unloader sits between compressor and tank (loads below
        # cut-in, unloads at cut-out), and the tank is a real vessel at
        # its own declared working pressure: a truck's 8 bar reservoir or
        # a ship's 30 bar starting-air receiver.
        pn = engine.pneumatics
        if pn.compressor_drive == "crank-belt" and "pneumatic_compressor" in mount_points:
            pneu_pos = mount_points["pneumatic_compressor"]
            _add_belt_driven_compressor(node, edge, "pneumatic_compressor", pneu_pos, pn.compressor_rated_w,
                                         crank_identity="powertrain.crank_shaft.front")
            compressor_id = "pneumatic_compressor"
        else:
            pneu_pos = (crank_x_min - 0.25, -0.20, -(block_half_z + 0.35))
            compressor_id = "electrical.pneumatic_compressor"
            node(compressor_id, pneu_pos, "electric-compressor", fixed_to="chassis",
                 compressor_kind=pn.compressor_kind, drive=pn.compressor_drive,
                 rated_w=pn.compressor_rated_w, circuit_role="pneumatic_compressor",
                 mass_kg=max(5.0, pn.compressor_rated_w * 0.01), mass_in_total=True,
                 body_half_extent_m=[0.12 + pn.compressor_rated_w * 2e-6] * 3)
            if pn.compressor_drive == "electric-motor":
                edge("electrical.wire.compressor_feed", "electrical.fusebox", compressor_id,
                     "insulated-copper-wire", radius=.0035, palette="active", circuit="tire-and-shock-air",
                     maximum_current_a=48.0, electrical_authority="vehicle-computer-fusebox-relay-dispatch",
                     routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")
        # what the tank holds when charged to its working pressure (ideal
        # gas: 1 atm air density scaled by the pressure ratio)
        tank_capacity_kg = (pn.reserve_tank_capacity_l / 1000.0
                            * AIR_DENSITY_AT_1_ATM_KG_M3 * pn.tank_pressure_pa / 101_325.0)
        # real pressure-vessel mass: thin-wall hoop stress -- wall
        # thickness = p*r/(2*sigma), steel at 150 MPa allowable, on a
        # cylinder of the declared volume (L/D = 3)
        volume_m3 = pn.reserve_tank_capacity_l / 1000.0
        radius_m = (volume_m3 / (3.0 * math.pi)) ** (1.0 / 3.0)
        length_m = 3.0 * 2.0 * radius_m
        wall_m = max(0.002, pn.tank_pressure_pa * radius_m / (2.0 * 150e6))
        shell_mass_kg = 2.0 * math.pi * radius_m * length_m * wall_m * 7850.0
        node("pneumatic_regulator", (pneu_pos[0] - 0.15, pneu_pos[1] - 0.05, pneu_pos[2]),
             "pressure-switch-unloader", cut_in_frac=pn.regulator_cut_in_frac,
             cut_out_frac=pn.regulator_cut_out_frac, mass_kg=0.6, mass_in_total=True,
             body_half_extent_m=[0.04, 0.04, 0.04])
        node("pneumatic_reserve_tank", (pneu_pos[0] - 0.35 - length_m / 2.0, -0.30, -0.40 - radius_m),
             "high-pressure-canister", tank_kind=pn.tank_kind, working_pressure_pa=pn.tank_pressure_pa,
             mass_kg=shell_mass_kg + 2.0, mass_in_total=True, capacity_kg=max(tank_capacity_kg, 0.01),
             material="pressed-steel", body_half_extent_m=[length_m / 2.0, radius_m, radius_m])
        edge("pneumatic_compressor_to_regulator", compressor_id, "pneumatic_regulator",
             "compressed-air-line", radius=0.006, circuit_identity="pneumatic-reserve",
             material="steel-pipe")
        edge("pneumatic_regulator_to_tank", "pneumatic_regulator", "pneumatic_reserve_tank",
             "compressed-air-line", radius=0.006, circuit_identity="pneumatic-reserve",
             material="steel-pipe")

    # The real belt loop itself, drawn -- accessory_mount_points already
    # placed every one of these at the SAME x, arranged around a circle;
    # this is that circle made actually visible (a discrete set of
    # points that merely satisfy a circle equation reads as scattered
    # dots, not a ring, until something traces the curve between them).
    # Connected in the SAME angular order accessory_mount_points used,
    # not insertion order, so the loop wraps the circle instead of
    # crossing it. Purely visual -- "cosmetic-belt-wrap" is never
    # dispatched by DrivetrainSolver.step() and carries no physics; the
    # real torque coupling for AC/pneumatics is the friction-clutch-shaft
    # edge _add_belt_driven_compressor already built above.
    # the universal bolt-on parts every real crank engine carries (engine_
    # parts.py): exhaust downstream of the collectors, damper/flywheel/
    # ring gear + the starter hardware that meshes with it, belt pulleys
    # and tensioner, timing cover and drive, coolant bottle/heater core,
    # PCV/EGR, bellhousing -- emitted here, after headers and dressing,
    # so every anchor node already sits at its final real position
    from engine_parts import emit_universal_parts
    emit_universal_parts(engine, layout, nodes, edges, node, edge)

    accessory_node_identity = {
        "alternator": "electrical.alternator", "water_pump": "powertrain.water_pump",
        "fan": "powertrain.cooling_fan", "ac_compressor": "ac_compressor",
        "pneumatic_compressor": "pneumatic_compressor",
    }
    ring_ids = [accessory_node_identity[name] for name in mount_points
                if name in accessory_node_identity and accessory_node_identity[name] in node_by_identity]
    if len(ring_ids) >= 2:
        for i, node_id in enumerate(ring_ids):
            next_id = ring_ids[(i + 1) % len(ring_ids)]
            edge(f"cosmetic_belt_wrap_{i}", node_id, next_id, "cosmetic-belt-wrap", radius=0.004)

    if fi.kind == "supercharger":
        # no forced-induction concept anywhere in the real engine preset
        # system to extend -- this stays genuinely this toy's own.
        # Right off the block's front face too (front, the same
        # reference every other accessory above mounts from), not a
        # separate hand-picked offset -- a real blower sits directly on
        # the block, driven off the same crank pulley.
        node("supercharger_rotor", (front[0], front[1] + 0.06, front[2]),
             "rotating-mass", mass_kg=1.5, inertia_kg_m2=0.0015)
        edge("engine_to_supercharger_belt", "powertrain.engine", "supercharger_rotor", "accessory-drive-belt",
             ratio_coordinate="supercharger_belt_ratio", ratio=fi.belt_ratio,
             stiffness_nm_per_rad=engine.peak_torque_nm * 40.0,
             damping_nm_per_rad_s=engine.peak_torque_nm * 0.3,
             max_torque_nm=engine.peak_torque_nm * 0.20,
             backlash_rad=0.004)

    # The dyno absorber: test equipment, not a vehicle part -- no real
    # vehicle graph would ever have one, so it stays genuinely this toy's
    # own. A real bench dyno roller is hollow steel tube, not solid cast
    # iron -- light enough that the engine under test dominates the
    # dynamics, not the drum's own inertia. Two real, different coupling
    # mechanisms to the crank are offered, selected by the caller:
    #   friction-clutch-shaft    -- the same tanh-saturating slip formula
    #                               as the real game's clutch_torque
    #   rolling-friction-contact -- a real chassis-dyno roller: dry
    #                               Coulomb friction, not a bolted shaft
    # Radius stays a real, bounded, purely visual/geometric dyno-roller
    # size (0.16-0.25m, a real benchtop-to-engine-dyno range) -- mass is
    # what actually carries the scaling now, DERIVED to hit a real
    # target inertia proportional to THIS ENGINE'S OWN crank inertia
    # (engine.inertia_kg_m2), not a fixed 8kg/0.16m floor. That fixed
    # floor was fine at car-engine scale but became a genuine bug at
    # the catalogue's real span: a 25cc trimmer's crank inertia is
    # 0.00006 kg*m^2, and the old floor gave it a 0.156 kg*m^2 roller --
    # 2600x heavier than the crank it was supposed to load. The real,
    # already-scaled clutch coupling (_build_brake_junction, ~3x peak
    # torque) could never spin a roller that heavy up at all: it just
    # saturated back and forth every tick, pinning the crank near idle
    # regardless of throttle -- the trimmer "not changing speed" bug.
    # A roller a real, disclosed 3x the engine's own crank inertia keeps
    # the design intent above (light enough that the engine dominates)
    # while actually scaling across the same 9 orders of magnitude the
    # rest of this catalogue already spans.
    drum_radius_m = 0.16 + 0.09 * min(1.0, engine.peak_torque_nm / 400.0)
    target_drum_inertia_kg_m2 = max(1e-8, engine.inertia_kg_m2 * 3.0)
    drum_mass_kg = target_drum_inertia_kg_m2 / max(0.75 * drum_radius_m * drum_radius_m, 1e-9)
    # Shifted by the same driveline_shift_x the clutch/transmission/
    # transfer-case chain above got, so the dyno stays beyond that real
    # chain's own new end instead of drifting back inside it once the
    # chain moved out to clear the real cylinder bank.
    node("dyno_absorber", (0.6 + driveline_shift_x, 0.0, 0.0), "rotating-mass",
         mass_kg=drum_mass_kg, inertia_kg_m2=target_drum_inertia_kg_m2,
         drum_radius_m=drum_radius_m)
    peak = max(engine.peak_torque_nm, 1.0)

    edge("dyno_friction_clutch", "powertrain.engine", "dyno_absorber", "friction-clutch-shaft",
         stiffness_nm_per_rad_s=peak * 8.0, max_torque_nm=peak * 3.0)
    edge("dyno_roller_contact", "powertrain.engine", "dyno_absorber", "rolling-friction-contact",
         friction_coefficient=0.9, normal_force_n=peak * 45.0, contact_radius_m=drum_radius_m)

    if engine.accessories.electric_fan and engine.accessories.water_pump:
        # A real electric cooling fan: its own small BLDC motor, no
        # mechanical coupling to the crank at all -- no belt/clutch edge
        # to anything. The concrete case proving the fan/rotor load
        # abstraction covers "with or without connection to a drive
        # shaft": DrivetrainSolver.step() commands this node's omega
        # directly off a real thermostat each tick (see
        # ELECTRIC_FAN_ON_TEMP_K), the same way a real relay/PWM
        # controller drives one, rather than the torque solver
        # integrating it off a shaft that doesn't exist.
        node("powertrain.cooling_fan_electric",
             (front[0] - 0.3, front[1] + 0.05, 0.0), "rotating-mass",
             mass_kg=0.6, inertia_kg_m2=0.0004,
             fan_airflow_m3_s_per_rad_s=ELECTRIC_FAN_FLOW_COEFF_M3_S_PER_RAD_S,
             fan_rated_omega_rad_s=ELECTRIC_FAN_RATED_OMEGA_RAD_S)
        # its feed off the fusebox, the ECU's relay closing it (see
        # ecu.EngineControlUnit.electric_fan_command) -- production's
        # insulated-copper-wire kind, the new "cooling-fan" circuit
        edge("electrical.wire.fan_feed", "electrical.fusebox", "powertrain.cooling_fan_electric",
             "insulated-copper-wire", radius=.0035, palette="active", circuit="cooling-fan",
             maximum_current_a=30.0, electrical_authority="vehicle-computer-fusebox-relay-dispatch",
             routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")

    propeller_rated_torque_nm = 0.0
    propeller_rated_omega_rad_s = 0.0
    if engine.identity in PROPELLER_COUPLED_ENGINES:
        # Real "propeller law": a fixed-pitch prop is sized so its own
        # hydrodynamic/aerodynamic drag exactly absorbs the engine's
        # rated torque at its rated speed -- calibrated here straight
        # from the engine's own already-declared peak_torque_nm/
        # redline_rpm, no new per-engine tuning. Applied on the
        # dyno_absorber's own shaft (this toy's stand-in for "whatever
        # the output shaft is actually turning" -- there's no separate
        # propeller node), additional to any manual dyno brake_load_nm
        # a test still dials in on top of it, exactly like a real ship's
        # engine always sees its own propeller curve plus whatever extra
        # sea/current load is present.
        propeller_rated_torque_nm = engine.peak_torque_nm
        propeller_rated_omega_rad_s = engine.redline_rpm * RPM_TO_RAD_S

    # Final real safety net, not a new independent placement scheme: the
    # cooling stack (radiator/condenser) still carries the production
    # subunit's own small FIXED offset when there's no mechanical fan to
    # carry it along with (see the "fan" in mount_points cascade above,
    # which only fires when a real belt-driven fan exists to anchor it
    # to). For almost the whole catalogue that fixed offset already
    # clears the now-correctly-scaled block; for the one real outlier
    # (a 25,327L 14-cylinder marine diesel whose own block is 10.5m
    # long) it doesn't, and no small offset could -- the whole "small
    # front-of-block cluster" convention itself stops being physically
    # meaningful at that scale (a real engine like this has a separate
    # heat-exchanger plant, not a car-style radiator bolted nearby).
    # Pushed out along X just enough to clear the block's own real
    # length instead of silently sitting inside it.
    # block_half_x is the same real half-length whether the block ended
    # up as one monolithic node or split into per-cylinder segments
    # above -- using it directly here instead of looking up a single
    # "powertrain.engine_block_body" node, which no longer exists once
    # the block is segmented.
    if block_half_x > 0.0:
        clear_x = block_half_x + 0.05
        for n in nodes:
            if n["identity"] in ("powertrain.radiator", "powertrain.condenser",
                                  "powertrain.engine_block_port.condenser_refrigerant"):
                if abs(n["reference_position"][0]) < clear_x:
                    n["reference_position"][0] = -clear_x if n["reference_position"][0] <= 0 else clear_x

    return {"schema": "engine-toy-drivetrain-graph-v1", "identity": f"{engine.identity}/drivetrain",
            "cylinder_layout": cylinder_layout,
            "dressing": {k: (v if isinstance(v, str) else v.__dict__) for k, v in dressing_report.items()},
            "nodes": nodes, "edges": edges,
            "propeller_rated_torque_nm": propeller_rated_torque_nm,
            "propeller_rated_omega_rad_s": propeller_rated_omega_rad_s}


# Everything past the block's own freewheel (the direct_drive_bypass dog
# clutch, alongside the main clutch and its flywheel wrench) -- the
# transmission, transfer case, their mounts, and the toy-only dyno rig --
# is real driveline, but it isn't "the engine" a mesh view of the engine
# itself is for. Excluded from the DEFAULT engine mesh view only;
# DrivetrainSolver still always gets the real, complete, unfiltered
# graph everywhere else.
ENGINE_VIEW_EXCLUDE_PREFIXES = (
    "dyno_absorber", "powertrain.transmission", "powertrain.transfer_case",
    "mount.transmission", "mount.transfer_case",
)


def engine_mesh_view_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """The real graph, cropped to the engine itself -- everything up
    through the clutch/flywheel/freewheel, minus
    ENGINE_VIEW_EXCLUDE_PREFIXES. This is a display default, not a
    different graph: callers that need the full driveline (physics,
    the bay-packing envelope) should keep using build_drivetrain_graph's
    own return value directly."""
    keep_nodes = [n for n in graph["nodes"]
                  if not any(n["identity"].startswith(p) for p in ENGINE_VIEW_EXCLUDE_PREFIXES)]
    keep_ids = {n["identity"] for n in keep_nodes}
    keep_edges = [e for e in graph["edges"] if e["a"] in keep_ids and e["b"] in keep_ids]
    return {**graph, "nodes": keep_nodes, "edges": keep_edges}


@dataclass
class _EdgeState:
    relative_angle_rad: float = 0.0
    last_torque_nm: float = 0.0
    engaged: bool = False
    wear: float = 0.0
    glaze: float = 0.0
    missing_endpoint: bool = False


@dataclass
class FluidCircuit:
    """One closed fluid system -- a connected component of fluid-line
    edges, discovered from the graph itself, not hand-listed. Each
    circuit gets its own lumped state and steps independently, at
    whatever rate its own physics actually needs (thermal-liquid and
    hydraulic are slow; compressible-gas needs the volume-derived fill
    time constant, still far coarser than the torque solver's substeps)."""
    kind_class: str        # "thermal-liquid" | "compressible-gas" | "incompressible-hydraulic"
    nodes: frozenset[str]
    edges: tuple[dict, ...]
    pump_node: str | None = None
    fan_nodes: tuple[str, ...] = ()
    thermal_mass_kj_per_k: float = 10.0
    passive_heat_loss_w_per_k: float = 0.0
    active_heat_exchange_w_per_k: float = 0.0   # a real radiator's own coefficient, if present
    heat_share: float = 0.0        # this circuit's share of engine waste heat, 0..1
    supply_pressure_pa: float = 0.0  # compressible-gas / hydraulic target pressure
    flow_capacity_kg_s: float = 0.0  # real valve-curtain/port choke limit, compressible-gas only
    heat_soak_w_per_k: float = 0.0  # a plenum/manifold body's own passive rejection, gas circuits
    relief_pressure_pa: float = 0.0  # a real relief-valve cap, oil circuit only
    # a real wax-pellet thermostat valve in this circuit (abstract_ui_
    # vehicles.py's powertrain.coolant_thermostat), if any: flow to the
    # radiator is gated proportionally by its opening, the rest bypasses
    thermostat_opening_start_k: float = 0.0
    thermostat_full_open_k: float = 0.0
    thermostat_tau_s: float = 15.0
    thermostat_open_frac: float = field(default=0.0, init=False)
    exchanger_medium: str = "air"   # "air" (fan-blown radiator) | "seawater" (pumped plate exchanger)
    pump_design_flow_lpm: float = 0.0     # the pump node's own declared sizing (0 = undeclared, old relation)
    pump_rated_omega_rad_s: float = 0.0
    # compressed-air circuits: the tank's working pressure, the unloader's
    # hysteresis, and a non-belt compressor's own rating/drive
    working_pressure_pa: float = 0.0
    regulator_cut_in_frac: float = 0.84
    regulator_cut_out_frac: float = 1.0
    compressor_rated_w: float = 0.0
    compressor_drive: str = "crank-belt"
    compressor_loaded: bool = field(default=True, init=False)
    compressor_running_w: float = field(default=0.0, init=False)
    # a real progressive flow regulator on a high-pressure-liquid-supply
    # circuit (a WMI kit's own metering valve, powertrain.auxiliary_
    # injection_regulator's own declared rating) -- 0 means undeclared,
    # falls back to the old tank-fraction-derived estimate (real for a
    # genuinely un-metered, simply on/off kit like nitrous)
    regulated_flow_kg_s: float = 0.0
    regulator_onset_map_frac: float = 0.0
    regulator_full_map_frac: float = 0.0
    volume_l: float = 1.0
    temp_k: float = field(default=293.15, init=False)
    pressure_pa: float = field(default=101_325.0, init=False)
    # the real choke-capped equilibrium _step_fluid_circuits just solved
    # for -- what draw_cylinder_charge references as its upstream value,
    # NOT raw supply_pressure_pa (that's the commanded, pre-choke value;
    # using it here let a firing event's many-times-per-tick pulls drag
    # the plenum back toward it, overpowering the once-per-tick choke
    # relaxation and defeating the whole real flow-capacity ceiling)
    achievable_target_pressure_pa: float = field(default=101_325.0, init=False)
    # high-pressure-liquid-supply only: a real depletable bottle
    bottle_capacity_kg: float = 0.0
    fill_level_frac: float = field(default=1.0, init=False)
    valve_open: bool = field(default=False, init=False)
    delivered_flow_kg_s: float = field(default=0.0, init=False)
    # high-pressure-liquid-supply only: what the bottle's contents
    # actually are (see step_reservoir_composition) -- 1.0 = the pure
    # nominal fluid, which is what every pre-filled tank/bottle starts
    # as; an on-site generator's gasholder is the one real circuit
    # that gets filled with something OTHER than pure fluid (its own
    # unpurged air at first, then whatever purity the generator is
    # actually making -- see engine_cycle_sim._step_atmospheric)
    composition_frac: float = field(default=1.0, init=False)
    # thermal-liquid only: the real pump-speed-derived volumetric flow
    # _step_fluid_circuits already computes each tick to weight its own
    # heat-rejection term -- stored here so a caller can report it too,
    # instead of it being thrown away as a local
    flow_lpm: float = field(default=0.0, init=False)


def _circuit_thermal_capacity(edges: list[dict], node_ids: frozenset[str], node_by_id: dict[str, dict],
                              kind_class: str) -> tuple[float, float]:
    """(thermal capacity kJ/K, fluid volume L) of one fluid circuit from
    its declared construction -- production's FLUID_LINE_MATERIALS /
    FLUID_MEDIA tables, read off each line's `material`/`fluid`/`radius`
    (wall thickness from the material) and each vessel's `material`/
    `fluid`/`fluid_volume_l`/`mass_kg`. A line's length is the real
    distance between its endpoints (through its waypoints if it has
    them). Anything undeclared falls back to the old lumped guess so an
    older graph still steps."""
    default_fluid = "engine-oil" if any("oil" in nid for nid in node_ids) else "coolant-water-glycol"
    capacity_j_per_k = 0.0
    fluid_m3 = 0.0
    declared = False
    for e in edges:
        material = FLUID_LINE_MATERIALS.get(str(e.get("material", "")))
        if material is None:
            continue
        declared = True
        length = _line_length_m(e, node_by_id)
        if length <= 0.0:
            continue
        r = float(e.get("radius", 0.01))
        wall = float(material.get("wall_thickness_m", 0.0015))
        bore_m3 = math.pi * r * r * length
        wall_m3 = math.pi * ((r + wall) ** 2 - r * r) * length
        medium = FLUID_MEDIA.get(str(e.get("fluid", default_fluid)), FLUID_MEDIA[default_fluid])
        capacity_j_per_k += bore_m3 * medium["density_kg_m3"] * medium["specific_heat_j_kg_k"]
        capacity_j_per_k += wall_m3 * material["density_kg_m3"] * material["specific_heat_j_kg_k"]
        fluid_m3 += bore_m3
    for nid in node_ids:
        if nid == "powertrain.engine":
            continue   # the block's own mass is its own thermal state, not any circuit's
        n = node_by_id.get(nid, {})
        material = FLUID_LINE_MATERIALS.get(str(n.get("material", "")))
        if material is not None:
            declared = True
            capacity_j_per_k += float(n.get("mass_kg", 0.0)) * material["specific_heat_j_kg_k"]
        vessel_l = float(n.get("fluid_volume_l", 0.0))
        if vessel_l > 0.0:
            medium = FLUID_MEDIA.get(str(n.get("fluid", default_fluid)), FLUID_MEDIA[default_fluid])
            capacity_j_per_k += vessel_l / 1000.0 * medium["density_kg_m3"] * medium["specific_heat_j_kg_k"]
            fluid_m3 += vessel_l / 1000.0
    if not declared:
        # the old lumped guess: every member node's mass at the fluid's specific heat
        specific_heat = 2.0 if kind_class == "thermal-liquid" and default_fluid == "engine-oil" else 4.2
        lumped = sum(float(node_by_id.get(nid, {}).get("mass_kg", 0.0)) for nid in node_ids
                     if nid != "powertrain.engine") * specific_heat
        return lumped, 0.0
    return capacity_j_per_k / 1000.0, fluid_m3 * 1000.0


def _line_length_m(e: dict, node_by_id: dict[str, dict]) -> float:
    a = node_by_id.get(e["a"], {}).get("reference_position")
    b = node_by_id.get(e["b"], {}).get("reference_position")
    if a is None or b is None:
        return 0.0
    points = [list(a)] + [list(w) for w in e.get("waypoints", ())] + [list(b)]
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def _line_wall_loss_w_per_k(edges: list[dict], node_by_id: dict[str, dict]) -> float:
    """Passive heat loss of every declared fluid line to the surrounding
    bay air, from the real cylindrical-wall heat equation: three
    resistances in series per unit length -- the inner liquid film
    (1 / h_in * 2*pi*r_i), the wall's own conduction (ln(r_o/r_i) /
    2*pi*k, k from the material table), and the outer still-air film
    (1 / h_out * 2*pi*r_o). A thick rubber hose (k=0.25) insulates; a
    thin cupronickel or steel pipe barely does -- exactly the difference
    a production run swapping materials/dimensions needs to see."""
    total = 0.0
    for e in edges:
        material = FLUID_LINE_MATERIALS.get(str(e.get("material", "")))
        if material is None:
            continue
        length = _line_length_m(e, node_by_id)
        if length <= 0.0:
            continue
        r_i = float(e.get("radius", 0.01))
        r_o = r_i + float(material.get("wall_thickness_m", 0.0015))
        k = float(material.get("conductivity_w_m_k", 50.0))
        r_inner_film = 1.0 / (LINE_INNER_FILM_W_M2K * 2.0 * math.pi * r_i)
        r_wall = math.log(r_o / r_i) / (2.0 * math.pi * k)
        r_outer_film = 1.0 / (LINE_OUTER_FILM_W_M2K * 2.0 * math.pi * r_o)
        total += length / (r_inner_film + r_wall + r_outer_film)
    return total


def _discover_fluid_circuits(graph: dict[str, Any]) -> list[FluidCircuit]:
    """Union-find over fluid-line edges: a real closed system is
    whatever the graph's own connectivity says it is, not a hand-picked
    list of 'the coolant loop' and 'the oil loop'."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    fluid_edges = [e for e in graph["edges"] if e["constraint"] in FLUID_LINE_KINDS]
    # Ports need their circuit identity as part of what keeps them from
    # joining systems they were never meant to share, even when they
    # touch a common hub node (both oil lines and the PCV line terminate
    # at "powertrain.engine", but that doesn't make crankcase gas and
    # engine oil the same fluid). An edge's own `circuit_identity`
    # (a real, explicit tag on the fluid edges in
    # abstract_ui_vehicles.py) is unioned into the connectivity key
    # itself: two edges only merge if they share a node AND the same
    # circuit identity. An edge with no tag gets its own identity as a
    # fallback key, so an untagged line never silently merges with
    # anything either.
    def circuit_key(node_id: str, identity: str) -> str:
        return f"{node_id}::{identity}"

    for e in fluid_edges:
        identity = e.get("circuit_identity", e["identity"])
        union(circuit_key(e["a"], identity), circuit_key(e["b"], identity))

    groups: dict[str, list[dict]] = {}
    for e in fluid_edges:
        identity = e.get("circuit_identity", e["identity"])
        groups.setdefault(find(circuit_key(e["a"], identity)), []).append(e)

    node_by_id = {n["identity"]: n for n in graph["nodes"]}
    circuits: list[FluidCircuit] = []
    for edges in groups.values():
        node_ids = frozenset({e["a"] for e in edges} | {e["b"] for e in edges})
        kinds = {e["constraint"] for e in edges}
        if kinds & THERMAL_LIQUID_KINDS:
            kind_class = "thermal-liquid"
        elif kinds & HIGH_PRESSURE_LIQUID_SUPPLY_KINDS:
            kind_class = "high-pressure-liquid-supply"
        elif kinds & INCOMPRESSIBLE_HYDRAULIC_KINDS:
            kind_class = "incompressible-hydraulic"
        else:
            kind_class = "compressible-gas"

        pump_node = next((nid for nid in node_ids
                          if node_by_id.get(nid, {}).get("kind") == "rotating-mass"
                          and "pump" in nid), None)
        # "powertrain.engine" is a shared hub several different fluid
        # lines terminate at (oil supply/return, PCV, intake air) -- the
        # circuit-identity partitioning above already keeps those from
        # merging into one circuit, but the engine's own structural mass
        # still must not be counted as any circuit's fluid mass just
        # because it's a member node.
        # Thermal capacity from what the circuit is actually MADE of and
        # HOLDS (abstract_ui_vehicles.py's FLUID_LINE_MATERIALS /
        # FLUID_MEDIA, declared per line and per vessel): each line's
        # fluid content from its bore x length and its wall from radius x
        # wall thickness x its material's own density/specific heat; each
        # vessel's declared fluid volume plus its casting mass at its
        # material's specific heat. Replaces counting a radiator's
        # aluminium and a pump's cast iron as if they were water.
        thermal_mass, fluid_volume_l = _circuit_thermal_capacity(edges, node_ids, node_by_id, kind_class)
        passive_loss = sum(float(node_by_id.get(nid, {}).get("passive_heat_loss_w_per_k", 0.0))
                           for nid in node_ids)
        # plus every declared line's own wall loss to the bay, parametric
        # in its material's conductivity and its radius/wall/length
        passive_loss += _line_wall_loss_w_per_k(edges, node_by_id)
        # a real declared heat-exchange-path edge (abstract_ui_vehicles.py's
        # thermal.engine_to_coolant / thermal.engine_to_oil), not a
        # guess made up on the toy side from matching substrings in node
        # names -- read whatever real fraction the graph itself declares,
        # 0.0 if this circuit has no such edge at all (a hydraulic/
        # pneumatic circuit genuinely doesn't receive engine waste heat)
        # matched on the "b" endpoint only (the pump each heat-exchange-
        # path edge targets) -- "a" is always "powertrain.engine", the
        # same shared hub every circuit's fluid edges also touch, so
        # matching on it too would double up every circuit's share with
        # every other circuit's, the same bug class as the thermal-mass
        # fix above
        heat_share = sum(
            float(e.get("heat_share_frac", 0.0)) for e in graph["edges"]
            if e["constraint"] == "heat-exchange-path" and e["b"] in node_ids
        )

        active_exchange = sum(float(node_by_id.get(nid, {}).get("heat_exchange_w_per_k", 0.0))
                              for nid in node_ids)
        # A fan is only ever connected to the rest of the graph by a
        # rotational belt edge to the crank (or, for an electric fan, by
        # nothing at all) -- never a fluid edge, so it's never actually
        # a member of this circuit's own fluid-node set and has to be
        # found by association instead (any circuit with a real radiator
        # gets EVERY fan node in the whole graph, not just the first
        # match -- any number of fans, of any type, genuinely able to
        # serve the same manifold at once).
        fan_nodes: tuple[str, ...] = ()
        if any("radiator" in nid for nid in node_ids):
            fan_nodes = tuple(n["identity"] for n in graph["nodes"]
                              if n.get("kind") == "rotating-mass" and "fan" in n["identity"])

        bottle_capacity = sum(float(node_by_id.get(nid, {}).get("capacity_kg", 0.0)) for nid in node_ids)
        # a real declared choke limit off the intake-air edge itself
        # (abstract_ui_vehicles.py's powertrain.intake_plenum_port), not
        # a toy-side guess
        flow_capacity = sum(float(e.get("flow_capacity_kg_s", 0.0)) for e in edges)
        # a real declared passive-rejection coefficient off the plenum/
        # manifold body itself (heat_soak_w_per_k) -- previously declared
        # in abstract_ui_vehicles.py and never read by anything
        heat_soak = sum(float(node_by_id.get(nid, {}).get("heat_soak_w_per_k", 0.0)) for nid in node_ids)
        # a real relief-valve setting off the oil gallery edge (a single
        # cap, not additive capacity, hence max not sum)
        relief_pressure = max((float(e.get("relief_pressure_pa", 0.0)) for e in edges), default=0.0)
        # the real thermostat valve node this circuit carries, if any --
        # its declared wax characteristics, read off the node itself
        thermostat = next((node_by_id[nid] for nid in node_ids
                           if node_by_id.get(nid, {}).get("kind") == "wax-pellet-thermostat-valve"), None)

        circuits.append(FluidCircuit(
            kind_class=kind_class, nodes=node_ids, edges=tuple(edges), pump_node=pump_node,
            fan_nodes=fan_nodes, thermal_mass_kj_per_k=max(3.0, thermal_mass),
            passive_heat_loss_w_per_k=passive_loss, active_heat_exchange_w_per_k=active_exchange,
            heat_share=heat_share, bottle_capacity_kg=bottle_capacity,
            flow_capacity_kg_s=flow_capacity, heat_soak_w_per_k=heat_soak,
            relief_pressure_pa=relief_pressure,
            volume_l=fluid_volume_l if fluid_volume_l > 0.0 else max(0.3, 0.15 * len(node_ids)),
            exchanger_medium=next((str(node_by_id[nid].get("cooling_medium", "air")) for nid in node_ids
                                   if node_by_id.get(nid, {}).get("heat_exchange_w_per_k", 0.0) > 0.0), "air"),
            pump_design_flow_lpm=float(node_by_id.get(pump_node, {}).get("design_flow_lpm", 0.0)) if pump_node else 0.0,
            pump_rated_omega_rad_s=float(node_by_id.get(pump_node, {}).get("rated_omega_rad_s", 0.0)) if pump_node else 0.0,
            working_pressure_pa=max((float(node_by_id[nid].get("working_pressure_pa", 0.0)) for nid in node_ids
                                     if nid in node_by_id), default=0.0),
            regulator_cut_in_frac=next((float(node_by_id[nid]["cut_in_frac"]) for nid in node_ids
                                        if node_by_id.get(nid, {}).get("kind") == "pressure-switch-unloader"), 0.84),
            regulator_cut_out_frac=next((float(node_by_id[nid]["cut_out_frac"]) for nid in node_ids
                                         if node_by_id.get(nid, {}).get("kind") == "pressure-switch-unloader"), 1.0),
            compressor_rated_w=next((float(node_by_id[nid].get("rated_w", 0.0)) for nid in node_ids
                                     if node_by_id.get(nid, {}).get("kind") == "electric-compressor"), 0.0),
            compressor_drive=next((str(node_by_id[nid].get("drive", "crank-belt")) for nid in node_ids
                                   if node_by_id.get(nid, {}).get("kind") == "electric-compressor"), "crank-belt"),
            regulated_flow_kg_s=next((float(node_by_id[nid]["rated_flow_kg_s"]) for nid in node_ids
                                      if node_by_id.get(nid, {}).get("kind") == "flow-regulator"), 0.0),
            regulator_onset_map_frac=next((float(node_by_id[nid]["onset_map_frac"]) for nid in node_ids
                                           if node_by_id.get(nid, {}).get("kind") == "flow-regulator"), 0.0),
            regulator_full_map_frac=next((float(node_by_id[nid]["full_map_frac"]) for nid in node_ids
                                          if node_by_id.get(nid, {}).get("kind") == "flow-regulator"), 0.0),
            thermostat_opening_start_k=float(thermostat.get("opening_start_k", 0.0)) if thermostat else 0.0,
            thermostat_full_open_k=float(thermostat.get("full_open_k", 0.0)) if thermostat else 0.0,
            thermostat_tau_s=float(thermostat.get("wax_time_constant_s", 15.0)) if thermostat else 15.0,
        ))
    return circuits


@dataclass
class DrivetrainSolver:
    """The real internal solver: it does not hand-roll physics per
    mechanism, it walks the graph document's own edges each substep and
    dispatches on each edge's `constraint` string -- the graph IS the
    program. Its internal substep rate is not a guessed constant: each
    stiff coupling edge (geared-timing-drive, accessory-drive-belt,
    friction-clutch-shaft) has its own natural frequency,
    omega_n = sqrt(stiffness / reduced_inertia), the way any spring-mass
    system does; explicit Euler needs dt well under that period to
    converge instead of diverging into a saturated limit cycle (a real
    bug this hit twice while empirically tuning a fixed constant: a
    tiny camshaft inertia against engine-scale timing-chain stiffness,
    then a light dyno drum against the clutch). STABILITY_MARGIN is how
    far under the fastest eigenfrequency's period the substep must be;
    the actual substep count is derived from it and whatever dt the
    caller passes each call, not fixed in advance."""
    graph: dict[str, Any]
    omega: dict[str, float] = field(default_factory=dict)

    STABILITY_MARGIN = 0.03   # dt * omega_n must stay under this
    SUBSTEP_CAP = 200
    # The two edge kinds that are ever actually integrated as a
    # position-tracking spring/damper (backlash angle state, stiffness *
    # angle + damping * relative_speed) -- the only kinds an ODE
    # stability requirement even applies to. friction-clutch-shaft and
    # rotational-bearing also carry a "stiffness_nm_per_rad_s"-shaped
    # attribute, but only ever as a rate coefficient inside an already-
    # unconditionally-stable saturating tanh formula (current relative
    # speed only, no position state) -- they were never really "stiff
    # ODEs" and must never be scanned for one, or (a real bug this used
    # to have) an edge that was already cheap and stable forces a
    # nonzero minimum substep count on the whole solver for nothing.
    SPRING_INTEGRATED_KINDS = ("geared-timing-drive", "accessory-drive-belt")

    def __post_init__(self) -> None:
        for n in self.graph["nodes"]:
            self.omega.setdefault(n["identity"], 0.0)
        self._edge_state: dict[str, _EdgeState] = {
            e["identity"]: _EdgeState() for e in self.graph["edges"]
        }
        self._inertia: dict[str, float] = {
            n["identity"]: float(n.get("inertia_kg_m2") or self._fallback_inertia(n))
            for n in self.graph["nodes"]
        }
        # THE ENSHRINED POLICY: stiffness/spring solving is off, by
        # default, for every coupling in this graph except the crank's
        # own combustion-driven dynamics -- which were never solved
        # through this spring-integrated path at all (the crank's
        # omega is integrated directly from real torque/inertia in
        # engine_cycle_sim.py's own fine per-cylinder substep loop, a
        # completely separate mechanism). No accessory coupling
        # (camshaft timing, oil pump gear, alternator/supercharger
        # belt, ...) is ever worth resolving the transient spring
        # dynamics of for this sim's purposes -- their own natural
        # frequencies (a few hundred to several thousand Hz) are far
        # above anything that changes crank rpm, torque, or combustion
        # timing at any timescale this sim reports on, and their real
        # transient character (backlash rattle, belt whip, gear whine)
        # belongs in a genuinely separate frequency-domain audio
        # treatment -- see the FFT/spectral discussion this policy
        # deliberately leaves room for -- calibrated from the SAME
        # (omega_n, reduced_inertia) pair computed once below, not
        # resolved by brute-force time integration at all.
        #
        # A specific edge can still opt into full spring/damper/
        # backlash resolution by declaring `resolve_stiffness=True` in
        # its real graph attributes -- a genuine, real escape hatch for
        # a future case where an accessory's own transient dynamics
        # actually matter to something this sim reports on, not a
        # blanket policy override.
        self._rigid_locked_edges: frozenset[str] = self._find_rigid_locked_edges()
        self._max_sub_dt_s = self._compute_stable_sub_dt()
        self.fluid_circuits: list[FluidCircuit] = _discover_fluid_circuits(self.graph)
        # Every real fan/rotor node in the graph, keyed by identity, with
        # its own declared aerodynamic spec -- one shared lookup used for
        # (a) the mechanical fan's real crank-reaction drag torque, (b)
        # the electric fan's thermostat-commanded omega, and (c) combining
        # any number of fans' airflow contribution to a manifold, so a
        # future third fan type needs nothing new here.
        self._fan_specs: dict[str, dict[str, float]] = {
            n["identity"]: {
                "flow_coeff": float(n.get("fan_airflow_m3_s_per_rad_s", 0.0)),
                "rated_omega": float(n.get("fan_rated_omega_rad_s", FAN_REFERENCE_OMEGA_RAD_S)),
            }
            for n in self.graph["nodes"]
            if n.get("kind") == "rotating-mass" and "fan" in n["identity"]
        }
        # Real "propeller law" rating (0 for any engine not direct-drive
        # to a fixed-pitch prop) -- see PROPELLER_COUPLED_ENGINES and
        # build_drivetrain_graph's own reasoning.
        self._propeller_rated_torque_nm = float(self.graph.get("propeller_rated_torque_nm", 0.0))
        self._propeller_rated_omega_rad_s = float(self.graph.get("propeller_rated_omega_rad_s", 0.0))
        # A real running engine's intake manifold is never actually AT
        # atmospheric the instant you start observing it -- it's already
        # been pulled down to something near its own real idle vacuum by
        # the very first few intake strokes. FluidCircuit's own dataclass
        # default (pressure_pa=101_325.0, literal atmospheric) is an
        # honest "we don't know yet" default for a circuit that's never
        # been stepped, but starting the SIM there manufactures a real,
        # if brief, artificial pressure-drop transient every single
        # start() -- invisible for a small/fast-responding engine, but
        # real enough to excite a lasting, lightly-damped rpm oscillation
        # on some larger/more coupled engines (a supercharged big-inch V8
        # among them) as the idle governor reacts to a MAP reading that
        # was never real combustion state, just an unprimed circuit's
        # generic default. Seed it at a real, typical idle MAP instead --
        # still an approximation (the genuinely correct value depends on
        # the engine's own idle_rpm/throttle-restriction, which this
        # constructor doesn't have access to), but a MUCH smaller,
        # honest error than starting cold at full atmospheric.
        intake_circuit = next((c for c in self.fluid_circuits
                               if c.kind_class == "compressible-gas" and any("intake" in nid for nid in c.nodes)),
                              None)
        if intake_circuit is not None:
            seed_pa = 101_325.0 * 0.30
            intake_circuit.pressure_pa = seed_pa
            intake_circuit.achievable_target_pressure_pa = seed_pa

    def _edge_omega_n(self, e: dict) -> float:
        stiffness = e.get("stiffness_nm_per_rad") or e.get("stiffness_nm_per_rad_s")
        if not stiffness:
            return 0.0
        ia = self._inertia.get(e["a"], 1.0)
        ib = self._inertia.get(e["b"], 1.0)
        reduced_inertia = max(min(ia, ib), 1e-9)
        return math.sqrt(float(stiffness) / reduced_inertia)

    def _find_rigid_locked_edges(self) -> frozenset[str]:
        return frozenset(
            e["identity"] for e in self.graph["edges"]
            if e["constraint"] in self.SPRING_INTEGRATED_KINDS
            and not e.get("resolve_stiffness", False)
        )

    def _compute_stable_sub_dt(self) -> float:
        fastest_omega_n = 0.0
        for e in self.graph["edges"]:
            if e["constraint"] not in self.SPRING_INTEGRATED_KINDS:
                continue  # never a stiff ODE in the first place -- see SPRING_INTEGRATED_KINDS
            if e["identity"] in self._rigid_locked_edges:
                continue  # solved via the saturating lock instead -- see __post_init__
            fastest_omega_n = max(fastest_omega_n, self._edge_omega_n(e))
        if fastest_omega_n <= 0.0:
            return 1e-3   # nothing left resolving a real spring -- coarse is fine
        return self.STABILITY_MARGIN / fastest_omega_n

    @staticmethod
    def _fallback_inertia(node: dict) -> float:
        mass = float(node.get("mass_kg", 1.0))
        return max(1e-6, mass * 0.01)   # a small effective radius-of-gyration guess

    def step(self, dt: float, crank_omega: float, alternator_shaft_load_w: float = 0.0,
              dyno_load_torque_nm: float = 0.0, waste_heat_kw: float = 0.0,
              electric_fan_on_frac: float | None = None,
              nitrous_active: bool = False, auxiliary_injection_active: bool = False,
              intake_demand_kg_s: float = 0.0,
              intake_supply_pressure_pa: float = 101_325.0,
              exhaust_demand_kg_s: float = 0.0, fuel_demand_kg_s: float = 0.0,
              fuel_production_kg_s: float = 0.0,
              fuel_production_composition_frac: float = 1.0,
              fuel_valve_open: bool = True,
              exhaust_brake_frac: float = 0.0,
              fuel_cooler_target_k: float | None = None, ac_active: bool = False,
              disabled_edges: frozenset[str] = frozenset(),
              starting_air_kg_s: float = 0.0,
              regulator_map_frac: float = 1.0) -> dict[str, float]:
        # the AC clutch coil is either energized or it isn't -- a real
        # magnetic clutch has no "half disabled" state, unlike the
        # tanh's own smooth torque saturation once it IS engaged. Same
        # real disabled_edges mechanism already used to keep the dyno's
        # own edges out of the loop when they're not the active coupling.
        # The pneumatic compressor (when fitted) has no separate on/off
        # coupling at all -- a real belt-driven air compressor just runs
        # continuously with the engine, same as a water pump.
        if not ac_active:
            # identity tracks _add_belt_driven_compressor's own naming
            # (f"{crank_identity.split('.')[-1]}_to_{identity}_clutch")
            # -- ac_compressor now couples off powertrain.crank_shaft.
            # front, not powertrain.engine directly
            disabled_edges = disabled_edges | {"front_to_ac_compressor_clutch"}
        # capped for real-time cost -- permanently-over-cap couplings
        # (see __post_init__'s _find_permanently_rigid_edges) are
        # already excluded from _max_sub_dt_s's own scan and handled by
        # their own cheap saturating-lock formula instead, so this cap
        # now only ever bounds an occasional transient stiffness spike
        # in the remaining edges, not a permanent per-call cost
        substeps = min(self.SUBSTEP_CAP, max(1, math.ceil(dt / self._max_sub_dt_s)))
        sub_dt = dt / substeps
        outputs = {"crank_reaction_torque_nm": 0.0, "cam_phase_lag_rad": 0.0,
                   "alternator_delivered_w": 0.0, "supercharger_belt_slack_frac": 0.0,
                   "cam_timing_slack_frac": 0.0, "dyno_absorber_omega": 0.0,
                   "dyno_locked": False, "ac_compressor_load_w": 0.0, "pneumatic_compressor_delivered_w": 0.0}
        # A real compressor load: compressing air to any pressure ratio
        # costs real mechanical work (the same isentropic relation the
        # intake circuit's own charge-heating uses -- specific work
        # scales with the pressure ratio, mass flow sets the power). This
        # was missing entirely: the belt coupling below only ever fights
        # to hold the rotor at its commanded speed ratio against generic
        # spring/damper/slip resistance, with zero connection to how much
        # actual compression work it's doing -- effectively a free-
        # spinning rotor as far as the crank could feel. Computed once
        # per outer call (constant across substeps, like the alternator's
        # delivered_w below) from the intake circuit's own current
        # pressure, since the fluid circuits step once per outer call too.
        compressor_load_torque_nm = 0.0
        if "supercharger_rotor" in {n["identity"] for n in self.graph["nodes"]}:
            intake_circuit = next((c for c in self.fluid_circuits
                                   if c.kind_class == "compressible-gas"
                                   and any("intake" in nid for nid in c.nodes)), None)
            if intake_circuit is not None:
                pressure_ratio = max(1.0, intake_circuit.pressure_pa / 101_325.0)
                specific_work_j_per_kg = 1005.0 * 293.15 * (pressure_ratio ** (0.4 / 1.4) - 1.0)
                compressor_power_w = max(0.0, intake_demand_kg_s) * specific_work_j_per_kg
                rotor_omega = max(50.0, self.omega.get("supercharger_rotor", 0.0))
                compressor_load_torque_nm = compressor_power_w / rotor_omega

        # The general fan/rotor aerodynamic load (rotor_aero_load_torque_nm)
        # applied to every real fan node this graph actually has, computed
        # once per outer call like the supercharger's own compressor load
        # above -- a crank-belted mechanical fan feeds a real resistive
        # torque back through its existing belt edge below, while an
        # electric fan (no shaft edge at all) has its omega commanded
        # directly here off a real coolant thermostat instead of being
        # integrated from any torque.
        mechanical_fan_load_torque_nm = 0.0
        mech_fan_spec = self._fan_specs.get("powertrain.cooling_fan")
        if mech_fan_spec is not None:
            mechanical_fan_load_torque_nm = rotor_aero_load_torque_nm(
                self.omega.get("powertrain.cooling_fan", 0.0),
                mech_fan_spec["flow_coeff"], COOLING_FAN_DISK_AREA_M2)
        electric_fan_spec = self._fan_specs.get("powertrain.cooling_fan_electric")
        if electric_fan_spec is not None:
            # the fan's duty is the ECU's thermostat program's decision
            # (ecu.EngineControlUnit.electric_fan_command), handed in --
            # a caller with no computer in the loop gets the bare
            # thermostatic-switch law as a fallback
            if electric_fan_on_frac is None:
                coolant_temp_k = self.coolant_temp_k()
                electric_fan_on_frac = max(0.0, min(1.0, (coolant_temp_k - ELECTRIC_FAN_ON_TEMP_K) / ELECTRIC_FAN_BAND_K))
            self.omega["powertrain.cooling_fan_electric"] = electric_fan_on_frac * electric_fan_spec["rated_omega"]

        for _ in range(substeps):
            self.omega["powertrain.engine"] = crank_omega
            if "powertrain.crank_shaft.front" in self.omega:
                # rigidly keyed to the crank nose -- unlike crank_shaft.
                # rear (only ever touched by physics-inert "torque-shaft"/
                # "synchronized-positive-dog-clutch-bypass" edges), .front
                # now carries REAL accessory-drive-belt/friction-clutch-
                # shaft torque, so its own omega has to track the crank's
                # exactly, every substep, for that real coupling to solve
                # against the crank's actual speed rather than a frozen
                # reference.
                self.omega["powertrain.crank_shaft.front"] = crank_omega
            net_torque = {n["identity"]: 0.0 for n in self.graph["nodes"]}
            if "dyno_absorber" in net_torque:
                net_torque["dyno_absorber"] -= dyno_load_torque_nm
                if self._propeller_rated_torque_nm > 0.0:
                    net_torque["dyno_absorber"] -= rotor_load_torque_from_rated_nm(
                        self.omega.get("dyno_absorber", 0.0),
                        self._propeller_rated_omega_rad_s, self._propeller_rated_torque_nm)
            if "supercharger_rotor" in net_torque:
                net_torque["supercharger_rotor"] -= compressor_load_torque_nm
            if "powertrain.cooling_fan" in net_torque:
                net_torque["powertrain.cooling_fan"] -= mechanical_fan_load_torque_nm
            if "electrical.alternator" in net_torque:
                # the electrical load is a real resistive torque ON the
                # alternator itself, not an instant reaction teleported
                # onto the crank -- however much of it actually reaches
                # the crank has to go through the belt's own stiffness/
                # damping/backlash, same as any other load on that belt,
                # real slip included if the load is more than it can hold
                # how much shaft power the alternator is actually pulling
                # is the electrical network's answer (electrical_network.
                # ElectricalNetwork: regulated output over real efficiency),
                # handed in one tick lagged like every cross-system reading
                # (the network's own capability curve makes that power zero
                # below cut-in speed, so the torque can never blow up as
                # the rotor slows -- the same floor is applied here so a
                # one-tick-lagged power figure can't either)
                alt_omega = self.omega.get("electrical.alternator", 0.0)
                net_torque["electrical.alternator"] -= alternator_shaft_load_w / max(alt_omega, ALTERNATOR_TORQUE_OMEGA_FLOOR_RAD_S)
                outputs["alternator_delivered_w"] = alternator_shaft_load_w
            crank_reaction = 0.0

            for e in self.graph["edges"]:
                if e["identity"] in disabled_edges:
                    continue
                kind, a, b = e["constraint"], e["a"], e["b"]
                st = self._edge_state[e["identity"]]
                if kind not in FLUID_LINE_KINDS and (a not in self.omega or b not in self.omega):
                    # an honest network can leave a drive with nothing on
                    # one end (a splash-lubricated engine has no oil pump
                    # for the cam gear to drive); a shaft into nothing
                    # carries nothing -- recorded, not invented
                    st.last_torque_nm = 0.0
                    st.missing_endpoint = True
                    continue

                if kind == "friction-clutch-shaft":
                    # the real game's clutch_torque formula exactly:
                    # max_torque * tanh(stiffness * relative_speed / max_torque)
                    relv = self.omega[a] - self.omega[b]
                    max_t = float(e.get("max_torque_nm", 500.0))
                    t = max_t * math.tanh(float(e.get("stiffness_nm_per_rad_s", 4000.0)) * relv / max(max_t, 1e-6))
                    st.last_torque_nm, st.engaged = t, abs(t) < 0.02 * max_t
                    net_torque[a] -= t
                    net_torque[b] += t
                    if a == "powertrain.engine" or a == "powertrain.crank_shaft.front":
                        # crank_shaft.front is rigidly keyed to the crank
                        # (omega forced equal every substep, see step()'s
                        # own top) -- torque reacting on it IS torque
                        # reacting on the crank, the same real rigid body
                        crank_reaction += t
                    if e["identity"] == "dyno_friction_clutch":
                        outputs["dyno_locked"] = st.engaged
                    elif e["identity"] == "front_to_ac_compressor_clutch":
                        outputs["ac_compressor_load_w"] = abs(t) * abs(self.omega.get(b, 0.0))
                    elif e["identity"] == "front_to_pneumatic_compressor_clutch":
                        outputs["pneumatic_compressor_delivered_w"] = abs(t) * abs(self.omega.get(b, 0.0))

                elif kind == "rolling-friction-contact":
                    # a real chassis-dyno roller: dry Coulomb friction, not
                    # a bolted shaft -- capped at mu * normal_force *
                    # radius regardless of how much relative slip there is,
                    # unlike the clutch's tanh (which keeps rising toward
                    # its cap smoothly). Sign follows relative surface speed.
                    relv = self.omega[a] - self.omega[b]
                    cap = (float(e.get("friction_coefficient", 0.9))
                           * float(e.get("normal_force_n", 1000.0))
                           * float(e.get("contact_radius_m", 0.2)))
                    t = cap * math.copysign(1.0, relv) if abs(relv) > 1e-3 else 0.0
                    st.last_torque_nm = t
                    net_torque[a] -= t
                    net_torque[b] += t
                    if a == "powertrain.engine":
                        crank_reaction += t

                elif kind == "rotational-bearing":
                    relv = self.omega[a] - self.omega[b]
                    drag = e.get("bearing_drag_nm", 0.0)
                    t = drag * math.tanh(relv * 5.0)
                    net_torque[a] -= t
                    net_torque[b] += t
                    if a == "powertrain.engine":
                        crank_reaction += t

                elif kind == "direct-torque-shaft":
                    # rigid, no belt, no compliance -- the driven node IS
                    # the crank's own speed, not a separately-integrated
                    # body; only its reaction torque matters
                    self.omega[b] = self.omega[a]

                elif kind in ("geared-timing-drive", "accessory-drive-belt"):
                    # ratio means omega_b_target = omega_a * ratio (a camshaft's
                    # 0.5 = half crank speed; a supercharger belt's 2.6 = 2.6x
                    # crank speed) -- relative_speed compared in the a-frame
                    # is omega_a - omega_b/ratio, the lock condition for that
                    ratio = max(float(e.get("ratio", 1.0)), 1e-6)
                    relative_speed = self.omega[a] - self.omega[b] / ratio
                    max_t = float(e.get("max_torque_nm", 15.0))
                    if e["identity"] in self._rigid_locked_edges:
                        # too stiff to usefully resolve even at this
                        # sim's finest real dt (__post_init__) -- a real
                        # gear/belt this stiff just doesn't meaningfully
                        # flex, so skip the backlash/angle state machine
                        # below entirely and drive it with the EXACT
                        # torque that reaches the target speed in this
                        # one substep, capped by the real max_torque_nm --
                        # not a fixed saturation rate. A fixed huge rate
                        # (this used to be tanh(stiffness * relative_speed
                        # / max_t), unconditionally near +-max_t for any
                        # nonzero error) is bang-bang: at only 1 substep
                        # per call (every permanently-rigid edge, now),
                        # it overshoots the target every tick and flips
                        # sign next tick -- a real limit-cycle oscillation
                        # (caught directly: a supercharger rotor's own
                        # omega alternating between two fixed values 2:1
                        # apart, tick after tick, feeding a noisy and
                        # sometimes huge reaction torque onto the crank).
                        # Solving for the exact torque has zero overshoot
                        # by construction when under the cap, and a clean,
                        # non-oscillating torque-limited slip when over it.
                        ib = self._inertia.get(b, 1e-6)
                        target_omega_b = self.omega[a] * ratio
                        required_t_on_b = (target_omega_b - self.omega[b]) * ib / sub_dt
                        t = max(-max_t, min(max_t, required_t_on_b * ratio))
                        st.engaged = True
                        gap = 0.0   # rigid-locked: no backlash slack concept, always "full lock"
                    else:
                        st.relative_angle_rad += relative_speed * sub_dt
                        gap = float(e.get("backlash_rad", 0.0))
                        if st.relative_angle_rad > gap:
                            engaged_angle, st.engaged = st.relative_angle_rad - gap, True
                        elif st.relative_angle_rad < -gap:
                            engaged_angle, st.engaged = st.relative_angle_rad + gap, True
                        else:
                            engaged_angle, st.engaged = 0.0, False
                        if st.engaged:
                            raw = (float(e.get("stiffness_nm_per_rad", 150.0)) * engaged_angle
                                   + float(e.get("damping_nm_per_rad_s", 3.0)) * relative_speed)
                            t = max_t * math.tanh(raw / max(max_t, 1e-6))
                        else:
                            t = 0.0
                    st.last_torque_nm = t
                    net_torque[a] -= t
                    net_torque[b] += t / ratio
                    if a == "powertrain.engine" or a == "powertrain.crank_shaft.front":
                        # see the friction-clutch-shaft branch above --
                        # crank_shaft.front is rigidly keyed to the
                        # crank (alternator/water_pump/fan belts attach
                        # here now)
                        crank_reaction += t
                    slack = min(1.0, abs(st.relative_angle_rad) / max(gap, 1e-9))
                    if e["identity"] == "camshaft_timing_drive":
                        outputs["cam_phase_lag_rad"] = st.relative_angle_rad
                        outputs["cam_timing_slack_frac"] = slack
                    elif kind == "accessory-drive-belt":
                        outputs["supercharger_belt_slack_frac"] = slack

            for identity, torque in net_torque.items():
                if identity == "powertrain.engine":
                    continue
                inertia = self._inertia.get(identity, 0.01)
                # no zero-floor here: a small driven mass under a stiff
                # coupling needs to correct THROUGH zero during a transient
                # (that's what keeps the coupling converged); only the
                # externally-driven crank is physically forbidden from
                # going negative in this sim's convention
                self.omega[identity] = self.omega[identity] + torque / inertia * sub_dt

            outputs["crank_reaction_torque_nm"] = crank_reaction
            outputs["dyno_absorber_omega"] = self.omega.get("dyno_absorber", 0.0)

        # fluid circuits step once per outer call, not per torque substep
        # -- their own physics (thermal mass, line volume fill lag) is
        # genuinely far slower than the rotational dynamics above, so
        # they don't need and shouldn't pay for that resolution
        for c in self.fluid_circuits:
            if c.kind_class == "high-pressure-liquid-supply" and any("nitrous" in nid for nid in c.nodes):
                c.valve_open = nitrous_active
            elif c.kind_class == "high-pressure-liquid-supply" and any("auxiliary_injection" in nid for nid in c.nodes):
                c.valve_open = auxiliary_injection_active
            elif c.kind_class == "high-pressure-liquid-supply" and any("fuel" in nid for nid in c.nodes):
                # fuel flows whenever the engine is running at all -- no
                # solenoid to command, unlike nitrous's genuinely optional
                # valve. See _step_fluid_circuits: this circuit's drain
                # rate is driven by real combustion demand (fuel_demand_
                # kg_s), not a fixed bottle-emptying heuristic -- fuel
                # isn't pressurized propellant, it only leaves the tank
                # because something downstream is actually burning it.
                # fuel_valve_open: a declared fuel network's own real
                # lock-off solenoid (fuel_network.LockoffSolenoid) --
                # True for every network-less engine.
                c.valve_open = fuel_valve_open
            elif c.kind_class == "compressible-gas" and any("intake" in nid for nid in c.nodes):
                c.supply_pressure_pa = intake_supply_pressure_pa
        self._step_fluid_circuits(dt, waste_heat_kw, intake_demand_kg_s, exhaust_demand_kg_s, fuel_demand_kg_s,
                                   exhaust_brake_frac, fuel_cooler_target_k,
                                   outputs.get("pneumatic_compressor_delivered_w", 0.0),
                                   starting_air_kg_s=starting_air_kg_s,
                                   regulator_map_frac=regulator_map_frac,
                                   fuel_production_kg_s=fuel_production_kg_s,
                                   fuel_production_composition_frac=fuel_production_composition_frac)
        pneumatic_circuit = next((c for c in self.fluid_circuits
                                  if c.kind_class == "high-pressure-liquid-supply"
                                  and any("pneumatic" in nid for nid in c.nodes)), None)
        outputs["pneumatic_reserve_pressure_pa"] = (
            101_325.0 + pneumatic_circuit.fill_level_frac
            * (max(pneumatic_circuit.working_pressure_pa, PNEUMATIC_RESERVE_PRESSURE_PA) - 101_325.0)
            if pneumatic_circuit else 0.0)
        outputs["pneumatic_compressor_running_w"] = pneumatic_circuit.compressor_running_w if pneumatic_circuit else 0.0
        outputs["pneumatic_compressor_drive"] = pneumatic_circuit.compressor_drive if pneumatic_circuit else "none"
        outputs["fluid_circuits"] = {
            (c.pump_node or next(iter(c.nodes))): {
                "kind_class": c.kind_class, "temp_k": c.temp_k, "pressure_pa": c.pressure_pa,
            }
            for c in self.fluid_circuits
        }
        nitrous_circuit = next((c for c in self.fluid_circuits
                                if c.kind_class == "high-pressure-liquid-supply"
                                and any("nitrous" in nid for nid in c.nodes)), None)
        outputs["nitrous_delivered_kg_s"] = nitrous_circuit.delivered_flow_kg_s if nitrous_circuit else 0.0
        outputs["nitrous_fill_frac"] = nitrous_circuit.fill_level_frac if nitrous_circuit else 0.0
        auxiliary_injection_circuit = next((c for c in self.fluid_circuits
                                            if c.kind_class == "high-pressure-liquid-supply"
                                            and any("auxiliary_injection" in nid for nid in c.nodes)), None)
        outputs["auxiliary_injection_delivered_kg_s"] = (
            auxiliary_injection_circuit.delivered_flow_kg_s if auxiliary_injection_circuit else 0.0)
        outputs["auxiliary_injection_fill_frac"] = (
            auxiliary_injection_circuit.fill_level_frac if auxiliary_injection_circuit else 0.0)
        fuel_circuit = next((c for c in self.fluid_circuits
                             if c.kind_class == "high-pressure-liquid-supply"
                             and any("fuel" in nid for nid in c.nodes)), None)
        outputs["fuel_delivered_kg_s"] = fuel_circuit.delivered_flow_kg_s if fuel_circuit else 0.0
        outputs["fuel_fill_frac"] = fuel_circuit.fill_level_frac if fuel_circuit else 1.0
        # the two real supply signals a consumer actually feels (a
        # gasholder bell holds its pressure at any fill level -- only
        # an EMPTY one starves, so fill_frac itself is the wrong
        # signal for charge strength): how much of the demanded flow
        # was actually delivered this tick, and what fraction of it is
        # genuinely the nominal fluid
        outputs["fuel_availability_frac"] = (
            min(1.0, fuel_circuit.delivered_flow_kg_s / fuel_demand_kg_s)
            if fuel_circuit and fuel_demand_kg_s > 0.0 else 1.0)
        outputs["fuel_composition_frac"] = fuel_circuit.composition_frac if fuel_circuit else 1.0
        outputs["fuel_pump_flow_capacity_kg_s"] = fuel_circuit.flow_capacity_kg_s if fuel_circuit else 0.0
        outputs["fuel_temp_k"] = fuel_circuit.temp_k if fuel_circuit else 293.15
        pneumatic_circuit = next((c for c in self.fluid_circuits
                                  if c.kind_class == "high-pressure-liquid-supply"
                                  and any("pneumatic" in nid for nid in c.nodes)), None)
        outputs["pneumatic_reserve_fill_frac"] = pneumatic_circuit.fill_level_frac if pneumatic_circuit else 0.0
        intake_circuit = next((c for c in self.fluid_circuits
                               if c.kind_class == "compressible-gas" and any("intake" in nid for nid in c.nodes)),
                              None)
        outputs["intake_manifold_pressure_pa"] = intake_circuit.pressure_pa if intake_circuit else intake_supply_pressure_pa
        outputs["intake_charge_temp_k"] = intake_circuit.temp_k if intake_circuit else 293.15
        # the real port/valve-curtain flow ceiling itself (flow_capacity_kg_s,
        # abstract_ui_vehicles.py's powertrain.intake_plenum_port), so the
        # caller can report real demand-vs-capacity -- how much the intake
        # is actually being restricted right now, not just the pressure
        # drop that restriction eventually produces
        outputs["intake_flow_capacity_kg_s"] = intake_circuit.flow_capacity_kg_s if intake_circuit else 0.0
        exhaust_circuit = next((c for c in self.fluid_circuits
                                if c.kind_class == "compressible-gas" and any("exhaust" in nid for nid in c.nodes)),
                               None)
        outputs["exhaust_temp_k"] = exhaust_circuit.temp_k if exhaust_circuit else 293.15
        outputs["exhaust_backpressure_frac"] = (
            exhaust_circuit.pressure_pa / 101_325.0 - 1.0) if exhaust_circuit else 0.0
        outputs["exhaust_flow_capacity_kg_s"] = exhaust_circuit.flow_capacity_kg_s if exhaust_circuit else 0.0
        oil_circuit = next((c for c in self.fluid_circuits
                            if c.kind_class == "thermal-liquid" and any("oil" in nid for nid in c.nodes)),
                           None)
        outputs["oil_pressure_pa"] = oil_circuit.pressure_pa if oil_circuit else 101_325.0
        # identified by real node membership, not a heat_share numeric
        # range -- that range broke the moment the exhaust circuit got
        # its own real heat_share_frac (0.25, landing inside what used to
        # be oil's exclusive 0.0-0.5 slot)
        coolant_circuit = next(
            (c for c in self.fluid_circuits
             if c.kind_class == "thermal-liquid" and c.heat_share > 0.0
             and not any("oil" in nid for nid in c.nodes)),
            None)
        outputs["coolant_temp_k"] = coolant_circuit.temp_k if coolant_circuit else 293.15
        outputs["coolant_flow_lpm"] = coolant_circuit.flow_lpm if coolant_circuit else 0.0
        outputs["oil_temp_k"] = oil_circuit.temp_k if oil_circuit else 293.15
        outputs["oil_flow_lpm"] = oil_circuit.flow_lpm if oil_circuit else 0.0

        return outputs

    def coolant_temp_k(self) -> float:
        """The coolant circuit's current bulk temperature (ambient if
        this engine has no liquid coolant circuit at all)."""
        coolant_circuit = next((c for c in self.fluid_circuits
                                if c.kind_class == "thermal-liquid"
                                and any("coolant" in nid or "radiator" in nid for nid in c.nodes)), None)
        return coolant_circuit.temp_k if coolant_circuit is not None else 293.15

    def coolant_thermostat_open_frac(self) -> float:
        coolant_circuit = next((c for c in self.fluid_circuits
                                if c.kind_class == "thermal-liquid"
                                and any("coolant" in nid or "radiator" in nid for nid in c.nodes)), None)
        return coolant_circuit.thermostat_open_frac if coolant_circuit is not None else 1.0

    def draw_cylinder_charge(self, cylinder_volume_m3: float) -> float:
        """The real cylinder cycle integrating the slower fluid transfer:
        called once per real intake event (from engine_cycle_sim's fast
        per-cylinder substep loop), this draws one cylinder's worth of
        charge out of the real intake-air circuit's CURRENT pressure --
        whatever the slow fluid solver last settled it to -- via the
        real parametric_volume_pressure_exchange law, and genuinely
        depletes that circuit's own pressure by the tiny amount one
        cylinder's volume actually takes (mass-conserving, cumulative:
        many fast firing events between slow fluid-circuit steps really
        do pull the plenum down a little, the same real effect a bank of
        cylinders drawing on one shared plenum has). Returns the
        resulting charge pressure as a fraction of atmospheric, for the
        caller's combustion-strength calculation.
        """
        ambient_pa = 101_325.0
        circuit = next((c for c in self.fluid_circuits
                        if c.kind_class == "compressible-gas" and any("intake" in nid for nid in c.nodes)),
                       None)
        if circuit is None:
            return 1.0
        port_volume_m3 = circuit.volume_l / 1000.0
        # the cylinder's own residual pressure before the intake valve
        # opens is whatever the throttle/boost system is actually
        # currently commanding, NOT literal atmospheric -- at closed
        # throttle the real commanded pressure is well below atmospheric
        # (that IS the throttle restriction, a real vacuum), and using
        # raw ambient here pulled every circuit back toward atmospheric
        # on every single firing event, which made idle run away since
        # idle's real target is a low vacuum, not atmospheric.
        #
        # Uses the circuit's real ACHIEVABLE (already choke-capped)
        # target, not raw supply_pressure_pa -- this function runs once
        # per real firing event (many times per physics tick at high
        # rpm), and using the uncapped commanded supply here let those
        # frequent pulls drag plenum pressure back toward it, overpowering
        # _step_fluid_circuits' own once-per-tick relaxation toward the
        # real flow-capacity-limited target and defeating the whole
        # choke ceiling under boost (a genuinely unbounded MAP, 2.18x
        # atmospheric, was the visible symptom).
        equilibrium_pa = parametric_volume_pressure_exchange(
            circuit.pressure_pa, port_volume_m3, circuit.achievable_target_pressure_pa, cylinder_volume_m3)
        circuit.pressure_pa = equilibrium_pa
        return equilibrium_pa / ambient_pa

    def _step_fluid_circuits(self, dt: float, waste_heat_kw: float, intake_demand_kg_s: float = 0.0,
                              exhaust_demand_kg_s: float = 0.0, fuel_demand_kg_s: float = 0.0,
                              exhaust_brake_frac: float = 0.0,
                              fuel_cooler_target_k: float | None = None,
                              pneumatic_compressor_delivered_w: float = 0.0,
                              starting_air_kg_s: float = 0.0,
                              regulator_map_frac: float = 1.0,
                              fuel_production_kg_s: float = 0.0,
                              fuel_production_composition_frac: float = 1.0) -> None:
        ambient_k = 293.15
        for c in self.fluid_circuits:
            pump_omega = self.omega.get(c.pump_node, 0.0) if c.pump_node else 0.0
            if not c.pump_node and any(e.get("splash") for e in c.edges):
                # splash lubrication: no pump at all -- the rod's dipper
                # throws oil out of the trough in proportion to crank
                # speed (a disclosed 0.4 of crank speed as the equivalent
                # pump speed, i.e. a fraction of what a gear pump would
                # move; it is also why these engines run low oil pressure)
                pump_omega = self.omega.get("powertrain.engine", 0.0) * 0.4
            if c.kind_class == "thermal-liquid":
                # a centrifugal pump: flow scales with speed up to its
                # declared design flow at its rated speed (both sized by
                # production from this engine's own rated heat and jacket
                # temperature rise -- see abstract_ui_vehicles.py's
                # powertrain.water_pump); the exchanger reaches full
                # effectiveness at design flow. The old 0.35 L/min per
                # rad/s over 40 L/min car-scale relation was starving a
                # marine engine's exchanger at 3.7 L/min.
                if c.pump_design_flow_lpm > 0.0 and c.pump_rated_omega_rad_s > 0.0:
                    flow_lpm = c.pump_design_flow_lpm * max(0.0, pump_omega) / c.pump_rated_omega_rad_s
                    flow_factor = max(0.05, min(1.0, flow_lpm / c.pump_design_flow_lpm))
                else:
                    flow_lpm = max(0.0, pump_omega) * 0.35
                    flow_factor = max(0.05, min(1.0, flow_lpm / 40.0))
                c.flow_lpm = flow_lpm
                delta_t = max(0.0, c.temp_k - ambient_k)
                # Any number of fans genuinely add real airflow to the
                # same manifold -- each one's own contribution normalized
                # against its own declared rated speed (a mechanical
                # crank fan and an electric fan don't share one), summed
                # and capped at the exchanger's own real practical
                # ceiling, same ceiling a single fan already had.
                fan_airflow_frac = 0.0
                for fan_nid in c.fan_nodes:
                    fan_omega = max(0.0, self.omega.get(fan_nid, 0.0))
                    rated_omega = self._fan_specs.get(fan_nid, {}).get("rated_omega", FAN_REFERENCE_OMEGA_RAD_S)
                    fan_airflow_frac += min(1.0, fan_omega / max(rated_omega, 1e-6))
                fan_airflow_frac = min(1.0, fan_airflow_frac)
                # a real radiator's own declared coefficient when this
                # circuit actually has one (the fan/radiator/condenser
                # stack), scaled by both coolant flow and fan airflow --
                # falls back to the flat approximation otherwise (no
                # radiator node at all, the lumped-external-circuit case)
                # the real wax-pellet thermostat: valve lift is proportional
                # to how far the coolant is into the wax's melt band, with
                # the pellet's own first-order thermal lag -- and it gates
                # only the RADIATOR/external leg (unpassed flow goes round
                # the bypass back to the pump, block circulation is never
                # starved). A circuit with no thermostat node is ungated.
                if c.thermostat_full_open_k > c.thermostat_opening_start_k > 0.0:
                    lift_target = max(0.0, min(1.0, (c.temp_k - c.thermostat_opening_start_k)
                                               / (c.thermostat_full_open_k - c.thermostat_opening_start_k)))
                    c.thermostat_open_frac += (lift_target - c.thermostat_open_frac) * min(
                        1.0, dt / max(c.thermostat_tau_s, 1e-3))
                    external_flow_factor = flow_factor * c.thermostat_open_frac
                else:
                    c.thermostat_open_frac = 1.0
                    external_flow_factor = flow_factor
                if c.active_heat_exchange_w_per_k > 0.0:
                    # a seawater plate exchanger's secondary side is a
                    # pumped liquid, not fan-blown air: full UA whenever the
                    # seawater pump runs (it runs with the engine)
                    airflow_factor = 1.0 if c.exchanger_medium == "seawater" else 0.4 + 0.6 * fan_airflow_frac
                    reject_kw = c.active_heat_exchange_w_per_k * delta_t * external_flow_factor * airflow_factor / 1000.0
                else:
                    reject_kw = 0.0018 * delta_t * external_flow_factor
                reject_kw += c.passive_heat_loss_w_per_k * delta_t / 1000.0
                net_kw = waste_heat_kw * c.heat_share - reject_kw
                c.temp_k = max(ambient_k, c.temp_k + (net_kw * dt) / c.thermal_mass_kj_per_k)

                if any("oil" in nid for nid in c.nodes):
                    # A real gear pump: delivered pressure tracks pump
                    # speed until the relief valve caps it (relief_pressure_pa,
                    # the real declared setting off the gallery edge
                    # itself -- abstract_ui_vehicles.py's
                    # powertrain.oil_pump_to_gallery). A real extra
                    # consumer sharing this same gallery (a turbo's
                    # bearing feed, when the engine has one) genuinely
                    # draws flow off it, eating into the pressure the
                    # rest of the circuit sees -- a coarse real orifice-
                    # style penalty per extra branch actually present in
                    # this circuit's own edges, not a free multiplication
                    # of plumbing.
                    pressure_per_rad_s = 900.0
                    raw_target_pa = 101_325.0 + max(0.0, pump_omega) * pressure_per_rad_s
                    relief_pa = c.relief_pressure_pa if c.relief_pressure_pa > 0.0 else 1e12
                    extra_consumers = sum(1 for e in c.edges if "turbo" in e["identity"])
                    target_pressure_pa = min(raw_target_pa, relief_pa) / (1.0 + 0.15 * extra_consumers)
                    pressure_tau_s = 0.15
                    c.pressure_pa += (target_pressure_pa - c.pressure_pa) * min(1.0, dt / pressure_tau_s)
            elif c.kind_class == "compressible-gas":
                is_exhaust = any("exhaust" in nid for nid in c.nodes)
                if is_exhaust:
                    # Real backpressure, mirroring the intake choke below
                    # but on the other side of atmospheric: exhaust gas
                    # restricted by the same real port/primary-pipe
                    # cross-section (flow_capacity_kg_s, off the real
                    # per-cylinder exhaust-primary edges) builds pressure
                    # ABOVE atmospheric when the engine's own exhaust
                    # demand exceeds it, instead of the intake side's
                    # vacuum-side restriction.
                    #
                    # exhaust_brake_frac (0..1) is a real, driver-operated
                    # valve pinching that same real cross-section further
                    # -- the actual mechanism a real diesel exhaust brake
                    # uses (a butterfly downstream of the turbo that closes
                    # on lift-off), not an abstract braking-torque number:
                    # it genuinely shrinks the port's effective flow
                    # capacity, and this exact same choked-flow formula
                    # below turns that into real backpressure, exactly as
                    # it already does for a merely undersized exhaust.
                    effective_flow_capacity_kg_s = c.flow_capacity_kg_s * max(0.05, 1.0 - exhaust_brake_frac)
                    target_pressure_pa = 101_325.0
                    if effective_flow_capacity_kg_s > 0.0 and exhaust_demand_kg_s > effective_flow_capacity_kg_s:
                        excess_frac = exhaust_demand_kg_s / effective_flow_capacity_kg_s - 1.0
                        target_pressure_pa = 101_325.0 * (1.0 + excess_frac)
                else:
                    # a real choke limit first: the valve-curtain/port cross-
                    # section (flow_capacity_kg_s, a real declared property
                    # off the intake-air edge) can only pass so much air
                    # regardless of how much pressure is stacked upstream
                    # (supercharger, nitrous) -- when the engine's own demand
                    # exceeds that capacity, the achievable pressure is
                    # reduced BELOW WHATEVER IS COMMANDED UPSTREAM,
                    # proportional to capacity/demand (a real restricted-
                    # orifice relationship), not blended toward atmospheric.
                    # That distinction matters: blending toward atmospheric
                    # meant this ceiling only ever did anything for a
                    # boosted supply (elevated above atmospheric) -- a
                    # naturally-aspirated engine's own WOT supply is
                    # already ~atmospheric, so the old formula left its
                    # MAP pinned at atmospheric even once real demand blew
                    # straight through the port's own real capacity, the
                    # opposite of what a real NA engine's MAP actually does
                    # (it droops well below atmospheric once the port
                    # itself becomes the restriction, not the throttle).
                    # Then fills/empties toward THAT achievable target with
                    # a real time constant derived from line volume -- same
                    # MAP-lag reasoning already used for manifold pressure
                    # elsewhere.
                    target_pressure_pa = c.supply_pressure_pa
                    if c.flow_capacity_kg_s > 0.0 and intake_demand_kg_s > c.flow_capacity_kg_s:
                        achievable_frac = c.flow_capacity_kg_s / intake_demand_kg_s
                        target_pressure_pa = c.supply_pressure_pa * achievable_frac
                    c.achievable_target_pressure_pa = target_pressure_pa
                tau_s = max(0.02, c.volume_l / 6.0)
                c.pressure_pa += (target_pressure_pa - c.pressure_pa) * min(1.0, dt / tau_s)

                # Real gas thermal state: exhaust gas is heated by its own
                # real combustion waste-heat share (thermal.engine_to_exhaust,
                # the same mechanism coolant/oil use); the intake side
                # gains genuine adiabatic-compression heat whenever
                # something is pushing its pressure above atmospheric
                # (turbo/supercharger/nitrous). Both are cooled by
                # whatever real passive heat-soak the plenum/manifold body
                # itself declares (heat_soak_w_per_k -- previously
                # declared and never read) plus ordinary convection. This
                # is the real link a charge-air cooler's benefit runs
                # through: a cooler, denser charge is worth more effective
                # boost at the same MAP, and the plenum's own passive
                # cooling capacity is what actually determines how much of
                # the compression heat gets rejected before that charge is
                # drawn -- the caller folds this circuit's own temp_k into
                # the real charge-density correction.
                pressure_ratio = max(1.0, c.pressure_pa / 101_325.0)
                compression_heat_kw = 0.0
                if not is_exhaust and pressure_ratio > 1.001:
                    # real adiabatic-compression relation for air (n=1.4):
                    # T2 = T1 * pressure_ratio^((n-1)/n), expressed as a
                    # heat-input rate driving temp_k toward that real
                    # endpoint rather than an instant jump
                    adiabatic_target_k = ambient_k * pressure_ratio ** (0.4 / 1.4)
                    compression_heat_kw = max(0.0, adiabatic_target_k - c.temp_k) * 0.02
                heat_in_kw = (waste_heat_kw * c.heat_share if is_exhaust else 0.0) + compression_heat_kw
                # Real Newton's-law-of-cooling convection is signed, not
                # one-directional: a body colder than its surroundings
                # gains heat from them just as readily as a hot one
                # loses it. Exhaust gas has no real mechanism that would
                # ever put it below ambient here, so clamping it stays
                # harmless; the intake side genuinely can run colder
                # than ambient now (real evaporative charge cooling,
                # fuel_evap_loss_kw below) and needs the real two-way
                # relation or it can never recover -- a one-way valve
                # that only ever removes heat, discovered by that
                # cooling term pinning the charge to an unrealistic
                # floor with nothing to pull it back.
                temp_delta_k = c.temp_k - ambient_k
                convective_loss_kw = 0.0015 * (max(0.0, temp_delta_k) if is_exhaust else temp_delta_k)
                soak_loss_kw = c.heat_soak_w_per_k * (max(0.0, temp_delta_k) if is_exhaust else temp_delta_k) / 1000.0
                # real evaporative + sensible charge cooling from the
                # fuel is NOT applied here -- this lumped lightweight-
                # plenum thermal mass, coupled only by a weak passive
                # convective term, has no real way to absorb a multi-kW
                # evaporative load and reach a sane equilibrium (tried;
                # it pins to a runaway low temperature). Real charge
                # cooling from evaporating fuel is a FLOW-THROUGH energy
                # balance (cooling power divided by the fresh air mass
                # flow's own heat capacity), not something a static
                # lumped-mass ODE models correctly -- applied directly
                # in engine_cycle_sim.py against the fresh air charge
                # instead, see fuel_cooling_kw's real use there.
                net_gas_kw = heat_in_kw - convective_loss_kw - soak_loss_kw
                # air's own real heat capacity at this line's volume, not
                # the metal body's mass -- ~1.2 kg/m^3 * 1.0 kJ/kg*K
                gas_thermal_mass_kj_per_k = max(0.05, c.volume_l * 0.0012)
                new_temp_k = c.temp_k + (net_gas_kw * dt) / gas_thermal_mass_kj_per_k
                # exhaust gas never really runs colder than ambient
                # (there's no real mechanism here that would cool it
                # below that); the intake charge genuinely CAN, and
                # should be allowed to -- that's the entire real point
                # of evaporative charge cooling (fuel_evap_loss_kw
                # above) and would otherwise silently discard it. A real
                # absolute floor (not ambient) guards against a runaway
                # numerical negative, nothing more.
                c.temp_k = max(ambient_k, new_temp_k) if is_exhaust else max(150.0, new_temp_k)
            elif c.kind_class == "high-pressure-liquid-supply" and any("fuel" in nid for nid in c.nodes):
                # a real tank, not a pressurized bottle: it drains by
                # whatever's actually being burned (fuel_demand_kg_s,
                # the caller's own real combustion-derived need), capped
                # by the pump's real rated flow_capacity_kg_s -- a weak
                # pump under a big demand genuinely can't keep up, same
                # as a real failing/undersized fuel pump -- and, for an
                # atmospheric engine's real gasholder specifically (see
                # engines.Engine.atmospheric_supply_tank_capacity_kg),
                # it can ALSO be filled by an on-site generator (gas_
                # works.GasGenerator/Boiler, via EngineCycleSim.
                # atmospheric_generator) at the same time -- a real
                # gasholder both fills and drains, unlike a combustion
                # engine's static pre-filled tank. fuel_production_kg_s
                # is 0.0 for every existing combustion/turbine engine
                # (no generator declared), so this changes nothing for
                # them. Uses step_depletable_reservoir -- the one real
                # mass-transfer convention this circuit, the pneumatic
                # reserve below, and the nitrous bottle further down all
                # share, not three hand-copied variations of the same
                # arithmetic.
                #
                # bottle_capacity_kg == 0.0 is a real, distinct case,
                # not just "a very small tank": no capacity_kg-bearing
                # node exists on this circuit at all, meaning there's no
                # real depletable vessel here -- an atmospheric engine's
                # unmetered piped town main (drivetrain_graph.py's own
                # "atmospheric" branch, no gasholder declared) forms
                # exactly this kind of zero-capacity circuit from its
                # bare connection fitting. Real infinite/unmetered
                # supply always satisfies demand and never depletes;
                # step_depletable_reservoir's own real "tiny reservoir"
                # math would otherwise drain a phantom near-zero tank to
                # empty almost instantly under any nonzero demand, which
                # is exactly wrong for a main that was never a vessel.
                if c.bottle_capacity_kg <= 0.0:
                    c.delivered_flow_kg_s = fuel_demand_kg_s
                    c.fill_level_frac = 1.0
                else:
                    c.fill_level_frac, c.delivered_flow_kg_s = step_depletable_reservoir(
                        c.fill_level_frac, c.bottle_capacity_kg, dt,
                        demand_kg_s=fuel_demand_kg_s, production_kg_s=fuel_production_kg_s,
                        flow_capacity_kg_s=c.flow_capacity_kg_s, valve_open=c.valve_open)
                    # and what those contents actually ARE -- the
                    # generator's own current purity mixing into
                    # whatever the holder already held (see
                    # step_reservoir_composition); inert for a pre-
                    # filled tank with no production (stays 1.0)
                    c.composition_frac = step_reservoir_composition(
                        c.composition_frac, c.fill_level_frac, c.bottle_capacity_kg, dt,
                        production_kg_s=fuel_production_kg_s,
                        production_composition_frac=fuel_production_composition_frac)
                # real fuel temperature: an uninsulated tank/line just
                # passively equilibrates toward ambient (slow -- a tank's
                # own real surface-to-volume ratio is small); a real
                # drag-racing cooler/ice-box actively pulls it toward
                # its own real target instead, much faster (an active
                # chiller loop, not passive soak). First-order lag
                # toward whichever target applies, the same real pattern
                # already used for MAP/coolant elsewhere in this file.
                target_k = fuel_cooler_target_k if fuel_cooler_target_k is not None else ambient_k
                tau_s = FUEL_COOLER_TAU_S if fuel_cooler_target_k is not None else FUEL_PASSIVE_TAU_S
                c.temp_k += (target_k - c.temp_k) * min(1.0, dt / tau_s)
            elif c.kind_class == "high-pressure-liquid-supply" and any("pneumatic" in nid for nid in c.nodes):
                # a real reserve tank that FILLS instead of draining --
                # the same depletable-reservoir bookkeeping nitrous/fuel
                # already use, just run in reverse. Real air mass flow
                # in is the compressor's own real delivered mechanical
                # power (pneumatic_compressor_delivered_w, read back
                # from the SAME belt-clutch mechanism ac_compressor_
                # load_w already uses) divided by the real specific work
                # to compress air to a typical reserve-tank pressure
                # (isothermal compression, w = R*T*ln(P2/P1) per kg --
                # PNEUMATIC_SPECIFIC_COMPRESSION_WORK_J_PER_KG is that
                # relation evaluated at a real, typical ~120psi target,
                # not a per-tank-configurable pressure this pass doesn't
                # carry). No consumption path yet (nothing in this toy
                # draws off the tank) -- fills and holds, honestly
                # disclosed as the real, current boundary rather than a
                # fabricated bleed-down.
                # the real pressure-switch unloader: the compressor loads
                # below cut-in and unloads at cut-out (fill fraction is
                # pressure fraction at fixed volume, ideal gas)
                if c.fill_level_frac >= c.regulator_cut_out_frac:
                    c.compressor_loaded = False
                elif c.fill_level_frac <= c.regulator_cut_in_frac:
                    c.compressor_loaded = True
                # compression work per kg to THIS tank's working pressure
                # (isothermal: R*T*ln(P/P0)), not one fixed figure
                p_work = max(c.working_pressure_pa, 2.0 * 101_325.0)
                specific_work_j_per_kg = 287.0 * ambient_k * math.log(p_work / 101_325.0)
                if c.compressor_drive == "crank-belt":
                    delivered_w = pneumatic_compressor_delivered_w if c.compressor_loaded else 0.0
                else:
                    # motor-driven: runs at rating whenever loaded (the
                    # sim charges its electrical draw off this figure)
                    delivered_w = c.compressor_rated_w if c.compressor_loaded else 0.0
                c.compressor_running_w = delivered_w
                if delivered_w > 0.0 and c.fill_level_frac < 1.0:
                    air_mass_flow_kg_s = delivered_w / specific_work_j_per_kg
                    c.delivered_flow_kg_s = air_mass_flow_kg_s
                    filled_kg = air_mass_flow_kg_s * dt
                    c.fill_level_frac = min(1.0, c.fill_level_frac + filled_kg / max(c.bottle_capacity_kg, 0.01))
                else:
                    c.delivered_flow_kg_s = 0.0
                # the one real consumer so far: starting air admitted to
                # the cylinders (starter.py's air-start), a real mass
                # drawn off the receiver every admission
                if starting_air_kg_s > 0.0:
                    c.fill_level_frac = max(0.0, c.fill_level_frac
                                            - starting_air_kg_s * dt / max(c.bottle_capacity_kg, 0.01))
            elif c.kind_class == "high-pressure-liquid-supply":
                # a real depletable bottle (nitrous): the solenoid
                # (valve_open, commanded by the caller -- engine_cycle_
                # sim's nitrous_active) either passes a real mass flow
                # rate (bottle pressure driven, tapering as the bottle
                # empties and its own vapor pressure drops) or passes
                # nothing at all. This is the slow-rate step the fluid
                # graph runs at; the actual combustion effect (extra
                # oxidizer) is read from delivered_flow_kg_s by the caller.
                if c.valve_open and c.fill_level_frac > 0.0:
                    if c.regulated_flow_kg_s > 0.0:
                        # a real progressive flow regulator (a WMI kit's
                        # own metering valve): duty ramps 0->1 between
                        # its declared onset and full boost points, off
                        # this tick's real manifold pressure -- not
                        # derived from tank size at all, so it doesn't
                        # dose during low-boost/idle/ramp conditions the
                        # way an un-metered on/off valve genuinely would
                        span = max(1e-6, c.regulator_full_map_frac - c.regulator_onset_map_frac)
                        duty = max(0.0, min(1.0, (regulator_map_frac - c.regulator_onset_map_frac) / span))
                        rated_flow_kg_s = c.regulated_flow_kg_s * duty
                    else:
                        # no declared regulator: a real, genuinely
                        # un-metered on/off kit (nitrous) -- rated off
                        # its own bottle size, real for that hardware
                        rated_flow_kg_s = max(0.01, c.bottle_capacity_kg) * 0.018
                    c.delivered_flow_kg_s = rated_flow_kg_s * max(0.2, min(1.0, c.fill_level_frac * 1.3))
                    depleted_kg = c.delivered_flow_kg_s * dt
                    c.fill_level_frac = max(0.0, c.fill_level_frac - depleted_kg / max(c.bottle_capacity_kg, 0.01))
                else:
                    c.delivered_flow_kg_s = 0.0
            else:  # incompressible-hydraulic
                # near-instant: high bulk modulus means pressure follows
                # its source almost immediately, only flow is restricted
                tau_s = max(0.005, c.volume_l / 200.0)
                c.pressure_pa += (c.supply_pressure_pa - c.pressure_pa) * min(1.0, dt / tau_s)
