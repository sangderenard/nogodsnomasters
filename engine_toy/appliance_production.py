"""A washer and a dryer: the same cabinet with a drum behind the door.

THIS IS WHAT THE GENERIC BOX IS FOR. Nothing here sizes a wall, hangs a
door or declares a foot; cabinet.py does that. What a washer adds is a
suspended drum with the motor it is specced with, a pump and a
counterweight below, a detergent drawer above, a porthole in a solid
door -- and a cycle whose spin state carries a wet towel on one side.

THE COUNTERWEIGHT IS NOT DECORATION. A washer's tub hangs on springs of
a few kN/m so that a thousand rpm passes far above its bounce frequency
and the cabinet feels a small fraction of the unbalance; the mass on
those springs is what sets that frequency, and the concrete is there to
put it low without making the drum heavy. Take it out and the same
towel walks the machine across the floor. The mode table shows the
difference as the drum's transmissibility, per row, with numbers.

A DRYER'S DRUM IS ON ROLLERS AND ITS CABINET IS THE BEARING HOUSING.
Light, slow, and bolted: it has nothing to isolate. What it has instead
is a heater and a blower in the plant compartment, and the blower is
the only thing in it that spins fast.
"""
from __future__ import annotations

import cabinet as cb
from operating_states import Cycle, OperatingState, run_up, coast

WASHER_INSIDE_M = (0.56, 0.56, 0.50)
DRYER_INSIDE_M = (0.56, 0.56, 0.52)
WASH_RPM = 50.0
SPIN_RPM = 1200.0
TUMBLE_RPM = 50.0
#: A wet towel on one side: half a kilogramme at the drum wall.
TOWEL_UNBALANCE_KG_M = 0.5 * 0.24

# One fixed fictional service model.  Faults annotate the actual component
# nodes/edges below; they are not a parallel appliance state machine.
DRYER_FAULT_PART = {
    "broken-belt": "drum.belt",
    "high-drag-idler": "bottom.idler",
    "high-drag-drum-roller": "drum.roller_l",
    "failed-open-door-switch": "door_switch",
    "misaligned-latch-striker": "door_latch",
    "disconnected-control-connector": "top.control_connector",
    "open-thermal-fuse": "bottom.thermal_fuse",
    "open-heater-element": "bottom.heater",
}


def washer_spec(identity: str = "plant.washer", *, counterweight_kg: float = 20.0,
                drive: str = "belt") -> cb.CabinetSpec:
    plant = [cb.PlantItem("pump", "drain-pump", (0.05, 0.04, 0.05), 1.2, at=(-0.5, 0.5),
                          rotor={"rotor_axis": (1.0, 0.0, 0.0), "rotor_rated_rpm": 2800.0,
                                 "rotor_shape": "vaned-rotor", "rotor_mass_kg": 0.3,
                                 "rotor_radius_m": 0.03, "balance_grade_mm_s": 6.3,
                                 "runs_in": ("drain",)})]
    if counterweight_kg > 0.0:
        plant.append(cb.PlantItem("counterweight", "counterweight", (0.15, 0.04, 0.10),
                                  counterweight_kg, at=(0.0, -0.3), material="concrete"))
    return cb.CabinetSpec(
        identity=identity, inside_m=WASHER_INSIDE_M, construction="screwed-sheet",
        sheet_thickness_m=0.0008,
        door=cb.Door(kind="solid", hinge="piano-hinge", dogs=1,
                     window=cb.Window(kind="porthole", porthole_radius_m=0.15)),
        drum=cb.Drum(radius_m=0.24, length_m=0.32, mass_kg=9.0, rated_rpm=SPIN_RPM,
                     mount="suspended", suspension_n_per_m=8000.0, suspension_damping=0.25,
                     runs_in=("wash", "spin"), load_kg=8.0,
                     drive=cb.DrumDrive(kind=drive, motor_kw=0.5, rated_rpm=2800.0, mass_kg=8.0)),
        bottom=cb.Compartment(0.12, plant=tuple(plant), label="plant"),
        top=cb.Compartment(0.10, plant=(cb.PlantItem("drawer", "detergent-drawer",
                                                     (0.10, 0.03, 0.12), 1.5, at=(-0.6, 0.5),
                                                     material="plastic"),),
                           label="controls"),
        feet_per_side=2, label="washer")


def washer_cycle(identity: str = "plant.washer", *, towel: bool = True,
                 live_vibration: bool = False) -> Cycle:
    spec = washer_spec(identity)
    loaded = cb.charges(spec)
    drum = f"{identity}.drum"
    over = {drum: TOWEL_UNBALANCE_KG_M} if towel else {}
    return Cycle(machine=identity, states=(
        OperatingState("wash", speeds={drum: WASH_RPM / SPIN_RPM}, charges_kg=loaded,
                       note="tumbling, full of water"),
        OperatingState("drain", speeds={drum: WASH_RPM / SPIN_RPM}, charges_kg=loaded,
                       note="pump running"),
        OperatingState("spin", curve=run_up(12), charges_kg=loaded, unbalance_kg_m=over,
                       note="run-up to full spin with whatever is on one side"),
        OperatingState("coast", curve=coast(6), charges_kg=loaded, unbalance_kg_m=over,
                       note="motor off, drum coasting"),
    ), live_vibration=live_vibration)


def dryer_spec(identity: str = "plant.dryer") -> cb.CabinetSpec:
    plant = (cb.PlantItem("blower", "centrifugal-blower", (0.08, 0.06, 0.08), 3.0, at=(0.5, -0.3),
                          rotor={"rotor_axis": (0.0, 0.0, 1.0), "rotor_rated_rpm": 2800.0,
                                 "rotor_shape": "centrifugal-impeller", "rotor_mass_kg": 0.6,
                                 "rotor_radius_m": 0.07, "balance_grade_mm_s": 6.3,
                                 "runs_in": ("tumble", "cool")}),
             cb.PlantItem("heater", "heater-cassette", (0.10, 0.03, 0.06), 1.8, at=(-0.5, -0.3),
                          material="stainless-plate"),
             cb.PlantItem("idler", "belt-idler-assembly", (0.035, 0.035, 0.025), 0.45,
                          at=(0.0, 0.45), material="steel-plate"),
             cb.PlantItem("thermal_fuse", "one-shot-thermal-fuse", (0.018, 0.008, 0.008),
                          0.02, at=(-0.5, 0.35), material="unresolved-composite"))
    return cb.CabinetSpec(
        identity=identity, inside_m=DRYER_INSIDE_M, construction="screwed-sheet",
        sheet_thickness_m=0.0008,
        door=cb.Door(kind="solid", hinge="piano-hinge", dogs=1,
                     window=cb.Window(kind="porthole", porthole_radius_m=0.17)),
        drum=cb.Drum(radius_m=0.27, length_m=0.38, mass_kg=6.0, rated_rpm=TUMBLE_RPM,
                     mount="rollers", runs_in=("tumble", "cool"), load_kg=6.0,
                     drive=cb.DrumDrive(kind="belt", motor_kw=0.25, rated_rpm=2800.0,
                                        mass_kg=5.0)),
        bottom=cb.Compartment(0.14, plant=plant, label="plant"),
        top=cb.Compartment(0.10, plant=(
            cb.PlantItem("timer", "dryer-timer", (0.035, 0.035, 0.025), 0.35,
                         at=(-0.55, 0.0), material="unresolved-composite"),
            cb.PlantItem("controls", "dryer-controls", (0.055, 0.025, 0.04), 0.30,
                         at=(0.15, 0.0), material="unresolved-composite"),
            cb.PlantItem("control_connector", "keyed-control-connector",
                         (0.018, 0.012, 0.012), 0.03, at=(0.65, 0.0),
                         material="unresolved-polymer"),
        ), label="controls"),
        feet_per_side=2, label="dryer")


def dryer_cycle(identity: str = "plant.dryer", *, live_vibration: bool = False) -> Cycle:
    spec = dryer_spec(identity)
    loaded = cb.charges(spec)
    return Cycle(machine=identity, states=(
        OperatingState("tumble", charges_kg=loaded, note="heater on, drum turning, blower running"),
        OperatingState("cool", charges_kg=loaded, note="heater off, still tumbling"),
    ), live_vibration=live_vibration)


def build_washer(identity: str = "plant.washer", **kw):
    return cb.build(washer_spec(identity, **kw))


def _dryer_extras(built: cb.Built) -> None:
    """Put service and airflow components in the cabinet's real graph."""
    g, ident = built.graph, built.spec.identity

    def fixed_part(suffix, position, kind, half_extent, material, support, **attrs):
        name = f"{ident}.{suffix}"
        g.node(name, position, kind, half_extent_m=half_extent, material=material,
               mass_kg=float(attrs.pop("mass_kg", 0.05)), in_view="plant",
               solver_condensed_into=support, **attrs)
        g.edge(f"{name}.mount", name, support, "bolted-flange-mount", radius=0.003,
               part_role=f"{kind}-mount")
        return name

    door_switch = fixed_part(
        "door_switch", (0.25, 0.55, 0.265), "door-interlock-switch",
        (0.018, 0.025, 0.012), "unresolved-composite", built.wn("right", "dog_0"),
        contact_states=("open", "closed"), safety_interlock=True)
    latch = fixed_part(
        "door_latch", (0.26, 0.55, 0.285), "door-striker-latch",
        (0.012, 0.020, 0.010), "steel-plate", built.door_nodes["dog_0"],
        actuates=door_switch)
    lint = fixed_part(
        "lint_screen", (0.0, 0.28, 0.22), "removable-lint-screen",
        (0.18, 0.015, 0.015), "steel-mesh", built.door_nodes["seal"],
        separate_from_building_duct=True, fouling_mode="filter-cake", blocked_frac=0.0)
    collar = fixed_part(
        "outlet_collar", (0.24, 0.10, -0.24), "dryer-outlet-collar",
        (0.055, 0.055, 0.035), "galvanized-sheet", built.wn("back", "drum_bearing"),
        port_kind="dryer-exhaust", bore_m=0.1016, circuit_identity=f"{ident}.process-air")
    # The one actual process-air circuit.  Heater, drum/screen, blower and
    # collar are ordinary graph endpoints, so the fluid circuit discovers it.
    flow = {"circuit_identity": f"{ident}.process-air",
            "medium_rate_state": "air_flow_kg_s", "fluid": "air",
            "fluid_route_class": "rigid-orthogonal"}
    heater = built.plant_nodes["heater"]["base"]
    blower = built.plant_nodes["blower"]["base"]
    g.edge(f"{ident}.air.heater_to_screen", heater, lint, "air-line",
           radius=0.045, part_role="internal-air-passage", **flow)
    g.edge(f"{ident}.air.screen_to_blower", lint, blower, "air-line",
           radius=0.045, part_role="internal-air-passage", **flow)
    g.edge(f"{ident}.air.blower_to_outlet", blower, collar, "air-line",
           radius=0.045, part_role="internal-air-passage", **flow)
    # Fixed-model interlocks are components, not inferred booleans.
    fixed_part("belt_tension_interlock", (0.0, 0.12, 0.02), "belt-tension-interlock",
               (0.014, 0.010, 0.010), "unresolved-composite",
               built.plant_nodes["idler"]["base"], safety_interlock=True)
    fixed_part("motor_speed_heater_interlock", (0.12, 0.12, 0.0),
               "motor-speed-heater-interlock", (0.014, 0.010, 0.010),
               "unresolved-composite", built.drum_nodes["motor"]["base"],
               safety_interlock=True)
    # These parts are deliberately inside the two declared service bays.
    # A routed component can belong to more than one nested claim (the air
    # path leaves the bottom plant bay and enters the drum bay), so it names
    # both rather than being mistaken for an intruder in either.
    bottom_bay = f"{ident}.bottom_bay"
    drum_bay = f"{ident}.drum_bay"
    for edge in g.edges:
        if edge["identity"].startswith(f"{ident}.air."):
            edge["clears"] = [bottom_bay, drum_bay]
        elif edge["identity"] == f"{ident}.belt_tension_interlock.mount":
            edge["clears"] = [bottom_bay, drum_bay]
        elif edge["identity"] in {f"{ident}.lint_screen.mount",
                                  f"{ident}.motor_speed_heater_interlock.mount"}:
            edge["clears"] = [drum_bay]


def _apply_dryer_faults(built: cb.Built, faults) -> None:
    nodes = {n["identity"]: n for n in built.graph.nodes}
    edges = {e["identity"]: e for e in built.graph.edges}
    for fault in tuple(faults):
        suffix = DRYER_FAULT_PART.get(fault)
        if suffix is None:
            raise ValueError(f"unknown fixed-model dryer fault {fault!r}")
        identity = f"{built.spec.identity}.{suffix}"
        part = nodes.get(identity) or edges.get(identity)
        if part is None:
            raise RuntimeError(f"dryer fault target {identity!r} was not built")
        part["starting_condition"] = fault
        part["generated_fault"] = True


def build_dryer(identity: str = "plant.dryer", *, faults=()):
    built = cb.build(dryer_spec(identity), extras=_dryer_extras)
    _apply_dryer_faults(built, faults)
    return built
