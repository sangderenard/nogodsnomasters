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
             cb.PlantItem("heater", "heater-bank", (0.10, 0.03, 0.06), 1.8, at=(-0.5, -0.3),
                          material="stainless-plate"))
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


def build_dryer(identity: str = "plant.dryer"):
    return cb.build(dryer_spec(identity))
