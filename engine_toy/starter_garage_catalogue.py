"""Authoring catalogue for the starter garage.

The catalogue is the stable ledger behind the built scene.  An entry is a
kind of object and ``quantity`` says how many independent instances spawn;
``set`` is only a storage/selection grouping and never fuses those instances.
The forty constituent records retain the internal construction that matters
to service, load paths, routing, diagnosis, and later fabrication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class InventoryEntry:
    identity: str
    label: str
    category: str
    quantity: int = 1
    location: str = "garage"
    state: str = ""
    group: str | None = None
    game_binding: str = "physical-object"

    def __post_init__(self):
        if not self.identity or not self.label or self.quantity < 1:
            raise ValueError("inventory entries need an identity, label, and positive quantity")


@dataclass(frozen=True)
class ConstituentRecord:
    identity: str
    owner: str
    parts: tuple[str, ...]
    note: str = ""

    def __post_init__(self):
        if not self.identity or not self.owner or not self.parts:
            raise ValueError("constituent records need an identity, owner, and parts")


def _rows(category: str, rows: Iterable[tuple]) -> list[InventoryEntry]:
    out = []
    for row in rows:
        identity, label, *tail = row
        quantity = tail[0] if tail else 1
        location = tail[1] if len(tail) > 1 else "garage"
        state = tail[2] if len(tail) > 2 else ""
        group = tail[3] if len(tail) > 3 else None
        out.append(InventoryEntry(identity, label, category, quantity, location, state, group))
    return out


INVENTORY = tuple(
    _rows("garage-services", (
        ("garage.shell", "Garage shell", 1, "building", "layered concrete/timber/lining/roof/exterior"),
        ("garage.room-air", "Enclosed room air", 1, "garage", "finite heat/moisture/air region"),
        ("garage.window", "Small operable window", 1, "exterior wall", "closed and latched"),
        ("garage.house-door", "Door into house", 1, "house wall", "closed and locked"),
        ("garage.panel", "Garage electrical panel", 1, "service wall", "healthy incoming supply"),
        ("garage.wiring", "Installed wiring and boxes", 1, "walls/ceiling", "healthy"),
        ("garage.bench-receptacle", "Bench duplex receptacle", 3, "bench wall", "working; shared upstream circuit"),
        ("garage.utility-receptacle", "General utility receptacle", 2, "laundry and work bay", "working"),
        ("garage.washer-receptacle", "Dedicated washer receptacle", 1, "laundry wall", "working"),
        ("garage.dryer-receptacle", "Dedicated electric-dryer receptacle", 1, "laundry wall", "healthy four-contact split-phase"),
        ("garage.opener-receptacle", "Ceiling opener receptacle", 1, "ceiling", "working"),
        ("garage.lighting", "Two ceiling lights and wall switch", 3, "ceiling and exit wall", "on initially; three independent objects"),
        ("garage.hot-cold-service", "Hot and cold service connections", 2, "laundry wall", "independent working supplies with local shutoffs"),
        ("garage.washer-supply", "Washer supply connection", 2, "laundry wall", "valve and reinforced hose"),
        ("garage.washer-drainage", "Washer drainage", 1, "laundry wall", "connected"),
        ("garage.utility-sink", "Small utility sink", 1, "service wall", "working"),
        ("garage.leakage-path", "Room leakage paths", 1, "openings", "door/window seal gaps plus player openings"),
    ))
    + _rows("overhead-door", (
        ("door.sectional", "Sectional overhead door", 1, "vehicle opening", "closed"),
        ("door.counterbalance", "Counterbalance assembly", 1, "door head", "healthy and balanced"),
        ("door.opener", "Ceiling opener", 1, "ceiling", "powered drive mechanically immobilized"),
        ("door.trolley", "Door trolley and release latch", 1, "opener rail", "coupled initially"),
        ("door.release-cord", "Red release cord and handle", 1, "opener trolley", "within floor reach"),
        ("door.arm", "Door arm and bracket", 1, "door/opener", "coupled"),
        ("door.wall-button", "Wired wall button", 1, "exit wall", "connected; jammed drive cannot move"),
        ("door.sensors", "Obstruction sensor", 2, "door jambs", "aligned and working"),
        ("door.remote-clip", "Empty remote-control clip", 1, "wall", "remote absent"),
    ))
    + _rows("workbench", (
        ("bench.frame", "2x4-and-particleboard bench", 1, "bench wall", "sturdy; about 2.1 m"),
        ("bench.top", "Particleboard working top", 1, "bench", "thick and worn"),
        ("bench.shelf", "Lower shelf", 1, "bench", "supported by lower frame"),
        ("bench.pegboard", "Pegboard backing", 1, "bench", "on stand-off battens"),
        ("bench.hook", "Pegboard hook", 24, "pegboard", "individual; some unused"),
        ("bench.basket", "Wire pegboard basket", 2, "pegboard"),
        ("bench.socket-rail", "Socket rail", 2, "pegboard", "individual retaining clips"),
        ("bench.drawer", "Shallow bench drawer", 2, "bench", "unlocked"),
        ("bench.parts-cabinet", "Small-parts drawer cabinet", 1, "bench end"),
        ("bench.vise", "Bench vise", 1, "bench", "load carried through backing plate"),
        ("bench.soft-jaws", "Soft vise jaw", 2, "left drawer"),
        ("bench.stool", "Workshop stool", 1, "under bench"),
        ("bench.upper-shelf", "Upper wall shelf", 1, "bench wall"),
        ("bench.magnetic-tray", "Magnetic parts tray", 2, "bench", "empty"),
        ("bench.fastener-can", "Reused fastener can", 3, "bench", "removable lid and finite contents"),
    ))
    + _rows("hand-tools", (
        ("tool.ratchet", "3/8-inch-drive ratchet", 1, "pegboard"),
        ("tool.metric-socket", "Metric socket 6-17 mm", 12, "socket rail", "individual", "metric-socket-set"),
        ("tool.inch-socket", "Inch socket 1/4-11/16 inch", 8, "socket rail", "individual", "inch-socket-set"),
        ("tool.socket-extension", "Socket extension", 2, "pegboard", "short and long", "socket-accessories"),
        ("tool.socket-universal", "Socket universal joint", 1, "pegboard"),
        ("tool.metric-wrench", "Metric combination wrench 8-17 mm", 10, "pegboard", "individual", "metric-wrench-set"),
        ("tool.adjustable-wrench", "Adjustable wrench", 1, "pegboard"),
        ("tool.bit-driver", "Interchangeable-bit screwdriver", 1, "pegboard"),
        ("tool.driver-bit", "Slotted/Phillips/Torx/hex driver bit", 16, "bit holder", "individual", "driver-bit-set"),
        ("tool.nut-driver", "Appliance nut driver", 2, "pegboard", "1/4 and 5/16 inch", "appliance-nut-drivers"),
        ("tool.hex-key", "Metric hex key", 8, "pegboard", "individual", "metric-hex-key-set"),
        ("tool.combination-pliers", "Combination pliers", 1, "pegboard"),
        ("tool.needle-pliers", "Needle-nose pliers", 1, "pegboard"),
        ("tool.groove-pliers", "Groove-joint pliers", 1, "pegboard"),
        ("tool.locking-pliers", "Locking pliers", 1, "pegboard"),
        ("tool.diagonal-cutters", "Diagonal cutters", 1, "pegboard"),
        ("tool.wire-stripper", "Wire stripper/crimper", 1, "pegboard"),
        ("tool.claw-hammer", "Claw hammer", 1, "pegboard"),
        ("tool.soft-mallet", "Soft-faced mallet", 1, "pegboard"),
        ("tool.torque-wrench", "Torque wrench in protective case", 1, "upper shelf"),
        ("tool.hacksaw", "Hacksaw", 1, "pegboard"),
        ("tool.file", "Hand file", 2, "pegboard", "flat and half-round", "file-pair"),
        ("tool.utility-knife", "Retractable utility knife", 1, "drawer"),
        ("tool.putty-knife", "Blunt putty knife", 1, "drawer"),
        ("tool.retaining-ring-pliers", "Retaining-ring pliers", 1, "drawer"),
        ("tool.f-clamp", "F-style bar clamp", 4, "bench"),
    ))
    + _rows("measuring", (
        ("measure.tape", "Tape measure", 1, "pegboard"),
        ("measure.calipers", "Mechanical calipers in sleeve", 1, "bench"),
        ("measure.ruler", "Steel ruler", 1, "pegboard"),
        ("measure.square", "Combination square", 1, "pegboard"),
        ("measure.mirror", "Articulating inspection mirror", 1, "left drawer"),
        ("measure.flashlight", "Working battery flashlight", 1, "bench", "working"),
        ("measure.meter-hook", "Empty hook labelled METER", 1, "pegboard", "dust outline"),
        ("measure.instrument-pouch", "Zipped instrument pouch", 1, "right drawer behind rags", "closed"),
        ("measure.multimeter", "Digital multimeter", 1, "instrument pouch", "working battery and fuses"),
        ("measure.probe-lead", "Multimeter probe lead", 2, "instrument pouch", "red and black", "meter-kit"),
        ("measure.clip-adapter", "Clip-on probe adapter", 2, "instrument pouch", "individual", "meter-kit"),
        ("measure.meter-card", "Meter quick-reference card", 1, "instrument pouch"),
    ))
    + _rows("power-and-cleanup", (
        ("power.drill", "Cordless drill/driver", 1, "bench", "working"),
        ("power.battery", "Compatible drill battery", 2, "bench", "charged", "drill-system"),
        ("power.charger", "Battery charger", 1, "bench", "working", "drill-system"),
        ("power.drill-bit", "Drill bit", 13, "drill-bit case", "individual useful size", "drill-bit-set"),
        ("power.grinder", "Corded angle grinder", 1, "bench", "guard and auxiliary handle"),
        ("power.grinder-disc", "Grinder disc", 3, "bench", "grinding/cut-off/flap", "grinder-disc-set"),
        ("cleanup.shop-vac", "Wet/dry shop vacuum", 1, "work bay", "empty drum; working"),
        ("cleanup.vac-hose", "Vacuum hose", 1, "beside vacuum", "coiled"),
        ("cleanup.vac-attachment", "Vacuum attachment", 4, "with vacuum", "two tubes/crevice/floor", "vacuum-attachments"),
        ("cleanup.dry-filter", "Dry filter cartridge", 1, "shop vacuum", "installed"),
        ("cleanup.wet-sleeve", "Wet-pickup sleeve", 1, "with vacuum"),
        ("cleanup.vent-brush", "Dryer-vent brush and extension rods", 1, "visible pegboard basket", "reaches authored vent run"),
        ("cleanup.broom-set", "Broom and dustpan", 1, "beside bench"),
    ))
    + _rows("lifting-and-engine", (
        ("lift.gantry", "Compact freestanding gantry", 1, "work bay", "assembled and stable"),
        ("lift.chain-hoist", "Hand-chain hoist", 1, "gantry", "installed and unloaded"),
        ("lift.tackle", "Compatible lifting tackle", 1, "gantry", "fits starter engine"),
        ("engine.stand", "Rolling engine-repair stand", 1, "work bay", "assembled and unloaded"),
        ("engine.mounting-hardware", "Matching engine-stand mounting hardware", 1, "bag tied to stand"),
        ("engine.starter", "Complete modest starter engine", 1, "timber pallet", "cold, off, fuel-drained, capped"),
        ("engine.pallet", "Timber pallet", 1, "under engine"),
        ("engine.wheel-chock", "Wheel chock", 2, "work bay"),
        ("engine.blocking", "Wood blocking", 6, "beside stand", "separate pieces"),
    ))
    + _rows("laundry", (
        ("laundry.washer", "Front-loading washer", 1, "laundry corner", "working, connected, empty, unlocked"),
        ("laundry.dryer", "Vented electric dryer", 1, "laundry corner", "connected and off; generated faults"),
        ("laundry.basket", "Laundry basket", 1, "on washer"),
        ("laundry.test-textile", "Cotton towel or test rag", 4, "basket", "two cotton towels and two test rags; clean and dry", "test-load"),
        ("laundry.detergent", "Detergent bottle", 1, "laundry shelf", "partly full and capped"),
        ("laundry.lint-bin", "Lidded lint bin", 1, "beside dryer", "empty"),
        ("laundry.reference-sheet", "Dryer service sheet or washer reference sheet", 2, "accessible sleeve and rear shelf", "independent appliance-specific documents"),
    ))
    + _rows("dryer-exhaust", (
        ("vent.dryer-collar", "Dryer outlet collar", 1, "dryer"),
        ("vent.transition", "Semi-rigid metal transition duct", 1, "dryer to wall", "intact"),
        ("vent.band-clamp", "Band clamp", 4, "vent service joints"),
        ("vent.wall-collar", "Wall inlet collar", 1, "garage wall", "nominal four inch"),
        ("vent.rigid-run", "Rigid smooth-metal duct run", 1, "wall/ceiling", "about 3 m developed length"),
        ("vent.elbow", "Vent elbow", 2, "duct run", "separate geometry/losses"),
        ("vent.support-strap", "Duct support strap", 3, "framing"),
        ("vent.deposit", "Persistent lint/debris deposit", 1, "outlet elbow/hood", "present at every new game"),
        ("vent.exterior-hood", "Exterior exhaust hood", 1, "outside wall", "gravity backdraft flap"),
        ("vent.service-joint", "Accessible vent service joint", 1, "garage wall", "opens full run to brush/vacuum"),
    ))
    + _rows("dryer-spares", (
        ("spare.drum-belt", "Correct replacement drum belt", 1, "APPLIANCE PARTS carton"),
        ("spare.idler-kit", "Compatible idler kit", 1, "APPLIANCE PARTS carton"),
        ("spare.drum-roller", "Compatible drum support roller and retainer", 2, "APPLIANCE PARTS carton"),
        ("spare.door-switch", "Correct door-switch assembly", 1, "APPLIANCE PARTS carton"),
        ("spare.latch-kit", "Door striker/latch kit and screws", 1, "APPLIANCE PARTS carton"),
        ("spare.thermal-fuse", "Matching one-shot thermal fuse", 3, "APPLIANCE PARTS carton"),
        ("spare.heater", "Complete compatible heater cassette", 1, "APPLIANCE PARTS carton"),
        ("spare.harness", "Preterminated control-harness section", 1, "APPLIANCE PARTS carton"),
        ("spare.hardware-kit", "Appliance screws, clips, and retainers", 1, "APPLIANCE PARTS carton", "finite kit"),
        ("spare.transition-kit", "Metal transition duct with clamps", 1, "APPLIANCE PARTS carton"),
    ))
    + _rows("stock-and-clutter", (
        ("stock.bucket", "Bucket with bail handle and lid", 1, "lower shelf"),
        ("stock.drain-pan", "Shallow drain pan", 1, "lower shelf"),
        ("stock.funnel", "Funnel", 1, "parts cabinet"),
        ("stock.utility-jug", "Empty capped utility jug", 2, "lower shelf"),
        ("stock.spray-bottle", "Water spray bottle", 1, "bench", "partly full"),
        ("stock.clear-tubing", "Transparent flexible tubing", 1, "upper shelf", "3 m"),
        ("stock.garden-hose", "Short garden hose", 1, "utility sink"),
        ("stock.hose-clamp", "Small hose clamp", 8, "parts cabinet"),
        ("stock.2x4-offcut", "Loose 2x4 offcut", 4, "stock rack"),
        ("stock.panel-offcut", "Particleboard, plywood, or thin sheet-metal offcut", 3, "stock rack", "one independent piece of each material"),
        ("stock.angle-bracket", "Small steel angle bracket", 6, "parts cabinet"),
        ("stock.threaded-rod", "Threaded rod", 2, "stock rack", "matching nuts in stock"),
        ("stock.fasteners", "Labelled screws, bolts, nuts, and washers", 1, "fastener cans", "finite batches"),
        ("stock.low-voltage", "Low-voltage wire, terminals, and clips", 1, "parts cabinet", "finite assortment"),
        ("stock.extension-cord", "Workshop extension cord", 1, "bench", "healthy"),
        ("stock.work-light", "Portable guarded work light", 1, "bench"),
        ("stock.power-strip", "Small workshop power strip", 1, "bench"),
        ("stock.cloth-tape", "Cloth adhesive tape", 1, "drawer", "part-used"),
        ("stock.foil-tape", "Metal HVAC foil tape", 1, "drawer", "part-used"),
        ("stock.insulating-tape", "Insulating tape", 1, "drawer", "part-used"),
        ("stock.wood-glue", "Wood glue", 1, "drawer", "part-full"),
        ("stock.marker-pencil", "Marker or pencil", 2, "bench", "one of each; independent objects"),
        ("stock.moving-blanket", "Old moving blanket", 1, "stock rack"),
    ))
    + _rows("safety-and-reference", (
        ("safety.glasses", "Safety glasses", 1, "visible hook"),
        ("safety.face-shield", "Face shield", 1, "visible hook"),
        ("safety.hearing", "Hearing protection", 1, "pegboard end"),
        ("safety.gloves", "Work gloves", 1, "shelf"),
        ("safety.dust", "Dust protection", 1, "clean storage"),
        ("safety.extinguisher", "Serviceable extinguisher and bracket", 1, "exit route", "visible and unobstructed"),
        ("safety.alarm", "Heat detection and CO alarm", 1, "ceiling", "working"),
        ("safety.first-aid", "First-aid box", 1, "wall", "visible and unlocked"),
        ("garage.window-ac", "Small window air conditioner", 1, "window", "working, off, separate air paths"),
        ("reference.manual-binder", "Workshop manual binder", 1, "shelf"),
        ("reference.owner-notebook", "Previous owner's notebook", 1, "bench", "observations only"),
    ))
)


CONSTITUENTS = tuple(ConstituentRecord(*row) for row in (
    ("garage-shell-layers", "garage.shell", ("concrete slab", "timber framing", "wall lining", "ceiling", "roof", "exterior skin"), "Layers remain distinct."),
    ("window-parts", "garage.window", ("glazing", "latch", "insect mesh", "weather seals")),
    ("house-door-parts", "garage.house-door", ("leaf", "frame", "hinges", "handle", "latch", "strike", "lock")),
    ("panel-parts", "garage.panel", ("incoming supply", "disconnect", "bus bars", "breakers", "neutral bar", "earth bar", "cover", "circuit schedule")),
    ("installed-wiring-parts", "garage.wiring", ("conductors", "individual insulation", "cable jackets", "terminal contacts", "clamps", "boxes", "protective earth")),
    ("washer-supply-parts", "garage.washer-supply", ("valve", "reinforced hose", "couplings", "washers", "liner", "reinforcement", "jacket")),
    ("washer-drain-parts", "garage.washer-drainage", ("drain hose", "support hook", "standpipe", "trap", "building-drain connection")),
    ("sink-parts", "garage.utility-sink", ("basin", "faucet", "supplies", "strainer", "trap", "drain")),
    ("sectional-door-parts", "door.sectional", ("four sections", "hinges", "rollers", "tracks", "brackets", "bottom seal", "lifting handles")),
    ("counterbalance-parts", "door.counterbalance", ("springs", "shaft", "drums", "cables", "bearings", "anchors")),
    ("opener-parts", "door.opener", ("motor", "reduction gearing", "drive chain", "rail", "powered carriage", "housing", "travel sensing", "overload protection")),
    ("trolley-parts", "door.trolley", ("trolley", "release latch", "coupling")),
    ("door-arm-parts", "door.arm", ("pivoting linkage", "pins", "retainers", "mounting hardware")),
    ("bench-frame-parts", "bench.frame", ("six legs", "upper rails", "lower rails", "crossmembers", "diagonal bracing")),
    ("pegboard-parts", "bench.pegboard", ("perforated hardboard", "stand-off battens", "fasteners")),
    ("vise-parts", "bench.vise", ("spindle", "moving jaw", "handle", "swivel lock", "mounting bolts", "backing plate")),
    ("drill-parts", "power.drill", ("chuck", "gearbox", "adjustable clutch", "trigger", "motor", "reversing control")),
    ("grinder-parts", "power.grinder", ("motor", "spindle", "guard", "auxiliary handle", "power cord")),
    ("vacuum-parts", "cleanup.shop-vac", ("drum", "motor/blower", "lid seal", "latches", "float shutoff", "casters", "intake", "exhaust", "power cord")),
    ("gantry-parts", "lift.gantry", ("uprights", "beam", "braces", "feet", "fasteners")),
    ("hoist-parts", "lift.chain-hoist", ("hand chain", "load chain", "gearing", "brake", "shafts", "bearings", "housing", "hooks")),
    ("tackle-parts", "lift.tackle", ("chain or sling", "connecting links", "shackles")),
    ("engine-stand-parts", "engine.stand", ("rotating head", "adjustable arms", "positive locking pin", "wheels", "frame")),
    ("starter-engine-boundaries", "engine.starter", ("declared lifting points", "capped fuel connection", "capped cooling connection", "capped exhaust connection"), "No invisible running installation."),
    ("washer-cabinet", "laundry.washer", ("cabinet", "feet", "outer tub", "perforated drum", "bearings")),
    ("washer-drive", "laundry.washer", ("motor", "belt", "suspension springs", "dampers", "counterweights")),
    ("washer-water", "laundry.washer", ("inlet valves", "detergent dispenser", "level sensing", "drain pump", "filter", "hoses")),
    ("washer-door", "laundry.washer", ("door", "boot", "interlock")),
    ("dryer-cabinet", "laundry.dryer", ("cabinet panels", "feet", "door", "latch", "door switch")),
    ("dryer-drum", "laundry.dryer", ("drum", "baffles", "glides", "support rollers", "belt", "idler assembly")),
    ("dryer-drive-air", "laundry.dryer", ("motor", "blower wheel", "blower housing", "internal air passages", "outlet collar")),
    ("dryer-heat", "laundry.dryer", ("heater cassette", "cycling thermostat", "over-temperature protection", "thermal fuse")),
    ("dryer-controls", "laundry.dryer", ("timer", "controls", "harness", "cord", "interlocks")),
    ("dryer-lint-screen", "laundry.dryer", ("removable screen", "screen frame", "screen seal"), "Separate from the building duct."),
    ("transition-duct-parts", "vent.transition", ("semi-rigid duct", "dryer coupling", "wall coupling")),
    ("rigid-duct-parts", "vent.rigid-run", ("smooth-metal segments", "seams", "serviceable joints")),
    ("vent-deposit-state", "vent.deposit", ("retained mass", "moisture", "geometry", "permeability", "attachment strength", "location")),
    ("exhaust-hood-parts", "vent.exterior-hood", ("housing", "collar", "gravity backdraft flap")),
    ("idler-kit-parts", "spare.idler-kit", ("pulley", "bearing", "arm", "spring")),
    ("window-ac-paths", "garage.window-ac", ("indoor air path", "outdoor air path", "compressor", "evaporator", "condenser", "fans")),
))


AUTHORING_RULES = {
    "room": {
        "air_is_finite_region": True,
        "exchanges": ["heat", "moisture", "air"],
        "leakage_sources": ["door seals", "window seals", "deliberate openings"],
        "indoor_dryer_discharge_is_not_outdoor_exhaust": True,
    },
    "electrical": {
        "bench_receptacles_share_upstream_capacity": True,
        "washer_and_dryer_use_direct_wall_connections": True,
        "dryer_service": "four-contact split-phase",
        "workshop_extension_cord_is_for_tools": True,
        "conductors_neutral_and_protective_earth_remain_distinct": True,
    },
    "door": {
        "missing_remote_does_not_cause_jam": True,
        "wall_button_is_present": True,
        "release_disengages_trolley_but_does_not_move_door": True,
        "manual_lift_requires_release": True,
        "counterbalance_and_tracks_start_healthy": True,
    },
    "dryer": {
        "one_fixed_fictional_model": True,
        "generated_fault_families": ["drive", "control", "thermal"],
        "additional_fault_count": [1, 3],
        "building_vent_obstruction_is_mandatory_but_not_a_dryer_fault": True,
        "choose_once_and_persist": True,
        "symptom_masking_allowed": True,
        "invented_follow_on_faults_forbidden": True,
        "available_repair_path_required": True,
        "live_electrical_diagnosis_not_required": True,
        "safety_bypass_not_required": True,
        "fixed_spare_carton_must_not_disclose_fault_selection": True,
        "service_fasteners_match_starting_tools": True,
    },
    "exhaust": {
        "owner": "garage building",
        "path": ["connected machine", "transition", "wall collar", "duct and elbows",
                 "lint/debris restriction", "outlet flap", "outdoors"],
        "law": "delta_p = R_v Q + R_i Q abs(Q)",
        "no_fixed_flow_ceiling": True,
        "deposit_is_finite_matter": True,
        "cleaning_requires_receiver": True,
        "replacement_machine_keeps_obstruction": True,
        "physical_bypass_avoids_original_path_without_removing_it": True,
        "new_games_start_clogged_but_cleaned_games_stay_clean": True,
        "large_pressure_can_move_debris_open_flap_or_fail_weak_joint": True,
    },
    "laundry": {
        "water_content_and_temperature_are_state": True,
        "drying_transfers_water_to_air": True,
        "washer_can_make_repeatable_wet_test_load": True,
        "lint_screen_and_building_duct_are_distinct": True,
    },
    "diagnosis": {
        "meter_location": "zipped pouch behind folded rags in unlocked right drawer",
        "meter_count": 1,
        "meter_readings_come_from_probe_connections": True,
        "finding_meter_does_not_reveal_faults": True,
        "vent_cleaning_equipment_is_visible": True,
        "notebook_is_observational_not_omniscient": True,
        "notebook_notes": ["Towels taking longer.", "Dryer quit.", "Remote missing."],
    },
    "engine_stand": {
        "stand_is_not_running_installation": True,
        "invisible_fuel_cooling_exhaust_and_torque_absorber_forbidden": True,
    },
}


def catalogue_document() -> dict:
    """Return the deterministic JSON-ready authoring manifest."""
    return {
        "schema": "engine-toy-starter-garage-v1",
        "kind": "authoring-specification",
        "inventory_entry_count": len(INVENTORY),
        "constituent_record_count": len(CONSTITUENTS),
        "quantity_semantics": "each quantity spawns independent objects; group labels do not fuse them",
        "inventory": [asdict(row) for row in INVENTORY],
        "constituents": [asdict(row) for row in CONSTITUENTS],
        "authoring_rules": AUTHORING_RULES,
    }


if len(INVENTORY) != 162:
    raise RuntimeError(f"starter garage catalogue has {len(INVENTORY)} entries, expected 162")
if len(CONSTITUENTS) != 40:
    raise RuntimeError(f"starter garage catalogue has {len(CONSTITUENTS)} constituents, expected 40")
if len({row.identity for row in INVENTORY}) != len(INVENTORY):
    raise RuntimeError("starter garage inventory identities must be unique")


__all__ = ["InventoryEntry", "ConstituentRecord", "INVENTORY", "CONSTITUENTS",
           "AUTHORING_RULES", "catalogue_document"]
