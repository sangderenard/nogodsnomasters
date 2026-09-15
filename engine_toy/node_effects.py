"""What each node does when it is hurt: the consequence API.

Four conditions, assessed from the real records every tick
(`assess(sim)`): for every graph node,

  a) FOULED      -- its contents are not its own fluid any more: the
                    circuit's FluidMix (hole_emitters) holds air, oil,
                    coolant, fuel, water, particulate in some fractions
  b) UNSEALED    -- it has a hole in its pressure boundary (a puncture
                    or spall recorded in damage_state, or it is gone)
  c) STRUCTURE LOST -- it can no longer carry load or hold shape: it
                    burst (absent), or the impact energy it has
                    absorbed exceeds its own budget (mass x the
                    material's toughness per kilogram, disclosed below)
  d) NEIGHBOUR MISSING -- a node it depends on (the far end of a drive
                    edge, a fluid supply line, a rigid mounting) is
                    absent or structurally gone

and a NodeBehaviour per node class decides what it cares about and
what follows -- as multiplicative levers on quantities the engine sim
already integrates (`Effects`): per-cylinder charge, intake flow and
mixture, exhaust restriction/openness, coolant heat exchange, oil
pump flow and pressure, fuel supply, alternator output, boost, and a
dead-engine verdict. A node that does not care (a bracket, a pulley,
a mount) returns nothing. Each rule is written as the physical reason,
not a hit-points table:

  oil pump   fouled with air -> cavitation, delivery falls with the
             gas fraction; unsealed -> its own leak (the emitters) and
             a pressure drop; structure gone or its suction/drive
             neighbour gone -> no delivery
  water pump / radiator  fouled with oil -> the exchanger's film
             insulates; structure gone -> no exchange, no flow
  plenum / throttle / filter  fouled with oil -> smoke and a richer
             charge; fouled with air (a vacuum leak's ingest) -> lean;
             structure gone -> a collapsed inlet; throttle gone ->
             unmetered air, lean
  exhaust    unsealed -> partly open (less restriction, brighter,
             gas into the bay); structure gone or the downstream
             neighbour gone -> open pipe
  fuel rail / filter / pump / tank  fouled with air or water ->
             misfires and starvation; structure gone or supply gone
             -> no fuel
  cylinder / head  unsealed -> compression loss on that bore;
             structure gone or the head gone -> a dead cylinder
  block / crankcase / crank  structure gone -> the engine is dead
  turbo      structure gone, or its oil feed gone -> no boost
  alternator / battery  structure gone -> no charge
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

import numpy as np

# accumulated impact energy a part can absorb before it no longer
# holds shape/load, per kilogram, by material class (a ductile
# casting takes a great deal of denting; a plastic box does not)
STRUCTURE_BUDGET_J_PER_KG = {"cast-iron": 1500.0, "steel": 4000.0, "aluminium": 2500.0, "plastic": 700.0,
                             "rubber": 3000.0, "brass": 2000.0, "copper": 2500.0, "default": 1800.0}
STRUCTURE_BUDGET_FLOOR_J = 600.0

DEPENDENCY_EDGE_KINDS = ("geared-timing-drive", "accessory-drive-belt", "rigid-keyed-hub", "rigid-bolted-joint")
DEPENDENCY_EDGE_WORDS = ("line", "flow-path", "drive", "keyed-hub", "hose", "pipe")


def is_dependency_edge(kind: str | None) -> bool:
    if not kind:
        return False
    return kind in DEPENDENCY_EDGE_KINDS or any(w in kind for w in DEPENDENCY_EDGE_WORDS)


@dataclass
class NodeCondition:
    identity: str
    fouling: dict = field(default_factory=dict)       # fluid -> mass fraction of the contents that is NOT its own
    unsealed: bool = False
    open_area_m2: float = 0.0
    structure_lost: bool = False
    absent: bool = False
    absorbed_j: float = 0.0
    budget_j: float = 0.0
    missing_neighbors: list = field(default_factory=list)   # (edge constraint, neighbour identity)
    contamination: dict = field(default_factory=dict)       # mechanism -> severity, from the air system's gunk

    @property
    def any(self) -> bool:
        return (bool(self.fouling) or self.unsealed or self.structure_lost
                or bool(self.missing_neighbors) or bool(self.contamination))


@dataclass
class Effects:
    """Multiplicative levers (1.0 = untouched) plus verdicts."""
    cylinder_factor: dict = field(default_factory=dict)   # cylinder number -> charge factor
    intake_flow_factor: float = 1.0
    mixture_factor: float = 1.0        # <1 lean (unmetered air), >1 rich (oil in the charge)
    smoke_factor: float = 0.0          # oil burning in the charge, 0..1
    exhaust_restriction_factor: float = 1.0
    exhaust_open_frac: float = 0.0     # 0 sealed .. 1 open pipe in the bay
    coolant_exchange_factor: float = 1.0
    coolant_flow_factor: float = 1.0
    oil_flow_factor: float = 1.0
    oil_pressure_factor: float = 1.0
    fuel_supply_factor: float = 1.0
    alternator_factor: float = 1.0
    boost_factor: float = 1.0
    # the compressed-air system's own condition (air_treatment.gunk_effects)
    air_flow_factor: float = 1.0        # fouled ports and stuck spools pass less
    air_seal_factor: float = 1.0        # worn seats stop holding pressure
    air_tool_factor: float = 1.0
    # real service verdicts, not multipliers
    engine_dead: str | None = None
    spark_lost: bool = False           # no ignition system left: it cannot fire at all
    brakes_lost: bool = False          # the air-brake reserve is gone
    startable: bool = True
    mounts_lost: int = 0               # how many engine mounts are gone
    notes: list = field(default_factory=list)

    def merge(self, other: "Effects") -> None:
        for cyl, f in other.cylinder_factor.items():
            self.cylinder_factor[cyl] = self.cylinder_factor.get(cyl, 1.0) * f
        for name in ("intake_flow_factor", "mixture_factor", "exhaust_restriction_factor", "coolant_exchange_factor",
                     "coolant_flow_factor", "oil_flow_factor", "oil_pressure_factor", "fuel_supply_factor",
                     "alternator_factor", "boost_factor", "air_flow_factor", "air_seal_factor", "air_tool_factor"):
            setattr(self, name, getattr(self, name) * getattr(other, name))
        self.smoke_factor = min(1.0, self.smoke_factor + other.smoke_factor)
        self.exhaust_open_frac = min(1.0, self.exhaust_open_frac + other.exhaust_open_frac)
        if other.engine_dead and not self.engine_dead:
            self.engine_dead = other.engine_dead
        self.spark_lost = self.spark_lost or other.spark_lost
        self.brakes_lost = self.brakes_lost or other.brakes_lost
        self.startable = self.startable and other.startable
        self.mounts_lost += other.mounts_lost
        for n in other.notes:            # one note per distinct consequence, however many nodes report it
            if n not in self.notes:
                self.notes.append(n)


class NodeBehaviour:
    """Base: cares about nothing. Subclasses override `evaluate`; the
    registry below maps identity patterns to behaviours."""
    cares: tuple = ()

    def evaluate(self, node: dict, cond: NodeCondition) -> Effects:
        return Effects()


def _foul(cond: NodeCondition, *fluids: str) -> float:
    return float(sum(cond.fouling.get(f, 0.0) for f in fluids))


class OilPumpBehaviour(NodeBehaviour):
    cares = ("fouled", "unsealed", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        air = _foul(cond, "air")
        if air > 0.02:
            e.oil_flow_factor *= max(0.0, 1.0 - 5.0 * air)           # cavitation: a gerotor pumps air badly
            e.notes.append(f"oil pump cavitating on {air * 100:.0f}% air")
        if cond.unsealed and not cond.structure_lost:
            e.oil_pressure_factor *= 0.6
            e.notes.append("oil pump housing holed: pressure down")
        if cond.structure_lost or any(k in ("geared-timing-drive", "oil-line", "rigid-keyed-hub") for k, _ in cond.missing_neighbors):
            e.oil_flow_factor = 0.0; e.oil_pressure_factor = 0.0
            e.notes.append("oil pump: no delivery" + (" (structure gone)" if cond.structure_lost else " (drive/suction gone)"))
        return e


class CoolantBehaviour(NodeBehaviour):
    cares = ("fouled", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        oil = _foul(cond, "engine-oil", "fuel")
        if oil > 0.05:
            e.coolant_exchange_factor *= max(0.3, 1.0 - 2.0 * oil)      # an oil film on the exchanger insulates
            e.notes.append(f"coolant fouled {oil * 100:.0f}% oil: exchanger losing effect")
        air = _foul(cond, "air")
        if air > 0.1:
            e.coolant_flow_factor *= max(0.0, 1.0 - 2.0 * air)          # an air-locked pump
            e.notes.append(f"coolant {air * 100:.0f}% air: pump air-locking")
        if cond.structure_lost:
            e.coolant_exchange_factor = 0.0 if "radiator" in cond.identity else e.coolant_exchange_factor
            e.coolant_flow_factor = 0.0
            e.notes.append(f"{cond.identity.split('.')[-1]} gone: no coolant circulation")
        if any(k in ("accessory-drive-belt", "rigid-keyed-hub") for k, _ in cond.missing_neighbors):
            e.coolant_flow_factor = 0.0
            e.notes.append("water pump: drive gone")
        return e


class IntakeBehaviour(NodeBehaviour):
    cares = ("fouled", "unsealed", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        oil = _foul(cond, "engine-oil")
        if oil > 0.002:
            e.smoke_factor += min(1.0, oil * 40.0)
            e.mixture_factor *= 1.0 + min(0.3, oil * 10.0)
            e.notes.append(f"intake breathing {oil * 100:.1f}% oil: smoke, rich")
        air = _foul(cond, "air")
        if air > 0.05:
            e.mixture_factor *= max(0.5, 1.0 - 0.5 * air)               # unmetered air leans the charge
            e.notes.append(f"vacuum leak: {air * 100:.0f}% unmetered air, lean")
        if cond.structure_lost:
            e.intake_flow_factor *= 0.3
            e.notes.append(f"{cond.identity.split('.')[-1]} collapsed: inlet choked")
        if any(n.endswith("throttle_body") or n.endswith("air_filter") for _, n in cond.missing_neighbors):
            e.mixture_factor *= 0.75
            e.notes.append("throttle/filter gone: unmetered inlet")
        return e


class ExhaustBehaviour(NodeBehaviour):
    cares = ("unsealed", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        if cond.unsealed and not cond.structure_lost:
            frac = min(0.6, cond.open_area_m2 / 1.2e-3)                 # holed vs a 40 mm pipe's own area
            e.exhaust_open_frac += frac
            e.exhaust_restriction_factor *= 1.0 - 0.7 * frac
            e.notes.append(f"exhaust holed: {frac * 100:.0f}% open into the bay")
        if cond.structure_lost or any("exhaust" in k for k, _ in cond.missing_neighbors):
            e.exhaust_open_frac = 1.0
            e.exhaust_restriction_factor = 0.2
            e.notes.append("exhaust: open pipe")
        return e


class FuelBehaviour(NodeBehaviour):
    cares = ("fouled", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        gas = _foul(cond, "air")
        water = _foul(cond, "water", "coolant")
        if gas > 0.01:
            e.fuel_supply_factor *= max(0.0, 1.0 - 4.0 * gas)          # vapour/air in the line: starvation, misfire
            e.notes.append(f"fuel aerated {gas * 100:.0f}%: starving")
        if water > 0.01:
            e.fuel_supply_factor *= max(0.0, 1.0 - 3.0 * water)
            e.notes.append(f"water in fuel {water * 100:.0f}%")
        if cond.structure_lost or any(("fuel" in k or "liquid" in k) for k, _ in cond.missing_neighbors):
            e.fuel_supply_factor = 0.0
            e.notes.append(f"{cond.identity.split('.')[-1]}: no fuel supply")
        return e


class CylinderBehaviour(NodeBehaviour):
    cares = ("unsealed", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        m = re.search(r"cylinder_(\d+)", cond.identity)
        cyl = int(m.group(1)) if m else int(node.get("cylinder", 0) or 0)
        if not cyl:
            return e
        if cond.unsealed and not cond.structure_lost:
            e.cylinder_factor[cyl] = 0.4                               # a holed bore/head: compression leaks past
            e.notes.append(f"cylinder {cyl} holed: compression loss")
        if cond.structure_lost or any("head" in n for _, n in cond.missing_neighbors):
            e.cylinder_factor[cyl] = 0.0
            e.notes.append(f"cylinder {cyl}: dead")
        return e


class BlockBehaviour(NodeBehaviour):
    cares = ("structure",)

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost:
            e.engine_dead = f"{cond.identity.split('.')[-1]} structure gone"
        return e


class TurboBehaviour(NodeBehaviour):
    cares = ("structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost:
            e.boost_factor = 0.0
            e.notes.append("turbo gone: no boost")
        elif any(k == "oil-line" for k, _ in cond.missing_neighbors):
            e.boost_factor *= 0.0
            e.notes.append("turbo oil feed gone: bearing failed, no boost")
        return e


class AlternatorBehaviour(NodeBehaviour):
    cares = ("structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost or any(k in ("accessory-drive-belt", "rigid-keyed-hub") for k, _ in cond.missing_neighbors):
            e.alternator_factor = 0.0
            e.notes.append("alternator: no charge")
        return e


class PneumaticBehaviour(NodeBehaviour):
    """A compressed-air system: the reserve and the brake reservoirs are
    what stop the vehicle and what starts a big diesel on air. Holed,
    they bleed down through the hole (the emitters already do that);
    gone, the service they feed is gone with them."""
    cares = ("unsealed", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        name = cond.identity
        if cond.structure_lost:
            if "brake" in name:
                e.brakes_lost = True
                e.notes.append(f"{name.split('.')[-1]} gone: air brakes lost")
            elif "reserve" in name or "compressor" in name:
                e.startable = False
                e.notes.append(f"{name.split('.')[-1]} gone: no stored air (idle assist and air start lost)")
        elif cond.unsealed:
            e.notes.append(f"{name.split('.')[-1]} holed: bleeding down")
        if any("compressed-air" in k or "pneumatic" in k for k, _ in cond.missing_neighbors):
            e.notes.append(f"{name.split('.')[-1]}: air supply gone")
        return e


class IgnitionBehaviour(NodeBehaviour):
    """Magneto, distributor, coil, plug leads: no spark, no fire. A
    compression-ignition engine does not care -- it has none of this."""
    cares = ("structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost or any(k in ("geared-timing-drive", "rigid-keyed-hub", "accessory-drive-belt")
                                      for k, _ in cond.missing_neighbors):
            e.spark_lost = True
            e.notes.append(f"{cond.identity.split('.')[-1]} gone: no ignition")
        return e


class TimingBehaviour(NodeBehaviour):
    """Camshaft, sprockets, the timing drive, an injection pump: without
    valve or injection timing the engine cannot run at all -- it is
    dead, not merely weak."""
    cares = ("structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost:
            e.engine_dead = f"{cond.identity.split('.')[-1]} gone: no valve/injection timing"
        elif any(k in ("geared-timing-drive", "rigid-keyed-hub") for k, _ in cond.missing_neighbors):
            e.engine_dead = f"{cond.identity.split('.')[-1]}: timing drive broken"
        return e


class MountBehaviour(NodeBehaviour):
    """An engine mount carries the block's weight and its torque
    reaction. One gone is a bad vibration; all of them gone and the
    engine is loose in the bay."""
    cares = ("structure",)

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost:
            e.mounts_lost = 1
            e.notes.append(f"{cond.identity.split('.')[-1]} broken: the engine is moving on its mounts")
        return e


class StarterBehaviour(NodeBehaviour):
    cares = ("structure",)

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost:
            e.startable = False
            e.notes.append("starter gone: it cannot be restarted")
        return e


class FanBehaviour(NodeBehaviour):
    """A cooling fan is airflow, not coolant flow: without it the
    exchanger keeps its coolant but loses most of its rejection at low
    road speed."""
    cares = ("structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost or any(k in ("accessory-drive-belt", "rigid-keyed-hub") for k, _ in cond.missing_neighbors):
            e.coolant_exchange_factor *= 0.35
            e.notes.append("cooling fan gone: little rejection at rest")
        return e


class DrivelineBehaviour(NodeBehaviour):
    """Clutch, transmission, transfer case, prop reduction: the load
    path out of the engine. Gone, there is nothing to drive."""
    cares = ("structure",)

    def evaluate(self, node, cond):
        e = Effects()
        if cond.structure_lost:
            e.notes.append(f"{cond.identity.split('.')[-1]} gone: no drive to the load")
        return e


class PlantBehaviour(NodeBehaviour):
    """The treatment train and its manifolds. A filter or cooler that is
    gone stops treating, which is not a failure you notice at once --
    it is a failure you notice later, in everything downstream."""
    cares = ("fouled", "unsealed", "structure", "neighbour")

    def evaluate(self, node, cond):
        e = Effects()
        name = cond.identity.split(".")[-1]
        if cond.structure_lost:
            e.notes.append(f"{name} gone: air leaving it untreated")
        elif cond.unsealed:
            e.notes.append(f"{name} holed: bleeding the treated air")
        if cond.fouling:
            e.notes.append(f"{name} fouled: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(cond.fouling.items())[:2]))
        return e


# identity pattern -> behaviour; first match wins; anything else does not care
BEHAVIOURS: list[tuple[str, NodeBehaviour]] = [
    (r"oil_pump|scavenge_pump|oil_filter|oil_pan|oil_reserve_tank|oil_cooler", OilPumpBehaviour()),
    (r"water_pump|seawater_pump|radiator|coolant|thermostat|heat_exchanger", CoolantBehaviour()),
    (r"intake_plenum|throttle_body|air_filter|intake_runner|intercooler|blower_case|supercharger_rotor", IntakeBehaviour()),
    (r"exhaust|muffler|catalytic|tailpipe|collector", ExhaustBehaviour()),
    (r"fuel_rail|fuel_filter|fuel_pump|fuel_tank|injector|carburet", FuelBehaviour()),
    (r"cylinder_\d+|head", CylinderBehaviour()),
    (r"powertrain\.engine$|crankcase|crank_shaft|bedplate|main_pedestal", BlockBehaviour()),
    (r"turbocharger", TurboBehaviour()),
    (r"alternator|battery", AlternatorBehaviour()),
    (r"magneto|distributor|ignition_coil|coil_pack|spark_plug|plug_lead", IgnitionBehaviour()),
    (r"camshaft|timing_drive|cam_sprocket|crank_sprocket|timing_(chain|belt|gear)|injection_pump", TimingBehaviour()),
    (r"pneumatic|brake_reservoir|air_tank|receiver", PneumaticBehaviour()),
    (r"mount_|engine_mount|isolator", MountBehaviour()),
    (r"starter_motor|starter|recoil", StarterBehaviour()),
    (r"cooling_fan|fan_", FanBehaviour()),
    (r"clutch|transmission|transfer_case|prop_reduction|final_drive|transaxle|differential", DrivelineBehaviour()),
    (r"^plant\.", PlantBehaviour()),
]


def behaviour_for(identity: str) -> NodeBehaviour | None:
    for pat, b in BEHAVIOURS:
        if re.search(pat, identity):
            return b
    return None


def _material_class(material: str | None) -> str:
    m = (material or "").lower()
    for k in ("cast-iron", "steel", "aluminium", "plastic", "rubber", "brass", "copper"):
        if k in m:
            return k
    return "default"


def assess(sim) -> tuple[dict, Effects]:
    """Every node's condition from the live records, and the merged
    Effects of the behaviours that care."""
    graph = sim._drivetrain.graph
    absent = set(getattr(sim.state, "absent_parts", ()))
    damage = sim.state.part_damage
    field_ = sim.hole_emitters
    fouling = field_.fouling() if field_.mix else {}
    from hole_emitters import circuit_identity
    circuit_of_node: dict[str, str] = {}
    nominal_of_circuit: dict[str, str] = {}
    for c in sim._drivetrain.fluid_circuits:
        cid = circuit_identity(c)
        for nid in c.nodes:
            circuit_of_node.setdefault(nid, cid)
        mix = field_.mix.get(cid)
        if mix is not None and mix.kg:
            nominal_of_circuit[cid] = max(mix.kg.items(), key=lambda kv: kv[1])[0] if cid not in fouling else \
                next(iter(k for k in mix.kg if k in (cid, "engine-oil", "coolant", "fuel", "water")), cid)
    nodes = {n["identity"]: n for n in graph["nodes"]}
    struct_gone: set = set(absent)
    # fold every damage record onto the node that owns the struck piece
    from damage_state import owning_part
    owned: dict[str, list] = {}
    for key, st in damage.items():
        if not st.punctures and not st.impacts:
            continue
        owner = key if key in nodes else (owning_part(key) or key)
        owned.setdefault(owner, []).append(st)

    # THE WORK SET. Damage is the causer's bill, not a standing tax on
    # every part in the machine: a part is only worth assessing if
    # something has actually happened to it. Four things can, and each
    # one names its own parts --
    #   a hole or an impact          -> the part the record is filed on
    #   a burst                      -> the absent part
    #   a leak fouling a circuit     -> the parts on that circuit
    #   a neglected air system       -> the parts its gunk reaches
    # everything else produces a clean NodeCondition that the merge at
    # the bottom then skips (`if not cond.any`), which is what made this
    # scan every node of the graph, forever, to conclude that an
    # undamaged engine is undamaged.
    gunk_per_part: dict = {}
    plant = getattr(sim, "plant", None)
    if plant is not None:
        import air_treatment as _at
        gunk_per_part = _at.gunk_damage_by_part(plant.treatment.gunk, list(nodes),
                                                plant.spec.wet_tank_capacity_l if plant.spec else 20.0)
    work: set = set(owned) | set(absent) | set(gunk_per_part)
    if fouling:
        work |= {nid for nid, cid in circuit_of_node.items() if cid in fouling}
    work &= set(nodes) | set(owned)

    conds: dict[str, NodeCondition] = {}
    for ident in work:
        node = nodes.get(ident, {})
        cond = NodeCondition(ident)
        cond.absent = ident in absent
        for st in owned.get(ident, ()):
            holes = [p for p in st.punctures if p.damage_mode in ("puncture", "spalling-puncture", "crater")]
            if holes:
                cond.unsealed = True
                cond.open_area_m2 += float(sum(math.pi * p.radius_m ** 2 * p.boundary_holes for p in holes))
            cond.absorbed_j += float(sum(i.energy_spent_j for i in st.impacts))
        mass = float(node.get("mass_kg", 0.0) or 0.0)
        if mass <= 0.0:
            # a casting piece with no node of its own: the engine's share
            eng = nodes.get("powertrain.engine", {})
            mass = float(eng.get("mass_kg", 100.0) or 100.0) * (0.06 if "cylinder" in ident else 0.15)
        cond.budget_j = max(STRUCTURE_BUDGET_FLOOR_J, STRUCTURE_BUDGET_J_PER_KG[_material_class(node.get("material"))] * mass)
        cond.structure_lost = cond.absent or cond.absorbed_j >= cond.budget_j
        if cond.structure_lost:
            struct_gone.add(ident)
        cid = circuit_of_node.get(ident)
        if cid in fouling:
            nominal = nominal_of_circuit.get(cid, cid)
            cond.fouling = {k: v for k, v in fouling[cid].items() if k != nominal and k != cid}
        conds[ident] = cond
    # a part next to something that is GONE has a real condition of its
    # own even though nothing happened to it directly -- so those
    # neighbours, and only those, join the work set now that struct_gone
    # is known. With nothing gone, this whole pass costs one bool.
    if struct_gone:
        for e in graph["edges"]:
            kind = e.get("constraint")
            if not is_dependency_edge(kind):
                continue
            for me, other in ((e["a"], e["b"]), (e["b"], e["a"])):
                if other not in struct_gone or me in struct_gone or me not in nodes:
                    continue
                cond = conds.get(me)
                if cond is None:
                    cond = NodeCondition(me)
                    mass = float(nodes[me].get("mass_kg", 0.0) or 0.0)
                    if mass <= 0.0:
                        eng = nodes.get("powertrain.engine", {})
                        mass = float(eng.get("mass_kg", 100.0) or 100.0) * (0.06 if "cylinder" in me else 0.15)
                    cond.budget_j = max(STRUCTURE_BUDGET_FLOOR_J,
                                        STRUCTURE_BUDGET_J_PER_KG[_material_class(nodes[me].get("material"))] * mass)
                    conds[me] = cond
                cond.missing_neighbors.append((kind, other))
    merged = Effects()
    # the air system's accumulated contamination is a real consequence
    # of the same kind: it is reported here, not in a separate place
    if plant is not None:
        g = plant.gunk_effects()
        merged.notes.extend(g["notes"])
        # ...and put that damage on the actual parts it reaches, so a
        # neglected air system shows up as named hardware in trouble
        # rather than only as a global multiplier (gunk_per_part was
        # computed above, where it helped choose the work set)
        for ident, sev in gunk_per_part.items():
            cond = conds.get(ident)
            if cond is None:
                continue
            cond.contamination = sev
            worst = max(sev.values())
            if "corrosion" in sev and sev["corrosion"] > 0.5:
                # a corroded vessel really does lose wall: it is weaker
                cond.budget_j *= max(0.2, 1.0 - 0.7 * sev["corrosion"])
                if cond.absorbed_j >= cond.budget_j:
                    cond.structure_lost = True
            if worst > 0.6:
                merged.notes.append(
                    f"{ident.split('.')[-1]}: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in sev.items()))
        merged.air_flow_factor *= g["flow_factor"]
        merged.air_seal_factor *= g["seal_factor"]
        merged.air_tool_factor *= g["tool_efficiency_factor"]
    for ident, cond in conds.items():
        if not cond.any:
            continue
        b = behaviour_for(ident)
        if b is None:
            continue
        merged.merge(b.evaluate(nodes.get(ident, {}), cond))
    return conds, merged


def _severity(c: NodeCondition) -> tuple:
    """Worst first: gone, then structurally lost, then holed, then
    fouled, then merely missing a neighbour."""
    return (not c.absent, not c.structure_lost, not c.unsealed, -len(c.fouling), -len(c.missing_neighbors))


def condition_lines(conds: dict, effects: Effects, limit: int = 8) -> list[str]:
    lines = []
    for ident, c in sorted(conds.items(), key=lambda kv: _severity(kv[1])):
        if not c.any:
            continue
        bits = []
        if c.absent:
            bits.append("ABSENT")
        elif c.structure_lost:
            bits.append(f"structure lost ({c.absorbed_j:.0f}/{c.budget_j:.0f} J)")
        elif c.absorbed_j > 0.0:
            bits.append(f"{c.absorbed_j:.0f}/{c.budget_j:.0f} J")
        if c.unsealed:
            bits.append(f"unsealed {c.open_area_m2 * 1e6:.0f} mm2")
        if c.fouling:
            bits.append("fouled " + ",".join(f"{k} {v * 100:.1f}%" for k, v in list(c.fouling.items())[:3]))
        if c.missing_neighbors:
            bits.append("missing " + ",".join(n.split(".")[-1] for _, n in c.missing_neighbors[:3]))
        if c.contamination:
            bits.append(", ".join(f"{k} {v * 100:.0f}%" for k, v in c.contamination.items()))
        b = behaviour_for(ident)
        lines.append(f"  NODE {ident.split('.')[-1]}: {'; '.join(bits)}"
                     + ("" if b else " (nothing depends on it)"))
    lines = lines[:limit]
    for n in effects.notes[:limit]:
        lines.append(f"  EFFECT {n}")
    if effects.engine_dead:
        lines.append(f"  ENGINE DEAD: {effects.engine_dead}")
    if effects.spark_lost:
        lines.append("  NO IGNITION: nothing left to make a spark")
    if effects.brakes_lost:
        lines.append("  NO AIR BRAKES")
    if not effects.startable:
        lines.append("  CANNOT BE RESTARTED")
    if effects.mounts_lost:
        lines.append(f"  MOUNTS BROKEN: {effects.mounts_lost}")
    if effects.air_flow_factor < 0.95 or effects.air_seal_factor < 0.95:
        lines.append(f"  AIR SYSTEM: flow x{effects.air_flow_factor:.2f}, seals x{effects.air_seal_factor:.2f}, "
                     f"tools x{effects.air_tool_factor:.2f}")
    return lines
