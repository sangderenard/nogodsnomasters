"""A climate-controlled separator: a centrifuge bowl inside a sealed,
gas-held cabinet.

WHY A CABINET AND NOT A BARE BOWL. centrifuges.py is explicit that a
cut depends on the oil's viscosity, and viscosity is exponential in
temperature -- "the same machine on cold oil cuts far coarser than on
hot". A bowl standing on an open frame (centrifuge_production.py) is
processing whatever the room happens to be that day. Put the same bowl
inside cabinet.py's sealed box, with its own coil and heater holding
the ENCLOSURE rather than the product, and the cut stops being a
function of the weather.

THE BOWL IS THE BOX'S OWN DRUM, NOT A NEW PART. cabinet.py already
declares a horizontal rotor behind the door -- a Drum on a
suspension, driven by a DrumDrive -- because that is what a washer's
tub and a dryer's drum both are: a rotor, a mount, a motor. A
horizontal decanter centrifuge is a real, common industrial form of
exactly that same machine, so this module reuses the substrate rather
than authoring a second one, and derives its separation physics
(sigma, cut size, g-force) from the SAME declared radius, length and
speed through climate_separator.centrifuge_for -- one set of numbers,
read twice, the way RefrigerantLoop.from_compressor_node already
insists the compressor's own rating be read off the graph rather than
guessed beside it.

THE MOUNT IS THE FIRST BALANCING TECHNIQUE, AND THE ONLY ONE BUILT
HERE. cabinet.py's suspended mount exists so a washer's tub can find
its own rotating axis rather than being forced to spin about a fixed
one -- which is exactly the passive balancing a soft-mounted
centrifugal extractor relies on, and it is what this machine ships
with: a declared `balance_grade_mm_s` on the bowl, held on springs
and dampers. Two further, real technologies are NOT built here and
are left as declared numbers this module does not yet touch:

    correction-mass ring   a ring of movable masses on the bowl an
                            operator (or a service task) sets against
                            a measured unbalance -- a real extra part,
                            with its own mass and its own claimed ring
                            of travel, the way centrifuge_production.py's
                            `bowl_unbalance_kg_m` is already a number a
                            build can carry without yet having a part
                            that sets it
    self-balancing ring    an active ring of fluid-filled chambers
                            that migrate at speed to cancel unbalance
                            on their own, the way some washer tubs do
                            -- a genuinely different, harder physics
                            (mass redistributing itself as a function
                            of the very vibration it is correcting)
                            and a separate model from either of the above

REFRIGERANT STAYS OUTSIDE; GAS CIRCULATES INSIDE. The compressor and
its condenser are heat-REJECTING machinery -- they want ambient air
and they vibrate -- so they sit in the box's own bottom compartment,
outside the sealed atmosphere, exactly where cabinet.py already puts
a washer's pump or a dryer's blower. Only the evaporator coil, the
electric trim heater and the recirculation blower are actually inside
the chamber, because they are the three things that have to touch the
gas the bowl spins in. Two refrigerant lines cross the wall to reach
them -- the same "one hole, one closure" crossing autoclave.py's own
steam line makes, not a new idea.

GAS INLET AND PURGE ARE PORTS, NOT A CYCLE -- see climate_separator.py
for why this deliberately does NOT reuse autoclave.PurgeCycle's
pump-and-load model: there is no load porosity here to pull a vacuum
out of, only a valve that is open or shut.
"""
from __future__ import annotations

import math

import cabinet as cb
from cabinet import Compartment, Door, Drum, DrumDrive, PlantItem, Window, CabinetSpec
from prism_bodies import Prism, emit_prism
from operating_states import Cycle, OperatingState, run_up, coast

INSIDE_W_M = 0.75
INSIDE_H_M = 0.95
INSIDE_D_M = 0.80
STIFFENER_PITCH_M = 0.30
#: A slight positive hold, the way a real purge-and-purge gas
#: enclosure is run so infiltration leaks OUT through a seal
#: imperfection rather than shop air leaking in -- and, because this
#: box also has to survive a bowl carrying real stored energy at
#: speed (centrifuges.stored_energy_j), it is built of welded plate
#: sized for pressure rather than screwed sheet even though the
#: pressure itself is modest. The wall's other job is containment.
WORKING_PRESSURE_PA = 250.0
BOTTOM_BAY_HEIGHT_M = 0.30

#: The bowl, as a cabinet.Drum -- suspended, so it finds its own axis.
BOWL_RADIUS_M = 0.16
BOWL_LENGTH_M = 0.42
BOWL_MASS_KG = 38.0
BOWL_RATED_RPM = 5500.0
#: A purifier-grade bowl, tighter than the 6.3 mm/s a washer tub is
#: balanced to (cabinet.DrumDrive's own default) -- see the module
#: docstring for the two real technologies that would tighten this
#: further without being built here.
BOWL_BALANCE_GRADE_MM_S = 4.0

#: The refrigerant side: a swash-plate compressor, the same
#: construction compressors.py names as "the automotive A/C
#: compressor" -- which is exactly the loop refrigeration.py models.
COMPRESSOR_RATED_W = 1800.0
COMPRESSOR_REFERENCE_OMEGA_RAD_S = 150.0
COMPRESSOR_MASS_KG = 14.0
HEATER_KW = 1.5
HEATER_MASS_KG = 6.0
BLOWER_FLOW_M3_S = 0.12
BLOWER_MASS_KG = 4.5

#: The chamber's wrap. cryogenics.INSULATION's "aerogel-blanket" needs
#: no vacuum and "works cold or hot" -- unlike every dewar in that
#: module, THIS chamber is above ambient as often as below it, so a
#: vacuum-dependent medium (mli, vacuum-perlite) would be the wrong
#: choice even though it conducts less.
INSULATION_MEDIA = "aerogel-blanket"
INSULATION_THICKNESS_M = 0.05


def spec_for(identity: str = "plant.separator") -> CabinetSpec:
    """The sealed box: the bowl behind the door, gas ports in the
    ceiling, refrigerant and heater power crossing the back wall."""
    drum = Drum(radius_m=BOWL_RADIUS_M, length_m=BOWL_LENGTH_M, mass_kg=BOWL_MASS_KG,
               rated_rpm=BOWL_RATED_RPM, mount="suspended",
               suspension_n_per_m=9000.0, suspension_damping=0.25,
               balance_grade_mm_s=BOWL_BALANCE_GRADE_MM_S,
               drive=DrumDrive(kind="belt", motor_kw=5.5, rated_rpm=2900.0,
                               mass_kg=41.0, rotor_mass_frac=0.35,
                               rotor_radius_m=0.045, balance_grade_mm_s=6.3,
                               belt_stiffness_n_per_m=22_000.0, at=(0.55, 0.25)))
    ports = {
        # the chamber's own atmosphere: a fill line and a valve to let
        # it down, and nothing in between -- see the module docstring
        "ceiling": {"gas_inlet": ("+y", (-0.3, 0.0)), "purge": ("+y", (0.3, 0.0))},
        # the heater's power, crossing the one wall the drum does not
        # already own ports on
        "back": {"heater_conduit": ("-z", (0.0, -0.85))},
        # the heater and blower sit ON the floor, inside the chamber;
        # the coil's two refrigerant lines cross THROUGH it, straight
        # down to the compressor and condenser in the bay below --
        # one boundary crossing, not two
        "floor": {"heater": ("+y", (0.0, -0.6)), "blower": ("+y", (0.0, 0.6)),
                  "coil_liquid": ("-y", (-0.5, -0.9)), "coil_suction": ("-y", (0.5, -0.9))},
    }
    bottom = Compartment(
        height_m=BOTTOM_BAY_HEIGHT_M, label="refrigeration bay",
        plant=(
            PlantItem(name="compressor", kind="refrigerant-compressor",
                     half_extent_m=(0.10, 0.09, 0.10), mass_kg=COMPRESSOR_MASS_KG,
                     at=(-0.5, -0.3), mount="isolated"),
            PlantItem(name="condenser", kind="condenser-coil",
                     half_extent_m=(0.22, 0.10, 0.05), mass_kg=9.0,
                     at=(0.4, 0.0), mount="bolted"),
            PlantItem(name="receiver_drier", kind="receiver-drier",
                     half_extent_m=(0.03, 0.09, 0.03), mass_kg=1.5,
                     at=(-0.5, 0.5), mount="bolted"),
        ))
    return CabinetSpec(identity=identity, inside_m=(INSIDE_W_M, INSIDE_H_M, INSIDE_D_M),
                       working_pressure_pa=WORKING_PRESSURE_PA, construction="welded-plate",
                       stiffener_pitch_m=STIFFENER_PITCH_M, door=Door(kind="solid", dogs=2),
                       drum=drum, bottom=bottom, feet_per_side=2, wall_ports=ports,
                       label="separator")


def chamber_for(identity: str = "plant.separator", *, target_k: float = 283.15):
    """The runtime state this machine is FOR: a climate_separator.
    ClimateChamber sized off the same box spec_for just declared,
    rather than a guessed volume and skin area beside it.

    `inlet_open`/`purge_open` on the object this returns are the
    runtime's own read of the graph's `gas_inlet`/`purge` ports -- set
    them from whatever is driving the machine (an operator, a
    servicing task, a test) the same way `centrifuge_for` reads the
    bowl off `spec.drum` rather than re-declaring it."""
    from climate_separator import (ClimateChamber, DewarAtmosphere, CHARGE_SPECIES,
                                   AMBIENT_K, ATM_PA, box_surface_area_m2,
                                   charge_mass_for_pressure_kg)
    spec = spec_for(identity)
    volume_m3 = spec.inside_m[0] * spec.inside_m[1] * spec.inside_m[2]
    # the fill this box's own declared WORKING_PRESSURE_PA actually IS,
    # in its own volume -- not a fixed mass left over from sizing a
    # different chamber, which is exactly the bug a flat default hides
    fill_kg = charge_mass_for_pressure_kg(ATM_PA + WORKING_PRESSURE_PA, volume_m3, AMBIENT_K)
    return ClimateChamber(volume_m3=volume_m3,
                          surface_area_m2=box_surface_area_m2(spec.inside_m),
                          insulation_media=INSULATION_MEDIA,
                          insulation_thickness_m=INSULATION_THICKNESS_M,
                          target_k=target_k, heater_kw=HEATER_KW,
                          atmosphere=DewarAtmosphere(kg={CHARGE_SPECIES: fill_kg},
                                                     temp_k=AMBIENT_K))


def build(identity: str = "plant.separator"):
    """Author the machine. Returns a turret_production.ProductionGraph."""
    spec = spec_for(identity)
    W, H, D = spec.inside_m
    hw, hh, hd = W / 2.0, H / 2.0, D / 2.0

    def extras(built: cb.Built):
        g = built.graph
        drum = spec.drum

        # ---- the chamber's own atmosphere: two valved ports ----------
        built.port("ceiling", "gas_inlet", "inlet-valve", 0.012)
        built.port("ceiling", "purge", "purge-valve", 0.012)

        # ---- the evaporator coil, inside, fed down through the floor --
        # a tube manifold rather than a drawn fin stack, the same
        # simplification autoclave.py's steam spreader makes: a real
        # coil's whole job here is a fed inlet and a real length, not a
        # picture of its fins. The lines are a connectivity claim
        # spanning real distance to the bay below, exactly the way
        # autoclave_production.py's own steam line reaches its spreader
        # -- not a literal drawn pipe.
        coil = Prism(identity=f"{identity}.coil", kind="evaporator-coil",
                    centre=(0.0, H - 0.10, -hd + 0.08), shape="tube",
                    axis=(1.0, 0.0, 0.0), radius=0.022, inner_radius=0.017,
                    length=W * 0.80, material="copper-tube",
                    attributes={"in_view": "plant", "inside": True})
        cn = emit_prism(g, coil, {"liquid": coil.side_point(math.pi, along=-0.4),
                                  "suction": coil.side_point(math.pi, along=0.4)},
                        assembly=spec.label)
        built.port("floor", "coil_liquid", "refrigerant-fitting", 0.010)
        built.port("floor", "coil_suction", "refrigerant-fitting", 0.014)
        g.edge(f"{identity}.coil.liquid_line", f"{identity}.port.coil_liquid", cn["liquid"],
              "refrigerant-line", radius=0.005, circuit_identity="refrigerant-liquid",
              part_role="through-port")
        g.edge(f"{identity}.coil.suction_line", f"{identity}.port.coil_suction", cn["suction"],
              "refrigerant-line", radius=0.007, circuit_identity="refrigerant-suction",
              part_role="through-port")

        # ---- the trim heater, on the floor ahead of the bowl ----------
        heater = Prism(identity=f"{identity}.heater", kind="heater-bank",
                       centre=(0.0, 0.05, hd - 0.14), shape="box",
                       half_extent=(0.20, 0.05, 0.07), material="stainless-plate",
                       mass_kg=HEATER_MASS_KG,
                       attributes={"in_view": "plant", "inside": True, "heater_kw": HEATER_KW})
        hn = emit_prism(g, heater, {"base": heater.face_point("-y"),
                                    "terminals": heater.face_point("+z")}, assembly=spec.label)
        g.edge(f"{identity}.heater.mount", hn["base"], built.wn("floor", "heater"),
              "bolted-flange-mount", radius=0.006, part_role="heater-mount")
        built.port("back", "heater_conduit", "gland-and-junction-box", 0.020)
        g.edge(f"{identity}.heater.conduit", f"{identity}.port.heater_conduit", hn["terminals"],
              "electrical-conduit", radius=0.010, wall_m=0.0015, alloy="a36",
              circuit_identity="heater-power", part_role="through-port")

        # ---- the recirculation blower: coil, bowl, heater, and back --
        # THE CIRCULAR GAS FLOW is this one part. Nothing about the coil
        # or the heater makes the chamber uniform on its own -- a still
        # gas volume stratifies exactly like HydraulicTank's oil would
        # without the pump moving it -- so the blower is not decorative,
        # it is the thing that makes climate_separator.ClimateChamber's
        # single lumped temperature an honest model rather than a wish.
        blower = Prism(identity=f"{identity}.blower", kind="recirculation-blower",
                       centre=(0.0, 0.14, -hd + 0.14), shape="cylinder",
                       axis=(0.0, 1.0, 0.0), radius=0.09, length=0.12,
                       material="cast-aluminium", mass_kg=BLOWER_MASS_KG,
                       attributes={"in_view": "plant", "inside": True,
                                   "flow_m3_s": BLOWER_FLOW_M3_S,
                                   "rotor_axis": (0.0, 1.0, 0.0), "rotor_rated_rpm": 2800.0,
                                   "rotor_shape": "solid-disc", "rotor_mass_kg": 1.0,
                                   "rotor_radius_m": 0.08, "balance_grade_mm_s": 6.3,
                                   "runs_in": ("run",)})
        bn = emit_prism(g, blower, {"base": blower.end_point("-")}, assembly=spec.label)
        g.edge(f"{identity}.blower.mount", bn["base"], built.wn("floor", "blower"),
              "bolted-flange-mount", radius=0.006, part_role="blower-mount")

        # ---- the compressor's own rating, read straight off the body
        # the bottom compartment already emitted -- RefrigerantLoop.
        # from_compressor_node reads exactly these two fields, so this
        # is the single declaration of how big the compressor is, not a
        # second number agreeing (or not) with one written elsewhere.
        compressor_ident = f"{identity}.bottom.compressor"
        comp = next((n for n in g.nodes if n["identity"] == compressor_ident), None)
        if comp is not None:
            comp.update(rated_w=COMPRESSOR_RATED_W,
                       reference_omega_rad_s=COMPRESSOR_REFERENCE_OMEGA_RAD_S,
                       rotor_axis=(1.0, 0.0, 0.0), rotor_rated_rpm=2900.0,
                       rotor_shape="solid-disc", rotor_mass_kg=COMPRESSOR_MASS_KG * 0.22,
                       rotor_radius_m=0.04, balance_grade_mm_s=6.3, runs_in=("run",))

    return cb.build(spec, extras=extras).graph


def cycle(identity: str = "plant.separator", *, live_vibration: bool = False) -> Cycle:
    """Charge with inert gas, chill to setpoint, spin up, run, coast,
    purge back to shop air. The bowl's own unbalance follows the
    single balance grade cabinet.Drum already declares; nothing here
    adds a second one."""
    charges = cb.charges(spec_for(identity))
    return Cycle(machine=identity, states=(
        OperatingState("charge", note="gas_inlet open, purge shut: filling the chamber"),
        OperatingState("chill", charges_kg=dict(charges),
                       note="coil and heater trimming to setpoint; bowl still"),
        OperatingState("spin-up", charges_kg=dict(charges), curve=run_up(24),
                       note="at temperature, bowl coming up to speed"),
        OperatingState("run", charges_kg=dict(charges), note="at speed and at setpoint"),
        OperatingState("coast", charges_kg=dict(charges), curve=coast(12),
                       speeds={f"{identity}.drum.motor": 0.0},
                       note="motor off, bowl coasting"),
        OperatingState("purge", note="purge open: chamber let down to shop air"),
    ), live_vibration=live_vibration)


def describe(g, identity: str = "plant.separator") -> str:
    from climate_separator import centrifuge_for
    doc = g.as_document()
    total = sum(float(n.get("mass_kg", 0.0)) for n in doc["nodes"] if not n.get("wrench_point"))
    spec = spec_for(identity)
    chamber = chamber_for(identity)
    unit = centrifuge_for(spec.drum, spec.drum.drive)
    cut_um = unit.cut_size_m(BLOWER_FLOW_M3_S, oil_temp_k=chamber.target_k) * 1e6
    lines = [f"{doc['identity']}: {INSIDE_W_M:.2f} x {INSIDE_H_M:.2f} x {INSIDE_D_M:.2f} m "
            f"sealed chamber, {WORKING_PRESSURE_PA:.0f} Pa hold",
            f"  bowl r {BOWL_RADIUS_M:.2f} m x {BOWL_LENGTH_M:.2f} m, {BOWL_RATED_RPM:.0f} rpm, "
            f"suspended, balance grade {BOWL_BALANCE_GRADE_MM_S:.1f} mm/s",
            f"  compressor {COMPRESSOR_RATED_W / 1000:.1f} kW, heater {HEATER_KW:.1f} kW, "
            f"blower {BLOWER_FLOW_M3_S * 1000:.0f} L/s recirculating",
            chamber.describe(),
            f"    at {chamber.target_k - 273.15:.0f} C: cut size {cut_um:.1f} um "
            f"({unit.g_force:.0f} g)",
            f"  {total:.0f} kg all told"]
    return "\n".join(lines)
