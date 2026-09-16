"""A hopper that can refuse to flow, and a shredder that can jam.

The site already accepts scrap -- aluminium, brass, copper, lead,
batteries, catalytic converters -- and the foundry already melts it.
What sits between the two is bulk handling, and bulk handling is
interesting for exactly one reason: GRANULAR MATERIAL IS NOT A FLUID.
It is the thing everyone models as a fluid and then wonders why the
silo is empty at the bottom and full at the top.

JANSSEN, AND WHY A TALL SILO DOES NOT CRUSH ITSELF

    Pour water into a column and the pressure at the bottom rises
    forever with depth. Pour grain in and it does not. The grains push
    sideways against the walls, the walls push back with friction, and
    that friction carries a growing share of the weight until the
    vertical stress SATURATES:

        sigma_v(h)  =  (rho.g.R / (mu.K)) . (1 - exp(-mu.K.h / R))

    Past about four diameters of depth the bottom of a silo stops
    caring how much more you add. This is Janssen's 1895 result, it is
    why silos are built tall and narrow, and it is also why the wall
    loads are what actually design them.

ARCHING, WHICH IS THE FAILURE EVERYONE HAS SEEN

    Cohesive material bridges over an outlet: the grains jam into a
    stable arch that carries its own load into the walls, and the hopper
    sits there full with nothing coming out. Whether it can happen is a
    competition between the material's own strength and the weight of
    the arch, and the outlet size decides the winner:

        D_critical  =  2 . sigma_c / (rho . g)

    Below that, it bridges. Above it, the arch cannot support itself and
    collapses under its own weight. There is a second, purely geometric
    limit as well -- an outlet less than about six times the largest
    lump will jam on interlocking alone, however free-flowing the
    material is -- and shredded scrap fails that one far more often than
    it fails the cohesive one.

MASS FLOW VERSUS FUNNEL FLOW

    A steep, smooth-walled hopper moves everything at once (mass flow).
    A shallow one carves a channel down the middle and leaves the rest
    standing (funnel flow), which is where RATHOLING comes from and why
    the last third of a shallow hopper never leaves on its own. The
    dividing wall angle comes from the wall friction, so it is a
    property of what the hopper is made of and what is in it, not a
    design preference.

THE SHREDDER, WHICH IS A TORQUE PROBLEM

    A shredder does not grind, it SHEARS: hammers or cutters drag
    material against an anvil until it fails in shear. So the force is
    the material's own shear strength times the area being cut, the
    torque is that times the rotor radius, and the machine either has
    that torque or it stalls. The energy per kilogram follows from the
    same shear work, which is why shredding steel costs several times
    what shredding aluminium does -- and why nobody shreds a solid
    engine block twice.

    Stalling is the interesting state, not a failure to be avoided: a
    real shredder jams, reverses, and tries again, and an operator who
    keeps feeding it a thing it cannot cut is the actual problem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

GRAVITY_M_S2 = 9.80665

#: Real shear strengths, Pa. Roughly 0.6 of tensile for ductile metals,
#: which is where these come from.
SHEAR_STRENGTH_PA = {
    "aluminium": 110e6,
    "magnesium": 110e6,
    "copper": 200e6,
    "brass": 250e6,
    "lead": 14e6,
    "mild-steel": 350e6,
    "cast-iron": 300e6,
    "plastic": 35e6,
    "rubber": 10e6,
}

#: Bulk density of the shredded product, kg/m3. Loose scrap is mostly
#: air, which is the entire reason it is shredded before it is shipped.
SHREDDED_BULK_DENSITY_KG_M3 = {
    "aluminium": 400.0, "mild-steel": 1100.0, "cast-iron": 1300.0,
    "copper": 900.0, "brass": 1000.0, "lead": 2000.0,
    "plastic": 250.0, "rubber": 300.0,
}

#: An outlet must clear the largest lump by this factor or the material
#: jams on interlocking alone, regardless of how free-flowing it is.
#: Real design practice, and the limit shredded scrap actually hits.
OUTLET_LUMP_CLEARANCE = 6.0

#: Janssen's lateral-to-vertical stress ratio for a free-flowing
#: granular solid. Real, standard, and only weakly material-dependent.
JANSSEN_K = 0.4

#: PUBLISHED ANCHOR: a real scrap shredder uses 15-30 kWh per tonne of
#: mild steel, so 20 kWh/t = 72 kJ/kg at a typical output size. The
#: shear-work coefficient below is SOLVED from this rather than chosen,
#: after a first version with a coefficient picked by eye reported a
#: 75 kW machine doing 3200 tonnes an hour of lead.
REFERENCE_SHRED_ENERGY_J_PER_KG = 72_000.0
REFERENCE_SHRED_MATERIAL = "mild-steel"
REFERENCE_SHRED_THICKNESS_M = 0.004
REFERENCE_SCREEN_M = 0.05


def _solve_shear_work_coefficient() -> float:
    """Solved from the published specific energy, not picked.

    The shape of the law is right -- shear work goes with the strength,
    the new surface created, and how far the cut has to tear -- but its
    scale is one unknown coefficient that nobody publishes. What IS
    published is kWh per tonne. One equation, one unknown."""
    tau = SHEAR_STRENGTH_PA[REFERENCE_SHRED_MATERIAL]
    rho = SHREDDED_BULK_DENSITY_KG_M3[REFERENCE_SHRED_MATERIAL]
    surface_per_m3 = 6.0 / REFERENCE_SCREEN_M
    unscaled = tau * surface_per_m3 * REFERENCE_SHRED_THICKNESS_M / rho
    return REFERENCE_SHRED_ENERGY_J_PER_KG / max(unscaled, 1e-9)


SHEAR_WORK_COEFFICIENT = _solve_shear_work_coefficient()


@dataclass
class Hopper:
    """A bin that holds bulk material and sometimes refuses to give it up.

    `wall_friction_deg` and `half_angle_deg` together decide whether it
    mass-flows or funnel-flows, which is the difference between emptying
    completely and emptying a third of the way and stopping."""
    identity: str = "site.hopper"
    diameter_m: float = 1.8
    height_m: float = 3.0
    outlet_m: float = 0.30
    half_angle_deg: float = 25.0        # from vertical; smaller = steeper
    wall_friction_deg: float = 22.0
    material: str = "mild-steel"
    fill_frac: float = 0.0
    #: cohesion of the stored material, Pa. Dry shredded metal is nearly
    #: zero; anything damp, oily or fine is not, and that is what turns
    #: a working hopper into a bridged one overnight.
    cohesion_pa: float = 0.0
    largest_lump_m: float = 0.04

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    def bulk_density(self) -> float:
        return SHREDDED_BULK_DENSITY_KG_M3.get(self.material, 800.0)

    def contents_kg(self) -> float:
        vol = math.pi * self.radius_m ** 2 * self.height_m * max(0.0, self.fill_frac)
        return vol * self.bulk_density()

    def janssen_stress_pa(self, depth_m: float | None = None) -> float:
        """Vertical stress at depth, which SATURATES rather than growing.

        The whole point: past a few diameters the bottom stops caring
        how much more is above it, because wall friction is carrying the
        rest."""
        h = self.height_m * self.fill_frac if depth_m is None else depth_m
        mu = math.tan(math.radians(self.wall_friction_deg))
        rho = self.bulk_density()
        denom = mu * JANSSEN_K
        if denom <= 1e-9:
            return rho * GRAVITY_M_S2 * max(0.0, h)
        saturation = rho * GRAVITY_M_S2 * self.radius_m / denom
        return saturation * (1.0 - math.exp(-denom * max(0.0, h) / self.radius_m))

    def hydrostatic_stress_pa(self, depth_m: float | None = None) -> float:
        """What it would be if this were a liquid -- for contrast only."""
        h = self.height_m * self.fill_frac if depth_m is None else depth_m
        return self.bulk_density() * GRAVITY_M_S2 * max(0.0, h)

    def critical_arch_m(self) -> float:
        """Smallest outlet that will not bridge, from cohesion."""
        if self.cohesion_pa <= 0.0:
            return 0.0
        return 2.0 * self.cohesion_pa / (self.bulk_density() * GRAVITY_M_S2)

    def interlock_limit_m(self) -> float:
        """Smallest outlet that will not jam on lump size alone."""
        return OUTLET_LUMP_CLEARANCE * self.largest_lump_m

    def mass_flow(self) -> bool:
        """Does everything move, or only a channel down the middle?

        The dividing line comes from the wall friction: a hopper must be
        steeper than its own wall friction allows, or the material at
        the walls simply stands still."""
        return self.half_angle_deg <= (45.0 - self.wall_friction_deg / 2.0)

    def flows(self) -> tuple:
        """Will anything come out, and if not, why not."""
        if self.fill_frac <= 0.0:
            return False, "empty"
        arch = self.critical_arch_m()
        lump = self.interlock_limit_m()
        if self.outlet_m < lump:
            return False, (f"JAMMED ON LUMP SIZE: a {self.outlet_m * 1000:.0f} mm "
                           f"outlet against {self.largest_lump_m * 1000:.0f} mm lumps "
                           f"needs {lump * 1000:.0f} mm to clear interlocking. "
                           "Shred finer or open the outlet")
        if self.outlet_m < arch:
            return False, (f"BRIDGED: cohesion {self.cohesion_pa / 1000:.1f} kPa "
                           f"supports an arch over anything under "
                           f"{arch * 1000:.0f} mm. Vibrate it, or keep the material dry")
        if not self.mass_flow():
            return True, (f"FUNNEL FLOW: at {self.half_angle_deg:.0f} deg with "
                          f"{self.wall_friction_deg:.0f} deg wall friction it carves a "
                          "channel and leaves the rest standing. It will empty part way "
                          "and then rathole")
        return True, "mass flow: the whole contents move together"

    def discharge_kg_s(self, discharge_coefficient: float = 0.58) -> float:
        """Beverloo-style outlet flow.

        Granular discharge is famously INDEPENDENT of head -- a hopper
        empties at the same rate full or nearly empty, which is the
        clearest possible demonstration that it is not a fluid. The rate
        goes with the outlet to the five-halves power."""
        ok, _ = self.flows()
        if not ok:
            return 0.0
        d = max(0.0, self.outlet_m - 1.4 * self.largest_lump_m)
        if d <= 0.0:
            return 0.0
        return (discharge_coefficient * self.bulk_density()
                * math.sqrt(GRAVITY_M_S2) * d ** 2.5)


@dataclass
class Shredder:
    """A rotor, hammers, and a torque limit it will genuinely hit.

    Sized by the torque it can deliver at the rotor, because that is
    what decides whether a given piece of scrap is shreddable at all --
    not by a throughput rating, which is a consequence rather than a
    property."""
    identity: str = "site.shredder"
    rotor_radius_m: float = 0.45
    rotor_torque_nm: float = 9000.0
    rotor_rpm: float = 700.0
    cutters: int = 12
    #: how wide a bite each cutter takes, and the screen that decides
    #: when a piece is small enough to leave
    bite_depth_m: float = 0.006
    screen_m: float = 0.05
    jammed: bool = False
    reversals: int = 0
    #: cutters go blunt, and a blunt cutter needs more force for the
    #: same cut -- the real reason throughput falls before anything
    #: actually breaks
    wear_frac: float = 0.0

    def shear_force_n(self, material: str, thickness_m: float) -> float:
        """Force to shear one bite: strength times the area being cut."""
        tau = SHEAR_STRENGTH_PA.get(material, 200e6)
        # the cut face is the bite width times the material thickness
        area = self.bite_depth_m * max(thickness_m, 1e-4)
        blunt = 1.0 + 2.0 * max(0.0, min(1.0, self.wear_frac))
        return tau * area * blunt

    def required_torque_nm(self, material: str, thickness_m: float,
                           simultaneous_cutters: int = 3) -> float:
        """Torque to keep cutting. Several cutters are engaged at once,
        which is why a shredder's peak demand is a multiple of one cut."""
        return (self.shear_force_n(material, thickness_m) * self.rotor_radius_m
                * max(1, simultaneous_cutters))

    def can_shred(self, material: str, thickness_m: float) -> tuple:
        need = self.required_torque_nm(material, thickness_m)
        if need <= self.rotor_torque_nm:
            return True, (f"{need / 1000.0:.1f} kNm needed against "
                          f"{self.rotor_torque_nm / 1000.0:.1f} available")
        return False, (f"STALLS: {need / 1000.0:.1f} kNm needed against "
                       f"{self.rotor_torque_nm / 1000.0:.1f} available. "
                       f"{material} at {thickness_m * 1000:.0f} mm is past this "
                       "machine -- cut it up first or feed it something softer")

    def energy_per_kg_j(self, material: str, thickness_m: float) -> float:
        """Shear work per kilogram of product.

        The cut face area per kilogram is set by how small the pieces
        end up: reducing to a 50 mm screen makes far less new surface
        than reducing to 10 mm, and the energy follows the surface."""
        tau = SHEAR_STRENGTH_PA.get(material, 200e6)
        rho = SHREDDED_BULK_DENSITY_KG_M3.get(material, 900.0)
        # new surface created per unit volume, for cubes of screen size
        surface_per_m3 = 6.0 / max(self.screen_m, 1e-3)
        # shear work is roughly strength x displacement over that area,
        # with displacement the material thickness it must tear through
        work_per_m3 = (tau * surface_per_m3 * max(thickness_m, 1e-4)
                       * SHEAR_WORK_COEFFICIENT)
        return work_per_m3 / max(rho, 1.0)

    def throughput_kg_s(self, material: str, thickness_m: float,
                        drive_power_w: float = 75_000.0) -> float:
        ok, _ = self.can_shred(material, thickness_m)
        if not ok or self.jammed:
            return 0.0
        e = self.energy_per_kg_j(material, thickness_m)
        return drive_power_w / max(e, 1.0)

    def feed(self, material: str, thickness_m: float) -> str:
        ok, why = self.can_shred(material, thickness_m)
        if not ok:
            self.jammed = True
            self.reversals += 1
            return (why + f". Jammed; reversing (reversal {self.reversals}). "
                    "A real machine backs off and tries again, and an operator "
                    "who keeps feeding it this is the actual problem")
        self.jammed = False
        self.wear_frac = min(1.0, self.wear_frac + 0.0005)
        return why


def report(hopper: Hopper, shredder: Shredder) -> str:
    L = [f"hopper {hopper.diameter_m:.1f} x {hopper.height_m:.1f} m, "
         f"{hopper.material}, {hopper.fill_frac * 100:.0f}% full "
         f"({hopper.contents_kg():.0f} kg)"]
    j = hopper.janssen_stress_pa()
    h = hopper.hydrostatic_stress_pa()
    L.append(f"  floor stress {j / 1000:.1f} kPa -- a LIQUID of the same density "
             f"would give {h / 1000:.1f} kPa")
    L.append(f"  (Janssen: wall friction carries {100 * (1 - j / max(h, 1e-9)):.0f}% "
             "of the weight)")
    ok, why = hopper.flows()
    L.append(f"  {'FLOWS' if ok else 'BLOCKED'}: {why}")
    if ok:
        L.append(f"  discharge {hopper.discharge_kg_s():.1f} kg/s, and it is the same "
                 "rate full or nearly empty -- granular flow does not care about head")
    L.append("")
    L.append(f"shredder, {shredder.rotor_torque_nm / 1000:.0f} kNm at "
             f"{shredder.rotor_radius_m * 1000:.0f} mm radius:")
    for mat, thk in (("aluminium", 0.004), ("mild-steel", 0.004),
                     ("cast-iron", 0.012), ("lead", 0.010)):
        good, msg = shredder.can_shred(mat, thk)
        rate = shredder.throughput_kg_s(mat, thk)
        L.append(f"  {mat:12s} {thk * 1000:4.0f} mm  "
                 f"{'ok   ' if good else 'STALL'}  {msg}"
                 + (f"  ->  {rate * 3.6:.1f} t/h" if good else ""))
    return "\n".join(L)


if __name__ == "__main__":
    print(report(Hopper(fill_frac=0.8, largest_lump_m=0.04),
                 Shredder()))
    print()
    print("=== damp fines in a shallow hopper ===")
    print(report(Hopper(fill_frac=0.8, half_angle_deg=40.0, cohesion_pa=3500.0,
                        largest_lump_m=0.002, outlet_m=0.15),
                 Shredder(wear_frac=0.9)))
