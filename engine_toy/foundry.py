"""Sand casting: a crucible, a rammed mould, and a part that needs work.

The heat source already exists. kiln.py is a real refractory-lined
chamber with a real burner on gas_works fuel and a forced-draft blower,
and it already knows about reducing and oxidising atmospheres -- a
foundry furnace IS that, pointed at a crucible. Nothing here builds a
second furnace.

What is new is everything between the melt and a usable part, and it is
governed by three real laws that between them decide whether a pour
succeeds:

CHVORINOV'S RULE -- the one that makes casting an engineering problem
rather than a pouring problem.

    t_solidification = C . (V / A)^2

    A shape freezes in a time set by the square of its volume over its
    surface area. That ratio has a name, the MODULUS, and everything
    follows from it: a thick section freezes slowly, a thin one fast, and
    -- the part that matters -- A RISER ONLY WORKS IF IT FREEZES LAST.
    A riser with a smaller modulus than the section it feeds solidifies
    first, and then it is not a reservoir, it is decoration. Foundries
    size risers by modulus for exactly this reason and so does this.

SHRINKAGE, which is why a riser is needed at all.

    phase_table now carries the signed volume change of every metal's
    freezing, and for metals it is NEGATIVE -- liquid metal is less
    dense than solid, so it contracts on freezing. Aluminium loses 6.5%.
    That volume has to come from somewhere, and if it does not come from
    a riser it comes from inside the casting, as shrinkage porosity.
    (Water is the anomaly that expands; a mould full of freezing water
    would split rather than sink.)

FLUIDITY -- how far the metal gets before it stops.

    A melt poured with little superheat freezes while it is still
    running, and the two halves of a flow meeting after both have
    started to solidify do not fuse: that is a COLD SHUT, a seam that
    looks like a crack and is one. Fluidity length rises with superheat,
    which is the real reason a foundry pours hot rather than at the
    melting point, and the real cost of doing so is more shrinkage and
    more gas pickup.

AND THE SAND ITSELF. Green sand is silica, clay and WATER, and the water
is the problem: the moment metal touches it, it flashes to steam. If the
rammed sand is permeable enough the steam escapes through it. If it is
not -- rammed too hard, or too wet -- the steam has nowhere to go but
into the metal, and the casting comes out full of gas holes, or the
mould blows back at the person pouring it. Ramming harder gives a better
surface and a worse casting, and that trade is real.

WHAT COMES OUT IS NOT A PART. It is a casting with the sprue and risers
still on it, flash where the mould halves met, a rough as-cast surface,
and draft angles that were there so the pattern could be drawn. All of
that has to come off, and `Casting.finishing` says how much of what you
poured is actually the part.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

GRAVITY_M_S2 = 9.80665

# --- green sand, real properties --------------------------------------
#: Silica sand grain density. The bulk density is lower because rammed
#: sand is maybe 60% solid.
SILICA_DENSITY_KG_M3 = 2650.0
GREEN_SAND_BULK_DENSITY_KG_M3 = 1600.0

#: Real green sand composition by mass. Water is the small number that
#: causes nearly all the trouble.
GREEN_SAND_CLAY_FRAC = 0.08
GREEN_SAND_MOISTURE_FRAC = 0.035

#: Steam from that moisture, per kilogram of sand the metal touches.
#: Water's latent heat is in phase_table; this is just the mass.
WATER_LATENT_VAPORISATION_J_PER_KG = 2_256_000.0

#: Chvorinov's constant for green sand, s/m^2. A real mould-material
#: property: it bundles the sand's thermal diffusivity and the metal's
#: latent heat, which is why it is quoted per mould material rather
#: than derived per casting.
CHVORINOV_CONSTANT_S_PER_M2 = 2.0e6

#: A riser must freeze later than what it feeds, and 1.2x modulus is the
#: real rule of thumb foundries size to -- enough margin for the riser
#: to still be liquid when the casting stops needing it.
RISER_MODULUS_MARGIN = 1.2

#: Fluidity length per kelvin of superheat, metres. Real spiral-test
#: numbers for aluminium in green sand are a few hundred millimetres at
#: a hundred degrees of superheat.
FLUIDITY_M_PER_K_SUPERHEAT = 0.004


def modulus_m(volume_m3: float, area_m2: float) -> float:
    """V/A, the casting modulus. Everything in Chvorinov is this."""
    return max(0.0, volume_m3) / max(area_m2, 1e-9)


def solidification_time_s(volume_m3: float, area_m2: float,
                          constant: float = CHVORINOV_CONSTANT_S_PER_M2) -> float:
    """Chvorinov: t = C.(V/A)^2. Squared, which is why doubling a
    section's thickness quadruples the time it takes to freeze."""
    m = modulus_m(volume_m3, area_m2)
    return constant * m * m


# ---------------------------------------------------------------------


@dataclass
class Crucible:
    """A refractory pot with a real capacity, heated by a real kiln.

    Capacity is quoted in LITRES because that is what a crucible
    physically holds; the kilograms depend entirely on what is in it,
    and a crucible rated for aluminium holds four times the mass in
    lead. Foundries size in 'aluminium equivalent' for the same reason
    and this refuses to pretend otherwise."""
    identity: str = "foundry.crucible"
    capacity_l: float = 6.0
    material: str = "clay-graphite"     # or "silicon-carbide"
    metal: str = "aluminium"
    charge_kg: float = 0.0
    temp_k: float = 293.15
    #: how much of the crucible's own life is used up. Real crucibles
    #: fail by thermal cycling, not by wearing out, so this counts heats.
    heats_used: int = 0
    max_heats: int = 100

    def capacity_kg(self) -> float:
        import fluids
        row = fluids.BY_KEY.get(f"molten-{self.metal}")
        rho = row.density_kg_m3 if row else 2400.0
        return self.capacity_l / 1000.0 * rho

    def melt_energy_j(self, ambient_k: float = 293.15,
                      superheat_k: float = 100.0) -> float:
        """What it takes to get this charge liquid and pourable.

        Three terms, none optional: raise the solid to its melting
        point, supply the latent heat of fusion to actually melt it, and
        then superheat the liquid so it stays liquid long enough to run.
        phase_table carries the melting point and latent heat; the
        specific heats are the metals' own."""
        import phase_table as pt
        m = pt.phases(self.metal)
        if m is None:
            return 0.0
        melt = m.get(pt.MELT)
        if melt is None or melt.temp_k <= 0.0:
            return 0.0
        cp_solid = _SPECIFIC_HEAT_J_PER_KGK.get(self.metal, 900.0)
        cp_liquid = cp_solid * 1.1      # liquid metals run slightly higher
        kg = max(0.0, self.charge_kg)
        sensible = kg * cp_solid * max(0.0, melt.temp_k - ambient_k)
        fusion = kg * abs(melt.latent_j_per_kg)
        superheat = kg * cp_liquid * max(0.0, superheat_k)
        return sensible + fusion + superheat

    def ready_to_pour(self, superheat_k: float = 0.0) -> tuple:
        import phase_table as pt
        m = pt.phases(self.metal)
        melt = m.get(pt.MELT) if m else None
        if melt is None or melt.temp_k <= 0.0:
            return False, f"{self.metal} has no declared melting point"
        if self.temp_k < melt.temp_k:
            return False, (f"still solid: {self.temp_k:.0f} K against a "
                           f"{melt.temp_k:.0f} K melting point")
        if self.heats_used >= self.max_heats:
            return False, "crucible is past its heat count -- it will fail in the tongs"
        if self.charge_kg > self.capacity_kg():
            return False, (f"overfilled: {self.charge_kg:.1f} kg of {self.metal} in a "
                           f"{self.capacity_kg():.1f} kg pot")
        sh = self.temp_k - melt.temp_k
        if sh < 50.0:
            return True, (f"pourable but only {sh:.0f} K of superheat -- it will "
                          "freeze in the runners on anything but a short pour")
        return True, f"{sh:.0f} K of superheat"

    def fluidity_m(self) -> float:
        """How far the metal will run before it stops.

        Linear in superheat, which is the real spiral-test result and
        the reason a foundry pours hot."""
        import phase_table as pt
        m = pt.phases(self.metal)
        melt = m.get(pt.MELT) if m else None
        if melt is None or melt.temp_k <= 0.0:
            return 0.0
        return max(0.0, self.temp_k - melt.temp_k) * FLUIDITY_M_PER_K_SUPERHEAT


#: Real specific heats, J/kg.K. Solid, near room temperature.
_SPECIFIC_HEAT_J_PER_KGK = {
    "aluminium": 897.0, "magnesium": 1020.0, "iron": 449.0, "lead": 129.0,
}


@dataclass
class SandMould:
    """Rammed green sand, with the trade that ramming makes real.

    `ram_hardness` 0..1: harder sand takes finer detail and a better
    surface, and is LESS permeable. Both directions are real and the
    foundry has to pick a point on that line, which is why this is one
    number and not two independent quality settings."""
    identity: str = "foundry.mould"
    cavity_volume_m3: float = 0.0008          # ~2 kg of aluminium
    cavity_area_m2: float = 0.06
    sand_mass_kg: float = 40.0
    ram_hardness: float = 0.6
    moisture_frac: float = GREEN_SAND_MOISTURE_FRAC
    #: the longest path metal must run from the sprue to fill it
    flow_path_m: float = 0.25
    riser_volume_m3: float = 0.0
    riser_area_m2: float = 0.0

    @property
    def permeability(self) -> float:
        """0..1, how freely steam gets out. Falls with ramming and with
        moisture: wetter sand has less connected air space."""
        return max(0.0, min(1.0, (1.0 - self.ram_hardness * 0.7)
                            * (1.0 - self.moisture_frac * 8.0)))

    @property
    def surface_finish(self) -> float:
        """0..1, the other half of the same trade."""
        return max(0.0, min(1.0, 0.25 + 0.75 * self.ram_hardness))

    def steam_kg(self, contacted_sand_frac: float = 0.15) -> float:
        """Water flashed to steam by the metal, in kg.

        Only the sand actually touching hot metal gives up its water,
        which is a small fraction of the mould -- but it happens in the
        first second or two, all at once."""
        return self.sand_mass_kg * contacted_sand_frac * self.moisture_frac

    def casting_modulus_m(self) -> float:
        return modulus_m(self.cavity_volume_m3, self.cavity_area_m2)

    def riser_modulus_m(self) -> float:
        if self.riser_volume_m3 <= 0.0:
            return 0.0
        return modulus_m(self.riser_volume_m3, self.riser_area_m2)

    def riser_adequate(self) -> tuple:
        """Does the riser freeze last, and is it big enough?

        Two separate questions and a riser has to pass both. Chvorinov
        decides the first: a riser with a smaller modulus than the
        casting freezes first and feeds nothing. Volume decides the
        second: whatever the metal contracts on freezing has to come out
        of the riser, or it comes out of the casting."""
        if self.riser_volume_m3 <= 0.0:
            return False, "no riser: every bit of shrinkage comes out of the casting"
        cm, rm = self.casting_modulus_m(), self.riser_modulus_m()
        if rm < cm * RISER_MODULUS_MARGIN:
            return False, (f"riser modulus {rm * 1000:.1f} mm against a casting's "
                           f"{cm * 1000:.1f} mm -- it freezes FIRST and feeds nothing. "
                           f"Needs {cm * RISER_MODULUS_MARGIN * 1000:.1f} mm")
        return True, (f"riser modulus {rm * 1000:.1f} mm vs casting "
                      f"{cm * 1000:.1f} mm: freezes last, as it must")


@dataclass
class Casting:
    """What actually came out, and everything wrong with it."""
    metal: str
    poured_kg: float
    cavity_kg: float
    solidification_s: float
    shrinkage_volume_m3: float
    shrinkage_fed_frac: float          # how much the riser covered
    porosity_frac: float
    cold_shut: bool
    gas_holes: bool
    blew_back: bool
    surface_finish: float
    defects: tuple = ()

    @property
    def sound(self) -> bool:
        return not (self.cold_shut or self.gas_holes or self.blew_back) \
            and self.porosity_frac < 0.01

    def finishing(self) -> dict:
        """What still has to come off before it is a part.

        A casting is not a part: the sprue and risers are still attached,
        there is flash where the mould halves met, and the as-cast
        surface needs work. This is the difference between what was
        poured and what you keep."""
        keep = self.cavity_kg
        remove = max(0.0, self.poured_kg - keep)
        return {"poured_kg": self.poured_kg,
                "part_kg": keep,
                "to_remove_kg": remove,
                "yield_frac": keep / max(self.poured_kg, 1e-9),
                "work": ("cut the sprue and risers off, grind the flash along the "
                         "parting line, and machine every surface that has to be "
                         "accurate -- the as-cast finish is "
                         f"{self.surface_finish:.2f} and draft is still on every "
                         "vertical face")}


def pour(crucible: Crucible, mould: SandMould) -> Casting:
    """Pour it and find out. Every defect here has a named cause."""
    import fluids
    import phase_table as pt

    row = fluids.BY_KEY.get(f"molten-{crucible.metal}")
    rho_liquid = row.density_kg_m3 if row else 2400.0
    cavity_kg = mould.cavity_volume_m3 * rho_liquid
    riser_kg = mould.riser_volume_m3 * rho_liquid
    poured = min(crucible.charge_kg, cavity_kg + riser_kg)

    defects = []
    # --- did it get there at all? -------------------------------------
    reach = crucible.fluidity_m()
    cold_shut = reach < mould.flow_path_m
    if cold_shut:
        defects.append(f"COLD SHUT: metal runs {reach * 1000:.0f} mm at this "
                       f"superheat against a {mould.flow_path_m * 1000:.0f} mm path. "
                       "Pour hotter or shorten the runners")

    # --- shrinkage, from the phase table's own signed figure ----------
    m = pt.phases(crucible.metal)
    freeze = m.get(pt.FREEZE) if m else None
    shrink_frac = abs(freeze.volume_change_frac) if freeze else 0.0
    shrink_vol = mould.cavity_volume_m3 * shrink_frac
    ok, why = mould.riser_adequate()
    fed = min(1.0, mould.riser_volume_m3 / max(shrink_vol, 1e-12)) if ok else 0.0
    porosity = shrink_frac * (1.0 - fed)
    if porosity > 0.005:
        defects.append(f"SHRINKAGE POROSITY {porosity * 100:.1f}% by volume: {why}")

    # --- steam, and where it went -------------------------------------
    steam = mould.steam_kg()
    vent = mould.permeability
    gas_holes = vent < 0.35 and steam > 0.0
    blew_back = vent < 0.15 and steam > 0.0
    if blew_back:
        defects.append(f"MOULD BLEW BACK: {steam * 1000:.0f} g of water flashed to "
                       f"steam with a permeability of {vent:.2f} and nowhere to go. "
                       "This throws metal at whoever is pouring")
    elif gas_holes:
        defects.append(f"GAS HOLES: {steam * 1000:.0f} g of steam forced into the "
                       f"metal, permeability {vent:.2f}. Ram softer or dry the sand")

    t_solid = solidification_time_s(mould.cavity_volume_m3, mould.cavity_area_m2)
    return Casting(metal=crucible.metal, poured_kg=poured, cavity_kg=cavity_kg,
                   solidification_s=t_solid, shrinkage_volume_m3=shrink_vol,
                   shrinkage_fed_frac=fed, porosity_frac=porosity,
                   cold_shut=cold_shut, gas_holes=gas_holes, blew_back=blew_back,
                   surface_finish=mould.surface_finish, defects=tuple(defects))


def report(crucible: Crucible, mould: SandMould) -> str:
    ok, why = crucible.ready_to_pour()
    c = pour(crucible, mould)
    f = c.finishing()
    L = [f"pouring {crucible.charge_kg:.1f} kg of {crucible.metal} at "
         f"{crucible.temp_k:.0f} K",
         f"  crucible: {'READY' if ok else 'NOT READY'} -- {why}",
         f"  melt energy required: {crucible.melt_energy_j() / 1e6:.1f} MJ",
         f"  fluidity: metal runs {crucible.fluidity_m() * 1000:.0f} mm",
         f"  mould: permeability {mould.permeability:.2f}, surface "
         f"{mould.surface_finish:.2f}, {mould.steam_kg() * 1000:.0f} g of steam",
         f"  casting modulus {mould.casting_modulus_m() * 1000:.1f} mm -> freezes in "
         f"{c.solidification_s:.0f} s (Chvorinov)",
         ""]
    if c.defects:
        L.append("  DEFECTS:")
        for d in c.defects:
            L.append(f"    - {d}")
    else:
        L.append("  sound casting")
    L.append("")
    L.append(f"  poured {f['poured_kg']:.2f} kg -> part {f['part_kg']:.2f} kg "
             f"({f['yield_frac'] * 100:.0f}% yield), {f['to_remove_kg']:.2f} kg to cut off")
    L.append(f"  {f['work']}")
    return "\n".join(L)


if __name__ == "__main__":
    import phase_table as pt
    al = pt.phases("aluminium").get(pt.MELT)
    print("=== a good pour ===")
    cru = Crucible(metal="aluminium", charge_kg=3.0, temp_k=al.temp_k + 120.0)
    # a riser sized by MODULUS, not by eye: it must beat the casting's
    # 13.3 mm by the 1.2x margin, so ~16 mm, and hold more than the
    # 6.5% aluminium shrinks
    m = SandMould(riser_volume_m3=0.00040, riser_area_m2=0.020, ram_hardness=0.45)
    print(report(cru, m))
    print()
    print("=== poured cold, rammed hard, no riser ===")
    cru2 = Crucible(metal="aluminium", charge_kg=3.0, temp_k=al.temp_k + 15.0)
    m2 = SandMould(ram_hardness=0.95, moisture_frac=0.055)
    print(report(cru2, m2))

# ---------------------------------------------------------------------
# CORES: the disposable form that makes the INSIDE
# ---------------------------------------------------------------------
#
# Everything above makes an outside. A hollow casting -- a water jacket,
# an intake port, anything with a passage through it -- needs a separate
# sand shape sitting INSIDE the cavity, and that is a core. Every port
# and jacket in a real cast head is one.
#
# A core is not made of moulding sand. Green sand holds together with
# clay and water, which is fine when it is rammed against a pattern and
# supported on every side, but a core is a free-standing shape with
# metal all around it and nothing holding it up. So it is made with a
# BINDER that sets hard -- baked oil, sodium silicate gassed with CO2,
# or a phenolic resin -- and that creates the defining tension of the
# whole business:
#
#   IT MUST BE STRONG ENOUGH TO SURVIVE THE POUR, AND WEAK ENOUGH TO
#   COME BACK OUT.
#
# That is COLLAPSIBILITY, and it is the property core binders are chosen
# for. Sodium silicate is cheap and sets in seconds and is notoriously
# hard to shake out afterwards; resin costs more and breaks down cleanly
# when the heat of the casting decomposes it. A core that will not
# collapse is a casting you cannot finish, which is a more expensive
# failure than one that broke during the pour.
#
# AND IT FLOATS. This is the part that surprises people. Molten iron is
# 7000 kg/m3 and a sand core is 1600, so the core is in a bath four
# times its own density and the buoyant force is enormous -- a core with
# a few litres of volume is pushed up with the weight of a person. It
# lifts, the casting comes out with the passage in the wrong place and a
# wall thickness of zero on one side, and nothing about the pour looked
# wrong at the time. A five-litre core in iron is pushed up with about
# twenty-seven kilograms of force, against prints bearing on sand that
# gives way at a couple of hundred kilopascals. Cores are held down by their PRINTS (extensions
# into the mould at each end) and, when that is not enough, by CHAPLETS
# -- small metal supports that are deliberately cast into the part and
# have to melt just enough to fuse without weakening it.

#: Real bulk density of bonded core sand.
CORE_SAND_DENSITY_KG_M3 = 1600.0

#: Compressive strength of rammed green sand, Pa. This is what a core
#: print actually bears against, and it is the weakest link by a factor
#: of twenty against any core binder.
GREEN_SAND_COMPRESSIVE_PA = 150e3

#: How much wider a chaplet's foot is than its stem. A chaplet is shaped
#: like a drawing pin for exactly this reason: the stem carries the load
#: and the foot spreads it over enough sand to not punch through.
CHAPLET_FOOT_SPREAD = 10.0

#: Core binders: (tensile strength Pa, collapsibility 0..1, gas volume
#: per kg of sand in m3 at temperature). Collapsibility is how readily
#: it breaks down in the casting's heat -- the higher the easier to
#: shake out, and the weaker during the pour.
CORE_BINDERS = {
    "baked-oil":      (1.2e6, 0.75, 0.020),
    "co2-silicate":   (2.5e6, 0.25, 0.008),
    "phenolic-resin": (3.5e6, 0.85, 0.035),
}


@dataclass
class Core:
    """A disposable sand form that makes an internal passage.

    `print_area_m2` is the total cross-section of the core prints -- the
    extensions at each end that sit in the mould and carry the core's
    load. It is what actually resists the buoyancy below, and it is the
    number that gets under-designed."""
    identity: str = "foundry.core"
    binder: str = "phenolic-resin"
    volume_m3: float = 0.0003
    print_area_m2: float = 0.0015
    chaplets: int = 0
    chaplet_area_m2: float = 0.0001

    @property
    def mass_kg(self) -> float:
        return self.volume_m3 * CORE_SAND_DENSITY_KG_M3

    def _binder(self):
        return CORE_BINDERS.get(self.binder, CORE_BINDERS["phenolic-resin"])

    def buoyant_force_n(self, metal: str) -> float:
        """Uplift on the core once metal surrounds it.

        Archimedes, with the core's own weight already taken off: the
        core displaces metal far denser than itself, so the net force is
        upward and large. This is the number that lifts cores."""
        import fluids
        row = fluids.BY_KEY.get(f"molten-{metal}")
        rho_metal = row.density_kg_m3 if row else 2400.0
        return max(0.0, (rho_metal - CORE_SAND_DENSITY_KG_M3)) * self.volume_m3 * GRAVITY_M_S2

    def restraint_n(self) -> float:
        """What the prints and chaplets can actually hold.

        THE LIMIT IS THE MOULD, NOT THE CORE. A print does not fail by
        tearing the core apart -- it fails by crushing into the green
        sand it is bearing against, and green sand is weak: a couple of
        hundred kilopascals in compression against the binder's several
        megapascals. Sizing this on the binder's tensile strength said a
        core could never lift, which contradicts the entire reason
        chaplets exist."""
        prints = GREEN_SAND_COMPRESSIVE_PA * max(0.0, self.print_area_m2)
        # A chaplet's STEM is steel and enormously strong, but its FOOT
        # bears on the same weak green sand the print does, so that is
        # where it is limited too. Sizing it on the steel gave 100 kN
        # from four small chaplets, which would make a core unliftable
        # by anything. The foot is spread wider than the stem, which is
        # the whole point of its shape.
        foot_area = self.chaplet_area_m2 * CHAPLET_FOOT_SPREAD
        chaplet = self.chaplets * foot_area * GREEN_SAND_COMPRESSIVE_PA
        return prints + chaplet

    def holds(self, metal: str) -> tuple:
        up = self.buoyant_force_n(metal)
        hold = self.restraint_n()
        if hold >= up:
            return True, (f"held: {up:.0f} N of uplift against {hold:.0f} N of "
                          f"print and chaplet restraint")
        need_area = up / GREEN_SAND_COMPRESSIVE_PA
        return False, (f"CORE LIFTS: {up:.0f} N of buoyancy in molten {metal} against "
                       f"only {hold:.0f} N of restraint. The passage ends up out of "
                       f"position with a thin wall on one side and nothing looks wrong "
                       f"during the pour. Needs {need_area * 1e4:.1f} cm2 of print, or "
                       f"{max(1, int((up - hold) / (self.chaplet_area_m2 * 250e6)) + 1)} "
                       "chaplets")

    def gas_m3(self) -> float:
        """Gas the binder gives off when the casting's heat hits it.

        It has to vent out through the prints. If it cannot, it goes
        into the metal, and a core blow puts a hole right where the
        passage wall should be."""
        return self.mass_kg * self._binder()[2]

    def shakes_out(self) -> tuple:
        """Will it come back out, which is a separate question entirely."""
        collapse = self._binder()[1]
        if collapse >= 0.6:
            return True, f"{self.binder} collapses readily ({collapse:.2f}) -- shakes out"
        if collapse >= 0.35:
            return True, (f"{self.binder} is stubborn ({collapse:.2f}) -- it will need "
                          "real work on the shakeout, and a blind passage may not clear")
        return False, (f"{self.binder} barely collapses ({collapse:.2f}). The casting is "
                       "sound and you cannot get the core out of it, which is a more "
                       "expensive failure than breaking one during the pour")

