"""Two things people fit purely for show, and the real physics of each.

Neither is decoration in the code. Both are ignition problems, and this
project already has everything an ignition problem needs: flammability
limits on every fluid (working_fluids.lfl/ufl), a real spark energy, a
combustion model that now computes how much fuel FAILS to burn, and a
fittings system where a hole with hardware screwed into it becomes a
port. All that was missing was pointing them at the tailpipe and at the
underbody.

EXHAUST FLAMES -- and why most kits do not work the way people think.

  The fuel is free: a rich engine cannot burn all of it, because past
  stoichiometric only 1/phi of the fuel finds oxygen
  (combustion_efficiency). That unburnt fraction leaves down the pipe.
  It is also why the trick is traditionally done on OVERRUN with the
  ignition cut -- fuel still goes in, nothing lights it, and the whole
  charge leaves unburnt.

  The oxygen is the hard part, and it is what separates a kit that works
  from one that clicks. Exhaust from a rich engine is oxygen-STARVED by
  definition: the reason the fuel did not burn is that there was no air
  for it. A spark in that stream finds a mixture far above its upper
  flammable limit and does nothing. Real kits therefore either run the
  engine rich in ALTERNATING cylinders, or add air, or -- most commonly
  -- rely on the flame front happening at the TAILPIPE MOUTH, where the
  stream finally meets the atmosphere and passes down through its
  flammable range on the way to being diluted away.

  So the model asks the real question at the real place: at the mouth,
  where entrained air is mixing in, is the local fuel fraction between
  lfl and ufl, is there a spark, and is there enough energy in it?

MAGNESIUM SCRAPE PLATES -- sparks with a real ignition temperature.

  A skid plate dragging on tarmac is a friction contact: the heat is
  mu.N.v watts, delivered into the tiny volume of metal actually being
  torn off. A particle that small has almost no thermal mass, so it
  reaches its ignition temperature almost immediately -- which is why
  the sparks appear the instant the plate touches down and stop the
  instant it lifts.

  Magnesium is used because of two real numbers. It ignites around
  travel 923 K, far below steel's practical spark temperature, and it
  burns at roughly 3400 K -- hot enough to be brilliant white rather
  than the orange of a steel spark. It also burns in CO2 and in
  nitrogen, which is why a magnesium fire cannot be put out with a
  normal extinguisher, and reacts with water to give hydrogen, which is
  why trying makes it considerably worse.

  That last part is not a warning bolted on. It is the reason the plate
  has a real `mass_kg` here: a magnesium plate that ignites as a PLATE
  rather than as particles is a fire with a real energy content and a
  real duration, and the model says how long it burns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- magnesium, real measured properties ------------------------------
MAGNESIUM_IGNITION_K = 923.0          # ~650 C, its melting point: bulk ignites here
MAGNESIUM_CHIP_IGNITION_K = 773.0     # torn particles light lower -- huge area for their mass
MAGNESIUM_FLAME_K = 3400.0            # why the spark is white, not orange
MAGNESIUM_HEAT_OF_COMBUSTION_J_PER_KG = 24.7e6
MAGNESIUM_DENSITY_KG_M3 = 1738.0
MAGNESIUM_SPECIFIC_HEAT_J_PER_KGK = 1020.0

STEEL_SPARK_K = 1700.0                # an orange steel spark, for comparison
STEEL_DENSITY_KG_M3 = 7850.0
STEEL_SPECIFIC_HEAT_J_PER_KGK = 490.0

#: Sliding friction of metal on dry tarmac. Real, and the same order as
#: any dry metal-on-mineral pair.
SKID_FRICTION_COEFFICIENT = 0.45

#: Fraction of friction heat that stays in the torn particle rather than
#: conducting into the plate or the road. A particle is separating from
#: both at the moment it is made, so most of the work done on it stays
#: with it.
PARTICLE_HEAT_RETENTION = 0.6


@dataclass
class ScrapePlate:
    """A skid plate that makes sparks, with the metal declared.

    `material` is the whole point: the same plate in steel and in
    magnesium behaves completely differently, and the difference is two
    real material properties rather than a flag."""
    identity: str = "underbody.scrape_plate"
    material: str = "magnesium"        # "magnesium" | "steel"
    contact_area_m2: float = 0.02
    mass_kg: float = 2.5
    thickness_m: float = 0.006
    #: how much of the plate has been worn away, 0..1. A scrape plate is
    #: a consumable and this is what makes it one.
    wear_frac: float = 0.0
    burning: bool = False
    burn_elapsed_s: float = 0.0

    @property
    def ignition_k(self) -> float:
        return (MAGNESIUM_CHIP_IGNITION_K if self.material == "magnesium"
                else STEEL_SPARK_K)

    @property
    def density(self) -> float:
        return (MAGNESIUM_DENSITY_KG_M3 if self.material == "magnesium"
                else STEEL_DENSITY_KG_M3)

    @property
    def specific_heat(self) -> float:
        return (MAGNESIUM_SPECIFIC_HEAT_J_PER_KGK if self.material == "magnesium"
                else STEEL_SPECIFIC_HEAT_J_PER_KGK)

    def particle_temp_k(self, normal_force_n: float, speed_m_s: float,
                        particle_mass_kg: float = 1e-8,
                        ambient_k: float = 293.15) -> float:
        """How hot one torn particle gets.

        The contact does mu.N.v watts of work. A particle being torn off
        carries away a share of it and has essentially no thermal mass,
        so its temperature rise is that energy over m.cp. The particle
        mass is tiny by construction -- that is what a spark IS -- which
        is why this reaches ignition temperature at walking pace on
        magnesium and not at all on steel."""
        if speed_m_s <= 0.0 or normal_force_n <= 0.0:
            return ambient_k
        friction_w = SKID_FRICTION_COEFFICIENT * normal_force_n * speed_m_s
        # the particle is in the contact for about the time it takes the
        # contact to slide its own length
        dwell_s = math.sqrt(max(self.contact_area_m2, 1e-9)) / max(speed_m_s, 1e-6)
        # ...sharing the heat with however many particles are made in
        # that time. One is the honest unit here: this asks what ONE
        # particle's own share does to it.
        energy_j = friction_w * dwell_s * PARTICLE_HEAT_RETENTION
        # a single particle takes a share proportional to its own mass
        # against the mass swept out of the contact in that dwell
        swept_kg = (self.contact_area_m2 * 1e-6 * self.density)   # a micron-deep layer
        share = min(1.0, particle_mass_kg / max(swept_kg, 1e-12))
        return ambient_k + energy_j * share / max(particle_mass_kg * self.specific_heat, 1e-12)

    def sparking(self, normal_force_n: float, speed_m_s: float,
                 ambient_k: float = 293.15) -> bool:
        return self.particle_temp_k(normal_force_n, speed_m_s,
                                    ambient_k=ambient_k) >= self.ignition_k

    def spark_temp_k(self) -> float:
        """What the spark burns at once lit -- the colour, effectively."""
        return MAGNESIUM_FLAME_K if self.material == "magnesium" else STEEL_SPARK_K

    def plate_fire_seconds(self) -> float:
        """If the PLATE itself lights, how long it burns.

        Not the same event as sparking. A plate reaches bulk ignition
        only if it gets to 923 K through its thickness, and then it is a
        real fire with the plate's own chemical energy behind it. Burn
        rate for bulk magnesium is roughly a millimetre of thickness a
        minute, so the duration is set by how thick the plate is."""
        if self.material != "magnesium":
            return 0.0
        mm = self.thickness_m * 1000.0 * (1.0 - self.wear_frac)
        return max(0.0, mm * 60.0)

    def fire_energy_j(self) -> float:
        if self.material != "magnesium":
            return 0.0
        return (self.mass_kg * (1.0 - self.wear_frac)
                * MAGNESIUM_HEAT_OF_COMBUSTION_J_PER_KG)

    def extinguishing_note(self) -> str:
        if self.material != "magnesium":
            return "steel: water is fine"
        return ("magnesium: burns in CO2 AND in nitrogen, so neither a CO2 "
                "extinguisher nor an inert blanket stops it, and water is "
                "decomposed to hydrogen by the burning metal -- which is a "
                "second fire on top of the first. Class D powder or sand.")


# ---------------------------------------------------------------------
# the exhaust flame kit
# ---------------------------------------------------------------------

#: A real flame-kit plug is a normal spark plug in a bung welded into
#: the pipe, fired by its own coil. Energy is the ordinary automotive
#: figure -- the point of the kit is placement, not power.
FLAME_KIT_SPARK_ENERGY_MJ = 40.0

#: How much atmospheric air has entrained into the stream by the time it
#: reaches the mouth of the pipe. This is why the flame happens THERE
#: and not inside: the stream has to come back down through its
#: flammable range, and inside the pipe it is far too rich.
TAILPIPE_ENTRAINMENT_FRAC = 0.35


@dataclass
class FlameKit:
    """A spark plug in the exhaust, its coil, and what it needs to work.

    Installs as real hardware: a bung in the pipe is a hole with
    something screwed into it, which fittings.py already calls a PORT.
    `air_pump_kg_s` is the optional extra that separates a kit that
    works at any mixture from one that only works on overrun."""
    identity: str = "exhaust.flame_kit"
    plug_position_m_from_mouth: float = 0.15
    spark_energy_mj: float = FLAME_KIT_SPARK_ENERGY_MJ
    firing: bool = False
    #: some kits add an air pump into the pipe. Without one the kit can
    #: only work where the atmosphere does the mixing for it.
    air_pump_kg_s: float = 0.0
    #: a wideband sensor in the same pipe, so the kit can be told when
    #: there is actually something to light. Real kits without one fire
    #: blindly and mostly click.
    richness_sensor: bool = False

    def unburnt_fuel_frac(self, phi: float, combustion_eff: float = 1.0) -> float:
        """Fuel leaving unburnt, as a fraction of the fuel that went in.

        Two independent sources, and the first dominates: past
        stoichiometric only 1/phi of the fuel can find oxygen, so
        1 - 1/phi of it leaves untouched. On top of that, whatever the
        combustion model says failed to burn for crevice and quench
        reasons also leaves. combustion_efficiency computes both."""
        p = max(float(phi), 1e-6)
        oxygen_starved = max(0.0, 1.0 - 1.0 / p) if p > 1.0 else 0.0
        incomplete = max(0.0, 1.0 - float(combustion_eff))
        return min(1.0, oxygen_starved + incomplete * (1.0 - oxygen_starved))

    def mouth_mixture_frac(self, unburnt_frac: float, phi: float,
                           entrainment: float = TAILPIPE_ENTRAINMENT_FRAC) -> float:
        """Fuel VOLUME fraction where the flame would actually happen.

        Inside the pipe the stream is its own exhaust: no air, and any
        unburnt fuel is far above its upper flammable limit. At the
        mouth, atmospheric air entrains and dilutes it back DOWN through
        the flammable range. That descent is the whole mechanism, and it
        is why the flame stands off the end of the pipe rather than
        living in it."""
        if unburnt_frac <= 0.0:
            return 0.0
        # rough volumetric fuel fraction in the raw stream, then diluted
        raw = unburnt_frac / max(phi, 1e-6) * 0.07   # stoich hydrocarbon vol frac ~7%
        pumped = self.air_pump_kg_s
        dilution = 1.0 + entrainment + pumped * 10.0
        return raw / dilution

    def fires(self, phi: float, combustion_eff: float, fluid_spec=None,
              spark_ok: bool = True) -> tuple:
        """Does it actually light? Returns (lit, why).

        Everything here is read from the fluid's own declared limits, so
        a kit on a methanol engine behaves differently from one on
        petrol without anything being special-cased: methanol's
        flammable range is much wider, which is exactly why alcohol
        cars flame so readily."""
        if not self.firing or not spark_ok:
            return False, "not firing"
        unburnt = self.unburnt_fuel_frac(phi, combustion_eff)
        if unburnt <= 1e-4:
            return False, (f"nothing to light: phi {phi:.2f} burns essentially "
                           "everything. Enrich it, or cut the ignition on overrun")
        frac = self.mouth_mixture_frac(unburnt, phi)
        lfl = float(getattr(fluid_spec, "lfl", 0.0) or 0.014)
        ufl = float(getattr(fluid_spec, "ufl", 0.0) or 0.076)
        mie = float(getattr(fluid_spec, "min_ignition_energy_mj", 0.25) or 0.25)
        if self.spark_energy_mj < mie:
            return False, f"spark too weak: {self.spark_energy_mj:.1f} mJ against {mie:.2f} needed"
        if frac < lfl:
            return False, (f"too lean at the mouth: {frac * 100:.2f}% against a "
                           f"{lfl * 100:.1f}% lower limit -- more fuel, or less dilution")
        if frac > ufl:
            return False, (f"too rich to burn: {frac * 100:.2f}% against a "
                           f"{ufl * 100:.1f}% upper limit. This is the one people "
                           "get wrong -- an oxygen-starved stream is ABOVE the "
                           "range, not below it. Add air or let it mix further out")
        return True, (f"lit: {frac * 100:.2f}% fuel at the mouth, inside "
                      f"{lfl * 100:.1f}-{ufl * 100:.1f}%")

    def advice(self) -> str:
        if self.richness_sensor:
            return ("wideband fitted: the kit fires only when the stream is "
                    "actually in range, instead of clicking through everything else")
        return ("no richness sensor: this fires blind and will mostly click. A "
                "wideband in the same pipe tells it when there is something to light")


def demo(engine, phis=(1.0, 1.15, 1.3, 1.6)) -> str:
    import working_fluids as wf
    import combustion_efficiency as ce
    spec = wf.WORKING_FLUIDS.get(getattr(engine, "preferred_fuel_profile", "") or "")
    kit = FlameKit(firing=True, richness_sensor=True)
    out = [f"flame kit on {getattr(engine, 'identity', '?')} "
           f"({getattr(spec, 'name', 'unknown fuel')})"]
    for phi in phis:
        eff = ce.for_engine(engine, phi).efficiency
        lit, why = kit.fires(phi, eff, spec)
        out.append(f"  phi {phi:.2f}  eff {eff:.3f}  ->  "
                   f"{'FLAME' if lit else 'no  '}  {why}")
    out.append("")
    out.append("  " + kit.advice())
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    import engines
    print(demo(engines.get(sys.argv[1] if len(sys.argv) > 1 else "amc-258-jeep-i6")))
    print()
    for mat in ("magnesium", "steel"):
        p = ScrapePlate(material=mat)
        lit = p.sparking(normal_force_n=1500.0, speed_m_s=15.0)
        print(f"{mat:10s} plate at 1500 N, 15 m/s: particle "
              f"{p.particle_temp_k(1500.0, 15.0):.0f} K against a "
              f"{p.ignition_k:.0f} K ignition -> {'SPARKS' if lit else 'no sparks'}"
              + (f", burning at {p.spark_temp_k():.0f} K" if lit else ""))
        if mat == "magnesium":
            print(f"           if the plate itself lights: {p.fire_energy_j() / 1e6:.1f} MJ "
                  f"over {p.plate_fire_seconds():.0f} s")
            print(f"           {p.extinguishing_note()}")
