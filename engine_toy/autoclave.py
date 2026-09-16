"""A pressure vessel heated by condensing steam, fed from a real boiler.

An autoclave is not an oven with a lid. Its defining property is that
SATURATED STEAM SETS ITS OWN TEMPERATURE: at a given pressure, steam and
water coexist at exactly one temperature and nowhere else, so choosing
the pressure chooses the temperature and no thermostat is involved.
gas_works.STEAM_TABLE already carries that relation -- pressure,
saturation temperature, latent heat -- because the boilers needed it,
and this reads the same table rather than a second copy.

    6 bar  ->  432 K.  Not approximately. Exactly, as long as there is
    liquid water present and nothing else in the vessel.

TWO THINGS MAKE IT WORTH THE PRESSURE VESSEL

  CONDENSING HEAT TRANSFER IS ENORMOUS. Steam giving up its latent heat
  on a cold surface runs at thousands of watts per square metre per
  kelvin, against roughly twenty-five for still hot air. That is not a
  marginal improvement, it is two and a half orders of magnitude, and it
  is why a steam cycle heats a load in minutes where a furnace takes
  hours. Every kilogram of steam that condenses delivers its entire
  latent heat -- over two megajoules -- at constant temperature.

  AND IT CANNOT OVERSHOOT. A furnace at 700 K will happily take a load
  to 700 K. Saturated steam at 6 bar cannot take anything past 432 K,
  because the moment the load reaches saturation temperature the steam
  stops condensing on it. The process temperature is bounded by
  physics rather than by control, which is exactly what you want for
  anything that is damaged by overheating.

AIR IS THE ENEMY, AND THIS IS THE PART PEOPLE GET WRONG

    A chamber that still contains air is a chamber where steam is only
    PART of the total pressure. Dalton: the gauge says 6 bar but if a
    third of that is trapped air then the steam partial pressure is 4
    bar, and the vessel is at 417 K rather than 432 -- fifteen degrees
    cold, with the gauge reading correct. Worse, air pockets sit against
    the load and insulate it, because air is the thing steam was
    supposed to replace.

    This is the entire reason real autoclaves purge: a downward-
    displacement cycle pushes air out of a bottom drain with steam
    (steam is lighter than air), and a pre-vacuum cycle pulls it out
    before admitting any steam at all. The difference between them is
    how well they do on a load with pockets and blind holes, which is
    why porous-load sterilisers are all pre-vacuum.

WHAT IT IS FOR HERE. moulding.InvestmentShell needs its wax melted from
the surface inward, fast, with somewhere for the melt to go -- which is
precisely what a condensing steam cycle does and why steam dewaxing is
the real technique. It is also how you cure anything that must not be
allowed to exceed a temperature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Condensing-film coefficient, W/m2.K. Real values for filmwise
#: condensation of saturated steam on a metal surface run 5-15 kW/m2.K;
#: this is a conservative mid figure. Compare engines.py's
#: CYLINDRICAL_PIPE_HTC_W_PER_M2K of 25 for still air -- the ratio is
#: the whole argument for steam.
CONDENSING_HTC_W_PER_M2K = 8000.0

#: The same coefficient once an air film is sitting on the surface.
#: Non-condensable gas at a condensing surface is catastrophic for heat
#: transfer -- a few per cent of air can halve it -- because the steam
#: has to diffuse through the air layer to reach the wall.
AIR_BLANKETED_HTC_W_PER_M2K = 400.0

AIR_R_J_PER_KGK = 287.0
AMBIENT_K = 293.15
ATMOSPHERE_PA = 101_325.0


def saturation(pressure_pa: float) -> tuple:
    """(temperature_k, latent_heat_j_per_kg) at this ABSOLUTE pressure.

    Reads gas_works' own steam table, which the boilers already use.
    A second table here would be a second thing to keep right."""
    import gas_works
    return gas_works._steam_properties(float(pressure_pa))


@dataclass
class PurgeCycle:
    """How the air gets out, which decides what temperature is reached.

    Three real cycles, in ascending order of how well they cope with a
    load that has pockets in it.

    A pre-vacuum cycle NEEDS A PUMP, and an earlier version of this
    simply assumed one: it claimed 2% residual air with nothing in the
    plant capable of producing a vacuum. Attach a real
    compressors.VacuumPump and the residual air becomes what that pump
    can actually reach against atmosphere, which is bounded by its own
    clearance volume and cannot be wished lower. Leave it None and this
    says so rather than pretending."""
    kind: str = "downward-displacement"   # | "pre-vacuum" | "none"
    pulses: int = 3                       # pre-vacuum only
    #: the real machine doing the pulling, if there is one
    pump: object | None = None

    def pump_floor_frac(self) -> float:
        """Residual air the attached pump can genuinely reach, as a
        fraction of an atmosphere. No pump, no vacuum."""
        if self.pump is None:
            return 1.0
        try:
            return max(0.0, min(1.0, self.pump.ultimate_pa() / ATMOSPHERE_PA))
        except Exception:
            return 1.0

    def residual_air_frac(self, load_porosity: float = 0.0) -> float:
        """Fraction of the chamber's gas that is still air when heating
        starts.

        `load_porosity` is how much of the load traps air in pockets and
        blind holes -- the thing displacement cannot reach and vacuum
        can. A solid billet is 0; an investment shell with blind
        passages is high, and that is exactly the case that needs
        pre-vacuum."""
        p = max(0.0, min(1.0, load_porosity))
        if self.kind == "none":
            return 1.0
        if self.kind == "downward-displacement":
            # steam is lighter than air and pushes it out of a bottom
            # drain -- good on open loads, poor wherever air can hide
            return 0.04 + 0.55 * p
        if self.kind == "pre-vacuum":
            # each pulse removes a fixed fraction of what remains, which
            # is why the cycle is pulsed rather than one long pull...
            pulsed = (0.25 ** max(1, self.pulses)) * (1.0 + p)
            # ...but no number of pulses beats the PUMP'S OWN ULTIMATE.
            # A pump cannot pull below the pressure at which its
            # clearance gas re-expands to fill the whole stroke, so that
            # is the floor, and with no pump attached there is no
            # vacuum at all.
            return max(pulsed, self.pump_floor_frac())
        raise KeyError(f"unknown purge {self.kind!r}; declared: none, "
                       "downward-displacement, pre-vacuum")


@dataclass
class Autoclave:
    """A parametric steam autoclave.

    Everything about its behaviour follows from `working_pressure_pa`
    and the load, because that is genuinely how the machine works: set
    the pressure and the temperature is decided for you."""
    identity: str = "plant.autoclave"
    chamber_volume_m3: float = 0.25
    working_pressure_pa: float = 600_000.0      # absolute, ~6 bar
    #: the vessel's own steel, which has to be heated too and is often
    #: the larger thermal mass on a small load
    shell_mass_kg: float = 180.0
    shell_specific_heat_j_per_kgk: float = 490.0
    #: how well lagged it is: W/K to ambient once hot
    heat_loss_w_per_k: float = 12.0
    purge: PurgeCycle = field(default_factory=PurgeCycle)
    #: real vessels are rated, and a rating is a real limit
    design_pressure_pa: float = 1_000_000.0

    # --- what the pressure decides ------------------------------------
    def temperature_k(self, load_porosity: float = 0.0) -> float:
        """The temperature actually reached, after air.

        Dalton: the steam only contributes its PARTIAL pressure, so
        residual air reads on the gauge without heating anything. This
        is the number that matters and it is not the one on the dial."""
        air = self.purge.residual_air_frac(load_porosity)
        steam_partial = self.working_pressure_pa * (1.0 - air)
        t, _ = saturation(max(steam_partial, ATMOSPHERE_PA * 0.1))
        return t

    def nominal_temperature_k(self) -> float:
        """What the gauge pressure implies with no air at all."""
        return saturation(self.working_pressure_pa)[0]

    def air_penalty_k(self, load_porosity: float = 0.0) -> float:
        return self.nominal_temperature_k() - self.temperature_k(load_porosity)

    def htc_w_per_m2k(self, load_porosity: float = 0.0) -> float:
        """Condensing coefficient, spoiled by whatever air remains."""
        air = self.purge.residual_air_frac(load_porosity)
        return (CONDENSING_HTC_W_PER_M2K * (1.0 - air)
                + AIR_BLANKETED_HTC_W_PER_M2K * air)

    def purge_time_s(self) -> float:
        """How long the pre-vacuum stage takes, from the real pump.

        Not free: compressors.VacuumPump.pumpdown_s integrates the real
        speed curve, and the speed collapses as the suction approaches
        ultimate -- which is why the last decade of a pumpdown costs as
        much as all the others put together."""
        if self.purge.kind != "pre-vacuum" or self.purge.pump is None:
            return 0.0
        target = max(self.purge.pump.ultimate_pa() * 1.5,
                     ATMOSPHERE_PA * 0.02)
        return self.purge.pulses * self.purge.pump.pumpdown_s(
            self.chamber_volume_m3, to_pa=target)

    def rated(self) -> tuple:
        if self.working_pressure_pa > self.design_pressure_pa:
            return False, (f"OVER RATING: {self.working_pressure_pa / 1e5:.1f} bar "
                           f"against a {self.design_pressure_pa / 1e5:.1f} bar vessel")
        return True, f"{self.working_pressure_pa / 1e5:.1f} bar, within rating"

    # --- what it costs -------------------------------------------------
    def come_up(self, load_mass_kg: float, load_cp_j_per_kgk: float,
                load_area_m2: float, load_start_k: float = AMBIENT_K,
                load_porosity: float = 0.0,
                load_conductivity_w_per_mk: float = 50.0,
                load_thickness_m: float = 0.05) -> dict:
        """Heating the load and the vessel to temperature.

        Two masses, not one: a small load in a big vessel spends most of
        the steam heating the STEEL, which is why autoclaves are run
        full and why the first cycle of the day is the slow one."""
        t_sat = self.temperature_k(load_porosity)
        htc = self.htc_w_per_m2k(load_porosity)
        dT = max(0.0, t_sat - load_start_k)

        load_j = load_mass_kg * load_cp_j_per_kgk * dT
        shell_j = self.shell_mass_kg * self.shell_specific_heat_j_per_kgk * dT
        total_j = load_j + shell_j

        # condensing transfer into the load: Q = h.A.dT, and dT falls as
        # the load approaches saturation, so this is exponential. The
        # time constant is the honest quantity.
        ua = htc * max(load_area_m2, 1e-6)
        tau_surface_s = (load_mass_kg * load_cp_j_per_kgk) / max(ua, 1e-9)

        # THE BIOT NUMBER DECIDES WHETHER THAT IS EVEN THE RIGHT
        # QUESTION. Lumping a load into one temperature is only valid
        # when the surface resists heat more than the interior does,
        # Bi = h.L/k below about 0.1. Condensing steam has almost no
        # surface resistance, so Bi is large and the load's OWN
        # conduction is the bottleneck -- the outside reaches
        # saturation almost at once and the middle takes as long as it
        # takes. Reporting the surface time constant alone said a 25 kg
        # billet comes to temperature in twelve seconds, which is the
        # answer to a question about a very thin sheet.
        biot = (htc * max(load_thickness_m, 1e-6)
                / max(load_conductivity_w_per_mk, 1e-9))
        alpha = (load_conductivity_w_per_mk
                 / max(load_cp_j_per_kgk * (load_mass_kg
                       / max(load_area_m2 * load_thickness_m, 1e-9)), 1e-9))
        # Fourier: the interior needs Fo ~ 0.4 to be substantially through
        tau_internal_s = 0.4 * load_thickness_m ** 2 / max(alpha, 1e-12)
        # whichever resistance is larger sets the time
        tau_s = max(tau_surface_s, tau_internal_s)
        come_up_s = 4.0 * tau_surface_s if biot < 0.1 else tau_internal_s

        _, latent = saturation(self.working_pressure_pa)
        steam_kg = total_j / max(latent, 1.0)
        return {"saturation_k": t_sat,
                "nominal_k": self.nominal_temperature_k(),
                "air_penalty_k": self.air_penalty_k(load_porosity),
                "htc_w_per_m2k": htc,
                "time_constant_s": tau_s,
                "biot": biot,
                "lumped_valid": biot < 0.1,
                "surface_limited_s": 4.0 * tau_surface_s,
                "conduction_limited_s": tau_internal_s,
                "come_up_s": come_up_s,
                "load_energy_j": load_j,
                "shell_energy_j": shell_j,
                "steam_kg": steam_kg,
                "condensate_kg": steam_kg,
                "load_share": load_j / max(total_j, 1e-9)}

    def hold_steam_kg_s(self) -> float:
        """Steam to hold temperature once there: just the losses."""
        t = self.nominal_temperature_k()
        _, latent = saturation(self.working_pressure_pa)
        return self.heat_loss_w_per_k * max(0.0, t - AMBIENT_K) / max(latent, 1.0)

    def boiler_can_feed(self, boiler, dt: float = 1.0,
                        engine_rpm: float = 0.0) -> tuple:
        """Will this boiler actually keep up?

        Asks gas_works.Boiler for its own real output rather than
        assuming a supply -- a boiler that cannot make steam as fast as
        the autoclave condenses it simply loses pressure, and the
        autoclave's temperature follows the pressure down."""
        try:
            avail = boiler.steam_output_kg_s(dt, engine_rpm=engine_rpm)
        except Exception:
            try:
                avail = boiler.steam_output_kg_s(dt)
            except Exception:
                return False, "boiler does not report a steam output"
        need = self.hold_steam_kg_s()
        if avail >= need:
            return True, (f"boiler makes {avail * 1000:.1f} g/s against "
                          f"{need * 1000:.1f} g/s to hold temperature")
        return False, (f"BOILER SHORT: {avail * 1000:.1f} g/s available against "
                       f"{need * 1000:.1f} g/s needed. Pressure falls, and so does "
                       "the temperature -- saturated steam has no other option")


def dewax(shell, autoclave: Autoclave | None = None,
          wax_mass_kg: float = 0.35) -> dict:
    """Steam-dewax an investment shell, which is what this is for.

    moulding.InvestmentShell needs its wax melted from the surface
    inward with somewhere for the melt to go. A condensing steam cycle
    does exactly that and nothing else does it as well: the steam gives
    up its latent heat at the shell's outer surface, the wax against the
    ceramic melts first, and it runs out of the open cup before the bulk
    has warmed enough to expand.

    A shell is exactly the load pre-vacuum exists for -- blind passages
    full of air that displacement will never reach."""
    import moulding
    a = autoclave or Autoclave(working_pressure_pa=600_000.0,
                               purge=PurgeCycle(kind="pre-vacuum", pulses=3))
    porosity = 0.6           # a shell is mostly blind passages
    t = a.temperature_k(porosity)
    melt_ok = t >= moulding.WAX_MELT_K
    # the gradient a condensing surface imposes: the whole temperature
    # drop happens across the shell wall, because the steam side has
    # essentially no resistance
    gradient = max(0.0, t - AMBIENT_K) / max(shell.thickness_m, 1e-6)
    good, note = shell.burnout(heating_rate_k_per_s=0.35,
                               gradient_k_per_m=gradient)
    _, latent = saturation(a.working_pressure_pa)
    wax_j = wax_mass_kg * (moulding.WAX_SPECIFIC_HEAT_J_PER_KGK
                           * max(0.0, moulding.WAX_MELT_K - AMBIENT_K)
                           + moulding.WAX_LATENT_J_PER_KG)
    return {"chamber_k": t,
            "wax_melts": melt_ok,
            "gradient_k_per_m": gradient,
            "shell_ok": good,
            "note": note,
            "steam_for_wax_kg": wax_j / max(latent, 1.0),
            "wax_recovered_kg": wax_mass_kg * shell.wax_removed_frac,
            "why": ("wax is recovered and re-used -- it is the one consumable in "
                    "the chain that does not have to be thrown away")}


def report(a: Autoclave, load_porosity: float = 0.3) -> str:
    ok, rating = a.rated()
    c = a.come_up(load_mass_kg=25.0, load_cp_j_per_kgk=900.0,
                  load_area_m2=1.2, load_porosity=load_porosity)
    L = [f"autoclave {a.chamber_volume_m3 * 1000:.0f} L at {rating}",
         f"  saturated steam sets the temperature: "
         f"{c['nominal_k']:.1f} K at {a.working_pressure_pa / 1e5:.1f} bar",
         f"  purge: {a.purge.kind}, leaving "
         f"{a.purge.residual_air_frac(load_porosity) * 100:.1f}% air on a load of "
         f"porosity {load_porosity:.2f}",
         f"  ACTUAL temperature {c['saturation_k']:.1f} K -- "
         f"{c['air_penalty_k']:.1f} K COLD, with the gauge reading correct",
         f"  condensing coefficient {c['htc_w_per_m2k']:.0f} W/m2.K "
         f"(still air is 25)",
         f"  Biot {c['biot']:.1f} -- "
         + ("surface limited, lumping valid" if c['lumped_valid'] else
            "CONDUCTION LIMITED: the load's own interior is the bottleneck, "
            "not the steam"),
         f"  come-up {c['come_up_s'] / 60.0:.1f} min "
         f"(surface would say {c['surface_limited_s'] / 60.0:.1f}), "
         f"{c['steam_kg']:.2f} kg of steam",
         f"  of which {c['load_share'] * 100:.0f}% heats the LOAD and the rest heats "
         "the vessel",
         f"  holding costs {a.hold_steam_kg_s() * 1000:.1f} g/s of steam"]
    return "\n".join(L)


if __name__ == "__main__":
    for purge in ("none", "downward-displacement", "pre-vacuum"):
        print(report(Autoclave(purge=PurgeCycle(kind=purge)), load_porosity=0.5))
        print()
    import moulding
    sh = moulding.InvestmentShell(coats=7)
    d = dewax(sh)
    print("=== steam dewaxing an investment shell ===")
    print(f"  chamber {d['chamber_k']:.0f} K, wax melts: {d['wax_melts']}")
    print(f"  gradient {d['gradient_k_per_m']:.0f} K/m across the shell wall")
    print(f"  {d['note']}")
    print(f"  {d['wax_recovered_kg'] * 1000:.0f} g of wax recovered -- {d['why']}")
