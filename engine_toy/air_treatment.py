"""Compressed-air treatment as a real stage chain, and the gunk it exists to stop.

Air out of a compressor is hot, saturated, oily and dusty. Everything
downstream of it -- reservoirs, valves, manifolds, brake chambers, air
tools -- is destroyed slowly by exactly those three things, and the
whole treatment train exists to take them out in a particular order.
That order is the point, so this models it as an ordered chain of
`Stage`s, each doing one real thing to one real air stream:

    compressor
      -> AFTERCOOLER (electric-fan air-to-air): drop most of the heat
      -> CHILLER (refrigerant-to-air, off the AC loop): drop it further,
         to a low pressure dewpoint -- this is where most of the water
         actually leaves
      -> WATER SEPARATOR (centrifugal): throw the condensed liquid out
      -> COALESCING FILTER: take out the oil aerosol the separator
         cannot, and the fine solids with it
      -> PARTICULATE FILTER: the remaining dust
      -> REHEATER: put some sensible heat back. This removes NO water
         at all -- it lowers the RELATIVE humidity of air that is
         already dry, so the tank and the lines downstream sit well
         above their pressure dewpoint and nothing re-condenses in
         them. (Reheating a wet stream would be useless; reheating a
         dried one is what stops the wet tank being wet.) The heat for
         it is FREE and comes from the right place: a real refrigerated
         dryer runs the incoming hot wet air and the outgoing cold dry
         air through one air-to-air recuperator, so the hot side is
         pre-cooled (which shrinks the chiller's duty) and the cold
         side is reheated by exactly that heat. An electric element is
         only a trim, for when there is no hot side to borrow from.
      -> WET TANK -> the isolated reserve set

The physics is ordinary psychrometrics, all of it standard:

  saturation pressure   Magnus/Tetens, p_sat(T)
  humidity ratio        w = 0.622 p_v / (p - p_v), kg water per kg dry air
  compression           T2 = T1 (p2/p1)^((g-1)/g) corrected by the
                        compressor's isentropic efficiency; the water
                        RIDES ALONG -- compression does not remove any,
                        it just raises the pressure that the same water
                        has to stay vapour against, which is why
                        compressed air condenses when a naturally humid
                        intake never would
  condensation          cooling to T with the stream at p: everything
                        above w_sat(T, p) becomes liquid
  pressure dewpoint     the temperature the treated stream would have
                        to reach before it condenses again -- the one
                        number that says whether the system is dry

Contamination carried per kilogram of dry air:
  water     from the intake's own humidity, as above
  oil       the compressor's real carryover (a lubricated reciprocating
            machine ~25 mg/m3, a screw ~3, an oil-free ~0), which a
            coalescing filter reduces by its own real efficiency
  dust      ambient dust past the intake filter's efficiency

`Contaminant` is what survives the chain and reaches the tank, and
`GunkLedger` accumulates it downstream: oil and water emulsify into
sludge, dust becomes abrasive, and a system run with a missing or
saturated filter genuinely fills up with it. `gunk_effects` turns that
into the same kind of consequence node_effects already applies, so
"nobody serviced the filters" ends in stuck valves and worn tools
rather than in a number nobody reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

ATM_PA = 101_325.0
R_DRY_AIR = 287.055
CP_AIR_J_KG_K = 1005.0
GAMMA_AIR = 1.4
WATER_LATENT_J_KG = 2.45e6
# how much dust ordinary air carries, by mass -- a clean shop is at the
# bottom of this, a quarry or a dirt road at the top
AMBIENT_DUST_MG_M3 = {"clean": 0.05, "workshop": 0.5, "roadside": 2.0, "quarry": 10.0}
# what a compressor puts into its own discharge, by type (mg per m3 of
# free air delivered) -- the real reason an oil-free machine costs more
OIL_CARRYOVER_MG_M3 = {"belt-piston": 25.0, "electric-piston": 25.0, "screw": 3.0,
                       "starting-air-compressor": 15.0, "oil-free": 0.0}


def saturation_pressure_pa(temp_k: float) -> float:
    """Magnus/Tetens saturation vapour pressure over water."""
    t_c = temp_k - 273.15
    if t_c < -40.0:
        return 0.0
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


def humidity_ratio(temp_k: float, pressure_pa: float, relative_humidity: float) -> float:
    """kg of water vapour per kg of dry air at these conditions."""
    p_v = max(0.0, min(1.0, relative_humidity)) * saturation_pressure_pa(temp_k)
    p_v = min(p_v, pressure_pa * 0.999)
    return 0.622 * p_v / max(pressure_pa - p_v, 1.0)


def saturation_humidity_ratio(temp_k: float, pressure_pa: float) -> float:
    return humidity_ratio(temp_k, pressure_pa, 1.0)


def dewpoint_k(humidity_ratio_kg_kg: float, pressure_pa: float) -> float:
    """The temperature at which a stream of this humidity, at this
    pressure, starts to condense -- the PRESSURE dewpoint."""
    w = max(humidity_ratio_kg_kg, 1e-12)
    p_v = pressure_pa * w / (0.622 + w)
    if p_v <= 0.0:
        return 0.0
    # invert Magnus
    ln = math.log(max(p_v, 1e-6) / 610.94)
    t_c = 243.04 * ln / (17.625 - ln)
    return t_c + 273.15


def compression_discharge_k(intake_k: float, pressure_ratio: float, isentropic_efficiency: float = 0.72) -> float:
    """Real discharge temperature: the isentropic rise divided by the
    machine's own efficiency (the losses all land in the air as heat)."""
    pr = max(pressure_ratio, 1.0)
    ideal_rise = intake_k * (pr ** ((GAMMA_AIR - 1.0) / GAMMA_AIR) - 1.0)
    return intake_k + ideal_rise / max(isentropic_efficiency, 0.2)


@dataclass
class Contaminant:
    """What a kilogram of dry air is carrying."""
    water_kg_per_kg: float = 0.0       # vapour still in the stream
    liquid_kg_per_kg: float = 0.0      # condensed, not yet separated out
    oil_kg_per_kg: float = 0.0
    dust_kg_per_kg: float = 0.0

    def copy(self) -> "Contaminant":
        return Contaminant(self.water_kg_per_kg, self.liquid_kg_per_kg, self.oil_kg_per_kg, self.dust_kg_per_kg)


@dataclass
class Stream:
    """The air between two stages."""
    temp_k: float = 293.15
    pressure_pa: float = ATM_PA
    mass_flow_kg_s: float = 0.0        # dry air
    carried: Contaminant = field(default_factory=Contaminant)

    @property
    def relative_humidity(self) -> float:
        w_sat = saturation_humidity_ratio(self.temp_k, self.pressure_pa)
        return min(1.0, self.carried.water_kg_per_kg / max(w_sat, 1e-12))

    @property
    def dewpoint_k(self) -> float:
        return dewpoint_k(self.carried.water_kg_per_kg, self.pressure_pa)

    def copy(self) -> "Stream":
        return Stream(self.temp_k, self.pressure_pa, self.mass_flow_kg_s, self.carried.copy())


@dataclass
class StageResult:
    name: str
    kind: str
    inlet: Stream
    outlet: Stream
    heat_removed_w: float = 0.0
    condensate_kg_s: float = 0.0
    drained_kg_s: float = 0.0
    oil_removed_kg_s: float = 0.0
    dust_removed_kg_s: float = 0.0
    note: str = ""


@dataclass
class Stage:
    """One real piece of treatment hardware.

    kind:
      "aftercooler"   air-to-air with a fan: approaches ambient to
                      within `approach_k`, and its capacity is limited
                      by the fan actually running
      "chiller"       refrigerant-to-air: pulls the stream down to
                      `target_k` so long as the refrigerant loop is
                      actually delivering (cooling_w)
      "separator"     centrifugal: throws out `efficiency` of the
                      LIQUID present (it cannot touch vapour)
      "coalescing"    takes out `efficiency` of the oil aerosol and
                      most fine solids; blinds slowly as it loads
      "particulate"   takes out `efficiency` of the dust
      "reheater"      adds heat, lowering relative humidity downstream
      "tank"          a volume; whatever is still liquid drops out here
                      and stays as gunk if nobody drains it
    """
    name: str
    kind: str
    efficiency: float = 0.9
    approach_k: float = 12.0           # aftercooler: how close to ambient it gets
    target_k: float = 276.15           # chiller: the pressure dewpoint it aims for (+3 C)
    reheat_k: float = 15.0             # reheater: how much sensible heat it puts back
    capacity_w: float = 8000.0
    fitted: bool = True
    powered: bool = True               # its fan/refrigerant/heater actually working
    # service state: a filter that is never changed stops working
    loading_kg: float = 0.0
    capacity_kg: float = 0.25          # how much it holds before it is blinded
    drained: bool = True               # a tank/separator with an open drain
    # reheater only: the hot compressor-discharge stream it borrows from
    # (set each tick by the chain), how well the recuperator couples the
    # two, and whether an electric trim element is fitted at all
    recuperator_source_k: float = 293.15
    effectiveness: float = 0.55
    electric_trim: bool = True
    recuperator_recovered_k: float = 0.0
    electric_k: float = 0.0

    @property
    def saturation_frac(self) -> float:
        return min(1.0, self.loading_kg / max(self.capacity_kg, 1e-6))

    # how loaded a filter gets before it stops holding what it catches
    FLOOD_FRAC = 0.80

    @property
    def live_efficiency(self) -> float:
        """A loading filter does NOT quietly pass more and more: a
        coalescer holds its rated efficiency and simply costs more and
        more pressure drop as it fills, right up until it floods, and
        then it re-entrains what it was holding and its efficiency
        collapses. A particulate element behaves the same way. So the
        real symptom of a neglected filter is first a rising pressure
        drop (`pressure_drop_pa`) and then, abruptly, dirty air."""
        if self.kind not in ("coalescing", "particulate"):
            return self.efficiency
        if not self.powered:
            return 0.0
        if self.saturation_frac <= self.FLOOD_FRAC:
            return self.efficiency
        over = (self.saturation_frac - self.FLOOD_FRAC) / max(1.0 - self.FLOOD_FRAC, 1e-6)
        return self.efficiency * max(0.0, 1.0 - over)

    @property
    def pressure_drop_pa(self) -> float:
        """A filter's real cost while it is still working: clean drop
        rising roughly with the square of how loaded it is."""
        if self.kind not in ("coalescing", "particulate"):
            return 0.0
        return 14_000.0 * (0.15 + 0.85 * self.saturation_frac ** 2)

    def apply(self, s: Stream, dt: float, ambient_k: float, cooling_available_w: float = 0.0) -> StageResult:
        out = s.copy()
        r = StageResult(self.name, self.kind, s.copy(), out)
        if not self.fitted:
            r.note = "not fitted"
            return r
        m = s.mass_flow_kg_s
        if self.kind in ("aftercooler", "chiller"):
            if self.kind == "aftercooler":
                target = ambient_k + self.approach_k if self.powered else max(ambient_k + 45.0, s.temp_k - 5.0)
                r.note = "fan running" if self.powered else "no fan: little rejection"
            else:
                if cooling_available_w <= 0.0 or not self.powered:
                    r.note = "no refrigerant: passing through"
                    return r
                target = self.target_k
            target = max(target, ambient_k - 40.0)
            new_t = min(s.temp_k, max(target, s.temp_k - self._max_drop(m, cooling_available_w if self.kind == "chiller" else self.capacity_w)))
            out.temp_k = new_t
            r.heat_removed_w = m * CP_AIR_J_KG_K * (s.temp_k - new_t)
            # whatever the cooled stream can no longer hold falls out
            w_sat = saturation_humidity_ratio(new_t, out.pressure_pa)
            if out.carried.water_kg_per_kg > w_sat:
                dropped = out.carried.water_kg_per_kg - w_sat
                out.carried.water_kg_per_kg = w_sat
                out.carried.liquid_kg_per_kg += dropped
                r.condensate_kg_s = dropped * m
                r.heat_removed_w += r.condensate_kg_s * WATER_LATENT_J_KG
            if self.kind == "chiller":
                r.note = f"to {new_t - 273.15:.0f} C"
        elif self.kind == "separator":
            got = out.carried.liquid_kg_per_kg * self.efficiency
            out.carried.liquid_kg_per_kg -= got
            # a centrifugal separator throws out entrained oil with the water
            oil_got = out.carried.oil_kg_per_kg * self.efficiency * 0.4
            out.carried.oil_kg_per_kg -= oil_got
            r.drained_kg_s = got * m
            r.oil_removed_kg_s = oil_got * m
            if not self.drained:
                self.loading_kg += (got + oil_got) * m * dt
                r.note = "drain shut: filling up"
        elif self.kind in ("coalescing", "particulate"):
            eff = self.live_efficiency
            if self.kind == "coalescing":
                oil_got = out.carried.oil_kg_per_kg * eff
                out.carried.oil_kg_per_kg -= oil_got
                liq_got = out.carried.liquid_kg_per_kg * eff
                out.carried.liquid_kg_per_kg -= liq_got
                dust_got = out.carried.dust_kg_per_kg * eff * 0.8
                out.carried.dust_kg_per_kg -= dust_got
                r.oil_removed_kg_s = oil_got * m
                r.drained_kg_s = liq_got * m
            else:
                dust_got = out.carried.dust_kg_per_kg * eff
                out.carried.dust_kg_per_kg -= dust_got
                oil_got = 0.0
                r.dust_removed_kg_s = dust_got * m
            # A coalescing element DRAINS the oil it coalesces, continuously,
            # through its own float or timer drain -- which is why a real
            # element lasts thousands of hours instead of filling up in a
            # fortnight. Only the solids it captures actually load it, and
            # only a blocked drain makes the liquid load it too.
            if self.kind == "coalescing":
                r.drained_kg_s += r.oil_removed_kg_s if self.drained else 0.0
                self.loading_kg += (r.dust_removed_kg_s + (0.0 if self.drained else r.oil_removed_kg_s)) * dt
            else:
                self.loading_kg += r.dust_removed_kg_s * dt
            r.dust_removed_kg_s = r.dust_removed_kg_s or dust_got * m
            if self.saturation_frac > self.FLOOD_FRAC:
                r.note = f"{self.saturation_frac * 100:.0f}% loaded -- FLOODED, passing contamination"
            elif self.saturation_frac > 0.4:
                r.note = f"{self.saturation_frac * 100:.0f}% loaded, {self.pressure_drop_pa / 1000:.0f} kPa drop"
        elif self.kind == "reheater":
            # a recuperator moves heat from the hot inlet stream to this
            # cold outlet one: what it can give is limited by how much
            # hotter that inlet actually is, times its effectiveness
            available_k = max(0.0, self.recuperator_source_k - s.temp_k) * self.effectiveness
            trim_k = self.reheat_k if (self.powered and self.electric_trim) else 0.0
            got_k = min(self.reheat_k, available_k + trim_k)
            if got_k > 0.0:
                out.temp_k = s.temp_k + got_k
                r.heat_removed_w = -m * CP_AIR_J_KG_K * got_k
                self.recuperator_recovered_k = min(got_k, available_k)
                self.electric_k = max(0.0, got_k - self.recuperator_recovered_k)
                how = "recuperated" if self.electric_k <= 0.01 else f"recuperated + {self.electric_k:.0f} K electric"
                r.note = f"+{got_k:.0f} K {how}: RH {out.relative_humidity * 100:.0f}%"
            else:
                self.recuperator_recovered_k = 0.0
                self.electric_k = 0.0
                r.note = "no heat to recover"
        elif self.kind == "tank":
            # anything still liquid arriving in a tank falls out of the
            # stream there -- but a tank only HOLDS what it can hold. An
            # undrained one fills to its own capacity and from then on
            # the water simply carries over into the lines downstream,
            # which is exactly how a neglected wet tank ruins everything
            # after it rather than quietly storing a swimming pool.
            got = out.carried.liquid_kg_per_kg
            out.carried.liquid_kg_per_kg = 0.0
            if self.drained:
                r.drained_kg_s = got * m
            else:
                room = max(0.0, self.capacity_kg - self.loading_kg)
                held = min(got * m * dt, room)
                self.loading_kg += held
                carried_over = got * m * dt - held
                r.drained_kg_s = held / max(dt, 1e-9)
                if carried_over > 0.0:
                    out.carried.liquid_kg_per_kg = carried_over / max(m * dt, 1e-12)
                    r.note = f"FULL ({self.loading_kg * 1000:.0f} g) -- carrying water over downstream"
                else:
                    r.note = f"undrained: {self.loading_kg * 1000:.0f} g standing"
        return r

    def _max_drop(self, m_kg_s: float, capacity_w: float) -> float:
        if m_kg_s <= 0.0:
            return 0.0
        return capacity_w / (m_kg_s * CP_AIR_J_KG_K)


# the chain the user asked for, in order
def default_chain(oil_free: bool = False) -> list[Stage]:
    return [
        Stage("aftercooler", "aftercooler", approach_k=12.0, capacity_w=9000.0),
        Stage("chiller", "chiller", target_k=276.15, capacity_w=6000.0),
        Stage("water_separator", "separator", efficiency=0.92),
        Stage("coalescing_filter", "coalescing", efficiency=0.999, capacity_kg=0.20, drained=True),
        Stage("particulate_filter", "particulate", efficiency=0.95, capacity_kg=0.35),
        Stage("reheater", "reheater", reheat_k=15.0, capacity_w=2500.0),
        Stage("wet_tank", "tank", capacity_kg=3.0),     # a 12 L tank can stand a few litres before it carries over
    ]


@dataclass
class GunkLedger:
    """What got past the treatment and is now living in the system.

    Oil and water do not stay separate: agitated together in a tank
    they emulsify into the sludge that actually blocks things, so the
    sludge here is the paired mass, and whatever is left over stays as
    free oil or free water. Dust in oil is the abrasive that wears
    valve seats and tool vanes."""
    water_kg: float = 0.0
    oil_kg: float = 0.0
    dust_kg: float = 0.0
    sludge_kg: float = 0.0
    abrasive_kg: float = 0.0
    corrosion_frac: float = 0.0        # standing water in a steel tank

    # What a system actually RETAINS. Most of what passes through blows
    # straight out of the tools and the drains; what stays is what
    # settles in the low points, and that reaches a standing level
    # rather than growing without bound. Above it, the rest carries
    # through -- which is what ruins the tools rather than the tank.
    retention_kg: float = 4.0

    def add(self, water_kg: float, oil_kg: float, dust_kg: float, dt: float) -> None:
        self.water_kg += max(0.0, water_kg)
        self.oil_kg += max(0.0, oil_kg)
        self.dust_kg += max(0.0, dust_kg)
        # emulsify what can pair up
        pair = min(self.water_kg, self.oil_kg) * 0.5
        self.water_kg -= pair
        self.oil_kg -= pair
        self.sludge_kg += pair * 2.0
        # dust bound into oil or sludge is what actually abrades
        bound = min(self.dust_kg, (self.oil_kg + self.sludge_kg) * 0.1)
        self.dust_kg -= bound
        self.abrasive_kg += bound
        # free water on steel corrodes, slowly and only while it is there
        if self.water_kg > 1e-6:
            self.corrosion_frac = min(1.0, self.corrosion_frac + 2e-5 * dt * min(1.0, self.water_kg * 100.0))
        # everything above what the low points hold simply blows through
        held = self.water_kg + self.oil_kg + self.sludge_kg
        if held > self.retention_kg:
            shed = (held - self.retention_kg) / held
            self.water_kg *= 1.0 - shed
            self.oil_kg *= 1.0 - shed
            self.sludge_kg *= 1.0 - shed

    @property
    def total_kg(self) -> float:
        return self.water_kg + self.oil_kg + self.dust_kg + self.sludge_kg + self.abrasive_kg

    def describe(self) -> str:
        return (f"water {self.water_kg * 1000:.1f} g, oil {self.oil_kg * 1000:.1f} g, "
                f"sludge {self.sludge_kg * 1000:.1f} g, abrasive {self.abrasive_kg * 1000:.2f} g"
                + (f", corrosion {self.corrosion_frac * 100:.1f}%" if self.corrosion_frac > 0.001 else ""))


@dataclass
class AirTreatment:
    """The whole train, stepped with the compressor's own real delivery."""
    stages: list = field(default_factory=default_chain)
    compressor_kind: str = "belt-piston"
    dust_environment: str = "workshop"
    intake_filter_efficiency: float = 0.99
    ambient_rh: float = 0.6
    gunk: GunkLedger = field(default_factory=GunkLedger)
    # live
    results: list = field(default_factory=list)
    delivered: Stream = field(default_factory=Stream)
    condensate_total_kg: float = 0.0
    discharge_k: float = 293.15
    # what the cold side was at last tick, for the recuperator
    last_cold_side_k: float = 293.15

    def stage(self, name: str):
        return next((s for s in self.stages if s.name == name), None)

    def step(self, dt: float, mass_flow_kg_s: float, intake_k: float, tank_pressure_pa: float,
             ambient_k: float = 293.15, chiller_cooling_w: float = 0.0) -> Stream:
        """Push one tick of the compressor's real delivery through the
        chain and bank whatever reaches the tank still dirty."""
        pr = max(1.0, tank_pressure_pa / ATM_PA)
        self.discharge_k = compression_discharge_k(intake_k, pr)
        # what the intake handed the compressor, now at tank pressure:
        # the same water per kilogram of dry air, a real oil carryover
        # from this machine, and the dust its filter let through
        w_in = humidity_ratio(intake_k, ATM_PA, self.ambient_rh)
        free_air_m3_per_kg = R_DRY_AIR * intake_k / ATM_PA
        oil_kg_per_kg = OIL_CARRYOVER_MG_M3.get(self.compressor_kind, 25.0) * 1e-6 * free_air_m3_per_kg
        dust_kg_per_kg = (AMBIENT_DUST_MG_M3.get(self.dust_environment, 0.5) * 1e-6 * free_air_m3_per_kg
                          * (1.0 - max(0.0, min(1.0, self.intake_filter_efficiency))))
        s = Stream(temp_k=self.discharge_k, pressure_pa=tank_pressure_pa, mass_flow_kg_s=max(0.0, mass_flow_kg_s),
                   carried=Contaminant(water_kg_per_kg=w_in, oil_kg_per_kg=oil_kg_per_kg, dust_kg_per_kg=dust_kg_per_kg))
        # compression itself condenses nothing until something cools it,
        # but the stream can already be over saturation at tank pressure
        w_sat_hot = saturation_humidity_ratio(s.temp_k, s.pressure_pa)
        if s.carried.water_kg_per_kg > w_sat_hot:
            s.carried.liquid_kg_per_kg += s.carried.water_kg_per_kg - w_sat_hot
            s.carried.water_kg_per_kg = w_sat_hot
        self.results = []
        rh = self.stage("reheater")
        if rh is not None:
            rh.recuperator_source_k = s.temp_k
            # the hot side gives up what the cold side takes: the
            # recuperator pre-cools the incoming air before the
            # aftercooler and chiller ever see it
            pre_cool_k = max(0.0, (s.temp_k - self.last_cold_side_k) * rh.effectiveness) if rh.fitted else 0.0
            if pre_cool_k > 0.0:
                s.temp_k -= min(pre_cool_k, s.temp_k - ambient_k)
        for st in self.stages:
            r = st.apply(s, dt, ambient_k, chiller_cooling_w)
            self.results.append(r)
            self.condensate_total_kg += (r.condensate_kg_s + r.drained_kg_s) * dt
            s = r.outlet
        self.delivered = s
        cold = next((r.inlet.temp_k for r in self.results if r.kind == "reheater"), s.temp_k)
        self.last_cold_side_k = cold
        # what still reaches the reserve set is what fills it with gunk
        m = s.mass_flow_kg_s
        self.gunk.add((s.carried.liquid_kg_per_kg + max(0.0, s.carried.water_kg_per_kg
                                                        - saturation_humidity_ratio(ambient_k, s.pressure_pa))) * m * dt,
                      s.carried.oil_kg_per_kg * m * dt, s.carried.dust_kg_per_kg * m * dt, dt)
        return s

    def summary(self) -> list[str]:
        out = []
        for r in self.results:
            bits = []
            if r.heat_removed_w:
                bits.append(f"{-r.heat_removed_w / 1000:.2f} kW in" if r.heat_removed_w < 0 else f"{r.heat_removed_w / 1000:.2f} kW out")
            if r.condensate_kg_s:
                bits.append(f"{r.condensate_kg_s * 3600:.2f} kg/h condensed")
            if r.drained_kg_s:
                bits.append(f"{r.drained_kg_s * 3600:.2f} kg/h drained")
            if r.oil_removed_kg_s:
                bits.append(f"{r.oil_removed_kg_s * 3.6e6:.1f} mg/h oil")
            if r.dust_removed_kg_s:
                bits.append(f"{r.dust_removed_kg_s * 3.6e6:.1f} mg/h dust")
            out.append(f"  {r.name:<20s} {r.outlet.temp_k - 273.15:6.1f} C  " + ", ".join(bits) + (f"  [{r.note}]" if r.note else ""))
        d = self.delivered
        out.append(f"  DELIVERED            {d.temp_k - 273.15:6.1f} C  pressure dewpoint {d.dewpoint_k - 273.15:.1f} C, "
                   f"RH {d.relative_humidity * 100:.0f}%, oil {d.carried.oil_kg_per_kg * 1e9:.1f} ug/kg")
        out.append(f"  GUNK                 {self.gunk.describe()}")
        return out


# Which real parts each contaminant actually ruins, and how. This is
# not a blanket "the air system is dirty" -- sludge collects in the
# vessels and the valve bodies it can settle in, abrasive dust wears
# the things with moving seats and sliding vanes, and standing water
# corrodes the steel vessels. A part that has none of those exposures
# is not harmed by any of it.
GUNK_EXPOSURE = {
    # pattern            -> (sludge, abrasive, corrosion) sensitivity
    "wet_tank":            (1.0, 0.2, 1.0),
    "reserve_tank":        (1.0, 0.2, 1.0),
    "brake_reservoir":     (0.9, 0.2, 1.0),
    "manifold":            (1.0, 0.6, 0.3),
    "protection_valve":    (1.0, 0.9, 0.2),
    "treadle_valve":       (1.0, 0.9, 0.2),
    "isolation_valve":     (1.0, 0.9, 0.2),
    "changeover_valve":    (1.0, 0.9, 0.2),
    "drain":               (1.0, 0.4, 0.4),
    "brake_chambers":      (0.6, 0.5, 0.6),
    "idle_assist":         (0.8, 0.9, 0.1),
    "regulator":           (0.9, 0.9, 0.2),
}


def gunk_damage_by_part(gunk: GunkLedger, identities, tank_capacity_l: float = 20.0) -> dict:
    """identity -> {mechanism: severity 0..1} for every part the
    contamination genuinely reaches, so the damage lands on the real
    hardware rather than on a global number."""
    e = gunk_effects(gunk, tank_capacity_l)
    fouling = 1.0 - e["flow_factor"]
    wear = 1.0 - e["tool_efficiency_factor"]
    out = {}
    for ident in identities:
        for pat, (s_w, a_w, c_w) in GUNK_EXPOSURE.items():
            if pat in ident:
                sev = {}
                if fouling * s_w > 0.05:
                    sev["sludge"] = min(1.0, fouling * s_w)
                if wear * a_w > 0.05:
                    sev["abrasive-wear"] = min(1.0, wear * a_w)
                if gunk.corrosion_frac * c_w > 0.02:
                    sev["corrosion"] = min(1.0, gunk.corrosion_frac * c_w)
                if sev:
                    out[ident] = sev
                break
    return out


# Which real parts each contaminant actually ruins, and how. This is
# not a blanket "the air system is dirty" -- sludge collects in the
# vessels and the valve bodies it can settle in, abrasive dust wears
# the things with moving seats and sliding vanes, and standing water
# corrodes the steel vessels. A part that has none of those exposures
# is not harmed by any of it.
GUNK_EXPOSURE = {
    # pattern            -> (sludge, abrasive, corrosion) sensitivity
    "wet_tank":            (1.0, 0.2, 1.0),
    "reserve_tank":        (1.0, 0.2, 1.0),
    "brake_reservoir":     (0.9, 0.2, 1.0),
    "manifold":            (1.0, 0.6, 0.3),
    "protection_valve":    (1.0, 0.9, 0.2),
    "treadle_valve":       (1.0, 0.9, 0.2),
    "isolation_valve":     (1.0, 0.9, 0.2),
    "changeover_valve":    (1.0, 0.9, 0.2),
    "drain":               (1.0, 0.4, 0.4),
    "brake_chambers":      (0.6, 0.5, 0.6),
    "idle_assist":         (0.8, 0.9, 0.1),
    "regulator":           (0.9, 0.9, 0.2),
}


def gunk_damage_by_part(gunk: GunkLedger, identities, tank_capacity_l: float = 20.0) -> dict:
    """identity -> {mechanism: severity 0..1} for every part the
    contamination genuinely reaches, so the damage lands on the real
    hardware rather than on a global number."""
    e = gunk_effects(gunk, tank_capacity_l)
    fouling = 1.0 - e["flow_factor"]
    wear = 1.0 - e["tool_efficiency_factor"]
    out = {}
    for ident in identities:
        for pat, (s_w, a_w, c_w) in GUNK_EXPOSURE.items():
            if pat in ident:
                sev = {}
                if fouling * s_w > 0.05:
                    sev["sludge"] = min(1.0, fouling * s_w)
                if wear * a_w > 0.05:
                    sev["abrasive-wear"] = min(1.0, wear * a_w)
                if gunk.corrosion_frac * c_w > 0.02:
                    sev["corrosion"] = min(1.0, gunk.corrosion_frac * c_w)
                if sev:
                    out[ident] = sev
                break
    return out


def gunk_effects(gunk: GunkLedger, tank_capacity_l: float = 20.0) -> dict:
    """What the accumulated contamination is actually doing, as the same
    kind of multiplier node_effects works in. Real mechanisms:
    sludge blocks ports and sticks valve spools; abrasive dust wears
    seats and vanes so tools lose power and valves stop sealing;
    standing water corrodes a steel tank from the inside, which is how
    reservoirs eventually burst rather than leak."""
    volume_l = max(tank_capacity_l, 0.1)
    fouling = min(1.0, gunk.sludge_kg / (volume_l * 0.004))     # kg of sludge per litre before it is a problem
    wear = min(1.0, gunk.abrasive_kg / (volume_l * 0.0002))
    return {
        "valve_response_factor": max(0.15, 1.0 - 0.85 * fouling),     # spools stick
        "flow_factor": max(0.2, 1.0 - 0.8 * fouling),                  # ports block
        "tool_efficiency_factor": max(0.3, 1.0 - 0.7 * wear),          # vanes and seats worn
        "seal_factor": max(0.0, 1.0 - wear),                           # valves stop sealing
        "vessel_strength_factor": max(0.3, 1.0 - 0.7 * gunk.corrosion_frac),
        "notes": [n for n in (
            f"air system fouled: {gunk.sludge_kg * 1000:.0f} g of oil/water sludge" if fouling > 0.15 else None,
            f"abrasive dust in the air system: {gunk.abrasive_kg * 1000:.1f} g" if wear > 0.15 else None,
            f"reservoir corroding from standing water ({gunk.corrosion_frac * 100:.0f}%)" if gunk.corrosion_frac > 0.02 else None,
        ) if n],
    }
