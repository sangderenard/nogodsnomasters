"""Desiccant beds: how dry they get you, how long they last, and how
they are killed.

Both systems here need one, for different reasons and on different
duties, so the bed itself is one component:

  the AIR TRAIN uses a twin-tower adsorption dryer to go ULTRA-DRY.
  A refrigerated chiller cannot take air below about +3 C pressure
  dewpoint without icing its own coil -- that is a hard floor. To get
  to -40 C you must adsorb the water rather than condense it, and that
  means a desiccant bed. Two towers: one on line drying, one
  regenerating, swapping every few minutes.

  the HYDRAULIC RESERVOIR uses a small cartridge as its EMERGENCY
  vacuum break. The tank is normally sealed and blanketed on plant air
  or nitrogen; if both are gone, the tank must still be able to
  equalise or it will pull a vacuum and collapse. So a check valve
  cracks at a small negative pressure and lets air in THROUGH the
  desiccant -- a last resort, used rarely, but the thing that decides
  whether an unattended machine ends up with wet oil.

What a bed actually does, and what actually ends one:

  CAPACITY    silica gel and activated alumina hold roughly 0.15-0.20
              kg of water per kg of desiccant at the partial pressures
              a compressed-air dryer works at. That capacity is what
              a cartridge's life is, and it is finite.
  DEWPOINT    a fresh bed with plenty of unused capacity gives its
              rated outlet dewpoint; as it loads, the mass-transfer
              zone reaches the outlet and the dewpoint climbs -- which
              is why a dryer does not fail gradually and obviously, it
              holds its number and then "breaks through".
  REGENERATION a twin-tower dryer regenerates the offline tower by
              purging it with a slice of its own dried product (a real
              and unavoidable cost: typically ~15 % of flow, which is
              why a desiccant dryer is expensive to run), or by heating
              it. Each regeneration is not free: the beads attrit.
  POISONING   OIL KILLS DESICCANT, permanently. Oil aerosol coats the
              beads and blocks the pores; no regeneration recovers it.
              That is precisely why the coalescing filter belongs
              UPSTREAM of the desiccant dryer, and why a flooded
              coalescer does not merely pass oil downstream -- it
              destroys the expensive thing behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

# how much water a bed holds per kilogram of desiccant, at the partial
# pressures a compressed-air dryer sees
CAPACITY_KG_PER_KG = {"silica-gel": 0.18, "activated-alumina": 0.15, "molecular-sieve": 0.22}
# the outlet dewpoint a healthy bed of each kind will hold
RATED_DEWPOINT_K = {"silica-gel": 233.15, "activated-alumina": 233.15, "molecular-sieve": 200.15}
BREAKTHROUGH_FRAC = 0.85        # of capacity, where the mass-transfer zone reaches the outlet
PURGE_FRACTION = 0.15           # of flow, the real cost of heatless regeneration
ATTRITION_PER_REGEN = 2.0e-4    # of the bed lost per cycle: beads grind themselves down
OIL_POISON_KG_PER_KG = 0.02     # oil this heavy, relative to the bed, finishes it


@dataclass
class DesiccantBed:
    """One bed. `regenerating` beds give their capacity back; cartridges
    that cannot regenerate simply fill up and are replaced."""
    kind: str = "activated-alumina"
    mass_kg: float = 8.0
    regenerates: bool = True
    # live
    water_kg: float = 0.0
    oil_kg: float = 0.0
    attrition_frac: float = 0.0      # bead loss: permanent
    regenerations: int = 0
    online: bool = True

    @property
    def live_mass_kg(self) -> float:
        return self.mass_kg * max(0.0, 1.0 - self.attrition_frac)

    @property
    def capacity_kg(self) -> float:
        """What it can still hold: its live mass, less whatever fraction
        of its pores the oil has permanently blocked."""
        poisoned = min(1.0, self.oil_kg / max(self.live_mass_kg * OIL_POISON_KG_PER_KG, 1e-9))
        return self.live_mass_kg * CAPACITY_KG_PER_KG.get(self.kind, 0.15) * (1.0 - poisoned)

    @property
    def loading_frac(self) -> float:
        return min(1.0, self.water_kg / max(self.capacity_kg, 1e-9))

    @property
    def poisoned_frac(self) -> float:
        return min(1.0, self.oil_kg / max(self.live_mass_kg * OIL_POISON_KG_PER_KG, 1e-9))

    @property
    def spent(self) -> bool:
        return self.capacity_kg <= 1e-6 or self.loading_frac >= 1.0

    def outlet_dewpoint_k(self, inlet_dewpoint_k: float) -> float:
        """What it can actually deliver. It holds its rated number while
        there is unused bed in front of the mass-transfer zone, then
        breaks through toward the inlet condition -- which is why a
        desiccant dryer's outlet goes from excellent to useless over a
        fairly short span rather than drifting."""
        rated = RATED_DEWPOINT_K.get(self.kind, 233.15)
        if self.capacity_kg <= 1e-9:
            return inlet_dewpoint_k
        f = self.loading_frac
        if f <= BREAKTHROUGH_FRAC:
            return min(inlet_dewpoint_k, rated)
        span = max(1.0 - BREAKTHROUGH_FRAC, 1e-6)
        t = (f - BREAKTHROUGH_FRAC) / span
        return min(inlet_dewpoint_k, rated + (inlet_dewpoint_k - rated) * t)

    def adsorb(self, water_kg: float, oil_kg: float = 0.0) -> float:
        """Take up water (and, unfortunately, any oil that reaches it).
        Returns what it could NOT take."""
        self.oil_kg += max(0.0, oil_kg)
        room = max(0.0, self.capacity_kg - self.water_kg)
        took = min(max(0.0, water_kg), room)
        self.water_kg += took
        return max(0.0, water_kg - took)

    def regenerate(self) -> None:
        """Drive the water back off. The oil stays: no regeneration
        removes it, which is the whole point of keeping it out."""
        if not self.regenerates:
            return
        self.water_kg = 0.0
        self.regenerations += 1
        self.attrition_frac = min(0.9, self.attrition_frac + ATTRITION_PER_REGEN)

    def describe(self) -> str:
        bits = [f"{self.kind} {self.live_mass_kg:.1f} kg", f"{self.loading_frac * 100:3.0f}% loaded"]
        if self.poisoned_frac > 0.01:
            bits.append(f"OIL-POISONED {self.poisoned_frac * 100:.0f}%")
        if self.attrition_frac > 0.01:
            bits.append(f"{self.attrition_frac * 100:.1f}% attrited over {self.regenerations} cycles")
        if self.spent:
            bits.append("SPENT")
        return ", ".join(bits)


@dataclass
class TwinTowerDryer:
    """Two beds, one drying and one regenerating, swapping on a timer.
    This is what takes the air train from the refrigerated floor of
    +3 C down to -40 C, and what it costs is a real slice of the flow."""
    a: DesiccantBed = None
    b: DesiccantBed = None
    fitted: bool = False
    cycle_s: float = 300.0
    purge_fraction: float = PURGE_FRACTION
    _elapsed: float = 0.0
    purge_kg_s: float = 0.0

    def __post_init__(self):
        if self.a is None:
            self.a = DesiccantBed()
        if self.b is None:
            self.b = DesiccantBed()
            self.b.online = False

    @property
    def online_bed(self) -> DesiccantBed:
        return self.a if self.a.online else self.b

    @property
    def offline_bed(self) -> DesiccantBed:
        return self.b if self.a.online else self.a

    def step(self, dt: float, mass_flow_kg_s: float, inlet_water_kg_per_kg: float,
             inlet_oil_kg_per_kg: float, inlet_dewpoint_k: float) -> tuple[float, float]:
        """Returns (outlet humidity ratio, outlet dewpoint)."""
        if not self.fitted or mass_flow_kg_s <= 0.0:
            return inlet_water_kg_per_kg, inlet_dewpoint_k
        self._elapsed += dt
        if self._elapsed >= self.cycle_s:
            self._elapsed = 0.0
            self.offline_bed.regenerate()
            self.a.online, self.b.online = not self.a.online, not self.b.online
        bed = self.online_bed
        # the purge is taken off the product: real flow that never
        # reaches the user
        self.purge_kg_s = mass_flow_kg_s * self.purge_fraction
        net_flow = max(0.0, mass_flow_kg_s - self.purge_kg_s)
        out_dp = bed.outlet_dewpoint_k(inlet_dewpoint_k)
        # what it has to hold is the difference between what came in and
        # what leaves, and the oil that arrives with it goes in too
        from air_treatment import saturation_humidity_ratio, ATM_PA
        out_w = min(inlet_water_kg_per_kg, saturation_humidity_ratio(out_dp, ATM_PA * 8.0))
        bed.adsorb((inlet_water_kg_per_kg - out_w) * net_flow * dt, inlet_oil_kg_per_kg * net_flow * dt)
        return out_w, out_dp

    def describe(self) -> list[str]:
        if not self.fitted:
            return []
        return [f"  DESICCANT DRYER  online: {self.online_bed.describe()}",
                f"                   offline: {self.offline_bed.describe()}, "
                f"purge {self.purge_kg_s * 3600:.1f} kg/h ({self.purge_fraction * 100:.0f}% of flow)"]
