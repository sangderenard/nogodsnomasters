"""The low-voltage DC plant: what stores the charge and what carries it.

The interesting failures in a small DC system are not in the source.
They are in the CABLE and in the CONNECTOR, because twelve volts is a
cruel bus: a drop that would be a rounding error at mains voltage is a
tenth of everything you have. Half a volt lost in a harness is four
percent of a twelve volt system and a third of the margin between a
charged battery and a flat one. So the wire is modelled as wire --
a length of copper of a gauge, with a resistance and a temperature
coefficient -- and the connectors are modelled too, because a corroded
contact is worth more resistance than the entire cable it joins.

WHAT IS HERE

  CONDUCTOR       copper of an AWG size and a length, with the real
                  resistance and the real ampacity.
  POWER PORT      a connector. May be an exterior receptacle, which is
                  the thing that lets a solar kit be a kit rather than
                  part of the vehicle.
  HARNESS         a chain of conductors and ports from a source to a
                  load, which reports what arrives and what was lost.
  DC BATTERY      a real chemistry with a real usable depth of
                  discharge, real internal resistance, and the real
                  refusal of lithium iron phosphate to accept charge
                  below freezing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

COPPER_RESISTIVITY_OHM_M = 1.724e-8          # at 20 C
COPPER_TEMPCO_PER_C = 0.00393

#: AWG to cross-sectional area in square millimetres, and a conservative
#: continuous ampacity in free air. Real table, not interpolated.
AWG_MM2 = {0: 53.5, 2: 33.6, 4: 21.2, 6: 13.3, 8: 8.37, 10: 5.26,
           12: 3.31, 14: 2.08, 16: 1.31, 18: 0.82, 20: 0.52}
AWG_AMPACITY_A = {0: 245, 2: 190, 4: 135, 6: 101, 8: 73, 10: 55,
                  12: 41, 14: 32, 16: 22, 18: 16, 20: 11}


@dataclass
class Conductor:
    """A run of copper. Both directions: current goes out and comes
    back, so a five metre run is ten metres of wire, and forgetting
    that halves every voltage drop you calculate."""
    identity: str = "cable"
    awg: int = 10
    length_m: float = 3.0
    temperature_c: float = 25.0
    both_directions: bool = True

    @property
    def area_mm2(self) -> float:
        return AWG_MM2.get(self.awg, 5.26)

    @property
    def ampacity_a(self) -> float:
        return AWG_AMPACITY_A.get(self.awg, 55)

    @property
    def resistance_ohm(self) -> float:
        run = self.length_m * (2.0 if self.both_directions else 1.0)
        rho = COPPER_RESISTIVITY_OHM_M * (
            1.0 + COPPER_TEMPCO_PER_C * (self.temperature_c - 20.0))
        return rho * run / (self.area_mm2 * 1e-6)

    def describe(self) -> list[str]:
        return [f"  {self.identity}: {self.awg} AWG ({self.area_mm2:.2f} mm2), "
                f"{self.length_m:.1f} m run, {self.resistance_ohm * 1000:.1f} mohm, "
                f"{self.ampacity_a:.0f} A continuous"]


@dataclass
class PowerPort:
    """A connector, and whether it is even fitted.

    THE EXTERIOR PORT IS WHAT MAKES A SOLAR KIT A KIT. A panel wired
    permanently into a vehicle's loom is part of the vehicle; a panel on
    a cord into an exterior receptacle is equipment, and can be moved,
    replaced, shared or left behind. Military vehicles carry exactly
    this as a NATO slave receptacle -- a two-pin exterior socket for
    external power and jump starting -- and it is the right model for
    this because it is the same job.

    `present` is the whole point: a harness asks for the exterior port
    and gets it if the vehicle has one, and quietly routes through the
    vehicle's own loom if it does not."""
    identity: str = "port"
    exterior: bool = False
    present: bool = True
    rating_a: float = 100.0
    contact_resistance_ohm: float = 0.0005     # a clean connector
    corrosion: float = 0.0                     # 0 clean, 1 badly corroded
    healthy: bool = True

    @property
    def resistance_ohm(self) -> float:
        # A CORRODED CONTACT OUT-RESISTS THE WHOLE CABLE. This is not an
        # exaggeration for effect: half a milliohm clean, a hundred times
        # that when it is green, against a few milliohms of copper.
        return self.contact_resistance_ohm * (1.0 + 100.0 * max(0.0, min(1.0, self.corrosion)))

    def describe(self) -> list[str]:
        where = "exterior receptacle" if self.exterior else "internal connector"
        if not self.present:
            return [f"  {self.identity}: {where} -- NOT FITTED"]
        state = "clean" if self.corrosion < 0.05 else f"{self.corrosion * 100:.0f}% corroded"
        return [f"  {self.identity}: {where}, {self.rating_a:.0f} A, "
                f"{self.resistance_ohm * 1000:.2f} mohm ({state})"
                + ("" if self.healthy else "  -- FAULTED")]


@dataclass
class Harness:
    """A path from a source to a load, through cable and connectors.

    Two routes may be offered. `prefer_exterior` decides which is taken
    when both are available, and the decision is made on whether the
    port is FITTED and HEALTHY -- declared facts about the hardware, not
    a guess from what anything is called."""
    identity: str = "harness"
    conductors: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    alternate_conductors: list = field(default_factory=list)
    alternate_ports: list = field(default_factory=list)
    prefer_exterior: bool = True

    # -----------------------------------------------------------------
    def _usable(self, ports) -> bool:
        return all(p.present and p.healthy for p in ports)

    def route(self) -> dict:
        """Which way the power actually goes, and why.

        The preferred route is taken when its ports are fitted and
        healthy. Otherwise the alternate is, and the reason is
        reported rather than silently swallowed -- a system that
        quietly fell back to its second choice and never said so is a
        system whose first fault is invisible until the second one."""
        primary_is_exterior = any(p.exterior for p in self.ports)
        want_primary = self._usable(self.ports)
        if self.prefer_exterior and primary_is_exterior and want_primary:
            return {"route": "exterior", "conductors": self.conductors,
                    "ports": self.ports,
                    "reason": "exterior port is fitted and healthy"}
        if want_primary and not self.alternate_conductors:
            return {"route": "exterior" if primary_is_exterior else "onboard",
                    "conductors": self.conductors, "ports": self.ports,
                    "reason": "only route available"}
        if not want_primary and self._usable(self.alternate_ports):
            missing = [p.identity for p in self.ports if not p.present]
            faulted = [p.identity for p in self.ports if p.present and not p.healthy]
            why = ("not fitted: " + ", ".join(missing)) if missing else \
                  ("faulted: " + ", ".join(faulted))
            return {"route": "onboard", "conductors": self.alternate_conductors,
                    "ports": self.alternate_ports,
                    "reason": f"fell back to the vehicle loom ({why})"}
        if want_primary:
            return {"route": "exterior" if primary_is_exterior else "onboard",
                    "conductors": self.conductors, "ports": self.ports,
                    "reason": "preferred route"}
        return {"route": "none", "conductors": [], "ports": [],
                "reason": "no usable route: neither the port nor the loom is available"}

    def resistance_ohm(self) -> float:
        r = self.route()
        if r["route"] == "none":
            return float("inf")
        return (sum(c.resistance_ohm for c in r["conductors"])
                + sum(p.resistance_ohm for p in r["ports"]))

    def deliver(self, source_v: float, current_a: float) -> dict:
        """What arrives at the far end, and what the wire kept."""
        r = self.resistance_ohm()
        if not math.isfinite(r):
            return {"volts": 0.0, "drop_v": 0.0, "loss_w": 0.0, "delivered_w": 0.0,
                    "over_ampacity": False, **self.route()}
        drop = current_a * r
        loss = current_a * current_a * r
        route = self.route()
        cap = min([c.ampacity_a for c in route["conductors"]] +
                  [p.rating_a for p in route["ports"]] or [0.0])
        return {"volts": max(0.0, source_v - drop), "drop_v": drop, "loss_w": loss,
                "delivered_w": max(0.0, (source_v - drop) * current_a),
                "drop_fraction": drop / max(source_v, 1e-9),
                "over_ampacity": current_a > cap, "ampacity_a": cap, **route}

    def describe(self) -> list[str]:
        r = self.route()
        out = [f"  {self.identity}: routing {r['route']} -- {r['reason']}",
               f"    {self.resistance_ohm() * 1000:.1f} mohm end to end"]
        for c in r["conductors"]:
            out.extend(c.describe())
        for p in r["ports"]:
            out.extend(p.describe())
        return out


# ---------------------------------------------------------------------
#  STORAGE
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Chemistry:
    key: str
    label: str
    cell_v: float
    cells: int
    usable_depth: float          # fraction of nameplate you may actually use
    internal_r_per_ah: float     # ohms, scaled by capacity
    charge_efficiency: float
    min_charge_c: float          # below this it must not be charged
    self_discharge_per_month: float
    note: str = ""


CHEMISTRIES: tuple[Chemistry, ...] = (
    Chemistry("lifepo4", "lithium iron phosphate", 3.2, 4, 0.90, 0.012, 0.98, 0.0, 0.03,
              note="flat discharge curve, most of the nameplate is usable, and it "
                   "MUST NOT be charged below freezing -- doing so plates lithium "
                   "onto the anode and destroys it permanently"),
    Chemistry("agm", "absorbed glass mat lead-acid", 2.13, 6, 0.50, 0.030, 0.85, -20.0, 0.03,
              note="half the nameplate is usable at best, sags under load, but it "
                   "will take a charge in the cold, which in a vehicle matters"),
    Chemistry("flooded", "flooded lead-acid", 2.10, 6, 0.40, 0.040, 0.80, -20.0, 0.08,
              note="cheapest, least usable, and it needs watering and ventilation"),
)
CHEMISTRY_BY_KEY = {c.key: c for c in CHEMISTRIES}


@dataclass
class DCBattery:
    """A battery on a circuit, with a real usable fraction.

    The nameplate amp-hours are a lie in the specific sense that you
    cannot have them: half of a lead-acid battery is reserve you must
    not touch if you want it to live, and even lithium keeps a little
    back. `usable_wh` is what is actually yours."""
    identity: str = "battery"
    chemistry: str = "lifepo4"
    capacity_ah: float = 100.0
    # HOW MANY PACKS IN SERIES. A chemistry fixes its cell voltage and
    # how many cells make a nominal block; a 24 V system is two of those
    # blocks, not a different chemistry. Leaving this out gave a 24 V
    # machine a 12.8 V lithium pack that could never have held an
    # isolator's input up.
    series_packs: int = 1
    state_of_charge: float = 0.80
    temperature_c: float = 20.0
    healthy: bool = True

    @property
    def chem(self) -> Chemistry:
        return CHEMISTRY_BY_KEY[self.chemistry]

    @property
    def nominal_v(self) -> float:
        return self.chem.cell_v * self.chem.cells * max(1, self.series_packs)

    @property
    def terminal_v(self) -> float:
        """Open-circuit terminal voltage at this charge. Lithium is
        nearly flat, which is why you cannot judge its charge by
        voltage; lead-acid sags steadily, which is why you can."""
        c = self.chem
        if c.key == "lifepo4":
            span = 0.08
        else:
            span = 0.22
        return self.nominal_v * (1.0 - span * (1.0 - self.state_of_charge))

    @property
    def internal_r_ohm(self) -> float:
        return self.chem.internal_r_per_ah / max(self.capacity_ah, 1e-6) * 100.0

    @property
    def usable_wh(self) -> float:
        return self.capacity_ah * self.nominal_v * self.chem.usable_depth

    @property
    def stored_wh(self) -> float:
        """Energy above the floor the chemistry will not let you cross."""
        floor = 1.0 - self.chem.usable_depth
        return max(0.0, (self.state_of_charge - floor)) * self.capacity_ah * self.nominal_v

    def may_charge(self) -> tuple[bool, str]:
        if not self.healthy:
            return False, "battery faulted"
        if self.temperature_c < self.chem.min_charge_c:
            return False, (f"{self.temperature_c:.0f} C is below the "
                           f"{self.chem.min_charge_c:.0f} C charge limit for "
                           f"{self.chem.label}")
        if self.state_of_charge >= 0.999:
            return False, "full"
        return True, "accepting charge"

    def step(self, dt_s: float, *, charge_w: float = 0.0, load_w: float = 0.0) -> dict:
        """Move the charge on by one tick."""
        ok, why = self.may_charge()
        accepted = charge_w * self.chem.charge_efficiency if ok else 0.0
        net_w = accepted - load_w
        wh = net_w * dt_s / 3600.0
        # self-discharge, which is small but never zero
        wh -= (self.capacity_ah * self.nominal_v
               * self.chem.self_discharge_per_month / (30.0 * 24.0 * 3600.0) * dt_s)
        floor = 1.0 - self.chem.usable_depth
        soc = self.state_of_charge + wh / max(self.capacity_ah * self.nominal_v, 1e-9)
        clipped_empty = soc < floor
        self.state_of_charge = max(floor, min(1.0, soc))
        return {"accepted_w": accepted, "rejected": not ok, "why": why,
                "net_w": net_w, "state_of_charge": self.state_of_charge,
                "stored_wh": self.stored_wh, "flat": clipped_empty}

    def describe(self) -> list[str]:
        ok, why = self.may_charge()
        return [f"  {self.identity}: {self.capacity_ah:.0f} Ah {self.chem.label} at "
                f"{self.nominal_v:.1f} V nominal, {self.terminal_v:.2f} V terminal",
                f"    {self.state_of_charge * 100:.0f}% charged = "
                f"{self.stored_wh:.0f} Wh usable of {self.usable_wh:.0f} "
                f"({self.chem.usable_depth * 100:.0f}% of nameplate is yours)",
                f"    charge: {why}"]
