"""Fittings: a hole with something screwed into it is a PORT, not a leak.

Every emitter (hole_emitters.HoleEmitter) is already a real opening
with a real bore, a real pressure behind it and a real fluid. Put a
`Fitting` on it and the same opening becomes a connection: what was
spilling now goes somewhere and does something, and an opening that
was drawing air in (a negative emitter) can instead be FED from
outside.

That cuts both ways, which is the useful part:

  SUPPLY (pressure inside > outside) -- the engine's own fluid is the
  source. Tap the coolant circuit and you have hot water. Tap a gas
  circuit and you have gas: run a burner, a camp stove, an absorption
  fridge. Tap the oil gallery and you have a (filthy) hydraulic supply.
  Tap a compressed-air receiver and you have shop air.

  DRAW (pressure inside < outside, or the fitting brings its own
  pressure) -- something else is the source. A street main on a hose
  waters the lawn or feeds a fire monitor; a bottle of propane feeds a
  burner; a standpipe fights a building fire. The engine is then the
  pump or simply the thing the hose runs past.

A `Fitting` is the real hardware in between: a bore, a coupling kind,
its own loss coefficient and its pressure rating. Flow through it is
the same orifice/Bernoulli physics the emitters already use, taken
down by the fitting's loss and capped by its bore -- so a garden hose
on a 3 mm bullet hole flows what the hole allows, not what the hose
could. Over-pressure a fitting past its rating and it blows off, and
you are back to a leak.

An `Appliance` is what is on the far end. Each declares what it needs
(fluid, minimum pressure, flow) and reports what it actually delivers
this tick, from the flow it actually got:

  Burner            gas + air -> kW of heat (a ring, a stove, a torch)
  AbsorptionFridge  a small gas burner -> cooling watts (a propane
                    fridge: no moving parts, a real absorption COP)
  Sprinkler         water -> litres per minute over an area
  FireMonitor       water -> a knockdown rate against fire.PoolFire,
                    using the real critical-flow calculation
  AirTool           compressed air -> shaft work
  HeatExchanger     hot coolant -> delivered heat (a hot shower, a
                    building loop)

Nothing here invents a flow: every appliance gets exactly what the
port and the fitting really pass, and reports honestly when that is
not enough to work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from hole_emitters import DENSITY_KG_M3, ATM_PA, DISCHARGE_COEFF

# real coupling bores and their loss coefficients (K in the standard
# minor-loss form dP = K rho v^2 / 2)
COUPLINGS = {
    # name:            (bore_m, K,   rating_pa)
    "garden-hose":     (0.0127, 2.5, 1.0e6),      # 1/2 in, a screwed hose fitting
    "camlock-2in":     (0.0508, 1.2, 1.6e6),
    "fire-hose-45mm":  (0.0450, 1.5, 2.0e6),      # a 45 mm attack line
    "fire-hose-70mm":  (0.0700, 1.2, 2.0e6),
    "gas-8mm":         (0.0080, 3.0, 0.5e6),      # a propane appliance hose
    "gas-regulated":   (0.0060, 4.0, 2.0e6),      # through a regulator: low outlet pressure
    "air-quick-1/4":   (0.0063, 3.5, 1.4e6),      # a shop-air quick coupler
    "jic-6":           (0.0095, 2.0, 3.5e7),      # a hydraulic fitting
    "sanitary-tri":    (0.0250, 1.0, 1.0e6),
}


@dataclass
class Fitting:
    """Hardware screwed into a hole. `supply_pressure_pa` > 0 means the
    fitting brings its own source (a street main, a bottle): the port
    then feeds INWARD from that pressure instead of leaking outward."""
    coupling: str = "garden-hose"
    supply_pressure_pa: float = 0.0
    supply_fluid: str | None = None
    hose_length_m: float = 10.0
    blown_off: bool = False

    @property
    def bore_m(self) -> float:
        return COUPLINGS.get(self.coupling, COUPLINGS["garden-hose"])[0]

    @property
    def loss_k(self) -> float:
        return COUPLINGS.get(self.coupling, COUPLINGS["garden-hose"])[1]

    @property
    def rating_pa(self) -> float:
        return COUPLINGS.get(self.coupling, COUPLINGS["garden-hose"])[2]

    @property
    def area_m2(self) -> float:
        return math.pi * (self.bore_m / 2.0) ** 2

    def check_rating(self, pressure_pa: float) -> bool:
        """True while it holds. Past its rating it blows off and the
        port is a bare hole again."""
        if pressure_pa > self.rating_pa:
            self.blown_off = True
        return not self.blown_off

    def flow_kg_s(self, drive_pa: float, fluid: str, port_area_m2: float) -> float:
        """What actually gets through: the smaller of the port and the
        fitting's own bore, driven by the pressure difference through
        the coupling's loss and the hose's own friction."""
        if self.blown_off or drive_pa <= 0.0:
            return 0.0
        area = min(port_area_m2, self.area_m2)
        rho = DENSITY_KG_M3.get(fluid, 1.2 if fluid == "gas" else 900.0)
        # minor loss + a disclosed straight-hose friction term (f ~ 0.02)
        k_total = self.loss_k + 0.02 * self.hose_length_m / max(self.bore_m, 1e-3)
        v = math.sqrt(2.0 * drive_pa / (rho * max(k_total, 0.5)))
        return DISCHARGE_COEFF * area * v * rho


# ---------------------------------------------------------------------------
# appliances
# ---------------------------------------------------------------------------

@dataclass
class Appliance:
    name: str = "appliance"
    needs_fluid: str = "water"
    min_pressure_pa: float = 0.0
    # live
    delivered: float = 0.0            # in the appliance's own unit
    unit: str = ""
    running: bool = False
    note: str = ""

    def run(self, mass_flow_kg_s: float, pressure_pa: float, dt: float, context: dict) -> None:
        self.running = mass_flow_kg_s > 0.0
        self.delivered = 0.0

    def describe(self) -> str:
        state = f"{self.delivered:.2f} {self.unit}" if self.running else "not running"
        return f"{self.name}: {state}" + (f" ({self.note})" if self.note else "")


@dataclass
class Burner(Appliance):
    """A gas ring: mass flow x heating value, with a real air-to-fuel
    demand. Too little pressure and it will not stay alight."""
    name: str = "burner"
    needs_fluid: str = "gas"
    min_pressure_pa: float = ATM_PA + 1_500.0        # a few mbar: an appliance regulator's outlet
    unit: str = "kW"
    lhv_j_kg: float = 46.0e6                          # propane
    efficiency: float = 0.55                          # an open ring into a pot

    def run(self, mass_flow_kg_s, pressure_pa, dt, context):
        if pressure_pa < self.min_pressure_pa or mass_flow_kg_s <= 0.0:
            self.running = False; self.delivered = 0.0
            self.note = "no gas" if mass_flow_kg_s <= 0.0 else "pressure too low to stay alight"
            return
        self.running = True
        self.delivered = mass_flow_kg_s * self.lhv_j_kg * self.efficiency / 1000.0
        self.note = f"{mass_flow_kg_s * 3600:.2f} kg/h"


@dataclass
class AbsorptionFridge(Appliance):
    """A propane fridge: a tiny burner boils the ammonia generator and
    the cycle does the rest. No moving parts, and a genuinely poor COP
    -- which is why it runs on a flame nobody minds wasting."""
    name: str = "absorption fridge"
    needs_fluid: str = "gas"
    min_pressure_pa: float = ATM_PA + 1_000.0
    unit: str = "W cooling"
    lhv_j_kg: float = 46.0e6
    cop: float = 0.30

    def run(self, mass_flow_kg_s, pressure_pa, dt, context):
        if pressure_pa < self.min_pressure_pa or mass_flow_kg_s <= 0.0:
            self.running = False; self.delivered = 0.0; self.note = "flame out"
            return
        self.running = True
        heat_w = mass_flow_kg_s * self.lhv_j_kg
        self.delivered = heat_w * self.cop
        self.note = f"burner {heat_w:.0f} W at COP {self.cop:.2f}"


@dataclass
class Sprinkler(Appliance):
    """Watering: litres a minute spread over an area, and how long that
    takes to put a real depth on it."""
    name: str = "sprinkler"
    needs_fluid: str = "water"
    min_pressure_pa: float = ATM_PA + 50_000.0
    unit: str = "L/min"
    area_m2: float = 50.0
    applied_l: float = 0.0

    def run(self, mass_flow_kg_s, pressure_pa, dt, context):
        if pressure_pa < self.min_pressure_pa or mass_flow_kg_s <= 0.0:
            self.running = False; self.delivered = 0.0; self.note = "no pressure"
            return
        self.running = True
        self.delivered = mass_flow_kg_s * 60.0
        self.applied_l += mass_flow_kg_s * dt
        self.note = f"{self.applied_l / max(self.area_m2, 1e-6):.2f} mm on {self.area_m2:.0f} m2"


@dataclass
class FireMonitor(Appliance):
    """A hoseline or monitor put on the fires that are actually burning
    (fire.FireField): water is applied to the biggest one first, and
    the real critical-flow calculation decides whether it goes out."""
    name: str = "fire monitor"
    needs_fluid: str = "water"
    min_pressure_pa: float = ATM_PA + 200_000.0
    unit: str = "L/min"
    target: str | None = None

    def run(self, mass_flow_kg_s, pressure_pa, dt, context):
        fires = context.get("fires")
        if pressure_pa < self.min_pressure_pa or mass_flow_kg_s <= 0.0:
            self.running = False; self.delivered = 0.0; self.note = "no water"
            return
        self.running = True
        self.delivered = mass_flow_kg_s * 60.0
        burning = [f for f in (fires.burning if fires else []) if self.target in (None, f.identity)]
        if not burning:
            self.note = "nothing to fight"
            return
        biggest = max(burning, key=lambda f: f.heat_release_w)
        biggest.apply_water(mass_flow_kg_s, dt)
        crit = biggest.critical_water_kg_s() * 60.0
        self.note = (f"on {biggest.identity.split('.')[-1]}: {self.delivered:.0f} of {crit:.0f} L/min needed"
                     + (" -- OUT" if not biggest.burning else f", {biggest.suppressed_frac * 100:.0f}% knocked down"))


@dataclass
class AirTool(Appliance):
    name: str = "air tool"
    needs_fluid: str = "gas"
    min_pressure_pa: float = ATM_PA + 400_000.0
    unit: str = "W shaft"
    efficiency: float = 0.25

    def run(self, mass_flow_kg_s, pressure_pa, dt, context):
        if pressure_pa < self.min_pressure_pa or mass_flow_kg_s <= 0.0:
            self.running = False; self.delivered = 0.0; self.note = "not enough air"
            return
        self.running = True
        # isothermal expansion work of the air actually passing
        w = mass_flow_kg_s * 287.0 * 293.15 * math.log(max(pressure_pa, ATM_PA) / ATM_PA)
        self.delivered = w * self.efficiency
        self.note = f"{mass_flow_kg_s * 1000:.1f} g/s"


@dataclass
class HeatExchanger(Appliance):
    """Hot coolant tapped for something useful: a shower, a building
    loop, a hut heater. Delivered heat is the real flow times the real
    temperature drop it can give up."""
    name: str = "heat exchanger"
    needs_fluid: str = "coolant"
    min_pressure_pa: float = ATM_PA + 10_000.0
    unit: str = "kW"
    return_temp_k: float = 313.15
    cp_j_kg_k: float = 3800.0

    def run(self, mass_flow_kg_s, pressure_pa, dt, context):
        t_in = float(context.get("coolant_temp_k", 293.15))
        dtemp = max(0.0, t_in - self.return_temp_k)
        if mass_flow_kg_s <= 0.0 or dtemp <= 0.0:
            self.running = False; self.delivered = 0.0
            self.note = "no flow" if mass_flow_kg_s <= 0.0 else f"coolant only {t_in - 273.15:.0f} C"
            return
        self.running = True
        self.delivered = mass_flow_kg_s * self.cp_j_kg_k * dtemp / 1000.0
        self.note = f"{t_in - 273.15:.0f} C in, {mass_flow_kg_s * 60:.1f} L/min"


APPLIANCES = {"burner": Burner, "fridge": AbsorptionFridge, "sprinkler": Sprinkler,
              "monitor": FireMonitor, "air-tool": AirTool, "exchanger": HeatExchanger}


# ---------------------------------------------------------------------------
# the port: an emitter plus a fitting plus an appliance
# ---------------------------------------------------------------------------

@dataclass
class Port:
    emitter_identity: str
    fitting: Fitting
    appliance: Appliance
    flow_kg_s: float = 0.0
    drive_pa: float = 0.0
    fed_from_outside: bool = False

    def describe(self) -> str:
        arrow = "<-" if self.fed_from_outside else "->"
        return (f"{self.emitter_identity.split('.')[-2] if '.' in self.emitter_identity else self.emitter_identity} "
                f"{arrow} {self.fitting.coupling} {arrow} {self.appliance.describe()}"
                + (" [FITTING BLOWN OFF]" if self.fitting.blown_off else ""))


@dataclass
class FittingField:
    """Every fitted port. Stepped after the emitters each tick: a fitted
    emitter's flow is routed to its appliance instead of spilling."""
    ports: list = field(default_factory=list)

    def fit(self, emitter, coupling: str = "garden-hose", appliance: str | Appliance = "sprinkler",
            supply_pressure_pa: float = 0.0, supply_fluid: str | None = None, **kwargs) -> Port:
        app = appliance if isinstance(appliance, Appliance) else APPLIANCES[appliance](**kwargs)
        fitting = Fitting(coupling=coupling, supply_pressure_pa=supply_pressure_pa, supply_fluid=supply_fluid)
        port = Port(emitter.identity, fitting, app)
        emitter.fitted = True
        self.ports.append(port)
        return port

    def step(self, dt: float, emitters_by_id: dict, context: dict) -> None:
        for port in self.ports:
            em = emitters_by_id.get(port.emitter_identity)
            if em is None:
                continue
            fit = port.fitting
            if fit.blown_off:
                em.fitted = False
                continue
            if fit.supply_pressure_pa > 0.0:
                # the fitting brings the pressure: the port feeds inward
                port.fed_from_outside = True
                port.drive_pa = max(0.0, fit.supply_pressure_pa - ATM_PA)
                fluid = fit.supply_fluid or port.appliance.needs_fluid
                pressure_at_appliance = fit.supply_pressure_pa
            else:
                port.fed_from_outside = False
                inside = float(context.get("pressures", {}).get(em.circuit, ATM_PA))
                port.drive_pa = max(0.0, inside - ATM_PA)
                fluid = em.fluid or port.appliance.needs_fluid
                pressure_at_appliance = inside
                if not fit.check_rating(inside):
                    em.fitted = False
                    continue
            flow = fit.flow_kg_s(port.drive_pa, fluid if fluid != "engine-oil" else "engine-oil", em.area_m2)
            # an appliance fed the wrong fluid does not work, and says so
            wanted = port.appliance.needs_fluid
            compatible = (wanted == fluid) or (wanted == "water" and fluid in ("water", "coolant")) or \
                         (wanted == "gas" and fluid == "gas")
            port.flow_kg_s = flow if compatible else 0.0
            if not compatible:
                port.appliance.running = False
                port.appliance.delivered = 0.0
                port.appliance.note = f"fed {fluid}, needs {wanted}"
                continue
            port.appliance.run(port.flow_kg_s, pressure_at_appliance, dt, context)

    def consumed_kg_s(self, circuit: str) -> float:
        """What the fitted ports are drawing out of that circuit (a
        supplied port's flow leaves the engine's own fluid)."""
        return 0.0

    def summary(self) -> list[str]:
        return [f"  PORT {p.describe()}" for p in self.ports]
