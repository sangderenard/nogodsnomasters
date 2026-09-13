"""The refrigerant loop, and everything hung off it.

An engine's air-conditioning compressor is a real vapour-compression
machine with real spare capacity, and once it is there the useful
thing is to hang more than a cabin evaporator on it. Here it feeds:

    compressor (belt clutch off the crank)
      -> CONDENSER (in the cooling stack, fan-assisted)
      -> RECEIVER-DRIER
      -> expansion valve per load:
           CABIN evaporator
           COMPRESSED-AIR CHILLER   (air_treatment's chiller stage)
           HYDRAULIC OIL CHILLER    (the hydraulic tank's cooler)
      -> back to the compressor

The physics is the ordinary cycle, kept honest rather than elaborate:

  saturation        a Clausius-Clapeyron style fit for R134a gives the
                    suction and discharge pressures from the evaporating
                    and condensing temperatures -- which is what the
                    pressure switches actually read
  condensing temp   ambient plus the condenser's own approach, and the
                    approach gets worse without airflow (a stopped fan
                    is what drives a system to its high-pressure cutout)
  COP               Carnot between evaporating and condensing
                    temperatures times a real machine efficiency
  capacity          shaft power in x COP, split across whichever
                    evaporators are actually calling

The CLUTCH is the control the user asked for, and it is the real one:
it pulls in only when the main switch is on, the loop has pressure
(a low-pressure switch stops it running with the charge leaked out --
which is what protects the compressor from running dry) and the high
side is below its cutout (a blocked condenser or a stopped fan trips
it), and it cycles on the evaporator's own thermostat. A leaked-out
loop simply never engages, and says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

ATM_PA = 101_325.0

# Pressure switch settings, real R134a service values
LOW_PRESSURE_CUTOUT_PA = 200_000.0       # below this the charge is gone: do not run
LOW_PRESSURE_CUTIN_PA = 230_000.0
HIGH_PRESSURE_CUTOUT_PA = 2_700_000.0    # above this something is wrong on the high side
HIGH_PRESSURE_CUTIN_PA = 2_000_000.0
CYCLE_EFFICIENCY = 0.45                  # of Carnot, a real automotive loop
CONDENSER_APPROACH_K = 12.0              # over ambient, with airflow
CONDENSER_APPROACH_NO_AIRFLOW_K = 45.0


def r134a_saturation_pa(temp_k: float) -> float:
    """Saturation pressure of R134a. A two-parameter Clausius-Clapeyron
    fit pinned to two real service points (0 C -> 293 kPa, 50 C ->
    1318 kPa absolute), which is the range a vehicle loop lives in."""
    t = max(temp_k, 200.0)
    # ln p = A - B/T through the two pinned points
    t1, p1 = 273.15, 293_000.0
    t2, p2 = 323.15, 1_318_000.0
    b = math.log(p2 / p1) / (1.0 / t1 - 1.0 / t2)
    a = math.log(p1) + b / t1
    return math.exp(a - b / t)


def r134a_saturation_temp_k(pressure_pa: float) -> float:
    t1, p1 = 273.15, 293_000.0
    t2, p2 = 323.15, 1_318_000.0
    b = math.log(p2 / p1) / (1.0 / t1 - 1.0 / t2)
    a = math.log(p1) + b / t1
    return b / (a - math.log(max(pressure_pa, 1.0)))


@dataclass
class CoolingLoad:
    """One evaporator on the loop."""
    name: str
    evaporating_k: float = 275.15      # what this load wants to evaporate at
    demand_w: float = 0.0              # how much it is calling for right now
    calling: bool = True
    delivered_w: float = 0.0
    priority: float = 1.0

    def describe(self) -> str:
        if not self.calling:
            return f"{self.name}: off"
        return f"{self.name}: {self.delivered_w / 1000:.2f} of {self.demand_w / 1000:.2f} kW at {self.evaporating_k - 273.15:.0f} C"


@dataclass
class RefrigerantLoop:
    """The whole circuit and its control."""
    charge_frac: float = 1.0           # 1.0 = correctly charged; a leak takes this down
    main_switch: bool = True           # the operator's master chiller switch
    # How much shaft power this compressor absorbs per rad/s. A fixed-
    # displacement machine's absorbed power rises with speed, so this is
    # its RATING divided by the speed that rating was quoted at -- both
    # of which the graph already declares on the compressor node
    # (`rated_w`, `reference_omega_rad_s`). Never guess this: the graph
    # is the single declaration of how big the compressor is.
    displacement_w_per_rad_s: float = 3.0
    max_shaft_w: float = 3500.0
    loads: list = field(default_factory=list)
    condenser_airflow: float = 1.0     # 0 = fan dead / blocked core
    # live state
    clutch_engaged: bool = False
    suction_pa: float = 0.0
    discharge_pa: float = 0.0
    evaporating_k: float = 275.15
    condensing_k: float = 313.15
    cop: float = 0.0
    shaft_w: float = 0.0
    cooling_w: float = 0.0
    lockout: str = ""
    _hp_tripped: bool = False

    @classmethod
    def from_compressor_node(cls, node: dict, **kw) -> "RefrigerantLoop":
        """Build a loop sized off the compressor the graph actually
        declares, rather than off a guess."""
        rated = float(node.get("rated_w", 0.0) or 0.0)
        ref_omega = float(node.get("reference_omega_rad_s", 0.0) or 150.0)
        loop = cls(**kw)
        if rated > 0.0:
            loop.displacement_w_per_rad_s = rated / max(ref_omega, 1.0)
            # a compressor driven above the speed its rating was quoted
            # at genuinely absorbs more, but not without limit
            loop.max_shaft_w = rated * 2.0
        return loop

    def load(self, name: str) -> CoolingLoad | None:
        return next((l for l in self.loads if l.name == name), None)

    def step(self, dt: float, compressor_omega_rad_s: float, ambient_k: float) -> None:
        """One tick: read the pressures, decide the clutch, split what
        capacity there is across whatever is calling."""
        # the pressures the switches actually see. A loop that has lost
        # charge sits low on both sides; a loop that cannot reject heat
        # climbs on the high side.
        self.condensing_k = ambient_k + (CONDENSER_APPROACH_K
                                         + (CONDENSER_APPROACH_NO_AIRFLOW_K - CONDENSER_APPROACH_K)
                                         * (1.0 - max(0.0, min(1.0, self.condenser_airflow))))
        calling = [l for l in self.loads if l.calling and l.demand_w > 0.0]
        self.evaporating_k = min((l.evaporating_k for l in calling), default=280.15)
        charge = max(0.0, min(1.0, self.charge_frac))
        # a partly charged loop cannot hold its suction pressure
        self.suction_pa = r134a_saturation_pa(self.evaporating_k) * charge
        self.discharge_pa = r134a_saturation_pa(self.condensing_k) * (0.4 + 0.6 * charge)

        # --- the clutch control, in the order the real switches sit ---
        self.lockout = ""
        if not self.main_switch:
            self.lockout = "master switch off"
        elif not calling:
            self.lockout = "nothing calling"
        elif self.suction_pa < LOW_PRESSURE_CUTOUT_PA:
            self.lockout = f"low-pressure cutout ({self.suction_pa / 1000:.0f} kPa): charge low"
        if self.discharge_pa >= HIGH_PRESSURE_CUTOUT_PA:
            self._hp_tripped = True
        elif self.discharge_pa <= HIGH_PRESSURE_CUTIN_PA:
            self._hp_tripped = False
        if self._hp_tripped and not self.lockout:
            self.lockout = f"high-pressure cutout ({self.discharge_pa / 1000:.0f} kPa): no condenser airflow?"
        self.clutch_engaged = not self.lockout and compressor_omega_rad_s > 5.0

        if not self.clutch_engaged:
            self.shaft_w = 0.0
            self.cooling_w = 0.0
            self.cop = 0.0
            for l in self.loads:
                l.delivered_w = 0.0
            return

        # --- capacity ---
        self.shaft_w = min(self.max_shaft_w, self.displacement_w_per_rad_s * compressor_omega_rad_s)
        lift = max(2.0, self.condensing_k - self.evaporating_k)
        self.cop = CYCLE_EFFICIENCY * self.evaporating_k / lift
        self.cooling_w = self.shaft_w * self.cop
        # split it: everyone gets their share of demand, by priority
        total = sum(l.demand_w * l.priority for l in calling)
        remaining = self.cooling_w
        for l in self.loads:
            if l not in calling:
                l.delivered_w = 0.0
                continue
            share = (l.demand_w * l.priority / total) if total > 0.0 else 0.0
            l.delivered_w = min(l.demand_w, self.cooling_w * share)
            remaining -= l.delivered_w
        # anything left over goes to whoever can still use it
        if remaining > 1.0:
            for l in calling:
                spare = l.demand_w - l.delivered_w
                if spare > 0.0:
                    take = min(spare, remaining)
                    l.delivered_w += take
                    remaining -= take

    def delivered(self, name: str) -> float:
        l = self.load(name)
        return l.delivered_w if l is not None else 0.0

    def describe(self) -> list[str]:
        head = (f"  REFRIGERANT  {'CLUTCH IN' if self.clutch_engaged else 'clutch out'}  "
                f"suction {self.suction_pa / 1000:5.0f} kPa  discharge {self.discharge_pa / 1000:6.0f} kPa  "
                f"charge {self.charge_frac * 100:3.0f}%")
        if self.lockout:
            head += f"  [{self.lockout}]"
        out = [head]
        if self.clutch_engaged:
            out.append(f"    {self.shaft_w / 1000:.2f} kW shaft x COP {self.cop:.2f} = {self.cooling_w / 1000:.2f} kW cooling, "
                       f"evap {self.evaporating_k - 273.15:.0f} C / cond {self.condensing_k - 273.15:.0f} C")
        out.extend(f"    {l.describe()}" for l in self.loads)
        return out


@dataclass
class HydraulicTank:
    """A hydraulic reservoir with its own chiller on the loop. Oil that
    runs hot oxidises and thins out; oil that is chilled below its
    pour behaves like treacle -- so a real chiller is thermostatted,
    not simply on."""
    capacity_l: float = 60.0
    oil_l: float = 55.0
    temp_k: float = 293.15
    target_k: float = 318.15           # 45 C: hot enough to be thin, cool enough to live
    chiller_fitted: bool = True
    return_heat_w: float = 0.0         # what the work the system is doing puts back in
    water_kg: float = 0.0              # condensation and ingress: the real hydraulic killer
    particulate_kg: float = 0.0
    cp_j_kg_k: float = 1900.0
    density_kg_m3: float = 870.0

    @property
    def mass_kg(self) -> float:
        return self.oil_l / 1000.0 * self.density_kg_m3

    @property
    def demand_w(self) -> float:
        """How hard the chiller is being asked to work: only above the
        setpoint, proportional to how far above it is."""
        if not self.chiller_fitted:
            return 0.0
        over = self.temp_k - self.target_k
        return max(0.0, min(12000.0, over * 600.0))

    @property
    def water_ppm(self) -> float:
        return self.water_kg / max(self.mass_kg, 1e-6) * 1e6

    def step(self, dt: float, cooling_w: float, ambient_k: float) -> None:
        m = self.mass_kg
        if m <= 0.0:
            return
        passive_w = (self.temp_k - ambient_k) * 8.0        # the tank's own skin
        net_w = self.return_heat_w - cooling_w - passive_w
        self.temp_k += net_w * dt / (m * self.cp_j_kg_k)
        self.temp_k = max(ambient_k - 5.0, self.temp_k)

    def describe(self) -> str:
        note = ""
        if self.water_ppm > 500.0:
            note = f"  WATER {self.water_ppm:.0f} ppm -- above the 500 ppm that starts etching bearings"
        elif self.water_ppm > 200.0:
            note = f"  water {self.water_ppm:.0f} ppm"
        return (f"  HYDRAULIC TANK {self.oil_l:.1f}/{self.capacity_l:.0f} L at {self.temp_k - 273.15:.1f} C "
                f"(target {self.target_k - 273.15:.0f} C, calling {self.demand_w / 1000:.2f} kW){note}")
