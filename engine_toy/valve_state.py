"""Valve/spring/rocker STATE as flat arrays -- one row per valve, every
quantity a numpy vector, so the whole valvetrain of a V12 is a few
array operations per tick and the game still gets a per-cylinder,
per-valve answer.

Each valve carries the nominal build (from the engine's own
LifterSpring and the cylinder layout's valve ports) PLUS its own
machine error: a seeded, per-valve draw within real shop tolerances
(spring rate a few percent, free length ~1.5%, installed height a few
tenths of a mm, lash a few hundredths, rocker ratio ~2%, seat
concentricity). Two engines of the same model differ; two cylinders
of the same engine differ -- which is exactly why real engines have a
"best" and a "worst" cylinder. The seed is the engine identity, so
the error is stable for that engine unless a part is replaced.

Derived per valve (vectorized):
  seat_load  = rate * (free - sag - installed + shim)
  open_load  = seat_load + rate * lift_eff
  lift_eff   = lift * (ratio / ratio_nominal) - max(0, lash - lash_spec)
  float_rpm  = engine's nominal float speed * sqrt(seat_load / seat_nominal)
               (float is a resonance the seat preload holds off; a
               tired spring floats earlier)
  seat_leak  = leak from seat recession / concentricity error, plus a
               low-seat-load term (a valve that isn't held shut
               leaks compression)
Reduced per cylinder:
  breathing_frac   product over the cylinder's intake valves of
                   sqrt(lift_eff / lift) -- flow follows curtain area
  float_rpm        the lowest float speed of any valve on the cylinder
  leak_frac        sum of seat leaks on the cylinder
  float_penalty(rpm)  per cylinder: 1 below its own float speed,
                   falling past it -- the piston loop multiplies its
                   combustion strength by breathing * (1 - leak) *
                   float_penalty, so every cylinder fires uniquely
Aging is vectorized too: sag grows with cycles (rpm) and block
temperature per cylinder; seat recession grows with cycles on the
exhaust valves (they run hot).

CARBON: every valve carries a deposit fraction (0 clean .. 1 fully
coked). Deposits form from oil and unburned fuel under the real
conditions that make them -- a cold engine, a rich mixture, idling
and light load -- and burn off when the valve runs hot enough under
load (exhaust valves first; intake valves only in a direct-injection
engine, where no fuel washes them, which is the real GDI/diesel
intake-coking problem; a port-injected or carbureted intake valve is
washed clean by its own fuel). Effects: an intake deposit chokes the
throat (breathing), a seat deposit holds the valve off its seat
(leak), the added mass lowers the float speed, and a glowing exhaust
deposit is a hot spot (hotspot_risk, for the knock/pre-ignition side).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

# real shop tolerances, disclosed (one-sigma of the per-valve draw)
TOL_RATE_FRAC = 0.04
TOL_FREE_LENGTH_FRAC = 0.015
TOL_INSTALLED_M = 0.0003
TOL_LASH_M = 0.00004
TOL_RATIO_FRAC = 0.02
TOL_SEAT_CONCENTRICITY_M = 0.00003
LASH_SPEC_M = 0.00025
SAG_LIMIT_FRAC = 0.05
SEAT_RECESSION_RATE_M_PER_CYCLE = 2.0e-12     # exhaust seats: ~0.2 mm per 1e8 cycles hot
# carbon: full coking after ~40 h of cold rich idling; burn-off within
# minutes above the burn temperature under load (disclosed rates)
CARBON_GROWTH_PER_S = 1.0 / (40.0 * 3600.0)
CARBON_BURN_PER_S = 1.0 / 300.0
CARBON_BURN_TEMP_K = 620.0
CARBON_COLD_TEMP_K = 340.0


def _seed(identity: str) -> int:
    return int(hashlib.sha1(identity.encode()).hexdigest()[:8], 16)


@dataclass
class ValveState:
    n_cyl: int
    cylinder: np.ndarray        # int, 1-based cylinder number per valve
    is_intake: np.ndarray       # bool
    rate: np.ndarray            # N/m
    free_len: np.ndarray        # m
    installed: np.ndarray       # m
    lift: np.ndarray            # m, nominal
    ratio: np.ndarray           # rocker ratio actual
    ratio_nominal: np.ndarray
    lash: np.ndarray            # m
    concentricity: np.ndarray   # m, seat runout error
    mass_kg: float
    float_rpm_nominal: float
    sag: np.ndarray = field(default=None)
    shim: np.ndarray = field(default=None)
    recession: np.ndarray = field(default=None)
    carbon: np.ndarray = field(default=None)     # 0 clean .. 1 fully coked
    identity: str = ""

    def __post_init__(self):
        n = len(self.cylinder)
        if self.sag is None: self.sag = np.zeros(n)
        if self.shim is None: self.shim = np.zeros(n)
        if self.recession is None: self.recession = np.zeros(n)
        if self.carbon is None: self.carbon = np.zeros(n)

    # ---- build ----
    @classmethod
    def from_engine(cls, engine, layout=None, seed: int | None = None) -> "ValveState":
        from cylinder_ports import cylinder_port_layout
        if layout is None:
            layout = cylinder_port_layout(engine)
        spring = engine.lifter_spring
        rate_nom = float(spring.spring_rate_n_per_mm) * 1000.0
        mass = max(float(spring.valvetrain_mass_g), 1.0) / 1000.0
        cyl, intake, lift, bore = [], [], [], []
        for g, ports in layout:
            if g.kind not in ("spark-piston", "compression-piston") or g.valvetrain == "none":
                continue
            for p in ports:
                if p.port_kind in ("intake-valve-port", "exhaust-valve-port"):
                    cyl.append(g.number); intake.append(p.fluid_role == "intake")
                    lift.append(p.radius_m * 0.55); bore.append(g.bore_m)
        n = len(cyl)
        rng = np.random.default_rng(_seed(engine.identity) if seed is None else seed)
        cyl = np.array(cyl, dtype=int); intake = np.array(intake, dtype=bool)
        lift = np.array(lift); bore = np.array(bore) if n else np.zeros(0)
        free_nom = bore * 0.55
        inst_nom = free_nom * 0.82
        ratio_nom = np.full(n, 1.6 if getattr(engine.architecture, "cylinders", 0) else 1.0)
        # a 0 rate on a two-stroke with no valves: n == 0 and everything is empty
        vs = cls(
            n_cyl=max(1, int(getattr(engine.architecture, "cylinders", 1) or 1)),
            cylinder=cyl, is_intake=intake,
            rate=rate_nom * (1.0 + TOL_RATE_FRAC * rng.standard_normal(n)),
            free_len=free_nom * (1.0 + TOL_FREE_LENGTH_FRAC * rng.standard_normal(n)),
            installed=inst_nom + TOL_INSTALLED_M * rng.standard_normal(n),
            lift=lift, ratio=ratio_nom * (1.0 + TOL_RATIO_FRAC * rng.standard_normal(n)), ratio_nominal=ratio_nom,
            lash=LASH_SPEC_M + TOL_LASH_M * rng.standard_normal(n),
            concentricity=np.abs(TOL_SEAT_CONCENTRICITY_M * rng.standard_normal(n)),
            mass_kg=mass, float_rpm_nominal=float(spring.max_safe_rpm()), identity=engine.identity)
        return vs

    # ---- per-valve derived (vectorized) ----
    @property
    def seat_load(self) -> np.ndarray:
        return np.maximum(0.0, self.rate * (self.free_len - self.sag - self.installed + self.shim))

    @property
    def seat_load_nominal(self) -> np.ndarray:
        return self.rate * (self.free_len - self.installed)

    @property
    def lift_eff(self) -> np.ndarray:
        # a coked intake throat behaves like less lift (curtain area choked)
        choke = np.where(self.is_intake, 1.0 - 0.35 * self.carbon, 1.0 - 0.10 * self.carbon)
        return np.maximum(0.0, (self.lift * (self.ratio / self.ratio_nominal) - np.maximum(0.0, self.lash - LASH_SPEC_M)) * choke)

    @property
    def open_load(self) -> np.ndarray:
        return self.seat_load + self.rate * self.lift_eff

    @property
    def float_rpm(self) -> np.ndarray:
        ref = np.maximum(self.seat_load_nominal, 1e-6)
        # deposit mass rides on the valve: omega_n = sqrt(k/m) falls with it
        mass_frac = 1.0 + 0.15 * self.carbon
        return self.float_rpm_nominal * np.sqrt(np.clip(self.seat_load / ref, 0.05, 1.5) / mass_frac)

    @property
    def seat_leak(self) -> np.ndarray:
        rec = (self.recession + self.concentricity) / 0.0005 * 0.02
        weak = np.maximum(0.0, 1.0 - self.seat_load / np.maximum(0.6 * self.seat_load_nominal, 1e-6)) * 0.05
        held_off = 0.03 * self.carbon                   # a deposit on the seat holds the valve off it
        return np.clip(rec + weak + held_off, 0.0, 0.5)

    @property
    def hotspot_risk(self) -> np.ndarray:
        """0..1 per valve: a glowing exhaust-valve deposit is a real
        pre-ignition source; intake deposits far less so."""
        return np.clip(np.where(self.is_intake, 0.2, 1.0) * self.carbon ** 1.5, 0.0, 1.0)

    @property
    def coil_bind_margin(self) -> np.ndarray:
        return (self.installed - self.shim - self.lift_eff) - self.free_len * 0.55

    # ---- per-cylinder reductions ----
    def per_cylinder(self) -> dict:
        n = self.n_cyl
        idx = self.cylinder - 1
        breathing = np.ones(n)
        if len(idx):
            lift_ratio = np.sqrt(np.clip(self.lift_eff / np.maximum(self.lift, 1e-9), 0.0, 1.5))
            # product over intake valves via log-sum
            logs = np.where(self.is_intake, np.log(np.maximum(lift_ratio, 1e-6)), 0.0)
            breathing = np.exp(np.bincount(idx, weights=logs, minlength=n))
            float_rpm = np.full(n, np.inf)
            np.minimum.at(float_rpm, idx, self.float_rpm)
            leak = np.bincount(idx, weights=self.seat_leak, minlength=n)
            hotspot = np.zeros(n); np.maximum.at(hotspot, idx, self.hotspot_risk)
            carbon = np.bincount(idx, weights=self.carbon, minlength=n) / np.maximum(np.bincount(idx, minlength=n), 1)
        else:
            float_rpm = np.full(n, np.inf); leak = np.zeros(n); hotspot = np.zeros(n); carbon = np.zeros(n)
        return {"breathing_frac": breathing, "float_rpm": float_rpm, "leak_frac": np.clip(leak, 0.0, 0.8),
                "hotspot_risk": hotspot, "carbon_frac": carbon}

    def float_penalty(self, rpm: float, float_rpm: np.ndarray) -> np.ndarray:
        """Per cylinder, 1.0 below its own float speed, dropping 0.6 per
        50% overspeed past it -- the same shape the engine-wide float
        penalty already used, now per cylinder."""
        risk = np.maximum(0.0, rpm / np.maximum(float_rpm, 1.0) - 1.0)
        return np.maximum(0.0, 1.0 - 0.6 * np.minimum(risk, 1.5))

    # ---- aging (vectorized) ----
    def age(self, dt: float, rpm: float, cylinder_temps_k, exhaust_temp_k: float | None = None,
            load_frac: float = 1.0, richness: float = 1.0, direct_injection: bool = False,
            oil_burn_frac=None) -> None:
        """Springs sag, seats recede, and carbon grows or burns -- all
        per valve from this tick's conditions: cylinder block temps,
        exhaust temperature (exhaust valves run toward it), load and
        mixture richness (fuel_quantity_frac: >1 rich), and whether
        the intake valves are fuel-washed (port/carb) or not (direct)."""
        if len(self.cylinder) == 0:
            return
        self._carbon_step(dt, rpm, cylinder_temps_k, exhaust_temp_k, load_frac, richness, direct_injection, oil_burn_frac)
        if rpm <= 0.0:
            return
        cycles = rpm / 120.0 * dt                     # four-stroke: a valve event every other rev
        temps = np.asarray(cylinder_temps_k, dtype=np.float64)
        if temps.size != self.n_cyl:
            temps = np.full(self.n_cyl, float(temps.mean()) if temps.size else 400.0)
        t_valve = temps[self.cylinder - 1]
        accel = np.exp((t_valve - 400.0) / 60.0)
        limit = self.free_len * SAG_LIMIT_FRAC
        self.sag = np.minimum(limit, self.sag + limit * (cycles / 1.0e8) * accel)
        hot = np.where(self.is_intake, 0.3, 1.0) * accel
        self.recession = self.recession + SEAT_RECESSION_RATE_M_PER_CYCLE * cycles * hot

    def _carbon_step(self, dt, rpm, cylinder_temps_k, exhaust_temp_k, load_frac, richness, direct_injection,
                     oil_burn_frac=None):
        temps = np.asarray(cylinder_temps_k, dtype=np.float64)
        if temps.size != self.n_cyl:
            temps = np.full(self.n_cyl, float(temps.mean()) if temps.size else 350.0)
        t_block = temps[self.cylinder - 1]
        t_exh = float(exhaust_temp_k) if exhaust_temp_k is not None else t_block
        # exhaust valves run between block and gas temperature; intake valves near the block
        t_valve = np.where(self.is_intake, t_block + 30.0, 0.45 * t_block + 0.55 * t_exh)
        cold = np.clip((CARBON_COLD_TEMP_K + 80.0 - t_valve) / 80.0, 0.0, 1.0)      # 1 stone cold .. 0 warm
        rich = max(0.0, float(richness) - 1.0) * 4.0                                 # 25% rich -> +1
        idle = max(0.0, 0.35 - float(load_frac)) / 0.35                              # light load / idling
        grow = CARBON_GROWTH_PER_S * (0.15 + cold + rich + 0.6 * idle) * (1.0 if rpm > 0.0 else 0.2)
        if oil_burn_frac is not None:
            # oil burning past the rings is its own deposit source (per
            # cylinder, from crankcase_state): heavier than fuel soot
            ob = np.asarray(oil_burn_frac, dtype=np.float64)
            if ob.size == self.n_cyl:
                grow = grow + CARBON_GROWTH_PER_S * 2.0 * ob[self.cylinder - 1]
        wash = np.where(self.is_intake, (1.5 if direct_injection else 0.25), 1.0)  # fuel-washed intake valves stay clean
        burn = CARBON_BURN_PER_S * np.clip((t_valve - CARBON_BURN_TEMP_K) / 60.0, 0.0, 1.0) * max(0.0, float(load_frac))
        self.carbon = np.clip(self.carbon + (grow * wash - burn * self.carbon) * dt, 0.0, 1.0)

    # ---- service ----
    def decarbonize(self, cylinder: int | None = None) -> None:
        """A walnut blast / cleaning: deposits gone on one cylinder or all."""
        sel = np.ones(len(self.cylinder), dtype=bool) if cylinder is None else (self.cylinder == cylinder)
        self.carbon[sel] = 0.0

    def replace_springs(self, cylinder: int | None = None, seed: int | None = None) -> None:
        """New springs (fresh machine error, no sag) on one cylinder or all."""
        rng = np.random.default_rng(seed if seed is not None else _seed(self.identity + ":new"))
        sel = np.ones(len(self.cylinder), dtype=bool) if cylinder is None else (self.cylinder == cylinder)
        n = int(sel.sum())
        nominal_rate = float(np.median(self.rate)) if len(self.rate) else 0.0
        self.rate[sel] = nominal_rate * (1.0 + TOL_RATE_FRAC * rng.standard_normal(n))
        self.sag[sel] = 0.0
        self.shim[sel] = 0.0

    def shim_valve(self, valve_index: int, shim_m: float) -> None:
        self.shim[valve_index] = shim_m

    def set_lash(self, valve_index: int, lash_m: float) -> None:
        self.lash[valve_index] = lash_m

    def spring_state(self, valve_index: int):
        """The same numbers as a valvetrain_parts.SpringState, for the manifest."""
        from valvetrain_parts import SpringState
        i = valve_index
        return SpringState(free_length_m=float(self.free_len[i]), rate_n_per_m=float(self.rate[i]),
                           installed_height_m=float(self.installed[i]), lift_m=float(self.lift_eff[i]),
                           sag_m=float(self.sag[i]), shim_m=float(self.shim[i]))

    def report(self) -> list[str]:
        pc = self.per_cylinder()
        out = []
        for c in range(self.n_cyl):
            out.append(f"cyl {c + 1}: breathing {pc['breathing_frac'][c]:.3f}  float {pc['float_rpm'][c]:6.0f} rpm  leak {pc['leak_frac'][c] * 100:.2f}%  carbon {pc['carbon_frac'][c] * 100:4.1f}%  hotspot {pc['hotspot_risk'][c]:.2f}")
        return out
