"""The air the engine lives in: a chain of real gas volumes.

    ambient  <->  garage / dyno cell (optional, enclosed)  <->  engine bay

Each volume is well-mixed air with its own temperature, CO content and
oxygen fraction, exchanging with its parent through a real ventilation
flow (the bay: the cooling fan's own real delivered flow through the
grille plus ram air with vehicle speed; the room: its door state, a
cell fan, and whatever leaks). The engine couples to the chain three
ways -- it DRAWS its intake from one volume (the bay for an under-hood
filter, the room/outside for a cold-air box or snorkel: that is the
whole reason cold-air boxes exist), it DUMPS heat into the bay (block
and radiator rejection), and it EXHAUSTS into wherever the pipe
actually ends: the room for a tailpipe, the bay itself for an open
header, or straight outdoors when a real exhaust extractor -- the big
orange high-temp duct clamped over the tailpipe in any dyno cell -- is
fitted and catches it.

What the engine inhales then matters honestly: hot bay air is a
denser-charge loss, oxygen-depleted room air is real power loss and
eventually a stall, and the CO in that room is what emissions.
Occupant breathes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AMBIENT_TEMP_K = 293.15
AMBIENT_O2_FRAC = 0.2095
AIR_DENSITY_KG_M3 = 1.2
AIR_VOLUMETRIC_HEAT_J_M3K = 1200.0
CO_KG_PER_M3_PPM = 1.145e-6

GARAGE_DOOR_ACH = {"closed": 0.5, "open": 25.0, "sealed": 0.1}
GRILLE_AREA_M2 = 0.25
GRILLE_RAM_EFFICIENCY = 0.5
FAN_THROUGH_GRILLE_FRACTION = 0.6      # what the fan actually pulls through the core vs recirculates
BAY_NATURAL_CONVECTION_M3_S = 0.04


@dataclass
class AirVolume:
    name: str
    volume_m3: float
    temp_k: float = AMBIENT_TEMP_K
    co_ppm: float = 0.0
    o2_frac: float = AMBIENT_O2_FRAC
    parent: "AirVolume | None" = None
    # per-tick accumulators, cleared by step()
    heat_in_w: float = field(default=0.0, init=False)
    exchange_m3_s: float = field(default=0.0, init=False)      # ventilation with parent
    draw_m3_s: float = field(default=0.0, init=False)          # air pulled OUT (engine intake), replaced from parent
    exhaust_m3_s: float = field(default=0.0, init=False)       # exhaust gas dumped IN
    exhaust_temp_k: float = field(default=AMBIENT_TEMP_K, init=False)
    exhaust_co_kg_s: float = field(default=0.0, init=False)
    exhaust_o2_frac: float = field(default=0.0, init=False)

    def add_exhaust(self, gas_kg_s: float, temp_k: float, co_kg_s: float, o2_frac: float) -> None:
        q = gas_kg_s / AIR_DENSITY_KG_M3
        if q <= 0.0:
            return
        # mass-weighted merge with anything already dumped this tick
        total = self.exhaust_m3_s + q
        self.exhaust_temp_k = (self.exhaust_temp_k * self.exhaust_m3_s + temp_k * q) / total
        self.exhaust_o2_frac = (self.exhaust_o2_frac * self.exhaust_m3_s + o2_frac * q) / total
        self.exhaust_m3_s = total
        self.exhaust_co_kg_s += co_kg_s

    def step(self, dt: float) -> None:
        p = self.parent
        pt = p.temp_k if p else AMBIENT_TEMP_K
        pco = p.co_ppm if p else 0.0
        po2 = p.o2_frac if p else AMBIENT_O2_FRAC
        v = max(self.volume_m3, 0.1)
        # everything leaving is replaced by parent air: ventilation
        # exchange + the intake draw; exhaust displaces an equal volume
        # out to the parent (well-mixed, constant pressure)
        q_in = self.exchange_m3_s + self.draw_m3_s
        q_ex = self.exhaust_m3_s
        d_temp = (q_in * (pt - self.temp_k) + q_ex * (self.exhaust_temp_k - self.temp_k)
                  + self.heat_in_w / AIR_VOLUMETRIC_HEAT_J_M3K) / v
        co_ppm_in = self.exhaust_co_kg_s / CO_KG_PER_M3_PPM       # m^3*ppm per second
        d_co = (q_in * (pco - self.co_ppm) + co_ppm_in - q_ex * self.co_ppm) / v
        d_o2 = (q_in * (po2 - self.o2_frac) + q_ex * (self.exhaust_o2_frac - self.o2_frac)) / v
        self.temp_k = max(200.0, self.temp_k + d_temp * dt)
        self.co_ppm = max(0.0, self.co_ppm + d_co * dt)
        self.o2_frac = max(0.0, min(AMBIENT_O2_FRAC, self.o2_frac + d_o2 * dt))
        # what left this volume by displacement carries into the parent
        if p is not None and q_ex > 0.0:
            p.add_exhaust(q_ex * AIR_DENSITY_KG_M3, self.temp_k, self.co_ppm * CO_KG_PER_M3_PPM * q_ex, self.o2_frac)
        self.heat_in_w = 0.0; self.exchange_m3_s = 0.0; self.draw_m3_s = 0.0
        self.exhaust_m3_s = 0.0; self.exhaust_co_kg_s = 0.0
        self.exhaust_temp_k = AMBIENT_TEMP_K; self.exhaust_o2_frac = 0.0


@dataclass
class AirStack:
    """ambient -> (garage) -> bay. `garage_mode`: "outdoors" (no room at
    all), "open" (door up), "closed", "sealed". `extractor_fitted`: the
    exhaust extraction duct over the tailpipe, venting straight to
    ambient at `extractor_capture_frac`. `cell_fan_m3_s`: a dyno-cell
    ventilation fan on the room."""
    bay: AirVolume
    garage: AirVolume | None = None
    garage_mode: str = "outdoors"
    extractor_fitted: bool = False
    extractor_capture_frac: float = 0.9
    cell_fan_m3_s: float = 0.0
    vehicle_speed_mps: float = 0.0

    @classmethod
    def build(cls, bay_volume_m3: float, garage_mode: str = "outdoors", garage_volume_m3: float = 94.0) -> "AirStack":
        garage = None if garage_mode == "outdoors" else AirVolume("garage", garage_volume_m3)
        bay = AirVolume("engine bay", bay_volume_m3, parent=garage)
        return cls(bay=bay, garage=garage, garage_mode=garage_mode)

    def set_garage_mode(self, mode: str, garage_volume_m3: float = 94.0) -> None:
        self.garage_mode = mode
        if mode == "outdoors":
            self.garage = None
        elif self.garage is None:
            self.garage = AirVolume("garage", garage_volume_m3)
        self.bay.parent = self.garage

    @property
    def outside(self) -> AirVolume | None:
        """The volume a cold-air box / snorkel breathes: the room if
        there is one, else ambient (None)."""
        return self.garage

    def ventilate_bay(self, fan_flow_m3_s: float) -> None:
        ram = GRILLE_AREA_M2 * max(self.vehicle_speed_mps, 0.0) * GRILLE_RAM_EFFICIENCY
        self.bay.exchange_m3_s += max(fan_flow_m3_s, 0.0) * FAN_THROUGH_GRILLE_FRACTION + ram + BAY_NATURAL_CONVECTION_M3_S

    def ventilate_garage(self) -> None:
        if self.garage is None:
            return
        ach = GARAGE_DOOR_ACH.get(self.garage_mode, 0.5)
        self.garage.exchange_m3_s += ach * self.garage.volume_m3 / 3600.0 + max(self.cell_fan_m3_s, 0.0)

    def exhaust_to(self, destination: str, gas_kg_s: float, temp_k: float, co_kg_s: float, o2_frac: float) -> float:
        """destination: "tailpipe" (the room, or outdoors) | "bay" (an open
        header ends under the hood). Returns the CO rate that actually
        stayed indoors (0 when it all went outside)."""
        if gas_kg_s <= 0.0:
            return 0.0
        if destination == "bay":
            self.bay.add_exhaust(gas_kg_s, temp_k, co_kg_s, o2_frac)
            return co_kg_s
        captured = self.extractor_capture_frac if self.extractor_fitted else 0.0
        leak = 1.0 - captured
        if self.garage is None or leak <= 0.0:
            return 0.0
        self.garage.add_exhaust(gas_kg_s * leak, temp_k, co_kg_s * leak, o2_frac)
        return co_kg_s * leak

    def step(self, dt: float) -> None:
        self.bay.step(dt)
        if self.garage is not None:
            self.garage.step(dt)
