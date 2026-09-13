"""Photovoltaic generation, and the DC plant that carries it.

A panel is not a number of watts. It is an area of a particular
semiconductor at a particular angle to a sun at a particular height,
getting hot while it works and losing efficiency because of it, feeding
a controller that is either clever or cheap. Every one of those is worth
modelling because every one of them is where the watts actually go, and
a "200 W panel" delivers 200 W on approximately no day of the year.

WHAT THE NUMBERS ARE

  IRRADIANCE falls off with AIR MASS -- how much atmosphere the light
  crossed to get here. Straight overhead is one atmosphere; thirty
  degrees above the horizon is two. The standard engineering model is
  Meinel's: 1.353 * 0.7^(AM^0.678) kW/m2, which gives about 950 W/m2
  with the sun overhead and 765 at thirty degrees. Above the atmosphere
  it is 1361 W/m2 and the panel would be much happier.

  COSINE LOSS is the other half and it is usually the bigger one. A
  panel lying flat on a vehicle roof at a sun elevation of 30 degrees
  collects half of what it would facing the sun squarely, on top of the
  air-mass loss it has already taken.

  HEAT. Silicon loses about 0.4% of its output per degree C above 25,
  and a panel in full sun runs 25-30 C above ambient. So a panel on a
  hot day makes noticeably less than the same panel on a cold bright
  one, which is why the best output of the year is often a clear day in
  spring rather than the middle of summer.

  THE CONTROLLER. A "12 V" panel's maximum-power point is around 18 V,
  and a 12 V battery wants about 13. A PWM controller simply connects
  the two, which drags the panel down to battery voltage and throws away
  the difference -- roughly a quarter of the harvest. An MPPT controller
  runs the panel at its own best point and converts the surplus voltage
  into current. That is the entire difference between the two, it is
  real, and it is worth about 25-30%.

Nothing here decides anything by looking at a name: a panel declares its
cell technology and the technology carries its own efficiency and
temperature coefficient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

SOLAR_CONSTANT_W_M2 = 1361.0          # above the atmosphere


@dataclass(frozen=True)
class CellTechnology:
    """What the panel is made of, and what that costs and buys."""
    key: str
    label: str
    efficiency: float                  # at 25 C, standard test conditions
    temperature_coefficient: float     # fractional loss per degree C over 25
    noct_c: float = 45.0               # nominal operating cell temperature
    vmp_per_vnominal: float = 1.45     # max-power voltage over nominal system
    note: str = ""


CELL_TECHNOLOGIES: tuple[CellTechnology, ...] = (
    CellTechnology("mono-si", "monocrystalline silicon", 0.21, 0.0035, 45.0, 1.45,
                   note="the default choice: best area efficiency of the cheap "
                        "technologies, and it hates heat the most"),
    CellTechnology("poly-si", "polycrystalline silicon", 0.17, 0.0040, 46.0, 1.45,
                   note="cheaper per watt, worse per square metre"),
    CellTechnology("cigs", "thin-film CIGS", 0.13, 0.0025, 44.0, 1.40,
                   note="poor per square metre, but it shrugs off heat and partial "
                        "shade far better than silicon"),
    CellTechnology("amorphous", "amorphous silicon", 0.07, 0.0020, 47.0, 1.50,
                   note="nearly heat-proof and nearly useless per square metre; "
                        "what you use when area is free and cold is not"),
    CellTechnology("gaas-3j", "triple-junction gallium arsenide", 0.30, 0.0020, 42.0, 1.60,
                   note="spacecraft and serious military kit: half again the output "
                        "per square metre, at many times the cost"),
)
CELL_BY_KEY = {c.key: c for c in CELL_TECHNOLOGIES}


# ---------------------------------------------------------------------
#  THE SUN
# ---------------------------------------------------------------------
def air_mass(sun_elevation_deg: float) -> float:
    """How many atmospheres the light crossed.

    Kasten-Young, which unlike the naive 1/sin does not go to infinity
    at the horizon -- it tops out near 38, which is about right."""
    e = max(float(sun_elevation_deg), -1.0)
    if e <= 0.0:
        return float("inf")
    return 1.0 / (math.sin(math.radians(e))
                  + 0.50572 * (e + 6.07995) ** -1.6364)


def direct_irradiance_w_m2(sun_elevation_deg: float, *, cloud_cover: float = 0.0,
                           altitude_m: float = 0.0) -> float:
    """Beam irradiance on a surface facing the sun squarely.

    Meinel's model, with the usual altitude correction: thinner air
    above you means more light gets through, worth about 1.5% per
    thousand metres."""
    am = air_mass(sun_elevation_deg)
    if not math.isfinite(am):
        return 0.0
    direct = 1353.0 * 0.7 ** (am ** 0.678)
    direct *= (1.0 + 0.00014 * max(altitude_m, 0.0))
    # CLOUD KILLS THE BEAM FAST, and not linearly -- thin cloud already
    # costs most of the direct light, and a full overcast leaves
    # essentially none of it. What still reaches the panel under
    # overcast arrives as diffuse, which `diffuse_irradiance_w_m2`
    # accounts for; the two together come to 10-25% of a clear day,
    # which is what is actually measured.
    c = max(0.0, min(1.0, cloud_cover))
    return direct * (1.0 - c) ** 2.2


def diffuse_irradiance_w_m2(sun_elevation_deg: float, *, cloud_cover: float = 0.0) -> float:
    """Sky light, which arrives from everywhere and so takes no cosine
    loss. It is a tenth of the beam on a clear day and essentially all
    of the light under overcast -- which is why a panel still makes
    something useful on a grey day instead of nothing."""
    clear = direct_irradiance_w_m2(sun_elevation_deg, cloud_cover=0.0)
    c = max(0.0, min(1.0, cloud_cover))
    return clear * (0.10 + 0.10 * c)


# ---------------------------------------------------------------------
#  THE PANEL
# ---------------------------------------------------------------------
@dataclass
class SolarPanel:
    """One panel: an area of a technology, pointed somewhere."""
    identity: str = "panel"
    area_m2: float = 1.05                  # a common 200 W-class module
    technology: str = "mono-si"
    tilt_deg: float = 0.0                  # 0 = lying flat, as on a vehicle roof
    azimuth_deg: float = 180.0             # where the tilt faces; 180 = south
    system_voltage_v: float = 12.0
    soiling_fraction: float = 0.04         # dust, which is never zero outdoors
    shaded_fraction: float = 0.0
    mass_kg: float = 11.0
    damaged: bool = False

    @property
    def cell(self) -> CellTechnology:
        return CELL_BY_KEY[self.technology]

    @property
    def rated_w(self) -> float:
        """What the sticker says: standard test conditions, 1000 W/m2,
        cell at 25 C, facing the light squarely. A number the panel will
        essentially never produce in service."""
        return self.area_m2 * self.cell.efficiency * 1000.0

    # -----------------------------------------------------------------
    def incidence_cosine(self, sun_elevation_deg: float, sun_azimuth_deg: float) -> float:
        """The cosine between the panel's normal and the sun. This is
        usually the largest single loss on a vehicle, because the panel
        is bolted flat to a roof and the sun is not overhead."""
        if sun_elevation_deg <= 0.0:
            return 0.0
        e, a = math.radians(sun_elevation_deg), math.radians(sun_azimuth_deg)
        t, p = math.radians(self.tilt_deg), math.radians(self.azimuth_deg)
        # sun and normal as unit vectors, then their dot product
        sun = (math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e))
        nrm = (math.sin(t) * math.sin(p), math.sin(t) * math.cos(p), math.cos(t))
        return max(0.0, sum(s * n for s, n in zip(sun, nrm)))

    def cell_temperature_c(self, ambient_c: float, plane_irradiance_w_m2: float) -> float:
        """A panel in full sun runs well above the air around it, and
        that is where a fifth of a hot day's output goes."""
        return ambient_c + (self.cell.noct_c - 20.0) / 800.0 * plane_irradiance_w_m2

    def output(self, *, sun_elevation_deg: float, sun_azimuth_deg: float,
               ambient_c: float = 20.0, cloud_cover: float = 0.0,
               altitude_m: float = 0.0) -> dict:
        """What this panel is making right now, and where the rest went."""
        if self.damaged:
            return {"power_w": 0.0, "plane_irradiance_w_m2": 0.0, "cell_c": ambient_c,
                    "vmp_v": 0.0, "losses": {"destroyed": 1.0}}
        beam = direct_irradiance_w_m2(sun_elevation_deg, cloud_cover=cloud_cover,
                                      altitude_m=altitude_m)
        sky = diffuse_irradiance_w_m2(sun_elevation_deg, cloud_cover=cloud_cover)
        cosine = self.incidence_cosine(sun_elevation_deg, sun_azimuth_deg)
        # the sky arrives from all over, so the tilted panel sees the
        # fraction of the sky dome it faces
        sky_view = (1.0 + math.cos(math.radians(self.tilt_deg))) / 2.0
        plane = beam * cosine + sky * sky_view
        plane *= (1.0 - self.soiling_fraction) * (1.0 - max(0.0, min(1.0, self.shaded_fraction)))
        t_cell = self.cell_temperature_c(ambient_c, plane)
        derate = 1.0 - self.cell.temperature_coefficient * (t_cell - 25.0)
        derate = max(0.0, derate)
        power = self.area_m2 * self.cell.efficiency * plane * derate
        return {
            "power_w": power,
            "plane_irradiance_w_m2": plane,
            "beam_w_m2": beam, "diffuse_w_m2": sky,
            "incidence_cosine": cosine,
            "cell_c": t_cell,
            "temperature_derate": derate,
            "vmp_v": self.system_voltage_v * self.cell.vmp_per_vnominal,
            "fraction_of_rated": power / max(self.rated_w, 1e-9),
        }

    def describe(self) -> list[str]:
        c = self.cell
        return [f"  {self.identity}: {self.area_m2:.2f} m2 of {c.label}, "
                f"{self.rated_w:.0f} W rated, tilt {self.tilt_deg:.0f} deg",
                f"    {c.efficiency * 100:.0f}% at 25 C, losing "
                f"{c.temperature_coefficient * 100:.2f}%/C above it; {c.note}"]


# ---------------------------------------------------------------------
#  THE CONTROLLER
# ---------------------------------------------------------------------
@dataclass
class ChargeController:
    """What stands between the panel and the battery.

    The choice between the two kinds is not a detail. A PWM controller
    is a switch: it ties the panel to the battery, which drags the panel
    off its maximum-power point down to battery voltage. The current is
    unchanged, so the power lost is exactly the voltage thrown away --
    about 28% for a panel whose best point is 18 V charging a battery at
    13. An MPPT controller is a converter: it holds the panel at its own
    best point and turns the surplus volts into amps, keeping all but
    its own conversion loss.
    """
    identity: str = "controller"
    kind: str = "mppt"                     # "mppt" | "pwm"
    conversion_efficiency: float = 0.97    # mppt only; a switch has none to lose
    maximum_input_w: float = 400.0
    standby_w: float = 0.6                 # it costs something just being awake
    healthy: bool = True

    def deliver(self, panel_w: float, vmp_v: float, battery_v: float) -> dict:
        if not self.healthy:
            return {"power_w": 0.0, "harvest_fraction": 0.0, "limited": False}
        available = min(panel_w, self.maximum_input_w)
        limited = panel_w > self.maximum_input_w
        if self.kind == "mppt":
            out = available * self.conversion_efficiency
        else:
            # the panel is dragged to battery voltage; current survives,
            # the surplus voltage does not
            out = available * min(1.0, battery_v / max(vmp_v, 1e-9))
        out = max(0.0, out - self.standby_w)
        return {"power_w": out, "harvest_fraction": out / max(panel_w, 1e-9),
                "limited": limited}

    def describe(self) -> list[str]:
        if self.kind == "mppt":
            return [f"  {self.identity}: MPPT, {self.conversion_efficiency * 100:.0f}% "
                    f"conversion, {self.maximum_input_w:.0f} W input limit"]
        return [f"  {self.identity}: PWM -- a switch, not a converter; the panel is "
                f"pulled down to battery voltage and the surplus is lost"]


@dataclass
class SolarArray:
    """Several panels on one controller."""
    identity: str = "array"
    panels: list = field(default_factory=list)
    controller: ChargeController = field(default_factory=ChargeController)

    @property
    def area_m2(self) -> float:
        return sum(p.area_m2 for p in self.panels)

    @property
    def rated_w(self) -> float:
        return sum(p.rated_w for p in self.panels)

    def output(self, *, battery_v: float = 13.2, **conditions) -> dict:
        total, vmp = 0.0, 0.0
        per_panel = {}
        for p in self.panels:
            o = p.output(**conditions)
            per_panel[p.identity] = o
            total += o["power_w"]
            vmp = max(vmp, o["vmp_v"])
        delivered = self.controller.deliver(total, vmp, battery_v)
        return {"panel_w": total, "delivered_w": delivered["power_w"],
                "harvest_fraction": delivered["harvest_fraction"],
                "controller_limited": delivered["limited"],
                "fraction_of_rated": delivered["power_w"] / max(self.rated_w, 1e-9),
                "panels": per_panel}

    def describe(self) -> list[str]:
        out = [f"  {self.identity}: {len(self.panels)} panel(s), {self.area_m2:.2f} m2, "
               f"{self.rated_w:.0f} W rated"]
        for p in self.panels:
            out.extend(p.describe())
        out.extend(self.controller.describe())
        return out


# ---------------------------------------------------------------------
#  A KEEP-ALIVE: ONE PANEL, ONE SMALL BATTERY, FEEDING AN ISOLATOR
# ---------------------------------------------------------------------
@dataclass
class SolarKeepAlive:
    """A small panel and its own battery, wired to the INPUT of a charge
    isolator -- and nothing else.

    This is deliberately the least ambitious way to put solar on a
    machine, and it is the right one, because the isolator already
    contains all of the logic it needs. A voltage-sensitive relay closes
    when it sees a charging voltage on its input and opens when that
    voltage sags. It does not know or care whether the voltage came from
    an alternator; it is a relay looking at a number. So a panel that
    can hold the input above the close threshold charges the accessory
    bank with the engine stopped, using a mechanism that was already
    there, and needs no new harness, no exterior port and no supervisor
    of its own.

    THE CONTROLLER'S OUTPUT VOLTAGE IS WHAT DOES THE WORK, not the
    panel's and not the little battery's. A charge controller drives its
    output to an absorption setpoint -- about 28.8 V on a 24 V system --
    and that is comfortably above the 25.2 V a VSR wants to see. The
    small battery is what carries that through a cloud and into the
    evening: without it the relay would chatter every time the sun went
    behind something, which is how relays die.

    WHAT IT IS FOR is the load that must never stop: the monitoring that
    reads the tanks, and enough current to trip a start solenoid. Those
    are watts, not kilowatts, and the panel is sized accordingly.
    """
    identity: str = "keep-alive"
    panel: SolarPanel = field(default_factory=lambda: SolarPanel(
        identity="keep-alive.panel", area_m2=0.60, system_voltage_v=24.0))
    controller: ChargeController = field(default_factory=lambda: ChargeController(
        identity="keep-alive.controller", kind="mppt", maximum_input_w=120.0))
    battery: object = None                 # dc_power.DCBattery
    absorption_v: float = 28.8             # what the controller drives to
    standby_load_w: float = 6.0            # the monitoring that never sleeps
    generating_w: float = 0.0
    supplying: bool = False

    def __post_init__(self) -> None:
        if self.battery is None:
            from dc_power import DCBattery
            # LiFePO4 because almost all of its nameplate is usable and
            # it barely self-discharges, which is the whole job here --
            # but it must not be charged below freezing, and this
            # module does not hide that: see DCBattery.may_charge.
            self.battery = DCBattery(identity="keep-alive.battery",
                                     chemistry="lifepo4", capacity_ah=20.0,
                                     state_of_charge=0.75)

    # -----------------------------------------------------------------
    def output_voltage_v(self) -> float:
        """What this presents at the isolator's input.

        While the panel is making more than the standby load, the
        controller holds its absorption setpoint and the relay sees a
        charging bus. Otherwise the input is just the little battery's
        own terminal voltage, which on a 24 V lithium pack sits near 26
        and will NOT close a relay looking for 25.2 with hysteresis --
        so the bank is not slowly drained by its own keep-alive
        overnight, which would be the obvious way to get this wrong."""
        if self.supplying:
            return self.absorption_v
        return 0.0

    def step(self, dt: float, *, sun_elevation_deg: float, sun_azimuth_deg: float = 180.0,
             ambient_c: float = 20.0, cloud_cover: float = 0.0) -> dict:
        out = self.panel.output(sun_elevation_deg=sun_elevation_deg,
                                sun_azimuth_deg=sun_azimuth_deg,
                                ambient_c=ambient_c, cloud_cover=cloud_cover)
        self.battery.temperature_c = ambient_c
        delivered = self.controller.deliver(out["power_w"], out["vmp_v"],
                                            self.battery.terminal_v)
        self.generating_w = delivered["power_w"]
        # the standby load comes first; only the surplus reaches the
        # battery, and only a charged battery is allowed to hold the
        # isolator's input up
        surplus = self.generating_w - self.standby_load_w
        b = self.battery.step(dt, charge_w=max(0.0, surplus),
                              load_w=max(0.0, -surplus))
        self.supplying = (surplus > 0.0 and self.battery.state_of_charge > 0.35)
        return {"panel_w": out["power_w"], "delivered_w": self.generating_w,
                "surplus_w": surplus, "supplying": self.supplying,
                "output_v": self.output_voltage_v(), **b}

    def describe(self) -> list[str]:
        out = [f"  {self.identity}: {self.panel.rated_w:.0f} W panel -> "
               f"{self.battery.capacity_ah:.0f} Ah {self.battery.chem.label} -> "
               f"isolator input at {self.absorption_v:.1f} V when generating"]
        out.extend(self.battery.describe())
        out.append(f"    standby load {self.standby_load_w:.0f} W; "
                   f"{'SUPPLYING' if self.supplying else 'idle'} "
                   f"({self.generating_w:.1f} W now)")
        return out
