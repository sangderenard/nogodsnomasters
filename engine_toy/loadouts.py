"""Loadouts: a named set of equipment an engine carries.

The other way to stop having to hand-build a cylinder onto the C18
every time you want to see one work. A loadout is the equipment
package a machine of some particular kind actually carries -- an
excavator's boom, stick, bucket and swing; a loader's lift and tilt; a
truck's tipping ram and air brakes; an aircraft's gear and its retract
actuators -- fitted in one go and run off the engine's own plant.

The useful part is not the fitting. It is the SIZING CHECK. A loadout
knows what it demands, the engine's plant knows what it can supply, and
the answer is frequently no: the pump has the pressure but not the
flow, so the machine works but crawls; or the loadout's peak demand
exceeds the relief setting, so one function simply will not move under
load while another is running. That is the real conversation between a
machine and its power source, and `check_against` has it.

A loadout is not mutually exclusive with the test bench (bench.py).
The bench exists to test ONE part with nothing else in the way; a
loadout exists to see a realistic set of parts fight over one engine's
finite pump. Both were wanted, and they answer different questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from actuators import LinearActuator, RotaryActuator, FluidMotor, Positioner, GasOverOilStrut

ATM_PA = 101_325.0


@dataclass
class LoadoutItem:
    """One piece of equipment in a package, with the duty it is
    expected to perform -- which is what decides the flow it wants."""
    name: str
    build: object                    # a callable returning the part
    duty_speed_m_s: float = 0.2      # the speed this function is designed to run at
    duty_rate_deg_s: float = 45.0    # ...or the swing rate, for a rotary
    duty_load_n: float = 0.0
    concurrent: bool = True          # does it run at the same time as the others?


@dataclass
class Loadout:
    identity: str
    label: str
    medium: str                      # the dominant medium
    items: list = field(default_factory=list)
    note: str = ""

    def build(self) -> list:
        """Make the parts. Each item builds its own, so a loadout is a
        recipe rather than a pile of shared objects -- fit the same
        loadout twice and you get two independent sets."""
        return [(it.name, it.build(), it) for it in self.items]

    # ---------------------------------------------------------------
    def demand(self) -> dict:
        """What this package asks for when everything that can run at
        once is running: peak flow, and the highest pressure any one
        function needs."""
        flow = 0.0
        peak_flow_single = 0.0
        pressure = ATM_PA
        for name, part, it in self.build():
            # a closed-loop axis IS a cylinder as far as the pump is
            # concerned -- it wraps one. Asking the wrapper for a bore
            # got nothing, so the press quietly reported needing no oil
            # at all, which is the kind of zero that looks like a pass.
            part = getattr(part, "actuator", part)
            if hasattr(part, "area_extend_m2"):
                q = part.area_extend_m2 * it.duty_speed_m_s * 60_000.0
                seal = getattr(part, "seal_friction_frac", 0.05)
                p = ATM_PA + it.duty_load_n / max(part.area_extend_m2 * (1.0 - seal), 1e-9)
            elif hasattr(part, "displacement_l_per_rev"):
                q = part.displacement_l_per_rev * (it.duty_rate_deg_s / 360.0) * 60.0
                at_rated = part.torque_at(part.rated_pressure_pa)
                p = ATM_PA + (it.duty_load_n / max(at_rated, 1e-9)) * part.rated_pressure_pa
            elif hasattr(part, "displacement_cc_rev"):
                q = part.displacement_cc_rev / 1000.0 * 60.0      # at 60 rpm of duty
                p = part.max_pressure_pa
            else:
                q, p = 0.0, ATM_PA
            peak_flow_single = max(peak_flow_single, q)
            if it.concurrent:
                flow += q
            pressure = max(pressure, p)
        return {"concurrent_flow_l_min": flow, "largest_single_l_min": peak_flow_single,
                "peak_pressure_pa": pressure}

    def check_against(self, supply_flow_l_min: float, supply_pressure_pa: float) -> list[str]:
        """Whether a given supply can actually run this package, and if
        not, which way it falls short. Flow and pressure fail
        differently and the difference matters: short of flow the
        machine still does everything, slowly; short of pressure it
        simply cannot do the thing at all."""
        d = self.demand()
        out = [f"  {self.label}: wants {d['concurrent_flow_l_min']:6.1f} L/min with everything moving "
               f"({d['largest_single_l_min']:5.1f} L/min for its biggest function alone) "
               f"at up to {d['peak_pressure_pa'] / 1e5:4.0f} bar"]
        out.append(f"  supply offers {supply_flow_l_min:6.1f} L/min at "
                   f"{supply_pressure_pa / 1e5:4.0f} bar")
        if d["peak_pressure_pa"] > supply_pressure_pa:
            out.append(f"    ! PRESSURE SHORT by {(d['peak_pressure_pa'] - supply_pressure_pa) / 1e5:.0f} bar"
                       f" -- the heaviest function will not move under its rated load at all")
        if d["concurrent_flow_l_min"] > supply_flow_l_min:
            frac = supply_flow_l_min / max(d["concurrent_flow_l_min"], 1e-9)
            out.append(f"    flow short: everything at once runs at {frac * 100:.0f} % of duty speed"
                       f" -- it all still works, just slowly, which is what a real machine does")
            if d["largest_single_l_min"] <= supply_flow_l_min:
                out.append(f"    (one function at a time runs at full speed: that is why an operator"
                           f" learns not to feather three levers at once)")
        elif d["peak_pressure_pa"] <= supply_pressure_pa:
            out.append("    supply is adequate for the whole package at full duty")
        return out


# =====================================================================
#  THE PRESETS -- each one a real machine's real equipment
# =====================================================================
def _cyl(**kw):
    return lambda: LinearActuator(**kw)


def _rot(**kw):
    return lambda: RotaryActuator(**kw)


def _mot(**kw):
    return lambda: FluidMotor(**kw)


LOADOUTS: dict[str, Loadout] = {}


def _add(lo: Loadout) -> Loadout:
    LOADOUTS[lo.identity] = lo
    return lo


_add(Loadout(
    identity="excavator-arm", label="excavator boom/stick/bucket + swing", medium="hydraulic",
    note="the classic three-cylinder arm: the boom cylinder is the big one, and the swing "
         "is a rotary actuator rather than a cylinder because it must turn continuously",
    items=[
        LoadoutItem("boom", _cyl(identity="boom", bore_m=0.120, rod_m=0.080, stroke_m=1.100),
                    duty_speed_m_s=0.25, duty_load_n=180_000.0),
        LoadoutItem("stick", _cyl(identity="stick", bore_m=0.110, rod_m=0.075, stroke_m=1.300),
                    duty_speed_m_s=0.30, duty_load_n=150_000.0),
        LoadoutItem("bucket", _cyl(identity="bucket", bore_m=0.100, rod_m=0.070, stroke_m=0.900),
                    duty_speed_m_s=0.30, duty_load_n=130_000.0),
        LoadoutItem("swing", _rot(identity="swing", kind="vane", swing_deg=360.0,
                                  vane_width_m=0.090, vane_outer_radius_m=0.110,
                                  vane_hub_radius_m=0.045, vanes=2),
                    duty_rate_deg_s=40.0, duty_load_n=9_000.0),
    ]))

_add(Loadout(
    identity="loader", label="wheel-loader lift + tilt", medium="hydraulic",
    note="paired cylinders either side of the machine, which is why the flow is doubled "
         "for what looks like two functions",
    items=[
        LoadoutItem("lift-left", _cyl(identity="lift-left", bore_m=0.125, rod_m=0.070, stroke_m=0.800),
                    duty_speed_m_s=0.25, duty_load_n=200_000.0),
        LoadoutItem("lift-right", _cyl(identity="lift-right", bore_m=0.125, rod_m=0.070, stroke_m=0.800),
                    duty_speed_m_s=0.25, duty_load_n=200_000.0),
        LoadoutItem("tilt-left", _cyl(identity="tilt-left", bore_m=0.140, rod_m=0.080, stroke_m=0.500),
                    duty_speed_m_s=0.20, duty_load_n=220_000.0),
        LoadoutItem("tilt-right", _cyl(identity="tilt-right", bore_m=0.140, rod_m=0.080, stroke_m=0.500),
                    duty_speed_m_s=0.20, duty_load_n=220_000.0),
    ]))

_add(Loadout(
    identity="tipper", label="tipping body + outriggers", medium="hydraulic",
    note="a telescopic ram gives a long stroke from a short closed length, and its force "
         "steps down as each stage takes over -- which is why a tipper is hardest at the start",
    items=[
        LoadoutItem("body-ram", _cyl(identity="body-ram", kind="telescopic", stages=4,
                                     bore_m=0.150, rod_m=0.120, stroke_m=3.600),
                    duty_speed_m_s=0.10, duty_load_n=260_000.0),
        LoadoutItem("outrigger-left", _cyl(identity="outrigger-left", bore_m=0.080, rod_m=0.050,
                                           stroke_m=0.600),
                    duty_speed_m_s=0.10, duty_load_n=60_000.0, concurrent=False),
        LoadoutItem("outrigger-right", _cyl(identity="outrigger-right", bore_m=0.080, rod_m=0.050,
                                            stroke_m=0.600),
                    duty_speed_m_s=0.10, duty_load_n=60_000.0, concurrent=False),
    ]))

_add(Loadout(
    identity="air-tools", label="workshop air tools", medium="pneumatic",
    note="air tools are sized by free-air consumption, not by power: an impact wrench that "
         "runs all day on a big compressor will empty a small receiver in seconds",
    items=[
        LoadoutItem("impact-wrench", _mot(identity="impact-wrench", medium="pneumatic",
                                          kind="vane-air", displacement_cc_rev=25.0,
                                          max_pressure_pa=6.2e5),
                    duty_load_n=0.0),
        LoadoutItem("air-ratchet", _mot(identity="air-ratchet", medium="pneumatic",
                                        kind="vane-air", displacement_cc_rev=12.0,
                                        max_pressure_pa=6.2e5),
                    duty_load_n=0.0, concurrent=False),
        LoadoutItem("clamp", _cyl(identity="clamp", medium="pneumatic", bore_m=0.063,
                                  rod_m=0.020, stroke_m=0.100, rated_pressure_pa=6.2e5),
                    duty_speed_m_s=0.30, duty_load_n=1_200.0),
    ]))

_add(Loadout(
    identity="landing-gear", label="retractable landing gear", medium="hydraulic",
    note="the struts are oleo-pneumatic (gas spring, oil damper, one body) and the retract "
         "actuators are ordinary cylinders -- two different jobs, two different parts",
    items=[
        LoadoutItem("nose-retract", _cyl(identity="nose-retract", bore_m=0.070, rod_m=0.040,
                                         stroke_m=0.700),
                    duty_speed_m_s=0.15, duty_load_n=45_000.0),
        LoadoutItem("main-retract-left", _cyl(identity="main-retract-left", bore_m=0.090,
                                              rod_m=0.050, stroke_m=0.900),
                    duty_speed_m_s=0.15, duty_load_n=90_000.0),
        LoadoutItem("main-retract-right", _cyl(identity="main-retract-right", bore_m=0.090,
                                               rod_m=0.050, stroke_m=0.900),
                    duty_speed_m_s=0.15, duty_load_n=90_000.0),
    ]))

_add(Loadout(
    identity="servo-press", label="servo-hydraulic press axis", medium="hydraulic",
    note="one closed-loop axis, which is where the fluid's stiffness stops being trivia and "
         "becomes the specification",
    items=[
        LoadoutItem("ram-axis", lambda: Positioner(
            identity="ram-axis",
            actuator=LinearActuator(identity="press-ram", bore_m=0.100, rod_m=0.070,
                                    stroke_m=0.250, seal_friction_frac=0.01),
            load_mass_kg=800.0, valve_rated_flow_l_min=60.0),
            duty_speed_m_s=0.05, duty_load_n=100_000.0),
    ]))


def get(identity: str) -> Loadout:
    if identity not in LOADOUTS:
        raise KeyError(f"unknown loadout {identity!r}; known: {', '.join(sorted(LOADOUTS))}")
    return LOADOUTS[identity]


def supply_from_engine(sim) -> dict | None:
    """What an engine's own plant can offer a loadout, if it has one.

    This is the question the loadouts exist to ask: can THIS engine run
    THAT equipment? It reads the real hydraulic circuit the engine
    already carries rather than assuming a supply."""
    plant = getattr(sim, "plant", None)
    h = getattr(plant, "hydraulics", None) if plant is not None else None
    if h is None:
        return None
    return {"flow_l_min": float(getattr(h, "pump_flow_lpm", 0.0) or 0.0),
            "pressure_pa": float(getattr(h, "relief_pressure_pa", 0.0) or 0.0),
            "identity": "engine hydraulic plant"}


def fit_to_bench(loadout: Loadout, rig) -> list[str]:
    """Put a whole loadout on the test bench, which will make up any
    supply it needs."""
    for name, part, item in loadout.build():
        rig.fit(part, name=name, duty_speed_m_s=item.duty_speed_m_s,
                duty_rate_deg_s=item.duty_rate_deg_s)
    return rig.connect_bench_supplies()
