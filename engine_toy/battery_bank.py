"""A series-parallel battery bank, sized by the duty it has to do.

HOW MANY BATTERIES IS NOT A PREFERENCE. It is two separate questions
with two different answers, and a bank that satisfies one and not the
other is useless:

  ENERGY   how many watt-hours the job takes. Strings in PARALLEL add
           amp-hours, so this decides how many strings.
  CURRENT  how many amps the job draws, which is power divided by bus
           voltage. Cells in SERIES raise the voltage and therefore
           LOWER the current for the same power. This decides how many
           batteries are in each string -- and it is usually the harder
           constraint, because current is what melts cable, sags the
           bus and cooks the battery's own internal resistance.

At 24 V a 7.5 kW motor draws 312 A. That is a starter-motor current
drawn continuously, it needs 95 mm2 cable, and the bank's own internal
resistance turns a meaningful fraction of the pack into heat. The same
motor on a 48 V bus draws 156 A, on 96 V it draws 78 A. This is the
entire reason vehicles that actually do hydraulic work on batteries do
not run them at 24 V.

THE DUTY IS UP AND DOWN, NOT UP. A machine that can raise itself and
cannot lower itself is stranded in the deployed position, so the cycle
that has to be survivable is both, plus the laying the gun does in
between, plus a reserve -- because the last thing to go should be the
ability to come down and drive away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from dc_power import DCBattery, CHEMISTRY_BY_KEY


@dataclass
class DutyItem:
    """One thing the bank has to run, and for how long."""
    name: str
    electrical_w: float
    seconds: float
    note: str = ""

    @property
    def wh(self) -> float:
        return self.electrical_w * self.seconds / 3600.0


@dataclass
class BatteryBank:
    """`series` batteries per string, `parallel` strings."""
    identity: str = "base.bank"
    chemistry: str = "lifepo4"
    module_ah: float = 100.0
    module_nominal_v: float = 12.8          # one LiFePO4 block of 4 cells
    series: int = 4
    parallel: int = 2
    state_of_charge: float = 1.0

    @property
    def modules(self) -> int:
        return self.series * self.parallel

    @property
    def bus_v(self) -> float:
        return self.module_nominal_v * self.series

    @property
    def capacity_ah(self) -> float:
        return self.module_ah * self.parallel

    @property
    def nameplate_wh(self) -> float:
        return self.bus_v * self.capacity_ah

    @property
    def usable_wh(self) -> float:
        return self.nameplate_wh * CHEMISTRY_BY_KEY[self.chemistry].usable_depth

    @property
    def internal_ohms(self) -> float:
        """Series adds resistance, parallel divides it -- which is the
        other reason a tall, narrow bank beats a short, wide one."""
        per = CHEMISTRY_BY_KEY[self.chemistry].internal_r_per_ah / max(
            self.module_ah, 1e-6) * 100.0
        return per * self.series / max(self.parallel, 1)

    def current_a(self, watts: float) -> float:
        return watts / max(self.bus_v, 1e-6)

    def sag_v(self, watts: float) -> float:
        return self.current_a(watts) * self.internal_ohms

    def loss_w(self, watts: float) -> float:
        i = self.current_a(watts)
        return i * i * self.internal_ohms

    def cable_mm2(self, watts: float, *, a_per_mm2: float = 4.0) -> float:
        """Copper section for the bus, at a conservative continuous
        current density."""
        return self.current_a(watts) / a_per_mm2

    # ------------------------------------------------------------------
    def report(self, duty: list[DutyItem], *, reserve: float = 0.30) -> dict:
        need_wh = sum(d.wh for d in duty)
        peak_w = max((d.electrical_w for d in duty), default=0.0)
        with_reserve = need_wh * (1.0 + reserve)
        return {
            "modules": self.modules,
            "arrangement": f"{self.series}S{self.parallel}P",
            "bus_v": self.bus_v,
            "capacity_ah": self.capacity_ah,
            "nameplate_wh": self.nameplate_wh,
            "usable_wh": self.usable_wh,
            "need_wh": need_wh,
            "need_with_reserve_wh": with_reserve,
            "spare_wh": self.usable_wh - with_reserve,
            "cycles_available": self.usable_wh / max(need_wh, 1e-9),
            "peak_w": peak_w,
            "peak_a": self.current_a(peak_w),
            "sag_v": self.sag_v(peak_w),
            "sag_frac": self.sag_v(peak_w) / max(self.bus_v, 1e-9),
            "loss_w": self.loss_w(peak_w),
            "cable_mm2": self.cable_mm2(peak_w),
            "enough": self.usable_wh >= with_reserve,
        }

    def describe(self, duty: list[DutyItem], *, reserve: float = 0.30) -> list[str]:
        r = self.report(duty, reserve=reserve)
        out = [
            f"{self.identity}: {r['modules']} x {self.module_ah:.0f} Ah "
            f"{CHEMISTRY_BY_KEY[self.chemistry].label}, wired {r['arrangement']}",
            f"  bus {r['bus_v']:.0f} V, {r['capacity_ah']:.0f} Ah "
            f"= {r['nameplate_wh'] / 1000:.2f} kWh nameplate, "
            f"{r['usable_wh'] / 1000:.2f} kWh usable",
            f"  duty {r['need_wh'] / 1000:.2f} kWh + {reserve * 100:.0f}% reserve "
            f"= {r['need_with_reserve_wh'] / 1000:.2f} kWh"
            + ("   OK" if r["enough"] else "   NOT ENOUGH"),
            f"  that is {r['cycles_available']:.1f} full up-and-down cycles "
            f"before it is flat",
            f"  peak {r['peak_w'] / 1000:.1f} kW = {r['peak_a']:.0f} A at "
            f"{r['bus_v']:.0f} V, sag {r['sag_v']:.2f} V "
            f"({r['sag_frac'] * 100:.1f}%), {r['loss_w']:.0f} W lost in the bank",
            f"  bus cable about {r['cable_mm2']:.0f} mm2 copper",
        ]
        for d in duty:
            out.append(f"    {d.name:28s} {d.electrical_w / 1000:5.1f} kW x "
                       f"{d.seconds:6.1f} s = {d.wh / 1000:5.2f} kWh"
                       + (f"   {d.note}" if d.note else ""))
        return out


def size_bank(duty: list[DutyItem], *, module_ah: float = 100.0,
              chemistry: str = "lifepo4", reserve: float = 0.30,
              max_bus_current_a: float = 200.0,
              max_modules: int = 16) -> BatteryBank:
    """The smallest bank that carries the energy AND the current.

    Series first, because current is the binding constraint: raise the
    bus until the peak draw is something cable can actually carry, then
    add strings until the energy is there."""
    nominal = {"lifepo4": 12.8, "agm": 12.78, "flooded": 12.6}[chemistry]
    peak_w = max((d.electrical_w for d in duty), default=0.0)
    series = 1
    while series * nominal < 1e-6 or peak_w / (series * nominal) > max_bus_current_a:
        series += 1
        if series > 16:
            break
    for parallel in range(1, max_modules // max(series, 1) + 1):
        bank = BatteryBank(chemistry=chemistry, module_ah=module_ah,
                           module_nominal_v=nominal, series=series,
                           parallel=parallel)
        if bank.report(duty, reserve=reserve)["enough"]:
            return bank
    return BatteryBank(chemistry=chemistry, module_ah=module_ah,
                       module_nominal_v=nominal, series=series,
                       parallel=max(1, max_modules // max(series, 1)))
