"""The automatic transmission, which is a hydraulic machine.

It is tempting to file this next to the manual gearbox, and that is the
wrong shelf. A manual box is a case of gears with a lubricant in it. An
automatic is a pump, a torque converter, a valve body and a stack of
hydraulically applied clutch packs, and the fluid in it is the WORKING
FLUID of all four: it transmits the torque, it applies the clutches, it
shifts the gears, and it carries out heat that the gears never made.

That difference is not a label. It shows up as behaviour:

  - The pump is crank-driven, so pressure exists only while the engine
    turns. An automatic with a dead engine has no line pressure, which
    is the real reason such a vehicle cannot be push-started and why
    towing one with the wheels down destroys it.
  - Line pressure APPLIES the clutch packs (couplings.py's own
    wet-multi-plate law, the same one the dyno's clutch uses). Low fluid
    or low pressure does not mean "slightly worse" -- it means the packs
    slip, and slipping packs cook the fluid that was supposed to apply
    them.
  - The converter is a fluid coupling with a stator: torque goes as the
    square of the speed difference and multiplies at stall, and all of
    that slip is heat dumped into the same fluid. This is why an
    automatic must have a cooler and a manual need not.
  - Overheated fluid oxidises and loses its friction characteristics
    permanently, so heat damage here is cumulative, not something that
    recovers when it cools down.

So the primitive here is a fluid machine with a pump, a converter, a
line-pressure circuit and a heat budget -- and its fluid comes from the
same registry as every other fluid (fluids.py), so it leaks, sprays,
burns and gauges without any of those systems knowing what a
transmission is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ATF_DENSITY_KG_M3 = 850.0
ATF_SPECIFIC_HEAT_J_PER_KG_K = 2000.0
# a real converter's capacity factor: torque = K * rho * D^5 * n^2, but
# expressed the way this sim already works, as a coefficient on the
# square of the speed difference
CONVERTER_K_NM_PER_RAD_S2 = 0.0021
STALL_TORQUE_RATIO = 2.1          # what the stator multiplies by at stall
# where the fluid stops being fluid: above this it oxidises, and the
# damage stays done
ATF_OXIDATION_K = 394.15          # 121 C -- the classic "every 20 F halves its life" knee
ATF_CHAR_K = 423.15               # 150 C -- varnish, and the packs start to glaze


@dataclass
class AutomaticTransmission:
    """One real automatic, as a fluid machine."""
    fluid_l: float = 9.5
    gear_ratios: tuple[float, ...] = (2.84, 1.62, 1.00, 0.70)
    final_drive_ratio: float = 3.42
    max_line_pressure_pa: float = 1.8e6      # ~18 bar at full throttle
    pump_displacement_l_per_rev: float = 0.012
    cooler_ua_w_per_k: float = 160.0
    lock_up_capable: bool = True
    lock_up_min_ratio: float = 0.86          # converter locks near coupling speed

    # live state
    fluid_kg: float = field(default=0.0, init=False)
    temp_k: float = field(default=293.15, init=False)
    line_pressure_pa: float = field(default=0.0, init=False)
    converter_slip_rad_s: float = field(default=0.0, init=False)
    converter_torque_nm: float = field(default=0.0, init=False)
    locked_up: bool = field(default=False, init=False)
    slip_heat_w: float = field(default=0.0, init=False)
    life_frac: float = field(default=1.0, init=False)
    packs_slipping: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.fluid_kg = self.fluid_l * ATF_DENSITY_KG_M3 / 1000.0

    # ---------------------------------------------------------------
    @property
    def fill_frac(self) -> float:
        nominal = max(self.fluid_l * ATF_DENSITY_KG_M3 / 1000.0, 1e-9)
        return max(0.0, min(1.0, self.fluid_kg / nominal))

    def lose(self, litres: float) -> None:
        """What a hole takes out. Kept as the one place fluid leaves, so
        the leak path and the gauge can never disagree."""
        self.fluid_kg = max(0.0, self.fluid_kg - litres * ATF_DENSITY_KG_M3 / 1000.0)

    def step(self, dt: float, crank_omega_rad_s: float, output_omega_rad_s: float,
             throttle: float, ambient_k: float = 293.15) -> float:
        """Advance the machine and return the torque it passes.

        The order is the real one: the pump makes pressure from crank
        speed, the pressure applies the packs, the converter passes what
        it passes, and everything that slipped becomes heat in the same
        fluid that was doing the work."""
        if dt <= 0.0:
            return 0.0
        fill = self.fill_frac
        # THE PUMP. Crank-driven, so no engine means no pressure at all.
        # Aerated fluid from a low level makes far less than its share,
        # which is why a slightly low automatic slips long before it
        # sounds low.
        speed_frac = min(1.0, crank_omega_rad_s / 180.0)
        self.line_pressure_pa = (self.max_line_pressure_pa * speed_frac
                                 * (0.55 + 0.45 * min(1.0, max(0.0, throttle)))
                                 * (fill ** 2))

        # THE CONVERTER. A fluid coupling: torque with the square of the
        # speed difference, multiplied by the stator while it is really
        # slipping, and nothing at all if the fluid is gone.
        slip = crank_omega_rad_s - output_omega_rad_s
        self.converter_slip_rad_s = slip
        ratio = (output_omega_rad_s / crank_omega_rad_s) if crank_omega_rad_s > 1e-6 else 0.0
        self.locked_up = bool(self.lock_up_capable and ratio >= self.lock_up_min_ratio
                              and self.line_pressure_pa > self.max_line_pressure_pa * 0.3)
        if fill <= 0.02:
            self.converter_torque_nm = 0.0
            self.slip_heat_w = 0.0
        elif self.locked_up:
            # the lock-up clutch removes the slip entirely: a mechanical
            # connection, no heat, which is the whole point of it
            self.converter_torque_nm = 0.0
            self.slip_heat_w = 0.0
        else:
            mult = 1.0 + (STALL_TORQUE_RATIO - 1.0) * max(0.0, min(1.0, 1.0 - ratio))
            self.converter_torque_nm = (CONVERTER_K_NM_PER_RAD_S2 * slip * abs(slip)
                                        * mult * fill)
            self.slip_heat_w = abs(self.converter_torque_nm * slip)

        # THE PACKS. Applied by line pressure; when there is not enough
        # of it they slip, and slipping packs make their own heat.
        self.packs_slipping = bool(self.line_pressure_pa < self.max_line_pressure_pa * 0.25
                                   and crank_omega_rad_s > 50.0)
        if self.packs_slipping:
            self.slip_heat_w += abs(self.converter_torque_nm) * 8.0

        # THE HEAT, into the fluid that is doing all of the above
        mass = max(self.fluid_kg, 1e-3)
        rejected = self.cooler_ua_w_per_k * max(0.0, self.temp_k - ambient_k)
        self.temp_k += (self.slip_heat_w - rejected) * dt / (mass * ATF_SPECIFIC_HEAT_J_PER_KG_K)
        self.temp_k = max(ambient_k, self.temp_k)

        # and the damage, which does not undo itself when it cools
        if self.temp_k > ATF_OXIDATION_K:
            over = self.temp_k - ATF_OXIDATION_K
            # the real rule of thumb: life halves for every ~10 K above
            # the knee
            self.life_frac = max(0.0, self.life_frac - (2.0 ** (over / 10.0)) * dt / 3.0e5)
        return self.converter_torque_nm

    def describe(self) -> list[str]:
        out = [f"  AUTO TRANS  fluid {self.fill_frac * 100:3.0f}%  {self.temp_k - 273.15:4.0f} C"
               f"  line {self.line_pressure_pa / 1e5:4.1f} bar"
               f"  {'LOCKED' if self.locked_up else f'slip {self.converter_slip_rad_s:5.1f} rad/s'}"]
        if self.packs_slipping:
            out.append("    ! line pressure too low: clutch packs slipping")
        if self.temp_k > ATF_CHAR_K:
            out.append(f"    ! fluid over {ATF_CHAR_K - 273.15:.0f} C: varnishing")
        if self.life_frac < 0.5:
            out.append(f"    ! fluid life {self.life_frac * 100:.0f}% -- oxidised, will not recover")
        return out
