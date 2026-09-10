"""The cutoff EXPANDER cylinder -- the one piece of engine hardware
steam and compressed air share, and the reason they're one engine
kind here ("expander"), not two: a working fluid already under
pressure is admitted for part of the stroke (the CUTOFF), expands
for the rest, and is exhausted -- no combustion in the cylinder at
all. What differs between steam and air is only the fluid's own
properties (working_fluids: gamma, R, and where it comes from) and
two real, fluid-specific failure modes modelled below:

  - steam: water hammer. A cold cylinder condenses the first steam
    into water; water is incompressible; the next stroke drives the
    piston into it and cracks the head or cover. Real engines have
    DRAIN COCKS the driver opens to blow the condensate out until the
    cylinder is warm. Admit steam to a cold cylinder with the cocks
    shut and the cylinder is destroyed -- the same "permanently down"
    consequence otto_langen.py gives a top-of-stroke impact.
  - compressed air: icing. Expansion cools the exhaust far below
    freezing (a 0.35 cutoff from room-temperature air leaves at
    ~190 K); moisture in the air freezes at the exhaust port and the
    ice throttles it, raising back pressure until it's thawed. A real
    dryer/separator upstream (fuel_network.CoolerFilter on the air
    line) is the real fix; without one the engine chokes itself.

The indicator diagram is the classic one, disclosed as ideal: admission
at chest pressure p_s to cutoff c (fraction of swept volume), then
polytropic expansion with the fluid's own exponent n to the end of the
stroke, then exhaust against back pressure p_b:

    MEP = p_s * c * (1 + (1 - c^(n-1)) / (n - 1)) - p_b

which is the textbook isothermal form p_s*c*(1 + ln(1/c)) - p_b in the
limit n -> 1. Clearance-volume recompression and valve throttling are
not modelled (disclosed); mechanical_efficiency stands in for the
crosshead/gland/valve-gear friction. Mean torque per revolution is
taken as the working quantity (a double-acting cylinder's torque
pulsation is mild compared with a single-acting combustion one's);
the per-stroke mass drawn from the chest is what the fuel network's
reservoir actually loses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

P_ATM_PA = 101_325.0
T_AMBIENT_K = 293.15
WATER_BOILING_K = 373.15
ICE_ONSET_K = 268.0


@dataclass(frozen=True)
class ExpanderCylinderSpec:
    bore_m: float
    stroke_m: float
    cylinders: int = 1
    double_acting: bool = True
    default_cutoff_frac: float = 0.35        # a real reversing/expansion link can vary it; this is the notch it sits in
    back_pressure_pa: float = P_ATM_PA        # exhaust to atmosphere (a condenser would pull this well below)
    mechanical_efficiency: float = 0.85
    drain_cocks_fitted: bool = True
    cylinder_metal_mass_kg: float = 60.0      # what the first steam has to warm
    warmup_tau_s: float = 45.0                # metal warm-up time constant while steaming through the cocks
    # a real dryer/separator on an air line keeps the exhaust from icing
    # up (see fuel_network.CoolerFilter); this is the cylinder's own
    # icing sensitivity without one
    icing_rate_per_s: float = 0.02
    thaw_rate_per_s: float = 0.05

    @property
    def swept_volume_m3(self) -> float:
        return math.pi * (self.bore_m / 2.0) ** 2 * self.stroke_m

    @property
    def strokes_per_rev(self) -> int:
        return 2 if self.double_acting else 1

    def build(self) -> "ExpanderCylinderBank":
        return ExpanderCylinderBank(spec=self)


def mean_effective_pressure_pa(supply_pressure_pa: float, cutoff_frac: float, n: float,
                               back_pressure_pa: float) -> float:
    c = max(1e-3, min(1.0, cutoff_frac))
    if abs(n - 1.0) < 1e-6:
        expansion = 1.0 + math.log(1.0 / c)
    else:
        expansion = 1.0 + (1.0 - c ** (n - 1.0)) / (n - 1.0)
    return supply_pressure_pa * c * expansion - back_pressure_pa


@dataclass
class ExpanderCylinderBank:
    spec: ExpanderCylinderSpec
    drain_cocks_open: bool = True             # a driver opens them for warm-up; the real default on a cold engine
    cylinder_metal_temp_k: float = T_AMBIENT_K
    ice_frac: float = 0.0
    destroyed: bool = False
    last_torque_nm: float = field(default=0.0, init=False)
    last_mass_kg_s: float = field(default=0.0, init=False)
    last_exhaust_temp_k: float = field(default=T_AMBIENT_K, init=False)
    last_mep_pa: float = field(default=0.0, init=False)
    condensate_present: bool = field(default=False, init=False)

    def step(self, dt: float, omega_rad_s: float, chest_pressure_pa: float, supply_temp_k: float,
             supply_density_kg_m3: float, gamma: float, is_steam: bool, cutoff_frac: float | None = None,
             direction: float = 1.0, dry_supply: bool = False) -> tuple[float, float]:
        """One tick: returns (mean shaft torque N*m, fluid drawn kg/s).
        `chest_pressure_pa` is what the throttle/regulator actually
        delivers to the steam chest this tick (absolute); `direction`
        is the reversing link (+1 forward, -1 reverse, 0 mid-gear)."""
        spec = self.spec
        c = spec.default_cutoff_frac if cutoff_frac is None else cutoff_frac
        if self.destroyed or chest_pressure_pa <= spec.back_pressure_pa or direction == 0.0:
            self.last_torque_nm = 0.0
            self.last_mass_kg_s = 0.0
            self.last_mep_pa = 0.0
            self._relax(dt, steaming=False, is_steam=is_steam)
            return 0.0, 0.0

        # --- steam: condensate and water hammer ---
        if is_steam:
            cold = self.cylinder_metal_temp_k < WATER_BOILING_K
            # a warm cylinder re-evaporates what it held; a cold one
            # condenses more every stroke
            self.condensate_present = cold
            if self.condensate_present and not self.drain_cocks_open and abs(omega_rad_s) > 0.5:
                # a stroke driven into trapped water: the real, documented
                # cracked cover/head. Permanently down.
                self.destroyed = True
                self.last_torque_nm = 0.0
                self.last_mass_kg_s = 0.0
                return 0.0, 0.0
            # the metal warms toward the steam's own temperature while
            # steam is flowing (through the cocks or through the engine)
            self.cylinder_metal_temp_k += (supply_temp_k - self.cylinder_metal_temp_k) * min(1.0, dt / spec.warmup_tau_s)

        # --- icing (air): the exhaust port chokes ---
        back_pressure = spec.back_pressure_pa * (1.0 + 2.0 * self.ice_frac)

        mep = mean_effective_pressure_pa(chest_pressure_pa, c, gamma, back_pressure)
        mep = max(0.0, mep)
        # with the drain cocks open a real share of each admission
        # blows straight out the cocks -- warm-up costs torque and fluid
        cock_loss = 0.5 if (is_steam and self.drain_cocks_open) else 0.0
        work_per_rev_j = mep * spec.swept_volume_m3 * spec.strokes_per_rev * spec.cylinders * (1.0 - cock_loss)
        torque = work_per_rev_j / (2.0 * math.pi) * spec.mechanical_efficiency * direction
        mass_per_rev = supply_density_kg_m3 * spec.swept_volume_m3 * c * spec.strokes_per_rev * spec.cylinders
        mass_kg_s = mass_per_rev * abs(omega_rad_s) / (2.0 * math.pi) * (1.0 + cock_loss)
        # polytropic expansion temperature at release -- for steam the
        # expansion runs wet and cannot fall below saturation at the
        # back pressure (~373 K exhausting to atmosphere), a disclosed
        # floor in place of a full steam-quality solve
        t_release = supply_temp_k * (max(c, 1e-3) ** (gamma - 1.0))
        self.last_exhaust_temp_k = max(t_release, WATER_BOILING_K) if is_steam else t_release
        self._relax(dt, steaming=True, is_steam=is_steam, dry_supply=dry_supply)
        self.last_torque_nm = torque
        self.last_mass_kg_s = mass_kg_s
        self.last_mep_pa = mep
        return torque, mass_kg_s

    def _relax(self, dt: float, steaming: bool, is_steam: bool, dry_supply: bool = False) -> None:
        spec = self.spec
        if is_steam:
            if not steaming:
                self.cylinder_metal_temp_k += (T_AMBIENT_K - self.cylinder_metal_temp_k) * min(1.0, dt / (spec.warmup_tau_s * 8.0))
            self.ice_frac = 0.0
        else:
            # no moisture in the supply (a real dryer/separator on the
            # line), nothing to freeze at the port
            if steaming and self.last_exhaust_temp_k < ICE_ONSET_K and not dry_supply:
                self.ice_frac = min(1.0, self.ice_frac + spec.icing_rate_per_s * dt)
            else:
                self.ice_frac = max(0.0, self.ice_frac - spec.thaw_rate_per_s * dt)

    @property
    def warm(self) -> bool:
        return self.cylinder_metal_temp_k >= WATER_BOILING_K
