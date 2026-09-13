"""Where hydraulic power actually goes, between the motor and the rod.

WHY THIS EXISTS. The outrigger model was reporting a pump pressure of
27 bar to lift 7.8 tonnes, and cheerfully lifting it. Real outrigger
circuits on machines this size run at 200-350 bar and are sized for it.
The model was not wrong about the CYLINDER -- load divided by piston
area really is 27 bar -- it was wrong because it charged nothing at all
for getting the oil to the cylinder and back.

The chain from the wall to the work has these losses, and the model had
exactly two of them:

    MODELLED ALREADY
      pump volumetric efficiency          0.93
      motor power limit at pressure       yes
      relief dumping, and the heat of it  yes, with oil temperature
      cylinder seal friction              running AND breakaway
      flow-limited rod speed              yes -- speed is flow / area

    NOT MODELLED AT ALL, and each one is real money
      directional / proportional valve    10-25 bar across the spool
      supply line and fittings            3-10 bar at these flows
      return line and filter              3-8 bar, acting on the
                                          ANNULUS, so it pushes back
      counterbalance / load-holding       15-30 bar of pilot pressure,
                                          and a real outrigger MUST
                                          have one or the machine falls
                                          when a hose bursts
      pump hydromechanical efficiency     ~0.90 -- only the volumetric
                                          part was applied, so shaft
                                          power was understated
      electric motor and drive            ~0.88 -- was being applied by
                                          hand outside the model
      cold oil                            temperature was tracked but
                                          never fed back into viscosity

Put together, a real circuit needs the pump to make something like 1.5
to 2 times the pressure the cylinder alone would suggest, and burns
about 1.4 times the electrical energy the ideal calculation gives. That
factor is not a fudge -- it is the sum of the list above, and each term
can be pointed at and argued with separately.

AND THE UNDERSTATED PRESSURE WAS ALSO THE STALL. The demand calculation
divided the load by the area and allowed for RUNNING seal friction. A
cylinder starting from rest has BREAKAWAY friction, which is a multiple
of that, so the pump was told to make a pressure that could not start
the rod moving. At 6 tonnes it scraped past; at 7.8 tonnes it stalled --
and it looked like the machine was too heavy when the truth was the
pressure command was too low.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

ATM_PA = 101_325.0
#: how much thicker cold oil is: ISO 46 at -10 C is roughly 20x its
#: viscosity at 40 C, and pressure drop in a line goes with viscosity
COLD_OIL_REFERENCE_K = 313.15


@dataclass
class CircuitLosses:
    """Everything between the pump outlet and the rod, declared.

    Defaults are ordinary mobile-hydraulic numbers for a machine of
    this size. They are deliberately separate fields rather than one
    efficiency, because when a number comes out wrong you want to know
    WHICH of these is responsible."""
    valve_drop_pa: float = 1.6e6            # proportional spool, both ways
    supply_line_pa: float = 0.7e6           # hose, fittings, manifold
    return_line_pa: float = 0.5e6           # return filter and cooler
    counterbalance_pa: float = 2.2e6        # load-holding valve pilot
    pump_hydromechanical: float = 0.90
    motor_and_drive: float = 0.88
    #: pressure drop scales with flow squared in a line, and with
    #: viscosity, so a cold machine is a slow machine
    reference_flow_l_min: float = 20.0

    def line_scale(self, flow_l_min: float, oil_temp_k: float) -> float:
        """Line losses rise with the square of flow and with cold oil."""
        q = (max(flow_l_min, 0.0) / max(self.reference_flow_l_min, 1e-6)) ** 2
        cold = 1.0 + 2.4 * max(0.0, (COLD_OIL_REFERENCE_K - oil_temp_k) / 40.0)
        return q * cold

    def supply_side_pa(self, flow_l_min: float, oil_temp_k: float) -> float:
        k = self.line_scale(flow_l_min, oil_temp_k)
        return (self.valve_drop_pa + self.supply_line_pa) * k

    def back_pressure_pa(self, flow_l_min: float, oil_temp_k: float) -> float:
        """What the return side pushes back with, on the annulus."""
        return self.return_line_pa * self.line_scale(flow_l_min, oil_temp_k)

    # ------------------------------------------------------------------
    def pump_pressure_for(self, *, load_n: float, piston_area_m2: float,
                          annulus_area_m2: float, flow_l_min: float,
                          oil_temp_k: float = COLD_OIL_REFERENCE_K,
                          seal_friction_frac: float = 0.05,
                          breakaway: float = 1.0,
                          holding: bool = True) -> dict:
        """What the pump must actually make to move this load.

        `breakaway` is 1.0 once moving and the stiction multiplier from
        rest -- and getting this wrong is what stalled the outriggers:
        the cylinder cannot start until the pump makes enough pressure
        to break its own seals loose, which is strictly more than the
        pressure that keeps it moving afterwards."""
        back = self.back_pressure_pa(flow_l_min, oil_temp_k)
        # the load, the return side pushing back on the annulus, and the
        # counterbalance valve you must open to let the load down at all
        at_piston = (abs(load_n) + back * annulus_area_m2) / max(piston_area_m2, 1e-12)
        if holding:
            at_piston += self.counterbalance_pa
        # seals: pressure times area times the friction fraction, and
        # from rest that is multiplied by stiction
        at_piston /= max(1.0 - seal_friction_frac * breakaway, 0.05)
        at_pump = at_piston + self.supply_side_pa(flow_l_min, oil_temp_k)
        ideal = abs(load_n) / max(piston_area_m2, 1e-12)
        return {
            "ideal_pa": ideal + ATM_PA,
            "at_piston_pa": at_piston + ATM_PA,
            "at_pump_pa": at_pump + ATM_PA,
            "back_pressure_pa": back,
            "supply_drop_pa": self.supply_side_pa(flow_l_min, oil_temp_k),
            "counterbalance_pa": self.counterbalance_pa if holding else 0.0,
            "pressure_multiplier": (at_pump + ATM_PA) / max(ideal + ATM_PA, 1.0),
        }

    def electrical_w(self, *, pump_pressure_pa: float, flow_l_min: float) -> float:
        """Wall power for that pressure and flow.

        Hydraulic power is pressure times flow; everything else here is
        the machinery that fails to deliver it."""
        hyd_w = pump_pressure_pa * flow_l_min / 60_000.0
        shaft_w = hyd_w / max(self.pump_hydromechanical, 1e-3)
        return shaft_w / max(self.motor_and_drive, 1e-3)

    def describe(self, r: dict) -> list[str]:
        return [
            f"  ideal (load / piston area)   "
            f"{r['ideal_pa'] / 1e5:6.1f} bar",
            f"  + return backpressure         "
            f"{r['back_pressure_pa'] / 1e5:6.1f} bar on the annulus",
            f"  + counterbalance valve        "
            f"{r['counterbalance_pa'] / 1e5:6.1f} bar",
            f"  + seals, from rest            (in the piston figure)",
            f"  = at the piston               "
            f"{r['at_piston_pa'] / 1e5:6.1f} bar",
            f"  + valve and supply line       "
            f"{r['supply_drop_pa'] / 1e5:6.1f} bar",
            f"  = AT THE PUMP                 "
            f"{r['at_pump_pa'] / 1e5:6.1f} bar"
            f"   ({r['pressure_multiplier']:.2f}x the ideal)",
        ]
