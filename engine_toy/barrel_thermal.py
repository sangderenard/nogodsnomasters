"""Holding the barrel straight by deciding what temperature it is.

TWO WAYS TO DEAL WITH A TUBE THAT BENDS, and the second is far better.

  MINIMISE THE GRADIENT. Wrap it evenly, flow hard, try to keep every
  part of it the same temperature. This is what a thermal sleeve does
  and it is what every tank gun does. It works, it is passive, and its
  ceiling is that you are always losing: the sun, the wind, the rain
  and your own firing keep putting gradients in and you keep taking
  them out. You end up a few kelvin off and a few kelvin is a few
  milliradians.

  COMMAND THE GRADIENT. Do not fight the bend -- aim it. If you can
  make the top of the tube warmer or cooler than the bottom on purpose,
  you can bend the barrel wherever you like, and "wherever you like"
  includes STRAIGHT. The muzzle reference system already measures the
  error; this gives it something to correct WITH.

AND THE GEOMETRY IS ALREADY THERE. The concentric barrel's screw-in
washers have four legs each, at 0, 90, 180 and 270 degrees. Those legs
already divide the coolant annulus into four channels around the tube;
all that was missing was sealing between them lengthwise so the
channels do not simply mix. Do that and every washer bay becomes four
independently valved cells -- axial station by quadrant -- each able to
warm or cool its own quarter of its own length of tube.

COUNTERFLOW, because it is free. Send the coldest water to the hottest
metal, which is the breech end, and let it warm as it travels toward
the muzzle where the tube is cooler. A parallel-flow arrangement puts
the coldest water against the coolest metal and wastes most of the
temperature difference it was given. Same pump, same pipe, more heat
moved, purely from which way round it is plumbed.

THE LIMIT IS TIME, NOT AUTHORITY. Four hundred kilograms of steel does
not change temperature quickly, so this holds a barrel against thermal
DRIFT -- sun, weather, heat soak between bursts -- and it cannot
correct anything that happens inside one burst. Which is fine, because
the fast stuff does not bend the tube; it is the slow soak that does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

WATER_CP_J_PER_KG_K = 4180.0
STEEL_CP_J_PER_KG_K = 470.0
STEEL_ALPHA_PER_K = 12.0e-6

#: the four legs of every screw-in washer, and therefore the four
#: channels they already divide the annulus into
QUADRANTS = ("top", "right", "bottom", "left")
QUADRANT_ANGLE = {"top": 90.0, "right": 0.0, "bottom": 270.0, "left": 180.0}


@dataclass
class ThermalCell:
    """One washer bay, one quadrant: the smallest thing that can be
    given its own temperature."""
    station: int
    quadrant: str
    steel_mass_kg: float
    temperature_k: float = 293.15
    #: what its valve is doing, 0 shut to 1 wide open
    valve: float = 0.5
    #: the thermostat's own target for this cell
    setpoint_k: float = 313.15

    @property
    def angle_deg(self) -> float:
        return QUADRANT_ANGLE[self.quadrant]


@dataclass
class BarrelThermalControl:
    """Every cell of the jacket, and the loop that decides them.

    The bend is computed from the cells, not declared: a temperature
    difference between the top and bottom quadrants of a station bends
    that length of tube, and the muzzle's pointing error is the sum of
    what every station is doing.
    """
    identity: str = "turret.barrel_thermal"
    stations: int = 6
    barrel_length_m: float = 5.28
    outer_diameter_m: float = 0.164
    barrel_mass_kg: float = 407.0
    #: the coolant side
    supply_temperature_k: float = 288.15
    flow_per_cell_kg_s: float = 0.12
    #: counterflow: station 0 is the breech end and gets the coldest water
    counterflow: bool = True
    #: how far open every valve sits before the straightness loop
    #: biases them. There is no heater, so this is the authority the
    #: loop has in BOTH directions: it can only open a valve further if
    #: it was not already wide, and only close it if it was not shut.
    base_valve: float = 0.55
    #: A PROPORTIONAL LOOP ALWAYS SETTLES WITH AN ERROR, because at zero
    #: error it commands zero differential -- and zero differential is
    #: what let the disturbance in. The first version took out 11% of
    #: the bend and sat there looking like it was working. The integral
    #: term is what actually drives it to zero: it keeps winding the
    #: valves apart for as long as any bend remains.
    integral_cross: float = 0.0
    integral_vert: float = 0.0
    integral_gain: float = 45.0
    integral_limit: float = 1.0
    cells: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        per = self.barrel_mass_kg / (self.stations * len(QUADRANTS))
        for s in range(self.stations):
            for q in QUADRANTS:
                self.cells[(s, q)] = ThermalCell(s, q, per)

    # ------------------------------------------------------------------
    @property
    def station_length_m(self) -> float:
        return self.barrel_length_m / self.stations

    def inlet_temperature_k(self, station: int) -> float:
        """What the water is when it reaches this station.

        COUNTERFLOW: the coldest water meets the hottest metal. Plumbed
        the other way the coldest water arrives where the tube is
        already coolest, and most of the temperature difference you
        paid a chiller for is simply never used."""
        if not self.counterflow:
            return self.supply_temperature_k
        # it enters at the breech and warms as it travels forward
        warmed = 0.0
        for s in range(station):
            hot = max(self.cells[(s, q)].temperature_k for q in QUADRANTS)
            warmed += max(0.0, hot - self.supply_temperature_k) * 0.06
        return self.supply_temperature_k + warmed

    def bend_rad(self) -> tuple:
        """Muzzle pointing error, in (cross, vertical) radians.

        Each station contributes its own curvature over its own length,
        and curvature comes from the temperature difference ACROSS the
        tube -- top against bottom, left against right."""
        vert = 0.0
        cross = 0.0
        for s in range(self.stations):
            dt_v = (self.cells[(s, "top")].temperature_k
                    - self.cells[(s, "bottom")].temperature_k)
            dt_h = (self.cells[(s, "right")].temperature_k
                    - self.cells[(s, "left")].temperature_k)
            k_v = STEEL_ALPHA_PER_K * dt_v / self.outer_diameter_m
            k_h = STEEL_ALPHA_PER_K * dt_h / self.outer_diameter_m
            # a station near the breech bends the whole tube ahead of it,
            # so its lever is longer
            lever = self.barrel_length_m - s * self.station_length_m
            vert += k_v * self.station_length_m * (lever / self.barrel_length_m)
            cross += k_h * self.station_length_m * (lever / self.barrel_length_m)
        return (cross, vert)

    def authority_rad(self, max_delta_k: float = 25.0) -> float:
        """How much bend it can command, if it drives top against bottom
        as hard as it is allowed to."""
        k = STEEL_ALPHA_PER_K * max_delta_k / self.outer_diameter_m
        return k * self.barrel_length_m * 0.5

    def time_constant_s(self, delta_k: float = 10.0) -> float:
        """How long to move one quadrant by that much.

        Four hundred kilos of steel is the flywheel here, and it is the
        reason this corrects drift and not transients."""
        mass = self.barrel_mass_kg / len(QUADRANTS)
        energy = mass * STEEL_CP_J_PER_KG_K * delta_k
        # what the water can carry away, at a 10 K approach
        watts = (self.flow_per_cell_kg_s * self.stations
                 * WATER_CP_J_PER_KG_K * 10.0)
        return energy / max(watts, 1.0)

    # ------------------------------------------------------------------
    def step(self, dt: float, firing_watts: float = 0.0,
             sun_watts: float = 0.0, sun_quadrant: str = "top") -> dict:
        """One tick of the real thing: heat in, heat out, temperatures move."""
        for (s, q), cell in self.cells.items():
            heat_in = firing_watts / (self.stations * len(QUADRANTS))
            # the breech end takes more of the firing heat
            heat_in *= 2.0 if s == 0 else (1.4 if s == 1 else 0.8)
            if q == sun_quadrant:
                heat_in += sun_watts / self.stations
            inlet = self.inlet_temperature_k(s)
            flow = self.flow_per_cell_kg_s * max(0.0, min(1.0, cell.valve))
            heat_out = flow * WATER_CP_J_PER_KG_K * (cell.temperature_k - inlet)
            cell.temperature_k += ((heat_in - heat_out) * dt
                                   / (cell.steel_mass_kg * STEEL_CP_J_PER_KG_K))
        cross, vert = self.bend_rad()
        return {"bend_cross_rad": cross, "bend_vert_rad": vert,
                "hottest_k": max(c.temperature_k for c in self.cells.values()),
                "coldest_k": min(c.temperature_k for c in self.cells.values())}

    def command_straight(self, measured_bend_rad: tuple, *, gain: float = 0.6,
                         max_delta_k: float = 25.0) -> dict:
        """Aim the bend at zero, using the MRS reading as the error.

        This is the whole point of segmenting the jacket: the valves are
        no longer trying to make the tube uniform, they are trying to
        make it STRAIGHT, and those are different objectives. A tube
        that is uniformly the wrong shape is still the wrong shape."""
        cross, vert = measured_bend_rad
        # to remove a downward bend, cool the bottom or warm the top
        want_dt_v = -vert * self.outer_diameter_m / (
            STEEL_ALPHA_PER_K * self.barrel_length_m * 0.5) * gain
        want_dt_h = -cross * self.outer_diameter_m / (
            STEEL_ALPHA_PER_K * self.barrel_length_m * 0.5) * gain
        want_dt_v = max(-max_delta_k, min(max_delta_k, want_dt_v))
        want_dt_h = max(-max_delta_k, min(max_delta_k, want_dt_h))
        # THERE IS NO HEATER. The tank is refrigerated and every cell can
        # do exactly one thing: remove heat, faster or slower. So the
        # command is a DIFFERENTIAL of flow, not a target temperature.
        #
        # The first version of this set an absolute setpoint of 313 K
        # against a tube sitting at 293 K, so every cell read eighteen
        # kelvin below target, every valve clamped shut, and the cooling
        # stopped dead -- the sun then heated the top with nothing
        # opposing it and the "corrected" barrel bent thirteen times
        # further than the uncorrected one. A controller that assumes a
        # heat source it has not got will drive the plant to the wrong
        # rail and look like it is working until something disturbs it.
        #
        # Cooling a side makes it SHORTER, which bends the tube toward
        # that side. To take out a downward bend, cool the bottom.
        self.integral_vert = max(-self.integral_limit, min(
            self.integral_limit,
            # PLUS. A positive bend means the top is hot, and the top is
            # cooled by OPENING its valve -- which is a positive bias.
            # Subtracting here shut the top valve instead, which made
            # the top hotter, which made the bend bigger: textbook
            # positive feedback, and with the integral gain at 45 it
            # buried the proportional term that had the sign right.
            self.integral_vert + vert * self.integral_gain))
        self.integral_cross = max(-self.integral_limit, min(
            self.integral_limit,
            self.integral_cross + cross * self.integral_gain))
        bias_v = (-want_dt_v / (2.0 * max(max_delta_k, 1e-9))
                  + self.integral_vert)
        bias_h = (-want_dt_h / (2.0 * max(max_delta_k, 1e-9))
                  + self.integral_cross)
        bias_v = max(-0.5, min(0.5, bias_v))
        bias_h = max(-0.5, min(0.5, bias_h))
        for (s, q), cell in self.cells.items():
            bias = {"top": +bias_v, "bottom": -bias_v,
                    "right": +bias_h, "left": -bias_h}[q]
            # the base flow is what the heat load needs; the bias is how
            # that flow is shared around the tube
            cell.valve = max(0.0, min(1.0, self.base_valve + bias))
            cell.setpoint_k = self.supply_temperature_k
        return {"commanded_dt_vertical_k": want_dt_v,
                "commanded_dt_cross_k": want_dt_h,
                "valve_bias_vertical": bias_v}

    def describe(self) -> list[str]:
        return [
            f"{self.identity}: {self.stations} washer bays x "
            f"{len(QUADRANTS)} quadrants = {len(self.cells)} valved cells",
            f"  the washer legs at 0/90/180/270 already divide the annulus; "
            f"sealing between them lengthwise is what makes them cells",
            f"  {'COUNTERFLOW' if self.counterflow else 'parallel flow'}: "
            f"coldest water to the breech end, warming toward the muzzle",
            f"  bend authority +/-{self.authority_rad() * 1000:.1f} mrad "
            f"at 25 K across the tube",
            f"  time constant {self.time_constant_s():.0f} s for 10 K "
            f"-- this holds against DRIFT, not transients",
        ]


# =====================================================================
#  THE JACKET'S REAL HEAT TRANSFER, FROM THE GRAPH
# =====================================================================
def jacket_conductance(document: dict, *, flow_kg_s: float = 0.35,
                       fluid_key: str = "water",
                       liner_outer_r_m: float = None,
                       bore_outer_r_m: float = None) -> dict:
    """UA for the barrel jacket, computed rather than declared.

    `WeaponStation.soak_barrel` was handed 680 W/K as a literal. That
    number decides how fast the barrel cools, which decides its
    temperature, which decides both its droop and its ringing frequency
    -- so it was a made-up figure with the whole thermal argument
    hanging off it.

    It is derivable, and every term comes from something already
    declared: the chamber nodes carry their own annulus and volume, the
    fluid registry carries conductivity, viscosity and specific heat,
    and the flow is a pump setting.

        Re  = rho v Dh / mu        for an annulus, Dh = 2 x gap
        Nu  = 0.023 Re^0.8 Pr^0.4  turbulent (Dittus-Boelter)
            = 4.36                 laminar, constant wall flux
        h   = Nu k / Dh
        UA  = h x wetted area

    AND THE GAP IS THE WHOLE GAME. A 20 mm liner in a 120 mm outer
    barrel leaves a 45 mm annulus; the flow crawls, it barely goes
    turbulent, and h is poor. Narrowing that gap with a filler sleeve
    raises the velocity, the Reynolds number and h together -- which is
    why real water-cooled barrels have tight jackets and not big baths.
    """
    import fluids
    f = next(x for x in fluids.FLUIDS if x.key == fluid_key)
    chambers = [n for n in document["nodes"]
                if n.get("part_role") == "barrel-jacket-chamber"]
    if not chambers:
        return {"ua_w_per_k": 0.0, "why": "no jacket chambers in the graph"}
    annulus = float(chambers[0]["annulus_m"])
    tube = next((n for n in document["nodes"]
                 if n["identity"] == "turret.outer_barrel"), None)
    r_out = bore_outer_r_m if bore_outer_r_m is not None else (
        float(tube["tube_inner_radius_m"]) if tube else 0.060)
    r_in = liner_outer_r_m if liner_outer_r_m is not None else (r_out - annulus)
    length = float(tube["drum_length_m"]) if tube else 1.70

    area_flow = math.pi * (r_out ** 2 - r_in ** 2)
    v = flow_kg_s / max(f.density_kg_m3 * area_flow, 1e-9)
    dh = 2.0 * (r_out - r_in)
    re = f.density_kg_m3 * v * dh / max(f.viscosity_pa_s, 1e-12)
    pr = (f.viscosity_pa_s * f.specific_heat_j_per_kg_k
          / max(f.conductivity_w_per_m_k, 1e-12))
    turbulent = re > 2300.0
    nu = 0.023 * re ** 0.8 * pr ** 0.4 if turbulent else 4.36
    h = nu * f.conductivity_w_per_m_k / max(dh, 1e-9)
    wetted = 2.0 * math.pi * r_in * length
    return {
        "ua_w_per_k": h * wetted,
        "h_w_per_m2_k": h, "reynolds": re, "prandtl": pr,
        "velocity_m_s": v, "hydraulic_diameter_m": dh,
        "wetted_area_m2": wetted, "annulus_m": annulus,
        "regime": "turbulent" if turbulent else "laminar",
        "chambers": len(chambers),
        "rise_k_at": lambda watts: watts / max(
            flow_kg_s * f.specific_heat_j_per_kg_k, 1e-9),
    }
