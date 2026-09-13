"""The hydraulic circuit: where its heat comes from, where its dirt
comes from, and what both of them do to it.

A hydraulic system is a pump, a relief valve, some actuators, a filter
and a tank. Almost everything interesting about running one comes from
three facts, and this models those three rather than a generic "oil
gets hot" number:

1. ALL THE LOST PRESSURE BECOMES HEAT, AND THE RELIEF VALVE IS A HEATER.
   A pump puts p x Q of hydraulic power into the oil. Whatever the
   actuators do not turn into useful work comes straight back as heat.
   The worst case is not a hard-working machine -- it is a machine held
   ON THE RELIEF at zero output: the pump is making full pressure at
   full flow and every watt of it is dumped across the relief valve as
   heat, with nothing at all to show for it. That is why a hydraulic
   system cooks while an operator holds a lever against a stop.

2. THE TANK BREATHES, AND THAT IS HOW WATER GETS IN.
   When a cylinder extends, oil leaves the tank and the tank draws air
   IN through its breather; when it retracts, it pushes air back out.
   The air it draws in is humid, and the tank is cooler than the air at
   night, so the moisture condenses on the tank walls and runs into the
   oil. Nobody spills water into a hydraulic tank -- it arrives on the
   air, one breath at a time. A desiccant breather is what stops it,
   and it stops it only until the desiccant is spent.

   Water in oil is not a cosmetic problem. Above roughly 200 ppm it is
   past what the additives hold in solution and starts coming out as
   free water; free water in a bearing contact flashes to steam under
   load and blows the metal out of the surface, and it is the single
   most destructive contaminant a hydraulic system sees.

3. TEMPERATURE DECIDES BOTH VISCOSITY AND HOW LONG THE OIL LASTS.
   Viscosity falls steeply with temperature (Walther/ASTM D341, the
   log-log relation this uses). Too hot and the film is too thin to
   keep metal apart; too cold and the pump cannot draw its own charge
   through the inlet and cavitates. Oxidation follows the usual
   chemical-rate rule: the oil's life roughly HALVES for every 10 K
   above 60 C, which is the real reason a chiller is worth its cost.

Dirt is the fourth thing, and it behaves like the air system's dust:
it comes in on the breather and from the system's own wear, the filter
takes out a fraction of it per pass, and what is left grinds. The
standard measure is the ISO 4406 cleanliness code, which is what this
reports, because that is the number a real machine is judged on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

ATM_PA = 101_325.0
OIL_DENSITY_KG_M3 = 870.0
OIL_CP_J_KG_K = 1900.0
WATER_SATURATION_PPM_AT_60C = 300.0     # how much water this oil can hold dissolved when warm
BREATHER_HUMID_AIR_G_PER_M3 = 12.0      # water carried by ordinary humid air
OXIDATION_REFERENCE_K = 333.15          # 60 C: the reference the halving rule is quoted at
OXIDATION_HALVING_K = 10.0


def viscosity_cst(temp_k: float, vg: float = 46.0, vi: float = 100.0) -> float:
    """Kinematic viscosity of a mineral hydraulic oil, by the Walther /
    ASTM D341 log-log relation that every oil datasheet is built on:
        log10(log10(nu + 0.7)) = A - B log10(T)
    pinned through the grade's own two defining points -- an ISO VG
    oil is DEFINED as its viscosity at 40 C, and a VI of ~100 fixes
    the 100 C end."""
    nu40 = vg
    nu100 = max(4.0, vg * 0.145 * (vi / 100.0))       # a VG46/VI100 oil is ~6.8 cSt at 100 C
    t1, t2 = 313.15, 373.15
    z1 = math.log10(math.log10(nu40 + 0.7))
    z2 = math.log10(math.log10(nu100 + 0.7))
    b = (z1 - z2) / (math.log10(t2) - math.log10(t1))
    a = z1 + b * math.log10(t1)
    z = a - b * math.log10(max(temp_k, 233.15))
    return max(1.0, 10.0 ** (10.0 ** z) - 0.7)


def iso4406_code(particles_per_ml_4um: float, particles_per_ml_6um: float, particles_per_ml_14um: float) -> str:
    """The ISO 4406 cleanliness code: three range numbers, one per size
    channel, each the log2 band the count per millilitre falls in."""
    def band(n: float) -> int:
        if n <= 0.01:
            return 0
        return max(0, min(28, int(math.floor(math.log2(max(n, 1e-6))) + 1)))
    return f"{band(particles_per_ml_4um)}/{band(particles_per_ml_6um)}/{band(particles_per_ml_14um)}"


@dataclass
class HydraulicCircuit:
    """A real open-centre hydraulic system on the engine.

    `commanded_flow_frac` is the operator's lever: how much of the
    pump's flow the actuators are actually taking. `load_frac` is how
    hard they are pushing against it. Hold the lever over with nothing
    moving (flow 0, load 1) and the whole pump output goes over the
    relief valve as heat -- which is the case worth being able to see.
    """
    # hardware
    pump_displacement_cc_rev: float = 60.0
    pump_drive_ratio: float = 1.0            # off the crank
    relief_pressure_pa: float = 21_000_000.0  # 210 bar
    tank_capacity_l: float = 60.0
    oil_l: float = 54.0
    oil_grade_vg: float = 46.0
    # A real machine rejects its hydraulic heat through a dedicated
    # oil-to-air cooler, sized off the installed power -- the chiller is
    # a supplement to that, not a substitute for it. UA is the cooler's
    # own conductance: a 240 W/K core rejects about 12 kW at a 50 K
    # approach, which is what a 37 kW system is normally given.
    # A REGULATED PILOT CIRCUIT. Clutches, brakes and valve pilots are
    # never applied off the implement circuit's working pressure -- that
    # rises and falls with whatever the machine happens to be doing, and
    # would drop a clutch the moment an operator centred a lever. They
    # run off a separate low-pressure pilot supply, taken through a
    # reducing valve (or its own small pump) and regulated to a steady
    # 25-40 bar that is available the whole time the pump is turning.
    # That is what applies the dyno coupling here.
    pilot_pressure_pa: float = 3_500_000.0
    pilot_available: bool = False
    cooler_ua_w_per_k: float = 240.0
    cooler_fan_running: bool = True
    return_filter_fitted: bool = True
    return_filter_beta: float = 200.0        # beta-200 at the rated size: it removes 99.5 % per pass
    desiccant_breather: bool = True
    breather_desiccant_capacity_kg: float = 0.15
    # A DRY-AIR BLANKET instead of (or as well as) a breather: the tank
    # is sealed and held at a slight positive pressure by treated air
    # from the plant's own dryer, through a regulator and a relief.
    #
    # Real practice, not an invention. Every transport aircraft
    # pressurises its hydraulic reservoirs with engine bleed air, for
    # exactly two reasons: nothing from outside can get in, and the
    # pump always has positive inlet pressure so it cannot cavitate.
    # Large industrial power units and power-station lube-oil systems
    # do the same with nitrogen or with instrument-quality dried air.
    #
    # It beats a desiccant breather because a desiccant is a consumable
    # that fails SILENTLY once spent, whereas a blanket cannot
    # saturate -- but it is only as dry as its supply. Feed it wet air
    # (the dryer off, the chiller unpowered) and it is WORSE than a
    # breather, because it pumps that moisture straight in rather than
    # merely letting it past. That coupling is the point of having it.
    # HOW MUCH A RESERVOIR ACTUALLY BREATHES. Not the pump flow -- that
    # oil comes straight back. A tank's level swings by the NET ROD
    # VOLUME of its cylinders: with a cylinder extended, the oil that
    # is now on the bore side minus what came off the rod side is out
    # of the tank, and the level drops by exactly that. Retract it and
    # the level comes back. So the gas exchanged per work cycle is the
    # swing volume, and the rate is how often the machine cycles.
    cylinder_swing_l: float = 18.0
    work_cycles_per_min: float = 2.0
    # A BLADDER (or diaphragm) reservoir seals the gas side off from
    # the oil entirely and simply changes shape as the level moves --
    # so it exchanges NO gas at all with anything. This is the real
    # answer for a machine whose cylinders cycle all day: a consumable
    # blanket gas cannot keep up with that (see the numbers this model
    # produces), whereas a bladder needs no supply, cannot saturate,
    # and cannot let anything in. Aircraft do the equivalent with a
    # piston driven by system pressure.
    bladder_separated: bool = False
    blanket_fitted: bool = False
    blanket_pressure_pa: float = ATM_PA + 50_000.0    # a reservoir is not a pressure vessel: ~0.5 bar gauge
    blanket_relief_pa: float = ATM_PA + 100_000.0
    blanket_supply_humidity_kg_kg: float = 0.0        # set each tick from the air train
    blanket_supply_available: bool = False
    blanket_air_used_kg: float = 0.0
    # A THREE-WAY SELECTOR on the reservoir's gas inlet, which is how a
    # real installation gets redundancy: plant air is the everyday
    # supply, a nitrogen bottle is the clean backup (and the only one
    # that is genuinely dry AND inert -- no oxygen over the oil means
    # no oxidation in the headspace at all), and a desiccant breather
    # is the last resort that needs no supply of any kind. The
    # selector's whole job is that the tank must NEVER be left with no
    # way to equalise: a sealed tank with no make-up gas will pull a
    # vacuum and collapse, or suck in through whatever seal is weakest.
    blanket_source: str = "auto"          # "auto" | "plant-air" | "nitrogen" | "breather"
    # THE LAST RESORT. The tank is normally sealed; if the plant air is
    # gone (or not dry enough to be worth using) and the nitrogen bottle
    # is empty, the tank still has to be able to equalise or it will
    # pull a vacuum and collapse. So a check valve cracks at a small
    # negative pressure and admits air THROUGH a desiccant cartridge.
    # It is sized for occasional use, not for a shift's breathing, and
    # its capacity is finite -- which is exactly the point of watching
    # it: if this is working hard, something upstream has failed.
    vacuum_break_crack_pa: float = 3_000.0       # below atmospheric
    vacuum_break_bed: object = None              # desiccant.DesiccantBed, made on demand
    vacuum_break_cartridge_kg: float = 0.4
    vacuum_break_events: int = 0
    blanket_gas_kg: float = 0.0          # everything the blanket has consumed, whatever the source
    blanket_active_source: str = "breather"
    nitrogen_fitted: bool = False
    nitrogen_capacity_kg: float = 1.2     # a 10 L bottle at 200 bar
    nitrogen_bottle_pressure_pa: float = 20_000_000.0
    nitrogen_fill_frac: float = 1.0
    nitrogen_used_kg: float = 0.0
    chiller_fitted: bool = True
    chiller_target_k: float = 318.15
    chiller_trim_capacity_w: float = 6000.0   # the chiller is a trim duty; the air cooler carries the rest
    # WARMING A COLD SYSTEM. Cold oil is a real problem -- below about
    # -10 C a VG46 is past 2000 cSt and the pump cannot draw its own
    # charge -- but an immersion heater is the crude answer and the one
    # that goes wrong. What matters is WATT DENSITY, not total watts: a
    # high-density element boils the oil film at its own surface, cokes
    # it on as varnish, and that varnish then insulates the element so
    # it runs hotter and cokes faster. Real lube/hydraulic immersion
    # heaters are specified at low watt density (~1.2-1.5 W/cm2) for
    # exactly that reason, with a thermostat AND a low-level cutout,
    # because an element running uncovered burns out and can start a
    # fire. So three ways to warm it, gentlest first:
    #   1. the engine's own jacket coolant through an oil/coolant
    #      exchanger -- free heat, no hot spot, cannot coke
    #   2. the relief valve, which is already a 37 kW heater: run the
    #      pump against a reduced relief setting and the oil warms
    #      itself (the classic warm-up mode)
    #   3. a low-watt-density element, last, for when the engine is
    #      cold too
    # A HYDRAULIC FAN DRIVE: the actuator that lives in the engine bay
    # and earns its place there. Big machines drive the cooling fan
    # through a hydraulic motor instead of a belt, and the reason is
    # that fan speed then has nothing to do with engine speed -- the
    # fan runs flat out at idle on a hot day and idles at full rpm on a
    # cold one, which a belt can never do. It also reverses, which is
    # how a machine blows chaff back out of its own radiator core.
    #
    # It closes a real loop with everything else here: hot oil wants
    # cooling, the fan drive burns hydraulic power to get it (and every
    # watt it burns is more heat into the same oil), but the air it
    # moves cools the oil cooler AND the refrigerant condenser, which
    # drops the loop's head pressure and gives the chillers back some
    # capacity.
    fan_drive_fitted: bool = False
    fan_drive_displacement_cc_rev: float = 30.0
    fan_drive_max_rpm: float = 2200.0
    fan_drive_command: float = 0.0           # 0..1, what the thermostat is asking for
    fan_thermostat: bool = True              # False lets a caller drive the command directly
    fan_start_k: float = 313.15              # 40 C: it starts winding up here
    fan_full_span_k: float = 25.0            # and is flat out 25 K above that
    fan_drive_reverse: bool = False
    fan_drive_rpm: float = 0.0
    fan_drive_w: float = 0.0
    fan_airflow_frac: float = 0.0            # 0..1, what the cores actually get
    warm_target_k: float = 288.15            # below this the system is too cold to work properly
    coolant_exchanger_fitted: bool = True
    coolant_exchanger_ua_w_per_k: float = 90.0
    relief_warmup_enabled: bool = True
    tank_heater_fitted: bool = True
    tank_heater_w: float = 1500.0
    tank_heater_area_cm2: float = 1200.0     # a long low-density element, not a kettle element
    tank_heater_on: bool = False
    heater_coking: float = 0.0               # varnish laid down on the element, 0..1
    heater_burned_out: bool = False
    warming_w: float = 0.0
    coupling_heat_w: float = 0.0        # a wet clutch slipping in this same oil
    # operator inputs
    commanded_flow_frac: float = 0.0
    load_frac: float = 0.0
    # live state
    temp_k: float = 293.15
    water_kg: float = 0.0
    particulate_kg: float = 0.0
    desiccant_used_kg: float = 0.0
    oil_life_frac: float = 1.0               # 1.0 = fresh, 0.0 = oxidised out
    cooler_w: float = 0.0
    seal_integrity: float = 1.0        # 1.0 = sound; heat destroys elastomer seals permanently
    oil_destroyed: bool = False
    burning: bool = False
    pump_flow_lpm: float = 0.0
    pressure_pa: float = 0.0
    useful_w: float = 0.0
    heat_w: float = 0.0
    relief_w: float = 0.0
    cavitating: bool = False

    # ------------------------------------------------------------------
    @property
    def mass_kg(self) -> float:
        return self.oil_l / 1000.0 * OIL_DENSITY_KG_M3

    @property
    def viscosity_cst(self) -> float:
        return viscosity_cst(self.temp_k, self.oil_grade_vg)

    @property
    def water_ppm(self) -> float:
        return self.water_kg / max(self.mass_kg, 1e-9) * 1e6

    @property
    def saturation_ppm(self) -> float:
        """How much water this oil holds in solution -- more when warm,
        which is why a system that looked fine hot drops free water in
        the tank overnight as it cools."""
        return WATER_SATURATION_PPM_AT_60C * max(0.15, self.temp_k / OXIDATION_REFERENCE_K) ** 3

    @property
    def free_water(self) -> bool:
        return self.water_ppm > self.saturation_ppm

    @property
    def cleanliness_code(self) -> str:
        # particles per ml from the carried mass, on a disclosed mean
        # particle size (5 um sphere of silica) -- the conversion is a
        # model, the CODE is the real reporting standard
        per_ml_4 = self.particulate_kg / max(self.oil_l, 1e-6) / 1000.0 / 1.7e-13
        return iso4406_code(per_ml_4, per_ml_4 * 0.25, per_ml_4 * 0.02)

    @property
    def chiller_demand_w(self) -> float:
        """What the refrigerant chiller is asked for -- which is only
        what the OIL COOLER cannot already do.

        A hydraulic system's heat normally leaves through an oil-to-air
        cooler; a refrigerant chiller is there to hold a precise
        temperature when ambient is too warm for air alone, not to carry
        the whole load. Asking a vapour-compression loop for 15 kW
        continuously would be asking for a machine nobody fits. So the
        demand is the heat the system is actually making, less what the
        cooler is actually rejecting, and only while the oil is above
        its setpoint."""
        if not self.chiller_fitted or self.temp_k <= self.chiller_target_k:
            return 0.0
        shortfall = self.heat_w - self.cooler_w
        # plus a proportional term so it actually pulls the temperature
        # back down rather than merely holding it wherever it drifted to
        pull_back = (self.temp_k - self.chiller_target_k) * 120.0
        return max(0.0, min(self.chiller_trim_capacity_w, shortfall + pull_back))

    # ------------------------------------------------------------------
    def step(self, dt: float, crank_rpm: float, ambient_k: float, chiller_cooling_w: float,
             ambient_humidity_g_m3: float = BREATHER_HUMID_AIR_G_PER_M3,
             coolant_temp_k: float = 293.15) -> None:
        # ---- the pump ----
        pump_rpm = max(0.0, crank_rpm) * self.pump_drive_ratio
        self.pump_flow_lpm = self.pump_displacement_cc_rev * pump_rpm / 1000.0
        # a cold, thick oil starves the pump inlet: real cavitation
        self.cavitating = self.viscosity_cst > 800.0 and self.pump_flow_lpm > 0.0
        # a blanketed reservoir feeds the pump at positive pressure, so
        # it will draw a far thicker oil before it starves -- the real
        # NPSH benefit, and the other reason aircraft do it
        if self.blanket_fitted and self.blanket_supply_available:
            self.cavitating = self.viscosity_cst > 2000.0 and self.pump_flow_lpm > 0.0
        flow_m3_s = self.pump_flow_lpm / 60000.0 * (0.5 if self.cavitating else 1.0)
        taken = max(0.0, min(1.0, self.commanded_flow_frac))
        # the fan drive takes its own share of the pump's flow first --
        # it is a real consumer on the same circuit, not a free extra
        fan_frac = 0.0
        if self.fan_drive_fitted and flow_m3_s > 0.0:
            # the fan is thermostatted off the oil it is there to cool:
            # nothing until the oil is warm, winding up from there. This
            # lives HERE, with the fan, so the circuit behaves correctly
            # on its own; a caller may still override the command.
            if self.fan_thermostat:
                over = self.temp_k - (self.fan_start_k)
                self.fan_drive_command = max(0.0, min(1.0, over / max(self.fan_full_span_k, 1.0)))
            cmd = max(0.0, min(1.0, self.fan_drive_command))
            demand_lpm = self.fan_drive_displacement_cc_rev * self.fan_drive_max_rpm * cmd / 1000.0
            fan_frac = min(1.0, demand_lpm / max(self.pump_flow_lpm, 1e-6))
            self.fan_drive_rpm = self.fan_drive_max_rpm * cmd * (1.0 if fan_frac <= 1.0 else 1.0 / fan_frac)
            # a fan's torque goes with the square of its speed, so its
            # power goes with the cube -- which is why running one at
            # half speed costs an eighth of the power
            self.fan_airflow_frac = max(0.0, min(1.0, self.fan_drive_rpm / max(self.fan_drive_max_rpm, 1.0)))
            self.fan_drive_w = 12_000.0 * self.fan_airflow_frac ** 3
        else:
            self.fan_drive_rpm = 0.0
            self.fan_airflow_frac = 0.0
            self.fan_drive_w = 0.0
        taken = min(1.0, taken + fan_frac)
        self.pressure_pa = self.relief_pressure_pa * max(0.0, min(1.0, self.load_frac))
        # the pilot supply is live whenever the pump is actually turning
        # and there is oil to draw -- independent of the implements
        self.pilot_available = self.pump_flow_lpm > 1.0 and self.oil_l > 0.5 and not self.cavitating
        hydraulic_w = self.pressure_pa * flow_m3_s
        # what the actuators actually take does work; the rest goes over
        # the relief valve, and ALL of that becomes heat in the oil
        self.useful_w = hydraulic_w * taken
        self.relief_w = hydraulic_w * (1.0 - taken)
        # everything the fan drive absorbs ends up as heat in this same
        # oil (the fan itself only moves air; the motor's losses and the
        # work against the air both come back thermally)
        # even the useful part is not free: line losses and actuator
        # inefficiency land in the oil too (a disclosed 20 %)
        self.heat_w = self.relief_w + self.useful_w * 0.20 + self.coupling_heat_w

        # ---- warming, if it is too cold to work ----
        self.warming_w = 0.0
        too_cold = self.temp_k < self.warm_target_k
        if too_cold and self.coolant_exchanger_fitted and coolant_temp_k > self.temp_k:
            # the gentlest source: the engine's own jacket water
            self.warming_w += (coolant_temp_k - self.temp_k) * self.coolant_exchanger_ua_w_per_k
        self.tank_heater_on = False
        if too_cold and self.tank_heater_fitted and not self.heater_burned_out:
            covered = self.oil_l > self.tank_capacity_l * 0.25      # the low-level cutout
            if covered:
                self.tank_heater_on = True
                # what the element actually delivers falls as varnish
                # builds on it, and that varnish is laid down faster the
                # higher the watt density is
                self.warming_w += self.tank_heater_w * (1.0 - 0.7 * self.heater_coking)
                w_per_cm2 = self.tank_heater_w / max(self.tank_heater_area_cm2, 1.0)
                if w_per_cm2 > 1.5:
                    self.heater_coking = min(1.0, self.heater_coking + (w_per_cm2 - 1.5) * dt / 3000.0)
                    if self.heater_coking >= 1.0:
                        self.heater_burned_out = True
            else:
                # uncovered: a real element destroys itself in minutes
                self.heater_burned_out = True

        # ---- temperature ----
        m = self.mass_kg
        if m > 0.0:
            over_k = self.temp_k - ambient_k
            skin_w = over_k * 8.0
            # the oil cooler: its full conductance with the fan running,
            # a fifth of it on natural convection alone
            cooler_w = over_k * self.cooler_ua_w_per_k * (1.0 if self.cooler_fan_running else 0.2)
            # the fan drive's own air is what the oil cooler gets, when
            # one is fitted -- a belt fan is the fallback
            if self.fan_drive_fitted:
                cooler_w = over_k * self.cooler_ua_w_per_k * (0.2 + 0.8 * self.fan_airflow_frac)
            self.cooler_w = max(0.0, cooler_w)
            net = self.heat_w + self.warming_w - chiller_cooling_w - skin_w - self.cooler_w
            self.temp_k += net * dt / (m * OIL_CP_J_KG_K)
            self.temp_k = max(ambient_k - 10.0, self.temp_k)
        self._age_thermally(dt)

        # ---- the breather: the real water ingress path ----
        # the tank breathes the volume the actuators displace
        # the level swings by the cylinders' net rod volume, at the rate
        # the machine is actually working -- not the pump's flow
        activity = max(0.0, min(1.0, self.commanded_flow_frac))
        breathed_m3 = (self.cylinder_swing_l / 1000.0) * (self.work_cycles_per_min / 60.0) * activity * dt
        if self.bladder_separated:
            breathed_m3 = 0.0       # sealed: the gas side never exchanges anything
        if breathed_m3 > 0.0:
            self._select_blanket_source()
            src = self.blanket_active_source
            if src == "nitrogen":
                # dry, inert, and metered off the bottle: the make-up gas
                # brings no water at all and no oxygen either
                gas_kg = breathed_m3 * 1.25 * self.blanket_pressure_pa / ATM_PA
                take = min(gas_kg, self.nitrogen_fill_frac * self.nitrogen_capacity_kg)
                self.nitrogen_used_kg += take
                self.blanket_gas_kg += take
                self.nitrogen_fill_frac = max(0.0, self.nitrogen_fill_frac
                                              - take / max(self.nitrogen_capacity_kg, 1e-9))
            elif src == "plant-air":
                # sealed and blanketed: the make-up gas is the plant's own
                # treated air, so what comes in is exactly as wet as that
                # air is -- and nothing else comes in at all
                air_kg = breathed_m3 * 1.2 * self.blanket_pressure_pa / ATM_PA
                self.blanket_air_used_kg += air_kg
                self.blanket_gas_kg += air_kg
                self.water_kg += air_kg * self.blanket_supply_humidity_kg_kg
            elif src == "vacuum-break":
                # the emergency: air in through the desiccant cartridge
                from desiccant import DesiccantBed
                if self.vacuum_break_bed is None:
                    self.vacuum_break_bed = DesiccantBed(kind="silica-gel", mass_kg=self.vacuum_break_cartridge_kg,
                                                         regenerates=False)
                self.vacuum_break_events += 1
                raw_water_kg = breathed_m3 * ambient_humidity_g_m3 / 1000.0
                passed = self.vacuum_break_bed.adsorb(raw_water_kg)
                self.water_kg += passed
            else:   # the breather: the fallback that needs no supply
                water_in_kg = breathed_m3 * ambient_humidity_g_m3 / 1000.0
                if self.desiccant_breather and self.desiccant_used_kg < self.breather_desiccant_capacity_kg:
                    caught = min(water_in_kg, self.breather_desiccant_capacity_kg - self.desiccant_used_kg)
                    self.desiccant_used_kg += caught
                    water_in_kg -= caught
                self.water_kg += water_in_kg
                # and the same breath brings dirt
                self.particulate_kg += breathed_m3 * 0.5e-6

        # ---- wear debris, and what the filter takes back out ----
        if flow_m3_s > 0.0:
            # wear rises when the film is thin (hot oil) or dirty
            film = max(0.2, min(2.0, self.viscosity_cst / 30.0))
            wear_kg_s = 2.0e-9 * (1.0 + 6.0 * (1.0 - min(1.0, film))) * (1.0 + 400.0 * self.particulate_kg)
            self.particulate_kg += wear_kg_s * dt
            if self.return_filter_fitted:
                # one pass of the whole tank takes beta of it out
                passes_per_s = flow_m3_s / max(self.oil_l / 1000.0, 1e-6)
                removed = self.particulate_kg * (1.0 - 1.0 / self.return_filter_beta) * passes_per_s * dt
                self.particulate_kg = max(0.0, self.particulate_kg - removed)

        # ---- oxidation: the oil's own life ----
        if self.temp_k > OXIDATION_REFERENCE_K:
            rate = 2.0 ** ((self.temp_k - OXIDATION_REFERENCE_K) / OXIDATION_HALVING_K)
        else:
            rate = 0.5 ** ((OXIDATION_REFERENCE_K - self.temp_k) / OXIDATION_HALVING_K)
        # a nominal 10 000 h life at the reference temperature, accelerated
        # by that rate, and water accelerates it further (hydrolysis)
        life_s = 10_000.0 * 3600.0
        self.oil_life_frac = max(0.0, self.oil_life_frac - dt / life_s * rate * (1.0 + 3.0 * min(1.0, self.water_ppm / 1000.0)))

    # Real thermal limits for a mineral oil on nitrile seals, which is
    # what almost every machine actually runs:
    SEAL_LIMIT_K = 393.15          # 120 C: nitrile hardens and takes a set above this
    SEAL_DESTROYED_K = 423.15      # 150 C: they are gone in minutes
    OIL_BREAKDOWN_K = 423.15       # 150 C: the oil is oxidising far faster than it can be replaced
    FLASH_POINT_K = 493.15         # 220 C: it will light off an ignition source

    def _age_thermally(self, dt: float) -> None:
        """Heat does permanent damage, and it is the seals that go
        first. Above the elastomer's limit they harden and take a
        compression set; they do not recover when the oil cools, which
        is why an overheated system leaks from then on."""
        if self.temp_k > self.SEAL_LIMIT_K:
            span = max(self.SEAL_DESTROYED_K - self.SEAL_LIMIT_K, 1.0)
            severity = min(4.0, (self.temp_k - self.SEAL_LIMIT_K) / span)
            self.seal_integrity = max(0.0, self.seal_integrity - severity * dt / 600.0)
        if self.temp_k > self.OIL_BREAKDOWN_K:
            self.oil_life_frac = max(0.0, self.oil_life_frac - dt / 300.0)
        if self.oil_life_frac <= 0.0:
            self.oil_destroyed = True
        self.burning = self.temp_k >= self.FLASH_POINT_K

    def _select_blanket_source(self) -> None:
        """The three-way valve. On "auto" it prefers plant air while the
        dryer is actually making dry air, falls back to the nitrogen
        bottle while there is any in it, and drops to the breather
        rather than ever sealing the tank with nothing to breathe."""
        if not self.blanket_fitted:
            self.blanket_active_source = "breather"
            return
        want = self.blanket_source
        air_ok = self.blanket_supply_available
        n2_ok = self.nitrogen_fitted and self.nitrogen_fill_frac > 0.01
        if want == "plant-air":
            self.blanket_active_source = "plant-air" if air_ok else ("nitrogen" if n2_ok else "breather")
        elif want == "nitrogen":
            self.blanket_active_source = "nitrogen" if n2_ok else ("plant-air" if air_ok else "breather")
        elif want == "breather":
            self.blanket_active_source = "breather"
        else:
            # auto: the everyday supply first, the bottle as backup, and
            # -- rather than ever sealing the tank with nothing to
            # breathe -- the emergency desiccated vacuum break, falling
            # back to a plain breather only once that cartridge is spent
            if air_ok:
                self.blanket_active_source = "plant-air"
            elif n2_ok:
                self.blanket_active_source = "nitrogen"
            else:
                bed = self.vacuum_break_bed
                spent = bed is not None and bed.spent
                self.blanket_active_source = "breather" if spent else "vacuum-break"

    # ------------------------------------------------------------------
    def condition(self) -> dict:
        """What the oil's state is doing to the machine."""
        visc = self.viscosity_cst
        film = max(0.0, min(1.0, (visc - 8.0) / 22.0))          # below ~8 cSt the film is gone
        water = min(1.0, max(0.0, (self.water_ppm - 200.0) / 800.0))
        dirt = min(1.0, self.particulate_kg / max(self.oil_l * 2e-5, 1e-9))
        return {
            "pump_efficiency_factor": max(0.2, 1.0 - 0.5 * (1.0 - film) - 0.3 * dirt) * (0.5 if self.cavitating else 1.0),
            "actuator_speed_factor": max(0.2, 1.0 - 0.6 * dirt),
            "bearing_wear_rate": 1.0 + 8.0 * water + 4.0 * (1.0 - film),
            "seal_integrity": self.seal_integrity,
            "notes": [n for n in (
                f"OIL OVER ITS FLASH POINT ({self.temp_k - 273.15:.0f} C) -- it will light from any source"
                if self.burning else None,
                f"oil destroyed by heat ({self.temp_k - 273.15:.0f} C)" if self.oil_destroyed else None,
                f"SEALS FAILING at {self.temp_k - 273.15:.0f} C: {self.seal_integrity * 100:.0f}% left"
                if self.seal_integrity < 0.99 else None,
                f"oil at {self.temp_k - 273.15:.0f} C: film thin ({visc:.1f} cSt)" if film < 0.4 else None,
                f"tank heater coked {self.heater_coking * 100:.0f}%" if 0.05 < self.heater_coking < 1.0 else None,
                "tank heater burned out" if self.heater_burned_out else None,
                f"warming: {self.warming_w / 1000:.2f} kW in" if self.warming_w > 100.0 else None,
                f"pump cavitating: oil too thick ({visc:.0f} cSt)" if self.cavitating else None,
                f"FREE WATER in the oil ({self.water_ppm:.0f} ppm, saturation {self.saturation_ppm:.0f})"
                if self.free_water else (f"water {self.water_ppm:.0f} ppm" if self.water_ppm > 150.0 else None),
                f"oil life {self.oil_life_frac * 100:.0f}% remaining" if self.oil_life_frac < 0.5 else None,
                "desiccant breather spent" if (self.desiccant_breather and not self.blanket_fitted and
                self.desiccant_used_kg >= self.breather_desiccant_capacity_kg * 0.99) else None,
                f"blanket on NITROGEN ({self.nitrogen_fill_frac * 100:.0f}% of the bottle left)"
                if self.blanket_active_source == "nitrogen" else None,
                (f"EMERGENCY vacuum break in use ({self.vacuum_break_bed.loading_frac * 100:.0f}% of its cartridge)"
                 if self.vacuum_break_bed is not None else "EMERGENCY vacuum break in use")
                if (self.blanket_active_source == "vacuum-break" and not self.bladder_separated) else None,
                "blanket and vacuum-break cartridge both gone: breathing raw ambient"
                if (self.blanket_fitted and not self.bladder_separated
                    and self.blanket_active_source == "breather") else None,
            ) if n],
        }

    def describe(self) -> list[str]:
        c = self.condition()
        out = [f"  HYDRAULICS {self.oil_l:.0f} L at {self.temp_k - 273.15:5.1f} C, {self.viscosity_cst:5.1f} cSt, "
               f"{self.pump_flow_lpm:5.1f} L/min at {self.pressure_pa / 1e5:4.0f} bar"
               f", pilot {self.pilot_pressure_pa / 1e5:2.0f} bar {'LIVE' if self.pilot_available else 'DEAD'}",
               f"    work {self.useful_w / 1000:5.2f} kW, over the relief {self.relief_w / 1000:5.2f} kW "
               + (f"+ clutch {self.coupling_heat_w / 1000:.1f} kW " if self.coupling_heat_w > 100.0 else "")
               + f"-> {self.heat_w / 1000:5.2f} kW of heat; cooler {self.cooler_w / 1000:5.2f} kW, "
               + f"chiller calling {self.chiller_demand_w / 1000:.2f} kW",
               f"    ISO {self.cleanliness_code}, water {self.water_ppm:4.0f} ppm, oil life {self.oil_life_frac * 100:3.0f}%"
               + (f", blanket: {self.blanket_active_source}" if self.blanket_fitted else "")]
        if self.bladder_separated:
            out.append("    bladder-separated reservoir: no gas exchange at all")
        elif self.blanket_fitted:
            out.append(f"    blanket gas used {self.blanket_gas_kg:.2f} kg"
                       + (f", nitrogen bottle {self.nitrogen_fill_frac * 100:.0f}%" if self.nitrogen_fitted else "")
                       + f"  (breathing {self.cylinder_swing_l:.0f} L swing x {self.work_cycles_per_min:.1f}/min)")
        if self.vacuum_break_bed is not None:
            out.append(f"    vacuum-break cartridge: {self.vacuum_break_bed.describe()} "
                       f"({self.vacuum_break_events} admissions)")
        if self.fan_drive_fitted:
            out.append(f"    fan drive {self.fan_drive_rpm:4.0f} rpm ({self.fan_airflow_frac * 100:3.0f}% airflow), "
                       f"taking {self.fan_drive_w / 1000:.2f} kW of the circuit")
        out.extend(f"    ! {n}" for n in c["notes"])
        return out
