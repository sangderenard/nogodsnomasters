"""The auxiliary plant at runtime: air treatment, refrigeration,
hydraulics, their controls and their own power.

`AuxiliaryPlantRuntime` is one object on the sim that owns the three
physics pieces (`air_treatment.AirTreatment`, `refrigeration.
RefrigerantLoop`, `refrigeration.HydraulicTank`), the switch panel
that decides what is allowed to run, and the accessory battery bank
that powers the electric half of it. It is stepped once per physics
tick from what the engine is really doing:

  the compressor's real delivered mass flow (the pneumatic circuit's
  own `delivered_flow_kg_s`) is what goes through the treatment train;
  the AC compressor's real shaft speed is what the refrigerant loop
  gets; the hydraulic system's real work is what heats its tank.

The two chillers are loads on one loop and genuinely compete for it:
when the hydraulic oil is hot the air chiller gets less, the pressure
dewpoint rises, and more water reaches the wet tank -- which is the
real reason a machine's air goes wet on a hot day under load.

The accessory bank matters for the same kind of reason. The fans, the
reheater and the drain solenoids all run off it, and it is charged
through an isolator that only closes when the engine's own bus is up.
Run the plant with the engine stopped and the bank goes down on its
own without touching the starting battery -- which is what an isolator
is for -- and when it is flat, the electric half of the treatment
train stops: no fan on the aftercooler, no reheat, no automatic
drains, and the air quality collapses accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import air_treatment
import desiccant
import hydraulics
import refrigeration

# what the plant's own electrical hardware draws when it runs
FAN_W = 90.0
REHEATER_W = 2500.0
SOLENOID_W = 18.0
CONTROL_W = 6.0
# A voltage-sensitive relay does not read cells, it reads whether the
# bus is clearly being CHARGED: it closes a little above the system's
# own nominal and drops out at or below it. Scaling off nominal rather
# than an assumed cell voltage keeps it right on 12 V and 24 V systems
# and on whatever a given alternator actually regulates to.
ISOLATOR_CLOSE_FRAC_OF_NOMINAL = 1.05
ISOLATOR_OPEN_FRAC_OF_NOMINAL = 1.00


@dataclass
class AccessoryBank:
    """A second battery bank behind a charge isolator, solved as a circuit.

    THIS IS A THEVENIN SOURCE DRIVING A THEVENIN LOAD, and it is worth
    writing as one because the shortcut it replaces was wrong in a way
    that mattered. Charge current used to be `(bus - bank) * 12.0`: a
    magic constant standing in for 1/R, implying 83 milliohms of total
    circuit resistance when the real figure for this bank and its cable
    is nearer 12. Off by five times, and worse, it made current depend
    on nothing but voltage difference -- so a fourteen watt solar panel
    presenting 28.8 V "charged" a 180 Ah bank at 45 A, which is a
    kilowatt out of nothing at all.

    The circuit is:

        V_source ---[ R_cable ]---+---[ R_internal ]--- EMF(soc)
                                  |
                              the isolator,
                          which sees THIS node

    and the law is Ohm's:

        I = (V_source - EMF) / (R_cable + R_internal)

    with two limits that are also real rather than fudged:

      THE SOURCE'S OWN CURRENT LIMIT. An alternator can hold its
      setpoint up to its rating and no further; a photovoltaic source is
      POWER limited, so its available current at the working voltage is
      simply P/V. That is not a correction bolted onto the model -- a PV
      array genuinely is a current source, and P/V is genuinely its
      current. Below its limit a source holds voltage (constant
      voltage); at it, it holds current and lets voltage fall
      (constant current). Every charger in the world works this way.

      THE CABLE. Its resistance is real copper of a real gauge, so the
      isolator does not see the source's setpoint -- it sees the
      setpoint minus I*R_cable. On a 24 V system that drop is what
      decides whether a relay stays in.

    No empirical taper is needed anywhere: as the bank charges its EMF
    rises, (V_source - EMF) shrinks, and the current tapers by itself.
    That is what a real absorption curve IS.
    """
    capacity_ah: float = 180.0
    nominal_voltage_v: float = 24.0
    soc_frac: float = 1.0
    isolator_kind: str = "voltage-sensitive-relay"
    isolator_closed: bool = False
    charge_a: float = 0.0
    load_a: float = 0.0
    max_charge_a: float = 60.0                 # the cable and breaker rating
    chemistry: str = "agm-lead-acid"
    #: 2 m of 6 AWG out and back: real copper, real gauge.
    cable_awg: int = 6
    cable_length_m: float = 2.0
    #: live circuit state, so a caller can see the whole solve
    terminal_v: float = 0.0
    isolator_sees_v: float = 0.0
    charge_w: float = 0.0

    @property
    def cells(self) -> int:
        return int(round(self.nominal_voltage_v / 2.0))

    @property
    def emf_v(self) -> float:
        """Open-circuit voltage: what the chemistry is, with no current
        flowing. A lead-acid cell runs 1.95 V flat to 2.12 V full."""
        return self.cells * (1.95 + 0.17 * max(0.0, min(1.0, self.soc_frac)))

    @property
    def internal_resistance_ohm(self) -> float:
        """Scaled the way `electrical_network.Battery` scales it: series
        cells add resistance, capacity divides it. About 6.7 milliohms
        for a 180 Ah 24 V AGM bank, which is the right order for real
        hardware."""
        k = 0.050 if self.chemistry.startswith("agm") else 0.065
        return k * self.nominal_voltage_v / max(self.capacity_ah, 1e-6)

    @property
    def cable_resistance_ohm(self) -> float:
        from dc_power import Conductor
        return Conductor(awg=self.cable_awg, length_m=self.cable_length_m).resistance_ohm

    @property
    def voltage_v(self) -> float:
        """Kept as the bank's terminal voltage, which is what anything
        outside this class means when it asks. Equal to the EMF only
        when no current is flowing."""
        return self.terminal_v or self.emf_v

    # ------------------------------------------------------------------
    def step(self, dt: float, source_voltage_v: float, load_w: float,
             engine_charging: bool, source_current_limit_a: float | None = None) -> None:
        """Solve the circuit for one tick.

        `source_current_limit_a` is what the thing presenting that
        voltage can actually push. None means "effectively unlimited",
        which is the honest description of an alternator with kilowatts
        behind it; a solar controller passes P/V, which is the honest
        description of a panel."""
        r_total = self.internal_resistance_ohm + self.cable_resistance_ohm

        # --- the load, solved rather than assumed ---
        # A constant-power load on a source with resistance is a
        # quadratic: V*I = P and V = EMF - I*R. Taking the physical
        # root rather than dividing by last tick's voltage.
        emf = self.emf_v
        disc = emf * emf - 4.0 * self.internal_resistance_ohm * max(load_w, 0.0)
        if disc <= 0.0:
            # the load cannot be supported at any current: the bank is
            # being asked for more power than its own resistance allows
            i_load = emf / (2.0 * self.internal_resistance_ohm)
        else:
            i_load = (emf - math.sqrt(disc)) / (2.0 * self.internal_resistance_ohm)
        self.load_a = i_load
        v_under_load = emf - i_load * self.internal_resistance_ohm

        # --- the isolator, looking at its own terminals ---
        # It sees the source's voltage less whatever the cable drops on
        # the way, which is why this is computed before the relay
        # decides and not after.
        i_available = (max(0.0, source_voltage_v - emf) / r_total) if r_total > 0 else 0.0
        if source_current_limit_a is not None:
            i_available = min(i_available, max(0.0, source_current_limit_a))
        i_available = min(i_available, self.max_charge_a)
        self.isolator_sees_v = (source_voltage_v - i_available * self.cable_resistance_ohm
                                if source_voltage_v > 0.0 else v_under_load)

        close_v = self.nominal_voltage_v * ISOLATOR_CLOSE_FRAC_OF_NOMINAL
        open_v = self.nominal_voltage_v * ISOLATOR_OPEN_FRAC_OF_NOMINAL
        if self.isolator_kind == "manual-switch":
            self.isolator_closed = engine_charging
        elif self.isolator_kind == "dc-dc-charger":
            self.isolator_closed = engine_charging and self.isolator_sees_v > open_v
        else:
            # a VSR: closes on a charging bus, drops out when it sags
            if self.isolator_sees_v >= close_v:
                self.isolator_closed = True
            elif self.isolator_sees_v <= open_v:
                self.isolator_closed = False

        # --- the charge, by Ohm's law ---
        if self.isolator_closed and source_voltage_v > emf:
            self.charge_a = i_available
            self.terminal_v = emf + self.charge_a * self.internal_resistance_ohm
        else:
            self.charge_a = 0.0
            self.terminal_v = v_under_load
        self.charge_w = self.charge_a * self.terminal_v

        # --- coulomb counting, with the charge acceptance efficiency ---
        # Not all the current that goes in comes back out: lead-acid
        # gasses some of it, and that is a real loss, not a fudge.
        eta = 0.85 if self.chemistry.startswith(("agm", "flooded", "lead")) else 0.98
        net_a = self.charge_a * eta - self.load_a
        self.soc_frac = max(0.0, min(1.0, self.soc_frac + net_a * dt / 3600.0
                                     / max(self.capacity_ah, 1e-6)))

    @property
    def flat(self) -> bool:
        return self.soc_frac <= 0.05

    def describe(self) -> str:
        state = "isolator CLOSED" if self.isolator_closed else "isolator open"
        return (f"  ACCESSORY BANK {self.soc_frac * 100:5.1f}%  {self.voltage_v:5.2f} V  "
                f"(emf {self.emf_v:5.2f}, {self.internal_resistance_ohm * 1000:.1f} mohm "
                f"internal + {self.cable_resistance_ohm * 1000:.1f} mohm cable)  "
                f"{state} seeing {self.isolator_sees_v:5.2f} V  "
                f"+{self.charge_a:5.1f} A / -{self.load_a:4.1f} A"
                + ("   FLAT" if self.flat else ""))


@dataclass
class PlantControls:
    """The panel. Every one of these is a real switch on it."""
    main_chiller: bool = True
    air_dryer_enable: bool = True        # the chiller stage on the air train
    reheater_enable: bool = True
    aftercooler_fan: bool = True
    separator_auto_drain: bool = True
    wet_tank_auto_drain: bool = True
    hydraulic_chiller_enable: bool = True
    reserve_isolation: bool = False      # shuts the reserve set off from the wet tank


@dataclass
class AuxiliaryPlantRuntime:
    spec: object = None
    treatment: air_treatment.AirTreatment = None
    loop: refrigeration.RefrigerantLoop = None
    hydraulics: refrigeration.HydraulicTank = None
    bank: AccessoryBank = None
    dryer: desiccant.TwinTowerDryer = None      # the ultra-dry adsorption stage
    # A SMALL PANEL ON THE ISOLATOR'S INPUT, and the weather it sees.
    # Ambient and sun are conditions, not settings: whatever drives the
    # plant should be setting these each tick the way it sets coolant
    # temperature, and the defaults are a dull temperate noon so that a
    # caller who never touches them still gets something honest.
    keep_alive: object = None
    sun_elevation_deg: float = 35.0
    cloud_cover: float = 0.25
    ambient_c: float = 18.0
    controls: PlantControls = field(default_factory=PlantControls)
    # live readings
    electrical_load_w: float = 0.0
    air_chiller_demand_w: float = 0.0
    delivered_dewpoint_k: float = 273.15
    running: bool = False

    @classmethod
    def build(cls, engine) -> "AuxiliaryPlantRuntime | None":
        spec = getattr(engine, "auxiliary_plant", None)
        if spec is None or not spec.fitted:
            return None
        chain = air_treatment.default_chain()
        fitted = {"aftercooler": spec.aftercooler_fitted, "chiller": spec.air_chiller_fitted,
                  "water_separator": spec.separator_fitted, "coalescing_filter": spec.coalescing_filter_fitted,
                  "particulate_filter": spec.particulate_filter_fitted, "reheater": spec.reheater_fitted}
        for st in chain:
            if st.name in fitted:
                st.fitted = bool(fitted[st.name])
        tr = air_treatment.AirTreatment(
            stages=chain, compressor_kind=engine.pneumatics.compressor_kind,
            dust_environment=spec.dust_environment, intake_filter_efficiency=spec.intake_filter_efficiency,
            ambient_rh=spec.ambient_relative_humidity)
        # size the loop off the AC compressor the graph really declares
        ac_node = {}
        try:
            from drivetrain_graph import build_drivetrain_graph
            for n in build_drivetrain_graph(engine)["nodes"]:
                if n["identity"] == "ac_compressor":
                    ac_node = n
                    break
        except Exception:
            ac_node = {}
        loop = refrigeration.RefrigerantLoop.from_compressor_node(
            ac_node, charge_frac=spec.refrigerant_charge_frac)
        loop.loads = [refrigeration.CoolingLoad("air-chiller", evaporating_k=274.15, priority=1.2)]
        if spec.hydraulic_fitted and spec.hydraulic_chiller_fitted:
            loop.loads.append(refrigeration.CoolingLoad("oil-chiller", evaporating_k=288.15, priority=1.0))
        if engine.accessories.air_conditioning:
            loop.loads.append(refrigeration.CoolingLoad("cabin", evaporating_k=277.15, priority=0.8, demand_w=2500.0))
        hyd = (hydraulics.HydraulicCircuit(tank_capacity_l=spec.hydraulic_tank_capacity_l,
                                           oil_l=spec.hydraulic_tank_capacity_l * 0.9,
                                           chiller_fitted=spec.hydraulic_chiller_fitted,
                                           blanket_fitted=spec.hydraulic_tank_blanketed,
                                           desiccant_breather=True,          # always kept as the last-resort fallback
                                           nitrogen_fitted=spec.hydraulic_nitrogen_backup,
                                           tank_heater_fitted=spec.hydraulic_tank_heater,
                                           fan_drive_fitted=spec.hydraulic_fan_drive,
                                           bladder_separated=spec.hydraulic_bladder_reservoir,
                                           cylinder_swing_l=spec.hydraulic_cylinder_swing_l)
               if spec.hydraulic_fitted else None)
        bank = (AccessoryBank(capacity_ah=spec.accessory_bank_ah, nominal_voltage_v=spec.accessory_bank_voltage_v,
                              isolator_kind=spec.isolator_kind) if spec.accessory_bank_fitted else None)
        keep_alive = None
        if getattr(spec, "solar_keep_alive_fitted", False):
            from solar import SolarKeepAlive, SolarPanel, ChargeController
            from dc_power import DCBattery
            keep_alive = SolarKeepAlive(
                identity="solar-keep-alive",
                panel=SolarPanel(identity="keep-alive.panel",
                                 area_m2=spec.solar_keep_alive_panel_m2,
                                 system_voltage_v=spec.accessory_bank_voltage_v),
                controller=ChargeController(identity="keep-alive.controller",
                                            kind="mppt", maximum_input_w=120.0),
                battery=DCBattery(identity="keep-alive.battery", chemistry="lifepo4",
                                  capacity_ah=spec.solar_keep_alive_battery_ah,
                                  # enough 12.8 V blocks to match the bank it feeds
                                  series_packs=max(1, int(round(
                                      spec.accessory_bank_voltage_v / 12.8))),
                                  state_of_charge=0.75),
                absorption_v=spec.accessory_bank_voltage_v * 1.20)
        dryer = desiccant.TwinTowerDryer(fitted=spec.desiccant_dryer_fitted,
                                         a=desiccant.DesiccantBed(kind=spec.desiccant_kind,
                                                                  mass_kg=spec.desiccant_bed_kg),
                                         b=desiccant.DesiccantBed(kind=spec.desiccant_kind,
                                                                  mass_kg=spec.desiccant_bed_kg, online=False))
        return cls(spec=spec, treatment=tr, loop=loop, hydraulics=hyd, bank=bank,
                   dryer=dryer, keep_alive=keep_alive)

    # ------------------------------------------------------------------
    def step(self, dt: float, *, compressor_flow_kg_s: float, ac_omega_rad_s: float, intake_k: float,
             tank_pressure_pa: float, ambient_k: float, bus_voltage_v: float, engine_charging: bool,
             condenser_airflow: float = 1.0, crank_rpm: float = 0.0, coolant_temp_k: float = 293.15) -> None:
        c = self.controls
        bank_ok = self.bank is None or not self.bank.flat
        # --- what the electric half is allowed to do ---
        fan_on = c.aftercooler_fan and bank_ok
        reheat_on = c.reheater_enable and bank_ok
        drains_on = bank_ok
        self.electrical_load_w = CONTROL_W
        if fan_on:
            self.electrical_load_w += FAN_W
        if reheat_on and compressor_flow_kg_s > 0.0:
            # only the electric TRIM draws power: the recuperator does
            # the rest of the reheat for nothing (air_treatment.Stage)
            rh = self.treatment.stage("reheater")
            trim_frac = (rh.electric_k / max(rh.reheat_k, 1e-6)) if rh is not None else 1.0
            self.electrical_load_w += REHEATER_W * max(0.0, min(1.0, trim_frac))
        if drains_on and (c.separator_auto_drain or c.wet_tank_auto_drain):
            self.electrical_load_w += SOLENOID_W

        # --- the hydraulic circuit, and what it is asking the loop for ---
        if self.hydraulics is not None:
            oil_load = self.loop.load("oil-chiller")
            if oil_load is not None:
                oil_load.calling = c.main_chiller and c.hydraulic_chiller_enable and self.hydraulics.chiller_fitted
                oil_load.demand_w = self.hydraulics.chiller_demand_w

        # --- what the air chiller needs to hit its dewpoint target ---
        air_load = self.loop.load("air-chiller")
        stage = self.treatment.stage("chiller")
        if air_load is not None:
            pre_k = ambient_k + (self.treatment.stage("aftercooler").approach_k if fan_on else 45.0)
            target = stage.target_k if stage is not None else 276.15
            self.air_chiller_demand_w = max(0.0, compressor_flow_kg_s * air_treatment.CP_AIR_J_KG_K * (pre_k - target))
            # plus the latent load of the water it is going to condense
            self.air_chiller_demand_w *= 1.6
            air_load.calling = c.main_chiller and c.air_dryer_enable and compressor_flow_kg_s > 0.0
            air_load.demand_w = self.air_chiller_demand_w

        # --- the loop ---
        self.loop.main_switch = c.main_chiller
        # the condenser sits in the same stack as the oil cooler, so a
        # hydraulically driven fan is what feeds BOTH -- more fan, lower
        # head pressure, more capacity for the chillers
        if self.hydraulics is not None and self.hydraulics.fan_drive_fitted:
            condenser_airflow = max(condenser_airflow * 0.3, self.hydraulics.fan_airflow_frac)
        self.loop.condenser_airflow = condenser_airflow
        self.loop.step(dt, ac_omega_rad_s, ambient_k)

        # --- the air train ---
        if stage is not None:
            stage.powered = self.loop.delivered("air-chiller") > 0.0
        ac_stage = self.treatment.stage("aftercooler")
        if ac_stage is not None:
            ac_stage.powered = fan_on
        rh_stage = self.treatment.stage("reheater")
        if rh_stage is not None:
            rh_stage.powered = reheat_on
        sep = self.treatment.stage("water_separator")
        if sep is not None:
            sep.drained = c.separator_auto_drain and drains_on
        wet = self.treatment.stage("wet_tank")
        if wet is not None:
            wet.drained = c.wet_tank_auto_drain and drains_on
        self.running = compressor_flow_kg_s > 0.0
        if self.running:
            delivered = self.treatment.step(dt, compressor_flow_kg_s, intake_k, tank_pressure_pa, ambient_k,
                                            self.loop.delivered("air-chiller"))
            # ULTRA-DRY: the refrigerated chain has a hard floor at about
            # +3 C dewpoint (below that the chiller ices), so anything
            # drier has to be ADSORBED. The desiccant towers sit after
            # the coalescing filter for a reason -- oil poisons them
            # permanently -- and they cost a real slice of the flow as
            # regeneration purge.
            if self.dryer is not None and self.dryer.fitted:
                w, dp = self.dryer.step(dt, compressor_flow_kg_s, delivered.carried.water_kg_per_kg,
                                        delivered.carried.oil_kg_per_kg, delivered.dewpoint_k)
                delivered.carried.water_kg_per_kg = w
                self.delivered_dewpoint_k = dp
            else:
                self.delivered_dewpoint_k = delivered.dewpoint_k

        # --- the hydraulics, blanketed on the air the train just made ---
        if self.hydraulics is not None:
            h = self.hydraulics
            # the fan drive thermostats itself off its own oil (see
            # hydraulics.HydraulicCircuit); nothing to do here
            if h.blanket_fitted:
                # the blanket is exactly as dry as the plant's own air: a
                # running, chilled train keeps the oil dry; a train that
                # is not running (or not drying) does not
                # only take plant air while it is actually dry enough to
                # be worth blanketing with -- wet air through the blanket
                # is worse than a breather, so the selector refuses it
                dry_enough = self.running and self.delivered_dewpoint_k <= self.spec.blanket_max_dewpoint_k
                h.blanket_supply_available = bool(dry_enough)
                h.blanket_supply_humidity_kg_kg = (self.treatment.delivered.carried.water_kg_per_kg
                                                   if dry_enough else 0.0)
            h.step(dt, crank_rpm, ambient_k, self.loop.delivered("oil-chiller"),
                   coolant_temp_k=coolant_temp_k)

        # --- the keep-alive, and then the bank ---
        # TWO SOURCES ON ONE INPUT: the higher wins, which is what
        # actually happens when a controller and an alternator are
        # paralleled through their diodes. With the engine running the
        # alternator is higher and the panel contributes nothing worth
        # naming; with the engine stopped the panel is the only thing
        # holding the relay in, and that is the whole point of it.
        source_current_limit_a = None
        if self.keep_alive is not None:
            ka = self.keep_alive.step(dt, sun_elevation_deg=self.sun_elevation_deg,
                                      ambient_c=self.ambient_c, cloud_cover=self.cloud_cover)
            if self.keep_alive.output_voltage_v() > bus_voltage_v:
                bus_voltage_v = self.keep_alive.output_voltage_v()
                # A PV SOURCE IS A CURRENT SOURCE. Its available current
                # at the working voltage is P/V -- that is the device,
                # not a correction to it.
                source_current_limit_a = (max(0.0, ka["surplus_w"])
                                          / max(bus_voltage_v, 1.0))
            engine_charging = engine_charging or self.keep_alive.supplying
        if self.bank is not None:
            self.bank.step(dt, bus_voltage_v, self.electrical_load_w, engine_charging,
                           source_current_limit_a=source_current_limit_a)

    # ------------------------------------------------------------------
    def summary(self) -> list[str]:
        out = ["  --- auxiliary plant ---"]
        out.extend(self.loop.describe())
        if self.hydraulics is not None:
            out.extend(self.hydraulics.describe())
        if self.running:
            out.extend(self.treatment.summary())
        else:
            out.append(f"  AIR TRAIN idle  {self.treatment.gunk.describe()}")
        if self.dryer is not None:
            out.extend(self.dryer.describe())
        if self.bank is not None:
            out.append(self.bank.describe())
        off = [n for n, v in (("main chiller", self.controls.main_chiller), ("air dryer", self.controls.air_dryer_enable),
                              ("reheater", self.controls.reheater_enable), ("aftercooler fan", self.controls.aftercooler_fan),
                              ("separator drain", self.controls.separator_auto_drain),
                              ("wet-tank drain", self.controls.wet_tank_auto_drain),
                              ("hydraulic chiller", self.controls.hydraulic_chiller_enable)) if not v]
        if off:
            out.append("  CONTROLS OFF: " + ", ".join(off))
        return out

    def gunk_effects(self) -> dict:
        return air_treatment.gunk_effects(self.treatment.gunk, self.spec.wet_tank_capacity_l if self.spec else 20.0)
