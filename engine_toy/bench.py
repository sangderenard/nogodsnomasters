"""A test bench for parts that are not engines.

Everything in this project so far has assumed the thing under test is
an engine: it has a subframe, it has mounts, it makes shaft power, and
a dyno measures that power. A cylinder has none of those. It bolts to
whatever is convenient, it has no mounts worth the name, and its output
is a force moving at a speed rather than a torque at an rpm. It still
needs testing, and it still needs feeding.

So this is the other half of the rig:

  A PART is anything with PORTS and a `step`. No subframe, no mounts,
  no shaft. `PartRig` holds one and runs it.

  A SUPPLY answers a port. `HydraulicPowerUnit` is a real power pack --
  motor, pump, relief valve, reservoir, cooler, and usually an
  accumulator. `AirSupply` is a real compressed-air service -- compressor,
  receiver, regulator. Each one is finite: ask for more flow than the
  pump makes and the pressure sags, exactly as it does on a real bench,
  and that sag is usually the most informative thing a test tells you.

  AN UNSATISFIED INLET is a port nothing is connected to. The bench
  finds them and feeds them from its own supplies, and says that it did.
  That is the whole point: without it, testing a hydraulic cylinder
  would mean bolting it onto the C18's plant and borrowing that engine's
  pump, which makes the engine part of every measurement. The bench
  makes the part testable on its own.

  AN OUTPUT is where the fluid goes afterwards, and it is not
  symmetric between the two media. Oil is a closed loop: every litre
  that leaves the pump comes back to the tank, carrying the heat the
  work put into it, which is why a power pack needs a cooler at all.
  Air is not: it is used once and exhausted to atmosphere, which is why
  compressed air is expensive to run and why an air tool is loud. The
  bench accounts for both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ATM_PA = 101_325.0
OIL_DENSITY_KG_M3 = 870.0
OIL_SPECIFIC_HEAT_J_PER_KG_K = 1900.0
AIR_GAMMA = 1.4
AIR_R_J_PER_KG_K = 287.0


@dataclass
class Port:
    """One connection on a part: what it carries and which way."""
    identity: str
    medium: str                  # "hydraulic" | "pneumatic"
    role: str                    # "supply-in" | "return-out" | "exhaust-out" | "drain-out"
    nominal_flow_l_min: float = 0.0
    connected_to: str | None = None

    @property
    def is_inlet(self) -> bool:
        return self.role == "supply-in"

    @property
    def satisfied(self) -> bool:
        return self.connected_to is not None


# =====================================================================
#  SUPPLIES
# =====================================================================
@dataclass
class HydraulicPowerUnit:
    """A real power pack, and finite in every way a real one is."""
    identity: str = "hpu"
    motor_kw: float = 7.5
    pump_displacement_cc_rev: float = 16.0
    pump_rpm: float = 1450.0                 # a 4-pole motor on 50 Hz
    volumetric_efficiency: float = 0.93
    relief_pressure_pa: float = 21_000_000.0
    reservoir_l: float = 60.0
    oil_l: float = 54.0
    cooler_ua_w_per_k: float = 240.0
    # An accumulator is a gas bladder holding oil under pressure. It
    # cannot make flow, only store it: it covers a demand SPIKE that is
    # briefer than the pump can answer, and then needs refilling. That
    # distinction is the whole of accumulator sizing.
    accumulator_l: float = 0.0
    accumulator_precharge_pa: float = 9_000_000.0
    # A TWO-STAGE PUMP, which is a genuinely clever piece of ordinary
    # engineering. Two pump sections share one shaft: both deliver
    # together for a fast, low-pressure approach, and when pressure
    # rises past the unloading valve's setting the large section is
    # dumped back to tank and only the small one keeps pumping. Same
    # motor, roughly ten times the force, roughly a tenth of the speed,
    # with no controls and no operator input -- which is why every log
    # splitter has one. 0 means a plain single-stage pump.
    second_stage_cc_rev: float = 0.0
    stage_switch_pressure_pa: float = 4_500_000.0
    on_low_stage: bool = field(default=False, init=False)
    # live state
    pressure_pa: float = field(default=ATM_PA, init=False)
    delivered_l_min: float = field(default=0.0, init=False)
    over_relief_l_min: float = field(default=0.0, init=False)
    accumulator_oil_l: float = field(default=0.0, init=False)
    temp_k: float = field(default=293.15, init=False)
    heat_w: float = field(default=0.0, init=False)
    starved: bool = field(default=False, init=False)
    over_relief: bool = field(default=False, init=False)

    @property
    def rated_flow_l_min(self) -> float:
        """What the pump actually makes: displacement times speed."""
        return (self.pump_displacement_cc_rev * self.pump_rpm / 1000.0
                * self.volumetric_efficiency)

    @property
    def low_stage_flow_l_min(self) -> float:
        """What is left once the big section has been dumped."""
        if self.second_stage_cc_rev <= 0.0:
            return self.rated_flow_l_min
        return (self.second_stage_cc_rev * self.pump_rpm / 1000.0
                * self.volumetric_efficiency)

    @property
    def rated_power_limited_flow_l_min(self) -> float:
        """What the MOTOR can drive at full pressure. A 7.5 kW motor
        cannot push the rated flow over a 210 bar relief -- power is
        pressure times flow, and something has to give. On a real pack
        that something is the motor stalling or its overload tripping,
        which is why power packs are sold by both numbers."""
        w = self.motor_kw * 1000.0 * 0.85            # pump + motor losses
        return w / max(self.relief_pressure_pa, 1e-6) * 60_000.0

    def step(self, dt: float, demand_l_min: float, load_pressure_pa: float) -> dict:
        """Serve what is asked for, as far as the hardware allows."""
        # the unloading valve, if this is a two-stage pump: past its
        # setting the big section goes back to tank and the ram slows
        # right down, which is exactly when it starts pushing hard
        self.on_low_stage = (self.second_stage_cc_rev > 0.0
                             and load_pressure_pa >= self.stage_switch_pressure_pa)
        rated = self.low_stage_flow_l_min if self.on_low_stage else self.rated_flow_l_min
        available = min(rated, max(self.rated_power_limited_flow_l_min, rated * 0.15))
        # the accumulator covers a brief spike, and only a brief one
        from_accumulator = 0.0
        if demand_l_min > available and self.accumulator_oil_l > 0.0:
            want_l = (demand_l_min - available) * dt / 60.0
            from_accumulator = min(want_l, self.accumulator_oil_l)
            self.accumulator_oil_l -= from_accumulator
        served = min(demand_l_min, available + from_accumulator * 60.0 / max(dt, 1e-9))
        self.starved = served < demand_l_min - 1e-6

        # pressure is set by the LOAD, up to the relief valve. An unloaded
        # pump makes almost no pressure at all: pressure is resistance to
        # flow, never something the pump decides.
        # PRESSURE AND FLOW ARE INDEPENDENT, and this is the thing to get
        # right. A pump short of flow does NOT lose pressure: it still
        # makes every bar the load asks for, and the actuator simply
        # moves more slowly because less volume per second is arriving.
        # An earlier version cut pressure in proportion to the flow
        # shortfall, which made an undersized pack unable to lift a load
        # it can certainly lift -- just slowly. Losing pressure is what
        # happens when the load exceeds the RELIEF setting, and that is
        # a different failure entirely.
        self.pressure_pa = min(self.relief_pressure_pa, max(ATM_PA, load_pressure_pa))
        self.over_relief = load_pressure_pa > self.relief_pressure_pa
        self.delivered_l_min = served
        self.over_relief_l_min = max(0.0, available - served)

        # refill the accumulator from whatever is spare
        if self.accumulator_l > 0.0 and self.over_relief_l_min > 0.0:
            room = self.accumulator_l * 0.75 - self.accumulator_oil_l
            if room > 0.0:
                take = min(room, self.over_relief_l_min * dt / 60.0)
                self.accumulator_oil_l += take
                self.over_relief_l_min -= take * 60.0 / max(dt, 1e-9)

        # EVERY LITRE OVER THE RELIEF VALVE BECOMES HEAT. Nothing is
        # being moved by it, so all of that power goes into the oil --
        # which is why a machine held against a stop cooks its own oil.
        self.heat_w = self.pressure_pa * self.over_relief_l_min / 60_000.0
        mass = max(self.oil_l * OIL_DENSITY_KG_M3 / 1000.0, 1e-3)
        rejected = self.cooler_ua_w_per_k * max(0.0, self.temp_k - 293.15)
        self.temp_k += (self.heat_w - rejected) * dt / (mass * OIL_SPECIFIC_HEAT_J_PER_KG_K)
        self.temp_k = max(293.15, self.temp_k)
        return {"pressure_pa": self.pressure_pa, "delivered_l_min": served,
                "starved": self.starved, "over_relief": self.over_relief,
                "heat_w": self.heat_w, "temp_k": self.temp_k}

    def describe(self) -> list[str]:
        out = [f"  {self.identity}: {self.motor_kw:.1f} kW power pack, "
               f"{self.pump_displacement_cc_rev:.0f} cc/rev at {self.pump_rpm:.0f} rpm "
               f"= {self.rated_flow_l_min:5.1f} L/min, relief {self.relief_pressure_pa / 1e5:.0f} bar"]
        if self.rated_power_limited_flow_l_min < self.rated_flow_l_min:
            out.append(f"    motor-limited to {self.rated_power_limited_flow_l_min:5.1f} L/min at full "
                       f"pressure ({self.motor_kw:.1f} kW cannot drive {self.rated_flow_l_min:.1f} L/min "
                       f"over {self.relief_pressure_pa / 1e5:.0f} bar)")
        if self.second_stage_cc_rev > 0.0:
            out.append(f"    two-stage: {self.rated_flow_l_min:4.1f} L/min until "
                       f"{self.stage_switch_pressure_pa / 1e5:.0f} bar, then "
                       f"{self.low_stage_flow_l_min:4.1f} L/min"
                       + ("  [ON THE LOW STAGE]" if self.on_low_stage else ""))
        if self.accumulator_l > 0.0:
            out.append(f"    accumulator {self.accumulator_l:.1f} L precharged "
                       f"{self.accumulator_precharge_pa / 1e5:.0f} bar -- covers spikes, makes no flow")
        out.append(f"    oil {self.oil_l:.0f} L at {self.temp_k - 273.15:4.0f} C, "
                   f"cooler {self.cooler_ua_w_per_k:.0f} W/K")
        return out


@dataclass
class AirSupply:
    """A compressed-air service: compressor, receiver, regulator.

    Air is used ONCE. Unlike oil it does not come back, so the receiver
    empties whenever demand beats the compressor, and the pressure at
    the tool falls as it does."""
    identity: str = "air"
    compressor_free_delivery_l_min: float = 500.0     # FAD, at atmosphere
    receiver_l: float = 270.0
    max_pressure_pa: float = 1_000_000.0              # 10 bar
    regulator_pressure_pa: float = 600_000.0          # 6 bar at the tool
    cut_in_frac: float = 0.80
    # live state
    receiver_pressure_pa: float = field(default=0.0, init=False)
    delivered_l_min: float = field(default=0.0, init=False)
    compressor_running: bool = field(default=True, init=False)
    consumed_free_l: float = field(default=0.0, init=False)
    starved: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.receiver_pressure_pa = self.max_pressure_pa

    @property
    def stored_free_l(self) -> float:
        """What the receiver holds, expressed as free air -- which is
        the only way a receiver's contents mean anything to a tool."""
        return self.receiver_l * self.receiver_pressure_pa / ATM_PA

    def step(self, dt: float, demand_free_l_min: float) -> dict:
        # the unloader: the compressor runs until the receiver is full,
        # then idles until it falls back to the cut-in point
        if self.receiver_pressure_pa <= self.max_pressure_pa * self.cut_in_frac:
            self.compressor_running = True
        elif self.receiver_pressure_pa >= self.max_pressure_pa:
            self.compressor_running = False
        made_l = (self.compressor_free_delivery_l_min if self.compressor_running else 0.0) * dt / 60.0
        want_l = demand_free_l_min * dt / 60.0
        have_l = self.stored_free_l + made_l
        served_l = min(want_l, have_l)
        self.starved = served_l < want_l - 1e-9
        new_free = have_l - served_l
        self.receiver_pressure_pa = max(ATM_PA, new_free / max(self.receiver_l, 1e-9) * ATM_PA)
        self.receiver_pressure_pa = min(self.receiver_pressure_pa, self.max_pressure_pa)
        self.delivered_l_min = served_l * 60.0 / max(dt, 1e-9)
        self.consumed_free_l += served_l
        return {"pressure_pa": min(self.regulator_pressure_pa, self.receiver_pressure_pa),
                "delivered_l_min": self.delivered_l_min, "starved": self.starved,
                "receiver_pressure_pa": self.receiver_pressure_pa}

    def describe(self) -> list[str]:
        return [f"  {self.identity}: {self.compressor_free_delivery_l_min:.0f} L/min free air, "
                f"{self.receiver_l:.0f} L receiver at {self.receiver_pressure_pa / 1e5:4.1f} bar, "
                f"regulated to {self.regulator_pressure_pa / 1e5:.1f} bar",
                f"    used {self.consumed_free_l:8.1f} L of free air so far "
                f"({'compressor running' if self.compressor_running else 'unloaded'})"]


# =====================================================================
#  THE RIG
# =====================================================================
@dataclass
class PartRig:
    """One part under test, with whatever supplies it turns out to need.

    No subframe, no mounts, no shaft: a part is here because it has
    ports and a `step`, and that is the only contract."""
    identity: str = "rig"
    hydraulic: HydraulicPowerUnit | None = None
    air: AirSupply | None = None
    parts: list = field(default_factory=list)         # (name, device, ports)
    duty: dict = field(default_factory=dict)          # name -> (m/s, deg/s) design duty
    # what the bench had to make up because nothing else was connected
    made_up: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    # live totals
    elapsed_s: float = 0.0
    peak_force_n: float = 0.0
    peak_torque_nm: float = 0.0
    peak_power_w: float = 0.0
    oil_returned_l: float = 0.0
    air_exhausted_free_l: float = 0.0

    def fit(self, device, name: str | None = None, ports: list[Port] | None = None,
            duty_speed_m_s: float = 0.2, duty_rate_deg_s: float = 60.0) -> None:
        """Put a part on the bench. Its ports are declared, not guessed
        from its type -- a part says what it needs."""
        name = name or getattr(device, "identity", "part")
        if ports is None:
            medium = getattr(device, "medium", "hydraulic")
            ports = [Port(f"{name}.supply", medium, "supply-in"),
                     Port(f"{name}.return", medium,
                          "return-out" if medium == "hydraulic" else "exhaust-out")]
        # what speed this function is DESIGNED to run at. Without it
        # every cylinder on the bench got the same nominal duty, so a
        # boom and a bucket sharing one pump came out moving at
        # identical speeds -- which is not what a machine with
        # different-sized functions does.
        self.duty[name] = (duty_speed_m_s, duty_rate_deg_s)
        self.parts.append((name, device, ports))

    def unsatisfied_inlets(self) -> list[Port]:
        """Every inlet with nothing connected to it. These are what the
        bench has to answer, and naming them is how a rig tells you what
        it is silently providing."""
        return [p for _, _, ports in self.parts for p in ports
                if p.is_inlet and not p.satisfied]

    def connect_bench_supplies(self) -> list[str]:
        """Feed every unsatisfied inlet from the bench's own supplies,
        making one if the bench does not have it yet. This is what
        removes the need to build the part onto an engine."""
        notes = []
        for port in self.unsatisfied_inlets():
            if port.medium == "hydraulic":
                if self.hydraulic is None:
                    self.hydraulic = HydraulicPowerUnit()
                    notes.append(f"  bench: no hydraulic supply for {port.identity} -- fitted a "
                                 f"{self.hydraulic.motor_kw:.1f} kW power pack")
                port.connected_to = self.hydraulic.identity
            else:
                if self.air is None:
                    self.air = AirSupply()
                    notes.append(f"  bench: no air supply for {port.identity} -- fitted a "
                                 f"{self.air.compressor_free_delivery_l_min:.0f} L/min compressor")
                port.connected_to = self.air.identity
            self.made_up[port.identity] = port.connected_to
        self.log.extend(notes)
        return notes

    def step(self, dt: float, commands: dict | None = None, loads: dict | None = None) -> dict:
        """Run every part for one tick against the supplies it is on."""
        commands = commands or {}
        loads = loads or {}
        self.elapsed_s += dt
        oil_demand = 0.0
        air_demand_free = 0.0
        results: dict = {}

        # two passes: ask what each part wants, serve it, then move it.
        # A real bench works this way too -- the pump does not know what
        # the cylinder wants until the valve opens.
        wants: list = []
        for name, device, ports in self.parts:
            medium = getattr(device, "medium", "hydraulic")
            cmd = float(commands.get(name, 0.0))
            want = abs(cmd) * self._nominal_demand_l_min(device, *self.duty.get(name, (0.2, 60.0)))
            if medium == "hydraulic":
                oil_demand += want
            else:
                # a pneumatic device swallows COMPRESSED air, and what it
                # costs the compressor is that volume expanded back to
                # atmosphere -- the ratio everyone forgets when sizing
                # a compressor for an air tool
                ratio = (self.air.regulator_pressure_pa if self.air else 600_000.0) / ATM_PA
                air_demand_free += want * ratio
            wants.append((name, device, ports, cmd, want, medium))

        oil_p = ATM_PA
        oil_flow = 0.0
        if self.hydraulic is not None:
            load_p = max((self._load_pressure(d, loads.get(n, 0.0)) for n, d, _, _, _, m in wants
                          if m == "hydraulic"), default=ATM_PA)
            out = self.hydraulic.step(dt, oil_demand, load_p)
            oil_p, oil_flow = out["pressure_pa"], out["delivered_l_min"]
        air_p = ATM_PA
        air_flow_free = 0.0
        if self.air is not None:
            out = self.air.step(dt, air_demand_free)
            air_p, air_flow_free = out["pressure_pa"], out["delivered_l_min"]

        for name, device, ports, cmd, want, medium in wants:
            if medium == "hydraulic":
                share = (want / oil_demand) if oil_demand > 1e-9 else 0.0
                supply_p, supply_flow = oil_p, oil_flow * share
            else:
                share = (want * ((self.air.regulator_pressure_pa if self.air else 6e5) / ATM_PA)
                         / air_demand_free) if air_demand_free > 1e-9 else 0.0
                supply_p = air_p
                supply_flow = air_flow_free * share * ATM_PA / max(air_p, ATM_PA)
            r = device.step(dt, supply_p, supply_flow, cmd, loads.get(name, 0.0))
            results[name] = r
            self.peak_force_n = max(self.peak_force_n, abs(r.get("force_n", 0.0)))
            self.peak_torque_nm = max(self.peak_torque_nm, abs(r.get("torque_nm", 0.0)))
            self.peak_power_w = max(self.peak_power_w, abs(r.get("power_w", 0.0)))
            # THE OUTPUT SIDE, and it is not symmetric
            spent = r.get("flow_l_min", 0.0) * dt / 60.0
            if medium == "hydraulic":
                self.oil_returned_l += spent            # closed loop: it comes back
            else:
                self.air_exhausted_free_l += spent * supply_p / ATM_PA   # used once, then gone
        return results

    def _nominal_demand_l_min(self, device, duty_speed_m_s: float = 0.2,
                              duty_rate_deg_s: float = 60.0) -> float:
        """What this part asks for at full command: its own swept area
        times the speed this function is meant to run at."""
        device = getattr(device, "actuator", device)   # a positioner wraps one
        if hasattr(device, "area_extend_m2"):
            return device.area_extend_m2 * duty_speed_m_s * 60_000.0
        if hasattr(device, "displacement_l_per_rev"):
            return device.displacement_l_per_rev * (duty_rate_deg_s / 360.0) * 60.0
        if hasattr(device, "displacement_cc_rev"):
            return device.displacement_cc_rev / 1000.0 * 1000.0  # 1000 rpm
        return 10.0

    def _load_pressure(self, device, load: float) -> float:
        """What pressure this part's load actually demands. Pressure is
        resistance: an unloaded actuator asks for almost none."""
        device = getattr(device, "actuator", device)
        if hasattr(device, "area_extend_m2") and device.area_extend_m2 > 0.0:
            # the pressure has to beat the load AND the seals, or the rod
            # will not move at all -- so the seal fraction belongs here,
            # not only in the force the actuator reports
            seal = getattr(device, "seal_friction_frac", 0.05)
            # A STATIONARY rod has to beat BREAKAWAY friction, not
            # running friction, and a real pump simply keeps building
            # pressure until it does (or reaches relief trying). Using
            # the running figure here left a cylinder sitting still
            # while exerting 97 % of the load it was asked to move --
            # forever, because it could never start.
            if abs(getattr(device, "velocity_m_s", 0.0)) < 0.002:
                seal *= 1.8
            return ATM_PA + abs(load) / (device.area_extend_m2 * max(1.0 - seal, 0.05))
        if hasattr(device, "torque_at"):
            at_rated = device.torque_at(getattr(device, "rated_pressure_pa", 21e6))
            if at_rated > 0.0:
                return ATM_PA + abs(load) / at_rated * getattr(device, "rated_pressure_pa", 21e6)
        return ATM_PA

    def summary(self) -> list[str]:
        out = [f"  --- {self.identity}: {len(self.parts)} part(s) on test, "
               f"{self.elapsed_s:.1f} s ---"]
        for name, device, ports in self.parts:
            if hasattr(device, "describe"):
                out.extend(device.describe())
        if self.made_up:
            out.append(f"  bench made up {len(self.made_up)} unsatisfied inlet(s): "
                       + ", ".join(sorted(self.made_up)))
        if self.hydraulic is not None:
            out.extend(self.hydraulic.describe())
        if self.air is not None:
            out.extend(self.air.describe())
        # what was MEASURED -- and a part with no shaft has no shaft
        # power, which is a real answer rather than a missing one
        if self.peak_force_n > 0.0:
            out.append(f"  measured: peak force {self.peak_force_n / 1000:7.2f} kN")
        if self.peak_torque_nm > 0.0:
            out.append(f"  measured: peak torque {self.peak_torque_nm:7.1f} Nm")
        out.append(f"  measured: peak power {self.peak_power_w / 1000:6.2f} kW"
                   if self.peak_power_w > 0.0
                   else "  measured: no mechanical output (this part makes none, which is not a fault)")
        if self.oil_returned_l > 0.0:
            out.append(f"  returned to tank {self.oil_returned_l:7.2f} L of oil")
        if self.air_exhausted_free_l > 0.0:
            out.append(f"  exhausted {self.air_exhausted_free_l:8.1f} L of free air "
                       f"(air is used once: this is the running cost)")
        return out
