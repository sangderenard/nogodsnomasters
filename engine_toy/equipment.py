"""Equipment fitted to a running engine, and the supply that makes up
whatever that engine cannot provide.

This is the piece that joins `actuators.py` and `loadouts.py` to the
engine itself. Without it the actuators were a separate world: correct
in isolation, tested on their own bench, and touching nothing the
engine actually does.

Two rules govern it, and they are the same two rules the rest of this
simulation already runs on.

ONE STATEFUL MACHINE. Fitted equipment does not get a private idea of
how much oil it is using. It reads the engine's real hydraulic circuit,
takes flow out of it, and sets the very levers (`hydraulic_flow_frac`,
`hydraulic_load_frac`) the plant already integrates -- so the pump load
on the crank, the relief-valve heating, the oil temperature and the
tank level are all consequences of what the equipment is really doing,
not a second ledger that happens to agree. Air works the same way: a
pneumatic tool drains the real compressed-air circuit, and the wet tank
and the brake reservoirs behind their protection valve drain with it.

COMPLEMENTARY SUPPLY FOR UNSATISFIED INLETS. An engine's own plant is
frequently too small for the equipment you want to watch, and the
alternative -- building every part onto the C18 until it fits -- makes
the engine part of every measurement. So the rig carries a bench supply
that makes up the shortfall and SAYS it is doing so. What the engine
provides and what the bench had to add are reported separately, every
tick, because the difference is the interesting part: it is exactly the
statement "this engine could not run this equipment on its own."
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bench import HydraulicPowerUnit, AirSupply, Port

ATM_PA = 101_325.0
AIR_DENSITY_KG_M3 = 1.2


@dataclass
class FittedItem:
    name: str
    part: object
    medium: str
    duty_speed_m_s: float = 0.2
    duty_rate_deg_s: float = 60.0
    command: float = 0.0
    load: float = 0.0
    ports: list = field(default_factory=list)


@dataclass
class EquipmentRig:
    """Everything bolted to this engine that runs on its fluids."""
    identity: str = "equipment"
    loadout: str = ""
    items: list = field(default_factory=list)
    # the bench supply that covers what the engine cannot
    supplement_hydraulic: HydraulicPowerUnit | None = None
    supplement_air: AirSupply | None = None
    auto_supplement: bool = True
    # live totals, all measured rather than assumed
    demand_l_min: float = 0.0
    engine_served_l_min: float = 0.0
    bench_served_l_min: float = 0.0
    required_pressure_pa: float = ATM_PA
    air_demand_free_l_min: float = 0.0
    air_from_engine_l_min: float = 0.0
    air_from_bench_l_min: float = 0.0
    engine_air_kg: float = 0.0           # taken out of the real air circuit
    peak_force_n: float = 0.0
    peak_torque_nm: float = 0.0
    peak_power_w: float = 0.0
    notes: list = field(default_factory=list)

    # ---------------------------------------------------------------
    @classmethod
    def from_loadout(cls, identity: str) -> "EquipmentRig":
        import loadouts
        lo = loadouts.get(identity)
        rig = cls(identity=f"{identity}-rig", loadout=identity)
        for name, part, item in lo.build():
            medium = getattr(getattr(part, "actuator", part), "medium", "hydraulic")
            rig.items.append(FittedItem(
                name=name, part=part, medium=medium,
                duty_speed_m_s=item.duty_speed_m_s, duty_rate_deg_s=item.duty_rate_deg_s,
                load=item.duty_load_n,
                ports=[Port(f"{name}.supply", medium, "supply-in"),
                       Port(f"{name}.return", medium,
                            "return-out" if medium == "hydraulic" else "exhaust-out")]))
        return rig

    def unsatisfied_inlets(self) -> list:
        return [p for it in self.items for p in it.ports if p.is_inlet and not p.satisfied]

    def command_all(self, value: float) -> None:
        for it in self.items:
            it.command = value

    # ---------------------------------------------------------------
    def _item_demand_l_min(self, it: FittedItem) -> float:
        part = getattr(it.part, "actuator", it.part)
        if hasattr(part, "area_extend_m2"):
            return part.area_extend_m2 * it.duty_speed_m_s * 60_000.0
        if hasattr(part, "displacement_l_per_rev"):
            return part.displacement_l_per_rev * (it.duty_rate_deg_s / 360.0) * 60.0
        if hasattr(part, "displacement_cc_rev"):
            return part.displacement_cc_rev / 1000.0 * 1000.0
        return 0.0

    def _item_pressure_pa(self, it: FittedItem) -> float:
        part = getattr(it.part, "actuator", it.part)
        if hasattr(part, "area_extend_m2") and part.area_extend_m2 > 0.0:
            seal = getattr(part, "seal_friction_frac", 0.05)
            if abs(getattr(part, "velocity_m_s", 0.0)) < 0.002:
                seal *= 1.8              # breakaway, or it never starts
            return ATM_PA + abs(it.load) / (part.area_extend_m2 * max(1.0 - seal, 0.05))
        if hasattr(part, "torque_at"):
            rated = getattr(part, "rated_pressure_pa", 21e6)
            at_rated = part.torque_at(rated)
            if at_rated > 0.0:
                return ATM_PA + abs(it.load) / at_rated * rated
        return ATM_PA

    # ---------------------------------------------------------------
    def step(self, dt: float, sim) -> dict:
        """Run the equipment against THIS engine's real fluids."""
        self.notes = []
        plant = getattr(sim, "plant", None)
        h = getattr(plant, "hydraulics", None) if plant is not None else None

        oil_demand = 0.0
        air_demand_free = 0.0
        need_p = ATM_PA
        for it in self.items:
            want = abs(it.command) * self._item_demand_l_min(it)
            if it.medium == "hydraulic":
                oil_demand += want
                need_p = max(need_p, self._item_pressure_pa(it))
            else:
                ratio = 6.2e5 / ATM_PA
                air_demand_free += want * ratio
        self.demand_l_min = oil_demand
        self.required_pressure_pa = need_p
        self.air_demand_free_l_min = air_demand_free

        # ---- what the ENGINE can give ----
        engine_flow = float(getattr(h, "pump_flow_lpm", 0.0) or 0.0) if h is not None else 0.0
        engine_relief = float(getattr(h, "relief_pressure_pa", 0.0) or 0.0) if h is not None else 0.0
        from_engine = min(oil_demand, engine_flow)
        engine_pressure_ok = need_p <= engine_relief if engine_relief > 0.0 else False

        # THE LEVERS THE PLANT ALREADY INTEGRATES. Setting these is what
        # makes the equipment's work show up as pump torque on the
        # crank, as relief-valve heat, and as oil temperature -- rather
        # than as a number this module keeps to itself.
        if h is not None and engine_flow > 0.0:
            sim.hydraulic_flow_frac = max(0.0, min(1.0, from_engine / engine_flow))
            sim.hydraulic_load_frac = (max(0.0, min(1.0, need_p / engine_relief))
                                       if engine_relief > 0.0 else 0.0)

        # ---- and what the BENCH has to make up ----
        shortfall = max(0.0, oil_demand - from_engine)
        from_bench = 0.0
        bench_pressure = 0.0
        if shortfall > 1e-6 or (oil_demand > 0.0 and not engine_pressure_ok):
            if self.supplement_hydraulic is None and self.auto_supplement:
                # sized to the shortfall rather than to a guess
                self.supplement_hydraulic = HydraulicPowerUnit(
                    identity="bench-hpu", motor_kw=max(7.5, shortfall * need_p / 60_000.0 / 1000.0 * 1.4),
                    pump_displacement_cc_rev=max(16.0, shortfall / 1.45),
                    relief_pressure_pa=max(21e6, need_p * 1.15))
                self.notes.append(
                    f"  bench: engine plant is short -- fitted a "
                    f"{self.supplement_hydraulic.motor_kw:.0f} kW pack to make it up")
            if self.supplement_hydraulic is not None:
                out = self.supplement_hydraulic.step(dt, shortfall, need_p)
                from_bench = out["delivered_l_min"]
                bench_pressure = out["pressure_pa"]
        self.engine_served_l_min = from_engine
        self.bench_served_l_min = from_bench

        supply_p = max(engine_relief if engine_pressure_ok else 0.0, bench_pressure)
        supply_p = min(supply_p, need_p) if supply_p > 0.0 else ATM_PA
        total_oil = from_engine + from_bench

        # ---- air, out of the engine's REAL compressed-air circuit ----
        air_p = ATM_PA
        air_served_free = 0.0
        if air_demand_free > 0.0:
            from_engine_air, air_p = self._take_engine_air(dt, sim, air_demand_free)
            air_served_free = from_engine_air
            self.air_from_engine_l_min = from_engine_air
            gap = air_demand_free - from_engine_air
            if gap > 1e-6:
                if self.supplement_air is None and self.auto_supplement:
                    self.supplement_air = AirSupply(identity="bench-air",
                                                    compressor_free_delivery_l_min=max(500.0, gap))
                    self.notes.append(f"  bench: engine air is short -- fitted a "
                                      f"{self.supplement_air.compressor_free_delivery_l_min:.0f} L/min compressor")
                if self.supplement_air is not None:
                    out = self.supplement_air.step(dt, gap)
                    air_served_free += out["delivered_l_min"]
                    self.air_from_bench_l_min = out["delivered_l_min"]
                    air_p = max(air_p, out["pressure_pa"])
        else:
            self.air_from_engine_l_min = self.air_from_bench_l_min = 0.0

        # ---- now move the parts with what they actually got ----
        results = {}
        for it in self.items:
            want = abs(it.command) * self._item_demand_l_min(it)
            if it.medium == "hydraulic":
                share = want / oil_demand if oil_demand > 1e-9 else 0.0
                got, p = total_oil * share, supply_p
            else:
                share = (want * 6.2e5 / ATM_PA / air_demand_free) if air_demand_free > 1e-9 else 0.0
                got = air_served_free * share * ATM_PA / max(air_p, ATM_PA)
                p = air_p
            r = it.part.step(dt, p, got, it.command, it.load)
            results[it.name] = r
            self.peak_force_n = max(self.peak_force_n, abs(r.get("force_n", 0.0)))
            self.peak_torque_nm = max(self.peak_torque_nm, abs(r.get("torque_nm", 0.0)))
            self.peak_power_w = max(self.peak_power_w, abs(r.get("power_w", 0.0)))
        return results

    def _take_engine_air(self, dt: float, sim, demand_free_l_min: float) -> tuple[float, float]:
        """Draw real air out of the engine's own compressed-air circuit.

        The circuit keeps the authoritative stored mass, so this reduces
        that mass and lets `air_vessels.py` decide which vessel it came
        out of -- which means a tool run long enough really does pull
        the wet tank down and stop at the protection valve rather than
        emptying the brakes."""
        from hole_emitters import circuit_identity as _cid
        drivetrain = getattr(sim, "_drivetrain", None)
        if drivetrain is None:
            return 0.0, ATM_PA
        pneu = next((c for c in drivetrain.fluid_circuits if _cid(c) == "pneumatic-reserve"), None)
        if pneu is None or pneu.bottle_capacity_kg <= 0.0:
            return 0.0, ATM_PA
        stored_kg = pneu.fill_level_frac * pneu.bottle_capacity_kg
        want_kg = demand_free_l_min / 60_000.0 * dt * AIR_DENSITY_KG_M3
        took_kg = min(stored_kg, want_kg)
        if took_kg > 0.0:
            pneu.fill_level_frac = max(0.0, (stored_kg - took_kg) / pneu.bottle_capacity_kg)
            self.engine_air_kg += took_kg
        pressure = ATM_PA + (float(getattr(pneu, "working_pressure_pa", 827_000.0)) - ATM_PA) * pneu.fill_level_frac
        served_free = took_kg / max(dt, 1e-9) * 60_000.0 / AIR_DENSITY_KG_M3
        return served_free, pressure

    # ---------------------------------------------------------------
    def summary(self) -> list[str]:
        out = [f"  -- EQUIPMENT: {self.loadout or self.identity} ({len(self.items)} parts) --"]
        for it in self.items:
            part = getattr(it.part, "actuator", it.part)
            if hasattr(part, "position_m"):
                out.append(f"   {it.name:16s} {part.position_m * 1000:6.0f} mm  "
                           f"{part.velocity_m_s * 1000:6.1f} mm/s  {part.force_n / 1000:7.1f} kN"
                           + ("  STALLED" if part.stalled else ""))
            elif hasattr(part, "angle_deg"):
                out.append(f"   {it.name:16s} {part.angle_deg:6.0f} deg  "
                           f"{part.rate_deg_s:6.1f} deg/s  {part.torque_nm:7.0f} Nm")
            elif hasattr(part, "rpm"):
                out.append(f"   {it.name:16s} {part.rpm:6.0f} rpm  {part.torque_nm:7.1f} Nm")
        if self.demand_l_min > 0.0:
            out.append(f"   oil: wants {self.demand_l_min:6.1f} L/min at "
                       f"{self.required_pressure_pa / 1e5:4.0f} bar -- engine "
                       f"{self.engine_served_l_min:6.1f}, bench {self.bench_served_l_min:6.1f}")
        if self.air_demand_free_l_min > 0.0:
            out.append(f"   air: wants {self.air_demand_free_l_min:6.0f} L/min free -- engine "
                       f"{self.air_from_engine_l_min:6.0f}, bench {self.air_from_bench_l_min:6.0f}"
                       f"   (taken from the engine so far {self.engine_air_kg:.3f} kg)")
        out.extend(self.notes)
        return out
