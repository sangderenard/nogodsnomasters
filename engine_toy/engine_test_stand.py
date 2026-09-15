"""External cooling equipment for engine capture and bench experiments.

The engine specification and its drivetrain graph remain the installed machine.
This stand connects a separate finite coolant reservoir, electric circulation
pump and ambient heat rejector only when a discovered thermal circuit lacks
its own pump or active exchanger. Oil uses a separate oil-to-coolant exchanger;
machines with no liquid thermal circuit keep their existing air cooling.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine_cycle_sim import EngineCycleSim
from station_cooling import PlatformCoolantRuntime


@dataclass
class TestStandCooling(PlatformCoolantRuntime):
    ambient_k: float = 293.15
    engine_heat_removed_w: float = 0.0
    rejected_heat_w: float = 0.0
    pump_flow_l_min: float = 0.0

    @classmethod
    def for_engine(cls, sim):
        circuit, _ = cls._engine_circuit(sim)
        if circuit is None:
            return None
        if circuit.pump_node and circuit.active_heat_exchange_w_per_k > 0.0:
            return None
        # Size the bench from the circuit's rated heat share, with 10 K
        # across the engine exchanger and 60 K across the ambient rejector.
        # These are stand design points, not imposed engine temperatures.
        heat_w = max(1000.0, sim.engine.peak_power_kw * 1000.0
                     * max(circuit.heat_share, 0.05))
        flow_l_min = heat_w * 60.0 / (1.04 * 3500.0 * 10.0)
        return cls(
            reservoir_l=max(5.0, flow_l_min * 0.25),
            pump_rated_flow_l_min=flow_l_min,
            engine_exchanger_ua_w_per_k=heat_w / 10.0,
            station_exchanger_ua_w_per_k=heat_w / 60.0)

    def advance(self, dt, sim):
        result = self.step(dt, sim, self.ambient_k)
        self.engine_heat_removed_w = result["engine_heat_removed_w"]
        self.rejected_heat_w = result["station_heat_return_w"]
        self.pump_flow_l_min = result["platform_pump_flow_l_min"]
        circuit, interface = self._engine_circuit(sim)
        # Publish the temperature from the actual circuit the stand cooled.
        # Do not manufacture a water jacket for an oil-cooled machine.
        name = "coolant_temp_k" if interface == "engine-coolant" else "oil_temp_k"
        setattr(sim.state, name, circuit.temp_k)
        sim._drivetrain_out[name] = circuit.temp_k


class EngineTestStand(EngineCycleSim):
    """The real engine solver with opt-in, separately owned bench services."""

    def __post_init__(self):
        super().__post_init__()
        self.test_stand_cooling = TestStandCooling.for_engine(self)
        # Include the external inventory in capture snapshots/deep copies.
        # It is not added to the engine specification or drivetrain graph.
        self.state.test_stand_cooling = self.test_stand_cooling

    def _step_once(self, dt):
        super()._step_once(dt)
        if self.test_stand_cooling is not None:
            self.test_stand_cooling.advance(dt, self)
