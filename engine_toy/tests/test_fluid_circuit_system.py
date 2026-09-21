import math

import pytest
import sympy as sp

from drivetrain_graph import (
    FluidCircuit,
    FluidCircuitInputs,
    FluidCircuitSystem,
    FluidVolume,
    choked_orifice_mass_flow_kg_s,
)
from fluid_circuit_laws import (
    compressible_orifice_mass_flow_kg_s,
    convected_enthalpy_w,
    hagen_poiseuille_mass_flow_kg_s,
    ideal_gas_pressure_pa,
    signed_compressible_orifice_mass_flow_kg_s,
)
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.time_contracts import HOLD


class _CircuitOwner:
    def __init__(self, circuit):
        self.fluid_circuits = [circuit]
        self.calls = []

    def _advance_fluid_circuits(self, *args):
        self.calls.append(args)
        self.fluid_circuits[0].delivered_flow_kg_s = 0.125
        self.fluid_circuits[0].pressure_pa += 2500.0


class _DelegatingOwner(_CircuitOwner):
    def __init__(self, circuit):
        super().__init__(circuit)
        self.fluid_system = FluidCircuitSystem(self, self.fluid_circuits)

    def machine_tick(self, dt, inputs):
        self.fluid_system.stage(inputs)
        if not self.fluid_system.externally_scheduled:
            self.fluid_system.advance(dt, inputs)


def test_top_level_system_owns_the_existing_circuit_objects_and_rolls_back():
    circuit = FluidCircuit(
        "compressible-gas", frozenset(("tank", "port")),
        ({"identity": "line", "a": "tank", "b": "port",
          "circuit_identity": "air"},),
    )
    owner = _CircuitOwner(circuit)
    system = FluidCircuitSystem(owner, owner.fluid_circuits)
    table = StateTable()
    system.register(
        table, lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
        (system,), group_label="fluid-system",
    )
    checkpoint = system.snapshot()
    inputs = FluidCircuitInputs(intake_demand_kg_s=0.02)
    system.stage(inputs)

    ok, metrics, returned = system.step_with_state(None, 0.25, state_table=table)

    assert ok
    assert returned is owner.fluid_circuits
    assert system.circuits[0] is circuit
    assert owner.calls[0][0] == pytest.approx(0.25)
    assert owner.calls[0][2] == pytest.approx(0.02)
    assert metrics.max_flux == pytest.approx(0.125)
    assert metrics.pub_contract.tolist() == [HOLD]
    assert metrics.pub_tau_present.tolist() == [0.0]
    assert circuit.pressure_pa == pytest.approx(103_825.0)

    system.restore(checkpoint)
    assert system.circuits[0] is circuit
    assert circuit.pressure_pa == pytest.approx(101_325.0)
    assert circuit.delivered_flow_kg_s == 0.0


def test_atmosphere_registers_as_one_externally_resolved_volume():
    circuit = FluidCircuit("compressible-gas", frozenset(), ())
    system = FluidCircuitSystem(_CircuitOwner(circuit), [circuit])
    atmosphere = FluidVolume(
        identity="lab.dewar.chamber_volume",
        volume_m3=1.953125,
        fluid="gas",
        pressure_pa=101_325.0,
        temperature_k=293.15,
        state_owner="atmosphere",
        spatial_model="voxel",
    )

    assert system.register_volume(atmosphere) is atmosphere
    assert system.volume(atmosphere.identity) is atmosphere
    assert system.volumes == (atmosphere,)
    with pytest.raises(ValueError, match="another contract"):
        system.register_volume(FluidVolume(
            identity=atmosphere.identity, volume_m3=2.0, fluid="gas"))


def test_dt_registration_makes_owner_stage_without_double_advancing():
    circuit = FluidCircuit("compressible-gas", frozenset(), ())
    owner = _DelegatingOwner(circuit)
    inputs = FluidCircuitInputs(intake_demand_kg_s=0.01)
    owner.machine_tick(0.1, inputs)
    assert len(owner.calls) == 1

    table = StateTable()
    owner.fluid_system.register(
        table, lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
        (owner.fluid_system,), group_label="fluid-system")
    owner.machine_tick(0.1, inputs)
    assert len(owner.calls) == 1
    owner.fluid_system.step_with_state(None, 0.1, state_table=table)
    assert len(owner.calls) == 2


def test_symbolic_compressible_law_retains_choked_and_subcritical_regimes():
    p0, p1 = sp.symbols("p0 p1", positive=True)
    expression = compressible_orifice_mass_flow_kg_s(
        p0, p1, 293.15, 1.0e-4, 287.0, 1.4, 0.65)
    evaluate = sp.lambdify((p0, p1), expression, "math")

    choked_low = evaluate(1.0e6, 1.0e5)
    choked_lower = evaluate(1.0e6, 0.5e5)
    subcritical = evaluate(1.0e6, 0.9e6)
    near_equal = evaluate(1.0e6, 0.999999e6)

    assert choked_low == pytest.approx(choked_lower)
    assert 0.0 < near_equal < subcritical < choked_low
    assert choked_low == pytest.approx(
        choked_orifice_mass_flow_kg_s(1.0e-4, 1.0e6, 293.15, 0.65),
        rel=2.0e-4,
    )


def test_symbolic_pipe_and_stream_laws_preserve_orientation_and_dimensions():
    forward = signed_compressible_orifice_mass_flow_kg_s(
        8.0e5, 1.0e5, 300.0, 280.0, 2.0e-5, 287.0, 1.4, 0.7)
    reverse = signed_compressible_orifice_mass_flow_kg_s(
        1.0e5, 8.0e5, 280.0, 300.0, 2.0e-5, 287.0, 1.4, 0.7)
    assert float(forward) == pytest.approx(-float(reverse))

    flow = hagen_poiseuille_mass_flow_kg_s(
        3.0e5, 1.0e5, 0.004, 2.0, 1.0e-3, 1000.0)
    wider = hagen_poiseuille_mass_flow_kg_s(
        3.0e5, 1.0e5, 0.008, 2.0, 1.0e-3, 1000.0)
    assert float(wider / flow) == pytest.approx(16.0)
    assert float(convected_enthalpy_w(flow, 42_000.0)) == pytest.approx(
        float(flow) * 42_000.0)

    pressure = ideal_gas_pressure_pa(10.0, 300.0, 0.25)
    assert float(pressure) == pytest.approx(10.0 * 8.31446261815324 * 300.0 / 0.25)
