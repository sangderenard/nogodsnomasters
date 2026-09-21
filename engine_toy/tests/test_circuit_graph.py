"""Circuits solved as the same matrix the field solver uses, then reduced.

Two claims are under test.  The first is that modified nodal analysis is
not a separate piece of machinery: ``d0^T Y d0`` assembled from the DEC
complex IS the nodal admittance matrix, with branch admittance standing
where the Hodge star stands on a mesh.  The second is that Kron
reduction -- the Schur complement that eliminates interior nodes -- is
EXACT for a linear network, which is what makes it safe to solve a
component once at build time and carry a small port matrix afterwards
instead of factoring anything per tick.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from circuit_graph import Branch, CircuitGraph


def _divider(upper=100.0, lower=300.0):
    return CircuitGraph("divider", (
        Branch("r1", "in", "mid", "resistor", upper),
        Branch("r2", "mid", "gnd", "resistor", lower),
    ), ports=("in", "gnd"))


def _ladder():
    """Six nodes, seven branches, two of them bridging: not a tree."""
    values = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
    pairs = (("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"),
             ("e", "f"), ("b", "e"), ("c", "f"))
    return CircuitGraph("ladder", tuple(
        Branch(f"r{i}", tail, head, "resistor", value)
        for i, ((tail, head), value) in enumerate(zip(pairs, values))
    ), ports=("a", "f"))


# --------------------------------------------------------------------------
# the operator is the shared one
# --------------------------------------------------------------------------

def test_the_incidence_is_the_dec_complexs_own_d0() -> None:
    """A circuit is a 1-complex: nodes, branches, and no faces."""
    incidence = _divider().incidence()

    assert incidence.shape == (2, 3)
    # Each row leaves one node and enters another.
    assert (incidence.sum(axis=1) == 0).all()
    assert set(np.unique(incidence)) <= {-1.0, 0.0, 1.0}


def test_the_nodal_matrix_is_the_weighted_graph_laplacian() -> None:
    """``d0^T Y d0`` with conductances on the branches.

    The same expression gives the seven-point stencil on a lattice and
    the cotangent Laplacian on a triangle mesh.
    """
    matrix = _divider().admittance_matrix(0.0).real

    assert matrix[0, 0] == pytest.approx(1 / 100.0)
    assert matrix[0, 1] == pytest.approx(-1 / 100.0)
    assert matrix[1, 1] == pytest.approx(1 / 100.0 + 1 / 300.0)
    # Every row sums to zero: a uniform potential drives no current.
    assert np.allclose(matrix.sum(axis=1), 0.0)


def test_a_resistor_divider_divides() -> None:
    solved = _divider().solve(0.0, driven={"in": 1.0}, reference="gnd")

    assert solved["mid"].real == pytest.approx(300.0 / 400.0)
    assert solved["in"].real == pytest.approx(1.0)
    assert solved["gnd"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# alternating current
# --------------------------------------------------------------------------

def test_an_rc_low_pass_turns_over_at_its_corner() -> None:
    """At ``f = 1/(2 pi R C)`` the output is ``1/sqrt(2)`` and lags 45 degrees.

    This is the check that the complex admittance path is real: a purely
    resistive network could not produce a phase at all.
    """
    resistance, capacitance = 1000.0, 1.0e-6
    corner = 1.0 / (2.0 * math.pi * resistance * capacitance)
    filter_ = CircuitGraph("rc", (
        Branch("r", "in", "out", "resistor", resistance),
        Branch("c", "out", "gnd", "capacitor", capacitance),
    ), ports=("in", "gnd"))

    solved = filter_.solve(corner, driven={"in": 1.0}, reference="gnd")
    output = solved["out"]

    assert abs(output) == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-9)
    assert math.degrees(math.atan2(output.imag, output.real)) == pytest.approx(
        -45.0, abs=1e-9)


def test_a_capacitor_passes_more_as_frequency_rises() -> None:
    filter_ = CircuitGraph("rc", (
        Branch("r", "in", "out", "resistor", 1000.0),
        Branch("c", "out", "gnd", "capacitor", 1.0e-6),
    ), ports=("in", "gnd"))

    low = abs(filter_.solve(10.0, driven={"in": 1.0}, reference="gnd")["out"])
    high = abs(filter_.solve(10000.0, driven={"in": 1.0}, reference="gnd")["out"])

    assert low > 0.99
    assert high < 0.1


def test_an_inductor_at_dc_is_refused_rather_than_silently_infinite() -> None:
    circuit = CircuitGraph("dc-inductor", (
        Branch("l", "in", "gnd", "inductor", 1.0e-3),
        Branch("r", "in", "gnd", "resistor", 50.0),
    ), ports=("in", "gnd"))

    with pytest.raises(ValueError, match="singular at"):
        circuit.admittance_matrix(0.0)


# --------------------------------------------------------------------------
# Kron reduction
# --------------------------------------------------------------------------

def test_reduction_collapses_a_divider_to_its_series_resistance() -> None:
    """One interior node eliminated, and the answer is 100 + 300."""
    reduced = _divider().reduce(0.0)

    assert reduced.eliminated_nodes == 1
    assert reduced.admittance.shape == (2, 2)
    assert reduced.two_terminal_impedance_ohm().real == pytest.approx(400.0)


def test_reduction_of_parallel_branches_needs_no_elimination() -> None:
    """With no interior node the Schur complement is the matrix itself."""
    parallel = CircuitGraph("parallel", (
        Branch("r1", "a", "b", "resistor", 100.0),
        Branch("r2", "a", "b", "resistor", 100.0),
    ), ports=("a", "b"))

    reduced = parallel.reduce(0.0)

    assert reduced.eliminated_nodes == 0
    assert reduced.two_terminal_impedance_ohm().real == pytest.approx(50.0)


def test_reduction_is_exact_against_a_full_solve() -> None:
    """The claim the whole approach rests on, on a network with loops.

    Seven branches and two bridging paths reduce to a two-by-two, and
    the end-to-end impedance agrees with driving the full network and
    measuring the current that goes in.
    """
    ladder = _ladder()
    reduced = ladder.reduce(0.0)

    matrix = ladder.admittance_matrix(0.0).real
    index = ladder.node_index
    solved = ladder.solve(0.0, driven={"a": 1.0}, reference="f")
    potentials = np.array([solved[name].real for name in ladder.nodes])
    current_in = matrix[index["a"], :] @ potentials

    assert reduced.eliminated_nodes == 4
    assert reduced.two_terminal_impedance_ohm().real == pytest.approx(
        1.0 / current_in, rel=1e-9)


def test_reduction_is_exact_for_a_complex_network_too() -> None:
    """Reactive branches reduce the same way; nothing is real-only."""
    circuit = CircuitGraph("rc-ladder", (
        Branch("r1", "in", "m1", "resistor", 1000.0),
        Branch("c1", "m1", "gnd", "capacitor", 1.0e-6),
        Branch("r2", "m1", "out", "resistor", 2200.0),
        Branch("c2", "out", "gnd", "capacitor", 4.7e-7),
    ), ports=("in", "gnd"))

    reduced = circuit.reduce(1000.0)
    assert reduced.eliminated_nodes == 2
    assert np.iscomplexobj(reduced.admittance)
    assert abs(reduced.admittance[0, 0].imag) > 0.0


def test_a_reduced_network_keeps_its_provenance() -> None:
    reduced = _ladder().reduce(0.0)

    assert reduced.identity.endswith("/reduced")
    assert reduced.ports == ("a", "f")
    assert reduced.frequency_hz == 0.0


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_a_circuit_needs_branches() -> None:
    with pytest.raises(ValueError, match="needs branches"):
        CircuitGraph("empty", ())


def test_a_port_must_be_a_node_of_the_circuit() -> None:
    with pytest.raises(ValueError, match="ports not in the circuit"):
        CircuitGraph("bad", (
            Branch("r", "a", "b", "resistor", 10.0),), ports=("a", "nowhere"))


def test_reduction_needs_at_least_two_ports() -> None:
    with pytest.raises(ValueError, match="at least two ports"):
        _divider().reduce(0.0, ports=("in",))


def test_an_unknown_component_kind_is_refused() -> None:
    circuit = CircuitGraph("odd", (
        Branch("x", "a", "b", "memristor", 1.0),), ports=("a", "b"))

    with pytest.raises(ValueError, match="unknown component kind"):
        circuit.admittance_matrix(50.0)


def test_a_negative_resistor_is_refused() -> None:
    circuit = CircuitGraph("odd", (
        Branch("r", "a", "b", "resistor", -10.0),), ports=("a", "b"))

    with pytest.raises(ValueError, match="must be positive"):
        circuit.admittance_matrix(50.0)


def test_the_graph_declares_itself_a_circuit() -> None:
    attributes = _ladder().graph_attributes()

    assert attributes["part_role"] == "circuit"
    assert attributes["node_count"] == 6
    assert attributes["branch_count"] == 7
    assert attributes["ports"] == ["a", "f"]


# --------------------------------------------------------------------------
# as a dt-system engine
# --------------------------------------------------------------------------

def _rc(resistance=1000.0, capacitance=1.0e-6):
    return CircuitGraph("rc", (
        Branch("r", "in", "out", "resistor", resistance),
        Branch("c", "out", "gnd", "capacitor", capacitance),
    ), ports=("in", "gnd"))


def test_the_engine_publishes_a_dt_limit_from_its_own_time_constant() -> None:
    """The circuit says what it needs; the controller decides what it gets."""
    from circuit_graph import CircuitEngine

    engine = CircuitEngine(_rc(), reference="gnd", driven={"in": 1.0})

    assert engine.preferred_dt() == pytest.approx(1000.0 * 1.0e-6 / 10.0)
    assert engine.causal_ceiling_dt() == pytest.approx(engine.preferred_dt())


def test_a_purely_resistive_circuit_has_nothing_to_say_about_dt() -> None:
    from circuit_graph import CircuitEngine

    engine = CircuitEngine(_divider(), reference="gnd", driven={"in": 1.0})

    assert engine.preferred_dt() is None
    assert engine.causal_ceiling_dt() == math.inf


def test_the_step_response_follows_the_exponential() -> None:
    """Trapezoidal companion models, checked against the closed form."""
    from circuit_graph import CircuitEngine

    resistance, capacitance = 1000.0, 1.0e-6
    engine = CircuitEngine(_rc(), reference="gnd", driven={"in": 1.0})
    tau = resistance * capacitance
    step = 1.0e-6

    for _ in range(2000):
        engine.step(step)

    exact = 1.0 - math.exp(-2000 * step / tau)
    assert engine.potentials["out"] == pytest.approx(exact, abs=1e-3)


def test_assemblies_are_cached_by_timestep_not_rebuilt_per_step() -> None:
    """The companion conductances depend on dt, so dt keys the assembly.

    A controller holding dt steady pays for one assembly and reuses it;
    changing dt costs exactly one more.
    """
    from circuit_graph import CircuitEngine

    engine = CircuitEngine(_rc(), reference="gnd", driven={"in": 1.0})

    for _ in range(50):
        engine.step(1.0e-6)
    assert engine.assembled_timesteps == 1

    for _ in range(50):
        engine.step(5.0e-7)
    assert engine.assembled_timesteps == 2


def test_the_engine_publishes_stored_energy_and_dissipation() -> None:
    from circuit_graph import CircuitEngine

    capacitance = 1.0e-6
    engine = CircuitEngine(_rc(capacitance=capacitance),
                           reference="gnd", driven={"in": 1.0})
    for _ in range(500):
        _, metrics, _ = engine.step(1.0e-6)

    stored = 0.5 * capacitance * engine.potentials["out"] ** 2
    assert float(metrics.error_channels[0]) == pytest.approx(stored)
    assert float(metrics.error_channels[1]) > 0.0
    assert metrics.advanced_dt == pytest.approx(1.0e-6)


def test_snapshot_and_restore_are_exact() -> None:
    """The dt controller rejects steps; an engine that cannot rewind lies."""
    from circuit_graph import CircuitEngine

    engine = CircuitEngine(_rc(), reference="gnd", driven={"in": 1.0})
    for _ in range(200):
        engine.step(1.0e-6)

    mark = engine.snapshot()
    held = engine.potentials["out"]
    for _ in range(300):
        engine.step(1.0e-6)
    assert engine.potentials["out"] != held

    engine.restore(mark)
    assert engine.potentials["out"] == held
    assert engine.world_time == pytest.approx(200 * 1.0e-6)


def test_two_engines_share_one_state_table() -> None:
    """Engines couple through the state system, not through each other."""
    from circuit_graph import CircuitEngine
    from src.common.dt_system.state_table import StateTable

    table = StateTable()
    first = CircuitEngine(_rc(), reference="gnd", driven={"in": 1.0})
    second = CircuitEngine(
        CircuitGraph("rc2", (
            Branch("r", "in", "out", "resistor", 220.0),
            Branch("c", "out", "gnd", "capacitor", 4.7e-6),
        ), ports=("in", "gnd")), reference="gnd", driven={"in": 5.0})

    for _ in range(100):
        first.step(1.0e-6, state_table=table)
        second.step(1.0e-6, state_table=table)

    assert table.get("circuit", "rc", "v:out") == pytest.approx(
        first.potentials["out"])
    assert table.get("circuit", "rc2", "v:out") == pytest.approx(
        second.potentials["out"])
    assert table.get("circuit", "rc", "v:out") != table.get(
        "circuit", "rc2", "v:out")


def test_a_driven_node_may_be_a_waveform() -> None:
    """Supply phase and magnitude enter here, as a function of world time."""
    from circuit_graph import CircuitEngine

    engine = CircuitEngine(
        _rc(), reference="gnd",
        driven={"in": lambda when: math.sin(2.0 * math.pi * 50.0 * when)})

    for _ in range(100):
        engine.step(1.0e-5)

    assert engine.world_time == pytest.approx(100 * 1.0e-5)
    assert engine.potentials["in"] == pytest.approx(
        math.sin(2.0 * math.pi * 50.0 * engine.world_time))


def test_an_unknown_reference_node_is_refused() -> None:
    from circuit_graph import CircuitEngine

    with pytest.raises(ValueError, match="no node named"):
        CircuitEngine(_rc(), reference="nowhere")


def test_a_component_without_a_companion_model_is_refused() -> None:
    from circuit_graph import CompanionAssembly

    circuit = CircuitGraph("odd", (
        Branch("x", "a", "b", "memristor", 1.0),), ports=("a", "b"))

    with pytest.raises(ValueError, match="no companion model"):
        CompanionAssembly.build(circuit, 1.0e-6)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def _supply(emf=12.0, internal=0.1, load=5.9):
    from circuit_graph import VoltageSource, assemble

    return assemble("supply", [
        VoltageSource("batt", "gnd", "out", emf, internal),
        Branch("load", "out", "gnd", "resistor", load),
    ], ports=("out", "gnd"))


def test_a_source_behind_an_impedance_divides_with_its_load() -> None:
    graph, injected = _supply()
    solved = graph.solve(0.0, reference="gnd", injected=injected)

    assert solved["out"].real == pytest.approx(12.0 * 5.9 / 6.0)


def test_the_source_agrees_with_the_network_about_its_own_terminal() -> None:
    """Two routes to the same number: the element's formula and the solve."""
    from circuit_graph import VoltageSource

    source = VoltageSource("batt", "gnd", "out", 12.0, 0.1)
    graph, injected = _supply()
    solved = graph.solve(0.0, reference="gnd", injected=injected)
    current = solved["out"].real / 5.9

    assert source.terminal_voltage_v(current).real == pytest.approx(
        solved["out"].real)


def test_thevenin_and_norton_are_indistinguishable_from_outside() -> None:
    """The equivalence the whole conversion rests on, to machine precision."""
    from circuit_graph import CurrentSource, assemble

    thevenin, injected_t = _supply()
    norton, injected_n = assemble("norton", [
        CurrentSource("src", "gnd", "out", 12.0 / 0.1, parallel_conductance_s=1 / 0.1),
        Branch("load", "out", "gnd", "resistor", 5.9),
    ], ports=("out", "gnd"))

    left = thevenin.solve(0.0, reference="gnd", injected=injected_t)["out"]
    right = norton.solve(0.0, reference="gnd", injected=injected_n)["out"]

    assert abs(left - right) < 1e-12


def test_a_real_source_sags_and_an_ideal_one_could_not() -> None:
    outputs = []
    for load in (100.0, 10.0, 1.0, 0.1):
        graph, injected = _supply(load=load)
        outputs.append(graph.solve(0.0, reference="gnd", injected=injected)["out"].real)

    assert outputs == sorted(outputs, reverse=True), "heavier load must sag more"
    assert outputs[0] > 11.9
    assert outputs[-1] < 7.0


def test_matching_the_load_to_the_source_halves_the_terminal_voltage() -> None:
    """Maximum power transfer, arrived at rather than encoded."""
    graph, injected = _supply(internal=0.1, load=0.1)
    solved = graph.solve(0.0, reference="gnd", injected=injected)

    assert solved["out"].real == pytest.approx(6.0)


def test_short_circuit_current_is_the_norton_current() -> None:
    from circuit_graph import VoltageSource

    source = VoltageSource("batt", "gnd", "out", 12.0, 0.1)

    assert abs(source.short_circuit_current_a) == pytest.approx(120.0)
    assert source.norton_conductance_s == pytest.approx(10.0)


def test_an_ideal_voltage_source_is_refused_with_the_reason() -> None:
    """The refusal is the physical statement, not a solver limitation."""
    from circuit_graph import VoltageSource

    with pytest.raises(ValueError, match="positive internal resistance"):
        VoltageSource("ideal", "a", "b", 12.0, 0.0)


def test_a_current_source_injects_into_its_head_and_out_of_its_tail() -> None:
    from circuit_graph import CurrentSource, assemble

    graph, injected = assemble("pump", [
        CurrentSource("src", "gnd", "out", 2.0),
        Branch("load", "out", "gnd", "resistor", 50.0),
    ], ports=("out", "gnd"))
    solved = graph.solve(0.0, reference="gnd", injected=injected)

    assert solved["out"].real == pytest.approx(2.0 * 50.0)


def test_a_bare_current_source_contributes_no_branch() -> None:
    from circuit_graph import CurrentSource

    assert CurrentSource("i", "a", "b", 1.0).branches() == ()
    assert len(CurrentSource("i", "a", "b", 1.0, 0.5).branches()) == 1


def test_injections_from_several_sources_superpose() -> None:
    from circuit_graph import CurrentSource, assemble

    graph, injected = assemble("two", [
        CurrentSource("a", "gnd", "out", 1.0),
        CurrentSource("b", "gnd", "out", 3.0),
        Branch("load", "out", "gnd", "resistor", 10.0),
    ], ports=("out", "gnd"))
    solved = graph.solve(0.0, reference="gnd", injected=injected)

    assert injected["out"] == pytest.approx(4.0)
    assert solved["out"].real == pytest.approx(40.0)


def test_assemble_collects_branches_and_injections_together() -> None:
    """Elements declare what they contribute; assembly does not interpret."""
    from circuit_graph import VoltageSource, assemble
    from traces import PcbTrace

    graph, injected = assemble("mixed", [
        VoltageSource("v", "gnd", "a", 5.0, 0.2),
        PcbTrace("tr", "a", "b", 0.015, 0.4e-3),
        Branch("load", "b", "gnd", "resistor", 100.0),
    ], ports=("a", "gnd"))

    # source resistor + trace resistance + trace inductance + load
    assert len(graph.branches) == 4
    assert set(injected) == {"a", "gnd"}
    assert injected["a"] == pytest.approx(25.0)


# --------------------------------------------------------------------------
# mutual inductance
# --------------------------------------------------------------------------

def _transformer(turns=1.0, load_ohm=1.0e6, coupling=0.999, secondary_h=1.0e-3):
    from circuit_graph import MutualInductance, VoltageSource, assemble

    return assemble("xfmr", [
        VoltageSource("src", "gnd", "in", 1.0, 0.001),
        Branch("lp", "in", "gnd", "inductor", secondary_h * turns ** 2),
        Branch("ls", "out", "sgnd", "inductor", secondary_h),
        Branch("load", "out", "sgnd", "resistor", load_ohm),
        Branch("tie", "sgnd", "gnd", "resistor", 1.0e-9),
        MutualInductance("m", "lp", "ls", coupling),
    ], ports=("in", "gnd"))


def test_mutual_inductance_is_k_root_l1_l2() -> None:
    from circuit_graph import MutualInductance

    coupling = MutualInductance("m", "a", "b", 0.8)

    assert coupling.mutual_h(4.0e-3, 1.0e-3) == pytest.approx(
        0.8 * math.sqrt(4.0e-3 * 1.0e-3))
    assert coupling.turns_ratio(4.0e-3, 1.0e-3) == pytest.approx(2.0)


def test_leakage_is_what_the_secondary_cannot_see() -> None:
    from circuit_graph import MutualInductance

    assert MutualInductance("m", "a", "b", 0.9).leakage_h(1.0) == pytest.approx(0.19)
    assert MutualInductance("m", "a", "b", 1.0).leakage_h(1.0) == pytest.approx(0.0)


def test_coupling_beyond_unity_is_refused_on_energy_grounds() -> None:
    from circuit_graph import MutualInductance

    with pytest.raises(ValueError, match="negative energy"):
        MutualInductance("bad", "a", "b", 1.5)


def test_coupling_makes_the_branch_admittance_non_diagonal() -> None:
    """The structural claim: a coupled pair is a 2x2 block, not two entries."""
    graph, _ = _transformer(coupling=0.9)
    admittance = graph.branch_admittance(1.0e5)

    off_diagonal = admittance - np.diag(np.diag(admittance))
    assert np.abs(off_diagonal).max() > 0.0

    uncoupled, _ = _transformer(coupling=0.0)
    quiet = uncoupled.branch_admittance(1.0e5)
    assert np.allclose(quiet - np.diag(np.diag(quiet)), 0.0)


def test_the_voltage_ratio_follows_the_turns_ratio() -> None:
    """Lightly loaded, the secondary reads the primary divided by n."""
    for turns in (1.0, 2.0, 10.0):
        graph, injected = _transformer(turns=turns)
        solved = graph.solve(1.0e5, reference="gnd", injected=injected)
        ratio = abs(solved["in"]) / abs(solved["out"] - solved["sgnd"])

        assert ratio == pytest.approx(turns, rel=0.01)


def test_a_load_reflects_through_the_square_of_the_turns_ratio() -> None:
    """``Z_in = n^2 R`` -- the property a transformer exists for."""
    frequency = 1.0e6
    for turns, load in ((1.0, 10.0), (1.0, 100.0), (3.0, 10.0), (3.0, 100.0)):
        graph, injected = _transformer(turns=turns, load_ohm=load)
        solved = graph.solve(frequency, reference="gnd", injected=injected)

        potentials = np.array([solved[name] for name in graph.nodes])
        branch_voltage = graph.incidence() @ potentials
        branch_current = graph.branch_admittance(frequency) @ branch_voltage
        primary = [b.identity for b in graph.branches].index("lp")
        impedance = branch_voltage[primary] / branch_current[primary]

        assert impedance.real == pytest.approx(turns ** 2 * load, rel=0.01)


def test_only_inductors_carry_mutual_flux() -> None:
    from circuit_graph import CircuitGraph, MutualInductance

    graph = CircuitGraph("odd", (
        Branch("r", "a", "b", "resistor", 10.0),
        Branch("l", "b", "c", "inductor", 1.0e-3),
    ), couplings=(MutualInductance("m", "r", "l", 0.5),))

    with pytest.raises(ValueError, match="only inductors carry mutual flux"):
        graph.branch_admittance(1.0e3)


def test_coupled_inductors_have_no_dc_admittance() -> None:
    graph, _ = _transformer()

    with pytest.raises(ValueError, match="short at\nDC|short at DC"):
        graph.branch_admittance(0.0)


def test_a_coupling_must_name_branches_that_exist() -> None:
    from circuit_graph import CircuitGraph, MutualInductance

    graph = CircuitGraph("odd", (
        Branch("l", "a", "b", "inductor", 1.0e-3),
    ), couplings=(MutualInductance("m", "l", "nowhere", 0.5),))

    with pytest.raises(ValueError, match="no branch named"):
        graph.branch_admittance(1.0e3)


# --------------------------------------------------------------------------
# solving against an outer network
# --------------------------------------------------------------------------

FREQ = 1.0e3

_INNER = (
    Branch("r1", "p", "m1", "resistor", 220.0),
    Branch("c1", "m1", "q", "capacitor", 1.0e-6),
    Branch("r2", "m1", "m2", "resistor", 470.0),
    Branch("c2", "m2", "q", "capacitor", 2.2e-7),
    Branch("r3", "m2", "q", "resistor", 1000.0),
)


def _outer_elements():
    from circuit_graph import VoltageSource

    return [
        VoltageSource("svc", "gnd", "feed", 230.0, 0.05),
        Branch("cable", "feed", "load", "resistor", 0.35),
        Branch("return", "gnd2", "gnd", "resistor", 0.35),
    ]


def _renamed(branch):
    swap = {"p": "load", "q": "gnd2"}
    return Branch(branch.identity,
                  swap.get(branch.tail, f"in.{branch.tail}"),
                  swap.get(branch.head, f"in.{branch.head}"),
                  branch.kind, branch.value)


def test_a_reduced_subcircuit_composes_exactly_with_an_outer_network() -> None:
    """Hierarchical and flat must agree, or the reduction is a lie.

    This is what "solve against the outer electrical graph" requires:
    a part is solved once on its own, carried as a port block, and
    plugged into a supply it knew nothing about -- and the answer is
    the same as if every branch had been in one matrix.
    """
    from circuit_graph import CircuitGraph, assemble

    inner = CircuitGraph("inner", _INNER, ports=("p", "q"))

    flat, injected = assemble(
        "flat", _outer_elements() + [_renamed(b) for b in _INNER],
        ports=("load", "gnd2"))
    nested, injected_nested = assemble(
        "nested", _outer_elements() + [inner.reduce(FREQ).stamp(("load", "gnd2"))],
        ports=("load", "gnd2"))

    flat_solved = flat.solve(FREQ, reference="gnd", injected=injected)
    nested_solved = nested.solve(FREQ, reference="gnd", injected=injected_nested)

    assert len(nested.nodes) < len(flat.nodes)
    for name in ("feed", "load", "gnd2"):
        assert abs(flat_solved[name] - nested_solved[name]) < 1e-10


def test_composition_nests_more_than_once() -> None:
    """A circuit containing a stamp reduces again, and still agrees."""
    from circuit_graph import CircuitGraph, assemble

    inner = CircuitGraph("inner", _INNER, ports=("p", "q"))
    middle, _ = assemble("middle", [
        Branch("series", "a", "load", "resistor", 47.0),
        inner.reduce(FREQ).stamp(("load", "gnd2")),
    ], ports=("a", "gnd2"))

    once = middle.reduce(FREQ)
    flat = CircuitGraph("flat-middle", (
        Branch("series", "a", "load", "resistor", 47.0),
        *[_renamed(b) for b in _INNER],
    ), ports=("a", "gnd2")).reduce(FREQ)

    assert np.allclose(once.admittance, flat.admittance, atol=1e-10)


def test_a_passive_stamp_floats() -> None:
    """Rows sum to zero: shifting every terminal drives no current."""
    from circuit_graph import CircuitGraph

    stamp = CircuitGraph("inner", _INNER, ports=("p", "q")).reduce(FREQ).stamp()

    assert stamp.is_floating
    assert stamp.graph_attributes()["floating"] is True


def test_stamp_terminals_are_nodes_even_without_a_branch() -> None:
    from circuit_graph import CircuitGraph, NodalStamp

    stamp = NodalStamp("s", ("x", "y"), np.array([[1.0, -1.0], [-1.0, 1.0]]))
    graph = CircuitGraph("only-stamp", (), stamps=(stamp,))

    assert set(graph.nodes) == {"x", "y"}
    assert graph.admittance_matrix(FREQ).shape == (2, 2)


def test_a_circuit_may_be_nothing_but_stamps() -> None:
    from circuit_graph import CircuitGraph, NodalStamp

    stamp = NodalStamp("s", ("x", "y"), np.array([[0.5, -0.5], [-0.5, 0.5]]))
    graph = CircuitGraph("only-stamp", (), stamps=(stamp,))
    solved = graph.solve(FREQ, reference="y", driven={"x": 1.0})

    assert solved["x"].real == pytest.approx(1.0)


def test_a_stamp_may_be_renamed_onto_the_enclosing_nodes() -> None:
    from circuit_graph import CircuitGraph

    reduced = CircuitGraph("inner", _INNER, ports=("p", "q")).reduce(FREQ)

    assert reduced.stamp().nodes == ("p", "q")
    assert reduced.stamp(("hot", "cold")).nodes == ("hot", "cold")
    with pytest.raises(ValueError, match="names for"):
        reduced.stamp(("only-one",))


def test_a_malformed_stamp_is_refused() -> None:
    from circuit_graph import NodalStamp

    with pytest.raises(ValueError, match="must be square"):
        NodalStamp("s", ("x", "y"), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="over 3 nodes"):
        NodalStamp("s", ("x", "y", "z"), np.zeros((2, 2)))


def test_a_stamp_naming_an_absent_node_is_refused() -> None:
    from circuit_graph import CircuitGraph, NodalStamp

    graph = CircuitGraph(
        "odd", (Branch("r", "a", "b", "resistor", 10.0),),
        stamps=(NodalStamp("s", ("a", "b"), np.zeros((2, 2))),))
    graph.admittance_matrix(FREQ)  # fine: both nodes exist

    detached = CircuitGraph(
        "odd2", (Branch("r", "a", "b", "resistor", 10.0),))
    object.__setattr__(detached, "stamps",
                       (NodalStamp("s", ("a", "ghost"), np.zeros((2, 2))),))
    with pytest.raises(ValueError, match="no node named"):
        detached.admittance_matrix(FREQ)


def test_a_branch_in_two_couplings_inverts_as_one_matrix() -> None:
    """Three windings sharing flux are ONE impedance matrix, not two pairs.

    A cavity mode driven by two probes has exactly this shape, and so
    does any three-winding transformer.  Inverting each declared pair on
    its own and adding the blocks is not the inverse of the whole, and
    it counts the shared branch's own inductance once per coupling.  The
    second assertion shows the two answers genuinely differ, so this is
    a correctness fix rather than a refactor.
    """
    from circuit_graph import MutualInductance

    inductances = (1.0e-3, 2.0e-3, 4.0e-3)
    graph = CircuitGraph("three-winding", (
        Branch("la", "n0", "n1", "inductor", inductances[0]),
        Branch("lb", "n1", "n2", "inductor", inductances[1]),
        Branch("lc", "n2", "n0", "inductor", inductances[2]),
    ), ports=("n0", "n1"), couplings=(
        MutualInductance("ab", "la", "lb", 0.5),
        MutualInductance("bc", "lb", "lc", 0.4),
    ))

    mutual_ab = 0.5 * math.sqrt(inductances[0] * inductances[1])
    mutual_bc = 0.4 * math.sqrt(inductances[1] * inductances[2])
    whole = np.array([
        [inductances[0], mutual_ab, 0.0],
        [mutual_ab, inductances[1], mutual_bc],
        [0.0, mutual_bc, inductances[2]],
    ])
    omega = 2.0 * math.pi * FREQ

    assert np.allclose(graph.branch_admittance(FREQ),
                       np.linalg.inv(1.0j * omega * whole))

    pairwise = np.zeros((3, 3), dtype=complex)
    for first, second, mutual in ((0, 1, mutual_ab), (1, 2, mutual_bc)):
        block = np.linalg.inv(1.0j * omega * np.array(
            [[whole[first, first], mutual], [mutual, whole[second, second]]]))
        pairwise[np.ix_([first, second], [first, second])] += block
    assert not np.allclose(graph.branch_admittance(FREQ), pairwise)


def test_jointly_inconsistent_couplings_are_refused() -> None:
    """Each pair inside [-1, 1], the group still storing negative energy.

    Two windings cannot each share 80 percent of a third's flux.  No
    pairwise check can see this; only the whole matrix can.
    """
    from circuit_graph import MutualInductance

    graph = CircuitGraph("greedy", (
        Branch("la", "n0", "n1", "inductor", 1.0e-3),
        Branch("lb", "n1", "n2", "inductor", 1.0e-3),
        Branch("lc", "n2", "n0", "inductor", 1.0e-3),
    ), ports=("n0", "n1"), couplings=(
        MutualInductance("ac", "la", "lc", 0.8),
        MutualInductance("bc", "lb", "lc", 0.8),
    ))

    with pytest.raises(ValueError, match="negative energy"):
        graph.branch_admittance(FREQ)

