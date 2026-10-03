"""Orbital craft build step 8: the craft as an engine_toy machine.

    python -m pytest tests/test_orbital_craft_machine.py -q -s   (from engine_toy/)

The first run compiles the machine's pieces (~100 s); later runs load them.
Each dt test runs its own state to completion before the next is built
(``llvm_dt_system`` runs the program of the state instantiated last; see
tests/test_orbital_jumper.py::test_interleaved_states_each_run_their_own_
program).
"""
import math

import numpy as np
import pytest

from orbital_actuation import allocate_wrench, machine_wrench
from orbital_craft_machine import (
    PAIRS,
    MachineCraft,
    craft_machine_columns,
    craft_machine_dt_pieces,
    orbital_craft,
)

CRAFT = orbital_craft()
FULL = {tank.identity: tank.fill_kg for tank in CRAFT.tanks}
#: the bipropellant nearly gone, the RCS tank full
DRAINED = {"tank.mmh": 5.0, "tank.nto": 8.0, "tank.hydrazine": 60.0}


def _cm(fill=None):
    return np.asarray(CRAFT.mass_properties(fill).center_of_gravity)


def _roles(allocation, tol=1e-9):
    return {CRAFT.thrusters[k].role
            for k, u in enumerate(allocation.throttles) if u > tol}


def _deflection_deg(angles):
    a, b = angles
    return math.degrees(math.acos(min(1.0, math.cos(a) * math.cos(b))))


# ------------------------------------------------------------ the document
def test_machine_passes_production_check_and_every_thruster_reaches_the_mount():
    assert CRAFT.check() == []
    paths = CRAFT.load_paths()
    tanks = {tank.identity for tank in CRAFT.tanks}
    assert all(path.kind != "unsupported" for path in paths.values())
    roles = {}
    for thruster in CRAFT.thrusters:
        path = paths[thruster.identity]
        roles.setdefault(thruster.role, 0)
        roles[thruster.role] += 1
        # through structure to the declared docking-ring mount, never
        # through its own feed line (routed lines carry no load)
        assert path.kind == "rigid"
        assert path.mount == "hull.docking_ring"
        assert not set(path.hops) & tanks
        assert thruster.feeds, thruster.identity
    assert roles == {"main": 1, "brake": 2, "navigation": 16}
    main = CRAFT.thrusters[CRAFT.thrusters_by_role("main")[0]]
    assert main.gimballed and main.cone_half_angle_rad > 0.0
    # the mixture ratio is the fuel lines' declared shares
    shares = dict(main.feeds)
    assert shares["tank.nto"] / shares["tank.mmh"] == pytest.approx(1.65)
    print(f"\n{len(paths)} massive bodies, all rigid to the docking ring; "
          f"dry {CRAFT.dry_mass_kg:.2f} kg, wet {CRAFT.mass_kg:.2f} kg")


def test_mass_properties_are_the_reduction_and_the_piece_reproduces_it():
    full, drained = CRAFT.mass_properties(), CRAFT.mass_properties(DRAINED)
    assert full.total_mass_kg - drained.total_mass_kg == pytest.approx(
        sum(FULL.values()) - sum(DRAINED.values()), rel=1e-15)
    # the off-axis bay and RCS tank: products of inertia, a full tensor
    tensor = np.asarray(full.inertia_tensor_kg_m2)
    assert abs(tensor[0, 1]) > 1.0
    # the compiled mass-properties and inverse pieces at a partial fill
    pieces, _labels = craft_machine_dt_pieces(CRAFT, 0)
    props, inverse = pieces[0], pieces[1]
    columns = craft_machine_columns(CRAFT)
    partial = {"tank.mmh": 101.5, "tank.nto": 167.3, "tank.hydrazine": 22.25}
    for t, tank in enumerate(CRAFT.tanks):
        columns[f"tank{t}_propellant"] = np.full(1, partial[tank.identity])
    columns["dry_mass"] = np.full(1, CRAFT.dry_mass_kg)
    out = dict(zip(props.output_names, props(*[
        np.ascontiguousarray(columns[n]) for n in props.argument_names])))
    rigid = CRAFT.mass_properties(partial)
    centre = np.asarray([out[f"centre_of_mass_{a}_next"][0] for a in "xyz"])
    assert centre == pytest.approx(np.asarray(rigid.center_of_gravity),
                                   abs=1e-12)
    expected = np.asarray(rigid.inertia_tensor_kg_m2)
    for pair in PAIRS:
        i, j = "xyz".index(pair[0]), "xyz".index(pair[1])
        assert out[f"inertia_{pair}_next"][0] == pytest.approx(
            expected[i, j], rel=1e-11, abs=1e-10)
        columns[f"inertia_{pair}"] = out[f"inertia_{pair}_next"]
    inv = dict(zip(inverse.output_names, inverse(*[
        np.ascontiguousarray(columns[n]) for n in inverse.argument_names])))
    product = np.empty((3, 3))
    for pair in PAIRS:
        i, j = "xyz".index(pair[0]), "xyz".index(pair[1])
        product[i, j] = product[j, i] = inv[f"inverse_inertia_{pair}_next"][0]
    assert product @ expected == pytest.approx(np.eye(3), abs=1e-12)
    print(f"\nfull: {full.total_mass_kg:.2f} kg, cm "
          f"{np.round(full.center_of_gravity, 4)}; drained: "
          f"{drained.total_mass_kg:.2f} kg, cm "
          f"{np.round(drained.center_of_gravity, 4)}")


# ------------------------------------------------------------- allocation
def test_pure_torque_is_met_by_rcs_couples_with_zero_net_force():
    centre = _cm()
    for torque in ((5.0, 0.0, 0.0), (0.0, 30.0, 0.0), (0.0, 0.0, -12.0),
                   (3.0, -8.0, 6.0)):
        allocation = allocate_wrench(
            CRAFT.thrusters, np.zeros(3), torque, centre_of_mass_m=centre,
            tank_propellant_kg=FULL, fuel_weight=1.0e-7)
        print(f"\ntorque {torque}: |F| {np.linalg.norm(allocation.force_n):.2e}"
              f" N, |tau short| "
              f"{np.linalg.norm(allocation.torque_shortfall_n_m):.2e} N m, "
              f"{int(np.sum(allocation.throttles > 1e-9))} RCS on")
        assert _roles(allocation) == {"navigation"}
        assert np.linalg.norm(allocation.force_n) < 1.0e-3      # of 22 N
        assert np.linalg.norm(allocation.torque_shortfall_n_m) < 1.0e-3


@pytest.mark.parametrize("fill", [None, DRAINED], ids=["full", "drained"])
def test_force_off_the_engine_axis_gimbals_the_main_engine_to_cancel_torque(
        fill):
    # the centre of mass is off the main engine's axis; a force along the
    # line from the gimbal pivot through the centre of mass, with no
    # torque, is met by gimballing the engine onto that line
    centre = _cm(fill)
    tanks = FULL if fill is None else fill
    k = CRAFT.thrusters_by_role("main")[0]
    main = CRAFT.thrusters[k]
    line = centre - np.asarray(main.position_m)
    line /= np.linalg.norm(line)
    trim = math.degrees(math.acos(line[0]))
    allocation = allocate_wrench(CRAFT.thrusters, 3000.0 * line, np.zeros(3),
                                 centre_of_mass_m=centre,
                                 tank_propellant_kg=tanks)
    deflection = _deflection_deg(allocation.gimbal_rad[k])
    print(f"\ntrim line {trim:.4f} deg, gimbal deflection {deflection:.4f} "
          f"deg (cone {math.degrees(main.cone_half_angle_rad):.1f}), |tau| "
          f"{np.linalg.norm(allocation.torque_n_m):.2e} N m, |F short| "
          f"{np.linalg.norm(allocation.force_shortfall_n):.3f} N")
    assert allocation.throttles[k] > main.deadband
    assert deflection == pytest.approx(trim, abs=5.0e-3)
    assert deflection <= math.degrees(main.cone_half_angle_rad) + 1e-9
    assert np.linalg.norm(allocation.torque_n_m) < 1.0e-2
    assert np.linalg.norm(allocation.force_shortfall_n) < 1.0
    # the engine itself points through the centre of mass
    direction = main.direction_at(*allocation.gimbal_rad[k])
    assert float(direction @ line) == pytest.approx(1.0, abs=1e-8)


def test_unreachable_request_returns_the_best_wrench_and_its_shortfall():
    centre = _cm()
    request = np.asarray((10000.0, 500.0, 0.0))
    allocation = allocate_wrench(CRAFT.thrusters, request, np.zeros(3),
                                 centre_of_mass_m=centre,
                                 tank_propellant_kg=FULL)
    k = CRAFT.thrusters_by_role("main")[0]
    forward = sum(t.max_thrust_n for t in CRAFT.thrusters
                  if t.direction[0] > 0.0)
    print(f"\nrequest {request}, achieved {np.round(allocation.force_n, 2)}, "
          f"shortfall {np.round(allocation.force_shortfall_n, 2)}, main at "
          f"{allocation.throttles[k]:.3f}")
    assert allocation.throttles[k] == pytest.approx(1.0)
    assert allocation.force_shortfall_n == pytest.approx(
        request - allocation.force_n, abs=0.0)
    assert 0.9 * 4000.0 < allocation.force_n[0] <= forward + 1e-9
    assert np.linalg.norm(allocation.force_shortfall_n) > 5000.0
    # achieved is exactly what the commanded states deliver
    force, torque = machine_wrench(CRAFT.thrusters, allocation.throttles,
                                   allocation.gimbal_rad, centre)
    assert allocation.force_n == pytest.approx(force, abs=1e-9)
    assert allocation.torque_n_m == pytest.approx(torque, abs=1e-9)


def test_slew_limits_bound_what_one_round_can_achieve():
    centre = _cm()
    k = CRAFT.thrusters_by_role("main")[0]
    main = CRAFT.thrusters[k]
    request = (3000.0, 0.0, 0.0)
    # from rest the main engine's throttle reaches 2/s * h: below its 0.4
    # deadband at h = 0.1 s, so only the RCS can push
    short = allocate_wrench(CRAFT.thrusters, request, np.zeros(3),
                            centre_of_mass_m=centre, tank_propellant_kg=FULL,
                            round_s=0.1)
    assert short.throttles[k] == main.throttle_min
    assert short.force_n[0] <= 4 * 22.0 + 1e-9
    # at h = 0.5 s it can reach 1.0
    long = allocate_wrench(CRAFT.thrusters, request, np.zeros(3),
                           centre_of_mass_m=centre, tank_propellant_kg=FULL,
                           round_s=0.5)
    assert long.throttles[k] > main.deadband
    # the gimbal: from (0, 0) it can move 10 deg/s * h per actuator
    h = 0.05
    tilted = allocate_wrench(
        CRAFT.thrusters, 3000.0 * np.asarray(
            (math.cos(0.1), 0.0, -math.sin(0.1))), np.zeros(3),
        centre_of_mass_m=centre, tank_propellant_kg=FULL, round_s=h,
        throttle_state=np.r_[0.75, np.zeros(18)])
    assert np.abs(tilted.gimbal_rad[k]).max() <= (
        main.gimbal_slew_rad_s * h + 1e-12)
    print(f"\nh=0.1: main {short.throttles[k]:.2f}, F_x {short.force_n[0]:.1f}"
          f" N; h=0.5: main {long.throttles[k]:.3f}, F_x "
          f"{long.force_n[0]:.1f} N; gimbal after 0.05 s "
          f"{np.degrees(tilted.gimbal_rad[k])} deg")


def test_centre_of_mass_shift_from_tank_drain_changes_the_torque_arms():
    full, drained = _cm(), _cm(DRAINED)
    shift = np.linalg.norm(drained - full)
    # an RCS thruster's moment per unit throttle about each centre of mass
    k = CRAFT.thrusters.index(next(t for t in CRAFT.thrusters
                                   if t.identity == "rcs.pz.fwd"))
    throttles = np.zeros(CRAFT.thruster_count)
    throttles[k] = 1.0
    _f, tau_full = machine_wrench(CRAFT.thrusters, throttles,
                                  np.zeros((CRAFT.thruster_count, 2)), full)
    _f, tau_drained = machine_wrench(CRAFT.thrusters, throttles,
                                     np.zeros((CRAFT.thruster_count, 2)),
                                     drained)
    main = CRAFT.thrusters[CRAFT.thrusters_by_role("main")[0]]
    trims = [math.degrees(math.acos((c - main.position_m)[0]
                                    / np.linalg.norm(c - main.position_m)))
             for c in (full, drained)]
    print(f"\ncm shift {shift * 1000:.1f} mm; rcs.pz.fwd torque {tau_full} -> "
          f"{tau_drained} N m; main trim {trims[0]:.3f} -> {trims[1]:.3f} deg")
    assert shift > 0.05
    assert np.linalg.norm(tau_drained - tau_full) > 1.0
    assert trims[1] - trims[0] > 2.0


# --------------------------------------------------------------- dynamics
def test_states_slew_at_their_declared_rates_and_drive_the_wrench():
    k = CRAFT.thrusters_by_role("main")[0]
    main = CRAFT.thrusters[k]
    window, rounds = 0.05, 14
    craft = MachineCraft([], CRAFT, position_m=(0.0, 0.0, 0.0),
                         velocity_m_s=(0.0, 0.0, 0.0), length_scale_m=1.0e6,
                         window_s=window)
    throttles = np.zeros(CRAFT.thruster_count)
    throttles[k] = 1.0
    angles = np.zeros((CRAFT.thruster_count, 2))
    angles[k] = (main.cone_half_angle_rad, 0.0)
    craft.throttle(throttles)
    craft.gimbal(angles)
    with pytest.raises(ValueError):
        too_far = angles.copy()
        too_far[k] = (main.cone_half_angle_rad, 0.01)
        craft.gimbal(too_far)
    worst_f = worst_t = 0.0
    lit_at = None
    for n in range(1, rounds + 1):
        before = craft.attitude()
        craft.advance()
        assert craft.substeps == n               # at rest: dt is the window
        t = n * window
        state = min(1.0, main.throttle_slew_per_s * t)
        a = min(main.cone_half_angle_rad, main.gimbal_slew_rad_s * t)
        assert craft.throttle_states()[k] == pytest.approx(state, abs=1e-12)
        assert craft.gimbal_states()[k] == pytest.approx((a, 0.0), abs=1e-12)
        delivered = state if state >= main.deadband else 0.0
        if delivered and lit_at is None:
            lit_at = t
        force_craft = main.max_thrust_n * delivered * main.direction_at(a, 0)
        expected_f = before @ force_craft
        expected_t = np.cross(np.asarray(main.position_m)
                              - craft.centre_of_mass(), force_craft)
        worst_f = max(worst_f, float(np.abs(craft.applied_force()
                                            - expected_f).max()))
        worst_t = max(worst_t, float(np.abs(craft.torque()
                                            - expected_t).max()))
    print(f"\nthrottle state 2/s, lit at {lit_at:.2f} s (deadband "
          f"{main.deadband}); gimbal 10 deg/s to "
          f"{math.degrees(main.cone_half_angle_rad):.0f} deg; worst |F - law| "
          f"{worst_f:.2e} N, |tau - law| {worst_t:.2e} N m")
    assert lit_at == pytest.approx(main.deadband / main.throttle_slew_per_s)
    assert worst_f < 1e-9 and worst_t < 1e-9


def test_a_burn_drains_each_tank_at_its_share_and_moves_the_centre_of_mass():
    k = CRAFT.thrusters_by_role("main")[0]
    main = CRAFT.thrusters[k]
    window, rounds = 1.0, 20
    craft = MachineCraft([], CRAFT, position_m=(0.0, 0.0, 0.0),
                         velocity_m_s=(0.0, 0.0, 0.0), length_scale_m=1.0e6,
                         window_s=window)
    start_cm = craft.centre_of_mass()
    throttles = np.zeros(CRAFT.thruster_count)
    throttles[k] = 1.0
    craft.throttle(throttles)
    for _ in range(rounds):
        craft.advance()
    left = craft.tank_propellant_kg()
    burned = {name: FULL[name] - left[name] for name in FULL}
    impulse = craft.thruster_impulses_n_s[k]
    total = burned["tank.mmh"] + burned["tank.nto"]
    exhaust = main.thruster_kind.exhaust_velocity_m_s
    print(f"\nburned mmh {burned['tank.mmh']:.6f} nto {burned['tank.nto']:.6f}"
          f" kg (ratio {burned['tank.nto'] / burned['tank.mmh']:.6f}); "
          f"impulse/c {impulse / exhaust:.6f} kg; cm "
          f"{np.round(start_cm, 5)} -> {np.round(craft.centre_of_mass(), 5)}")
    assert burned["tank.hydrazine"] == 0.0
    assert burned["tank.nto"] / burned["tank.mmh"] == pytest.approx(
        1.65, rel=1e-12)
    assert total == pytest.approx(impulse / exhaust, rel=1e-12)
    assert craft.mass_kg == pytest.approx(CRAFT.mass_kg - total, rel=1e-13)
    # cut off: the state spools down within the first round (2/s); after
    # a second, cold round the mass-property columns are the reduction at
    # the tanks as they now stand
    craft.throttle(np.zeros(CRAFT.thruster_count))
    craft.advance()
    tanks_now = craft.tank_propellant_kg()
    craft.advance()
    assert craft.tank_propellant_kg() == tanks_now
    rigid = CRAFT.mass_properties(tanks_now)
    assert craft.centre_of_mass() == pytest.approx(
        np.asarray(rigid.center_of_gravity), abs=1e-12)
    assert craft.inertia_tensor() == pytest.approx(
        np.asarray(rigid.inertia_tensor_kg_m2), rel=1e-11, abs=1e-10)
    assert np.linalg.norm(craft.centre_of_mass() - start_cm) > 1e-3


def test_the_allocate_seam_applies_its_commands_and_the_craft_delivers_them():
    from orbital_actuation import AchievedWrench
    window = 0.5
    craft = MachineCraft([], CRAFT, position_m=(0.0, 0.0, 0.0),
                         velocity_m_s=(0.0, 0.0, 0.0), length_scale_m=1.0e6,
                         window_s=window)
    k = CRAFT.thrusters_by_role("main")[0]
    main = CRAFT.thrusters[k]
    line = craft.centre_of_mass() - np.asarray(main.position_m)
    line /= np.linalg.norm(line)
    # the tracker's call: allocate(wrench) with .force_n / .torque_n_m
    allocation = craft.allocate(AchievedWrench(3000.0 * line, np.zeros(3)))
    assert craft.applies_allocation
    assert craft.throttles() == pytest.approx(allocation.throttles, abs=0.0)
    craft.advance()
    # the states arrived at the commands within the round
    assert craft.throttle_states() == pytest.approx(allocation.throttles,
                                                    abs=1e-12)
    assert craft.gimbal_states()[k] == pytest.approx(
        allocation.gimbal_rad[k], abs=1e-12)
    force, torque = craft.applied_force(), craft.torque()
    print(f"\nachieved |F| {np.linalg.norm(allocation.achieved.force_n):.4f}"
          f" N, delivered |F| {np.linalg.norm(force):.4f} N; delivered torque"
          f" {torque} N m (the round burned "
          f"{CRAFT.mass_kg - craft.mass_kg:.3f} kg, moving the cm)")
    assert np.linalg.norm(force) == pytest.approx(
        np.linalg.norm(allocation.achieved.force_n), rel=1e-12)
    # the torque about the centre of mass the round moved (measured
    # 4e-7 N m; the untrimmed engine would make ~120 N m)
    assert np.linalg.norm(torque) < 1.0e-3


def test_fuel_is_priced_by_propellant_flow_not_by_newtons():
    from orbital_actuation import Thruster
    # two identical 100 N thrusters on one line through the centre of mass,
    # bipropellant (310 s) and monopropellant (230 s): per newton they are
    # the same, per kilogram of propellant the bipropellant is cheaper
    pair = (Thruster("biprop", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 100.0,
                     kind="bipropellant"),
            Thruster("monoprop", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 100.0,
                     kind="monopropellant"))
    allocation = allocate_wrench(pair, (80.0, 0.0, 0.0), np.zeros(3),
                                 centre_of_mass_m=np.zeros(3),
                                 fuel_weight=1.0e-3)
    print(f"\nbiprop {allocation.throttles[0]:.6f}, monoprop "
          f"{allocation.throttles[1]:.6f}")
    # 0.799: the price itself trades a 1e-3 miss (fuel_weight 1e-3)
    assert allocation.throttles[0] == pytest.approx(0.8, abs=2e-3)
    assert allocation.throttles[1] == pytest.approx(0.0, abs=1e-6)
