import numpy as np

from hcu import (ACTIVE_LEVEL, FAULT, FORCE_PRELOAD, READY_TO_SHIP,
                 EnginePlatformAuthority, HCUBackupHydraulics,
                 HydraulicControlUnit, HydraulicSwitchboard, INNER,
                 NitrogenBottlePair, OUTER, URGENT,
                 StructureSensorFrame, emit_hcu)


def sensors(*, y=None, clearance=None, contact=None, force=None,
            extension=1.8, angle=34.0, unexpected=(0.0, 0.0, 0.0),
            supply=21.0e6, required=17.0e6):
    y = y or {name: 0.0 for name in INNER}
    positions = {
        "inner.fr": (1.7, y["inner.fr"], 1.7),
        "inner.fl": (-1.7, y["inner.fl"], 1.7),
        "inner.rl": (-1.7, y["inner.rl"], -1.7),
        "inner.rr": (1.7, y["inner.rr"], -1.7),
        "outer.a": (4.0, 0.0, 1.7), "outer.b": (4.0, 0.0, -1.7),
        "outer.c": (-4.0, 0.0, 1.7), "outer.d": (-4.0, 0.0, -1.7),
    }
    return StructureSensorFrame(
        positions,
        pillar_extensions_m={name: extension for name in INNER},
        corner_extensions_m={name: 0.5 for name in OUTER},
        corner_angles_deg={name: angle for name in OUTER},
        pad_clearance_m=clearance or {name: 0.0 for name in OUTER},
        pad_contact=contact or {name: True for name in OUTER},
        support_force_n=force or {name: 60_000.0 for name in OUTER},
        length_locked={name: True for name in (*INNER, *OUTER)},
        hydraulic_supply_pressure_pa=supply,
        hydraulic_required_pressure_pa=required,
        unexpected_force_n=unexpected)


def test_active_level_uses_structure_positions_to_correct_tilt():
    hcu = HydraulicControlUnit()
    hcu.active_level(target_deck_y_m=0.0)
    frame = sensors(y={"inner.fr": -0.02, "inner.fl": -0.02,
                       "inner.rl": 0.02, "inner.rr": 0.02})
    command = hcu.step(0.02, frame)
    assert command.mode == ACTIVE_LEVEL
    assert command.pillar_valves["inner.fr"] > 0.0
    assert command.pillar_valves["inner.fl"] > 0.0
    assert command.pillar_valves["inner.rl"] < 0.0
    assert command.pillar_valves["inner.rr"] < 0.0


def test_directional_preload_keeps_every_corner_engaged():
    hcu = HydraulicControlUnit()
    hcu.preload_for_force((0.0, 0.0, 400_000.0), base_preload_n=80_000.0,
                          target_deck_y_m=0.0)
    command = hcu.step(0.02, sensors())
    assert command.mode == FORCE_PRELOAD
    assert min(command.corner_preload_target_n.values()) >= 40_000.0
    forward = np.mean([command.corner_preload_target_n[n]
                       for n in ("outer.a", "outer.c")])
    rear = np.mean([command.corner_preload_target_n[n]
                    for n in ("outer.b", "outer.d")])
    assert forward > rear


def test_corner_walk_unlocks_only_one_and_holds_ten_mm_hover():
    hcu = HydraulicControlUnit()
    hcu.active_level(target_deck_y_m=0.0)
    hcu.walk_corners(deploy=False)
    command = hcu.step(0.02, sensors(
        clearance={name: (0.010 if name == "outer.a" else 0.0)
                   for name in OUTER}))
    assert command.moving_corner == "outer.a"
    assert command.length_locks["outer.a"] is False
    assert all(command.length_locks[n] for n in OUTER if n != "outer.a")
    assert command.corner_swing_valves["outer.a"] < 0.0


def test_unexpected_walk_force_seats_then_locks_the_moving_pad():
    hcu = HydraulicControlUnit()
    hcu.active_level(target_deck_y_m=0.0)
    hcu.walk_corners(deploy=False)
    first = hcu.step(0.02, sensors(
        clearance={name: (0.010 if name == "outer.a" else 0.0)
                   for name in OUTER}, unexpected=(8_000.0, 0.0, 0.0),
        contact={name: False for name in OUTER}))
    assert first.corner_telescope_valves["outer.a"] == 1.0
    assert first.length_locks["outer.a"] is False
    seated = hcu.step(0.02, sensors(unexpected=(8_000.0, 0.0, 0.0)))
    assert seated.mode == FAULT
    assert seated.length_locks["outer.a"] is True


def test_ready_to_ship_waits_for_real_pressure_margin():
    hcu = HydraulicControlUnit()
    hcu.ready_to_ship(1.8)
    command = hcu.step(0.02, sensors(extension=0.2,
                                    supply=21.0e6, required=21.18e6))
    assert command.mode == READY_TO_SHIP
    assert command.phase == "waiting-for-plant"
    assert not any(command.pillar_valves.values())


def test_ready_to_ship_uses_powerplant_flow_schedule_not_full_spool():
    hcu = HydraulicControlUnit()
    hcu.ready_to_ship(1.8)
    command = hcu.step(0.02, sensors(extension=0.2))
    assert set(command.pillar_valves.values()) == {0.25}


def test_urgent_starts_both_platforms_opens_every_lift_valve_and_backup():
    hcu = HydraulicControlUnit()
    hcu.urgent()
    command = hcu.step(0.02, sensors())

    assert command.mode == URGENT
    assert command.engine_start_request == {"low": True, "high": True}
    assert set(command.pillar_valves.values()) == {1.0}
    assert set(command.corner_telescope_valves.values()) == {1.0}
    assert not any(command.length_locks.values())
    assert command.backup_pump_command == 1.0


def test_platform_ignition_starts_but_rail_off_never_kills():
    authority = EnginePlatformAuthority(base_retry_s=2.0)
    first = authority.step(.1, running=False, starter_engaged=False,
                           ignition_live=True, kill_signal_sum=0.0,
                           kill_consensus_required=3.0)
    assert first.start and not first.kill
    no_rails = authority.step(.1, running=True, starter_engaged=False,
                              ignition_live=False, kill_signal_sum=0.0,
                              kill_consensus_required=3.0)
    assert not no_rails.start and not no_rails.kill


def test_active_engine_kill_requires_its_platform_rail_consensus_sum():
    authority = EnginePlatformAuthority()
    one = authority.step(.1, running=True, starter_engaged=False,
                         ignition_live=False, kill_signal_sum=2.0,
                         kill_consensus_required=3.0)
    both = authority.step(.1, running=True, starter_engaged=False,
                          ignition_live=False, kill_signal_sum=3.0,
                          kill_consensus_required=3.0)
    assert not one.kill
    assert both.kill


def test_failed_start_retry_delay_increases_and_waits_after_cranking():
    authority = EnginePlatformAuthority(base_retry_s=2.0)
    first = authority.step(.1, running=False, starter_engaged=False,
                           ignition_live=True, kill_signal_sum=0.0,
                           kill_consensus_required=3.0)
    assert first.start and first.retry_delay_s == 2.0
    cranking = authority.step(10.0, running=False, starter_engaged=True,
                              ignition_live=True, kill_signal_sum=0.0,
                              kill_consensus_required=3.0)
    assert cranking.retry_delay_s == 2.0
    waiting = authority.step(1.0, running=False, starter_engaged=False,
                             ignition_live=True, kill_signal_sum=0.0,
                             kill_consensus_required=3.0)
    second = authority.step(1.0, running=False, starter_engaged=False,
                            ignition_live=True, kill_signal_sum=0.0,
                            kill_consensus_required=3.0)
    assert not waiting.start
    assert second.start and second.retry_delay_s == 4.0


def test_hcu_graph_has_large_headers_backup_pumps_and_four_authority_rails():
    import stand
    from turret_production import ProductionGraph
    graph = ProductionGraph("hcu-witness")
    spec = stand.Stand()
    site = stand.emit_stand(graph, spec)
    # Header endpoints are the same real powerplant machine identities used
    # by station_reference; graph construction allows later node definition.
    for side in ("port", "starboard"):
        graph.node(f"engine.{side}", (-3.0 if side == "port" else 3.0, 0, 0),
                   "load-bearing-structure")
        graph.node(f"powerplant.{side}.hydraulic",
                   (-3.0 if side == "port" else 3.0, -.2, .4),
                   "shaft-hydraulic-gear-pump")
    graph.node("turret.accumulator", (0.6, 0.4, 0.5),
               "high-pressure-canister")
    graph.node("mount.absorber.reaction.pneumatic", (0.0, 1.0, 0.0),
               "load-bearing-structure")
    graph.node("turret.elevation_anchor", (0.2, 1.0, 0.0),
               "load-bearing-structure")
    made = emit_hcu(graph, spec, site)
    document = graph.as_document()
    nodes = {node["identity"]: node for node in document["nodes"]}
    edges = {edge["identity"]: edge for edge in document["edges"]}
    assert nodes[made["manifold"]]["pressure_header_nominal_id_m"] == .050
    assert nodes[made["emergency_hand_pump"]]["average_flow_l_min"] == .016
    assert nodes[made["pressure_booster"]]["pressure_ratio"] == 1.6
    assert nodes[made["backup_battery"]]["usable_energy_kwh"] == 12.0
    assert nodes[made["backup_electric_pump"]]["rated_flow_l_min"] == 7.2
    assert nodes[made["backup_reservoir"]]["oil_volume_l"] == 180.0
    assert edges["stand.hcu.header.return.port"]["nominal_id_m"] == .065
    assert len([name for name in edges if name.startswith("stand.hcu.ignition_")
                or name.startswith("stand.hcu.kill_")]) == 4


def test_backup_electric_pump_delivers_slow_pressure_flow_from_own_battery():
    backup = HCUBackupHydraulics()
    reading = backup.step(60.0, command=1.0, demand_l_min=20.0,
                          required_pressure_pa=8.9e6)
    assert reading["available"]
    assert reading["delivered_l_min"] == 7.2
    assert reading["battery_soc"] < 1.0
    assert backup.usable_battery_kwh == 12.0


def test_nitrogen_pair_tops_receiver_without_mixing_oil_circuits():
    bank = NitrogenBottlePair()
    reading = bank.top_off(60.0, receiver_pressure_pa=8.0e6,
                           receiver_gas_volume_l=40.0,
                           target_pressure_pa=12.0e6)
    assert 8.0e6 < reading["receiver_pressure_pa"] <= 12.0e6
    assert reading["bottle_pressure_pa"] < 30.0e6


def test_solenoid_switchboard_combines_sources_by_pressure_and_priority():
    board = HydraulicSwitchboard()
    result = board.dispatch(
        sources={
            "low": {"available_flow_l_min": 55.0, "pressure_pa": 9.0e6},
            "high": {"available_flow_l_min": 120.0, "pressure_pa": 20.0e6},
            "backup": {"available_flow_l_min": 7.2, "pressure_pa": 21.0e6}},
        draws={
            "support": {"requested_flow_l_min": 100.0,
                        "required_pressure_pa": 8.5e6, "priority": 1},
            "turret": {"requested_flow_l_min": 20.0,
                       "required_pressure_pa": 15.0e6, "priority": 2}})
    matrix = result["contributions_l_min"]
    assert matrix["turret"]["low"] == 0.0
    assert sum(matrix["turret"].values()) == 20.0
    assert sum(matrix["support"].values()) == 100.0
