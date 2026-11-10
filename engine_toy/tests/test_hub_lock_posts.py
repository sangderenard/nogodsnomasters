import numpy as np
import pytest
from machines import get, roster, hub_lock_posts
from assembly_ports import PartPort, mate_ports
from engine_mesh import build_engine_mesh


def test_posts_are_registered_real_machine_parts_and_mesh():
    assert 'hub-lock-posts' in roster()
    machine = get('hub-lock-posts')
    graph = machine.build_graph()
    assert len(graph['nodes']) == 20
    assert len(graph['edges']) == 16
    assert machine.hpu is None and machine.air is None
    assert not machine.moving_parts
    assert all(n['mass_kg'] > 0 and n['in_view'] for n in graph['nodes'])
    static, moving = build_engine_mesh(graph)
    assert len(static.vertices) > 0
    assert np.isfinite(static.vertices).all()
    assert len(moving.vertices) == 0
    assert len({n['identity'] for n in graph['nodes']}) == 20
    assert all(e['a'] in {n['identity'] for n in graph['nodes']} and
               e['b'] in {n['identity'] for n in graph['nodes']} for e in graph['edges'])


def test_real_mount_ports_mate_without_moving_or_preloading_hubs():
    machine = get('hub-lock-posts')
    ports = [port for part in machine.parts for port in part.ports]
    assert len(ports) == 8
    assert len(mate_ports(ports).open_ports) == 8
    targets = [PartPort('target.' + p.identity, 'target.' + p.part,
               p.kind, p.position.copy(), -p.direction, p.radius_m, True,
               fluid='', joint='bolted-flange') for p in ports]
    before = [p.position.copy() for p in ports + targets]
    result = mate_ports(ports + targets)
    assert len(result.seals) == 8 and not result.open_ports
    for p, original in zip(ports + targets, before):
        np.testing.assert_array_equal(p.position, original)
    assert all(p.joint == 'bolted-flange' for p in ports)


def test_authored_hub_positions_and_bad_layouts():
    positions = {'fl': (-1, .8, 2), 'fr': (1, .8, 2),
                 'rl': (-1, .7, -2), 'rr': (1, .7, -2)}
    machine = hub_lock_posts(positions)
    ports = [p for part in machine.parts for p in part.ports if p.role == 'hub-lock']
    for port, corner in zip(ports, ('fl', 'fr', 'rl', 'rr')):
        np.testing.assert_array_equal(port.position, positions[corner])
    with pytest.raises(ValueError):
        hub_lock_posts({})
    positions['fl'] = (float('nan'), .8, 2)
    with pytest.raises(ValueError):
        hub_lock_posts(positions)
