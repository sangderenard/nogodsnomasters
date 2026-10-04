"""Transcriptions own execution arenas; warm-start capture uses only NumPy."""
from types import SimpleNamespace

import numpy as np

import orbital_collocation as oc
from orbital_actuation import six_axis_jumper
from orbital_jumper import GravityCenter


def transcription():
    problem = oc.CollocationProblem(
        six_axis_jumper(1e5, 1000.0),
        (GravityCenter((0, 0, 0), 3.986004418e14),), 8e6,
        fuel_budget=2e6, slices=3)
    return oc._Transcription(problem, 0.0, (7e6, 0, 0), (0, 7500, 0), 0.0)


def test_all_four_row_families_drop_cached_execution_in_each_transcription(monkeypatch):
    owners = []
    for factory in ("_slice_rows", "_arrival_rows", "_fuel_rows", "_cost_rows"):
        rows = oc.ReverseRows(factory, object(), {}, (), (), {})
        rows.__dict__["_execution"] = object()
        owners.append(rows)
        monkeypatch.setattr(oc, factory, lambda *args, rows=rows: rows)
    a, b = transcription(), transcription()
    for name, original in zip(("slice_rows", "arrival_rows", "fuel_rows", "cost_rows"),
                              owners):
        first, second = getattr(a, name), getattr(b, name)
        assert first is not second and first is not original
        assert first.artifact is second.artifact is original.artifact
        assert "_execution" not in first.__dict__
        assert "_execution" not in second.__dict__
        assert getattr(a, name) is first


def test_remainder_capture_copies_all_read_arrays_and_transcription():
    tr = transcription()
    plan = oc.CollocationPlan(
        mu=3.986004418e14, target_radius_m=8e6,
        times=np.arange(4.0), positions=np.tile((7e6, 0, 0), (4, 1)),
        velocities=np.tile((0, 7500, 0), (4, 1)),
        propellant_kg=np.zeros(4), throttles=np.zeros((3, 6)),
        fuel=0, cost=0, max_defect=0, converged=True, message="capture",
        iterations=0, evaluations=0, jacobian_s=0, solve_s=0, transcription=tr)
    captured = oc.capture_remainder(plan)
    assert captured is not plan and captured.transcription is not tr
    for name in ("times", "positions", "velocities", "propellant_kg", "throttles"):
        old, copied = getattr(plan, name), getattr(captured, name)
        assert not np.shares_memory(old, copied)
        assert not copied.flags.writeable
        before = copied.copy()
        old[...] += 1
        assert np.array_equal(copied, before)
    assert captured._coast is not plan._coast
    assert captured.transcription.problem is tr.problem
