"""Lifecycle unit gates; analytic seam fixtures are not native game acceptance."""
import math
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import orbital_game as og


def wait_for(predicate):
    deadline = time.monotonic() + 3.0
    while not predicate():
        assert time.monotonic() < deadline, "worker did not complete"
        time.sleep(0.001)


class CircleCraft:
    time_s = 0.0
    propellant_kg = 12.0
    design = SimpleNamespace(thruster_count=1)
    thruster_impulses_n_s = np.zeros(1)

    def __init__(self):
        self.offset = np.zeros(3)

    def r(self):
        r, v = og._circular_state(og.MU_EARTH, 7e6,
                                 math.sqrt(og.MU_EARTH / 7e6**3) * self.time_s)
        return np.asarray(r) + self.offset, np.asarray(v)

    def advance(self, window):
        self.time_s += window


class CircleStations:
    time_s = 0.0

    def __init__(self, specs):
        self.specs = specs
        self.offset = np.zeros(3)

    def r(self):
        states = [og._circular_state(og.MU_EARTH, s.radius_m,
                                    s.angle_rad + math.sqrt(
                                        og.MU_EARTH / s.radius_m**3) * self.time_s)
                  for s in self.specs]
        return (np.asarray([s[0] for s in states]) + self.offset,
                np.asarray([s[1] for s in states]))

    def advance(self, window):
        self.time_s += window


class CapturedPlan:
    t_start = 20.0
    t_arrive = 100.0
    transfer_time = 80.0

    def reference(self, t):
        return tuple(np.asarray(v) for v in og._circular_state(
            og.MU_EARTH, 7e6, math.sqrt(og.MU_EARTH / 7e6**3) * t))

    def impulses(self):
        return (SimpleNamespace(time_s=self.t_start),
                SimpleNamespace(time_s=self.t_arrive))


@pytest.fixture
def make_game(monkeypatch):
    games = []

    def make(solve):
        monkeypatch.setattr(og, "_solve_planning", solve)
        game = object.__new__(og.OrbitalGame)
        game.mu, game.planner_name = og.MU_EARTH, "hohmann"
        game.planner = object()
        game.gains, game.round_s = og.MACHINE_GAINS, 2.0
        game.craft = CircleCraft()
        game.specs = (og.TargetSpec("one", 9e6, 0.5),
                      og.TargetSpec("two", 13e6, 1.0))
        game.stations = CircleStations(game.specs)
        game.order = game.report = None
        game.rendezvous, game.path = [], []
        game.max_thrust = np.ones(1)
        game.plume_throttles = np.zeros(1)
        game._init_planning()
        game._planning_service.cycle_s = 0.001
        game._planning_service.start()
        games.append(game)
        return game

    yield make
    for game in games:
        game.close()


def initial_order(request):
    return og.TransferOrder(request.target, request.time_s, CapturedPlan(),
                            og.Phasing(0.0, 20.0, 0.0), 0.0)


def test_select_and_step_keep_running_while_solver_is_held(make_game):
    entered, release = threading.Event(), threading.Event()

    def solve(request):
        entered.set()
        assert release.wait(3.0)
        return initial_order(request)

    game = make_game(solve)
    try:
        captured = game.select(0)
        assert entered.wait(3.0)
        assert game.order is None
        assert game.step(2.0) == 2.0
        assert game.time_s == game.stations.time_s == 2.0
        assert captured.payload.time_s == 0.0
        release.set()
        wait_for(lambda: game._planning_service.latest_result() is not None)
        assert game.poll_planning() is game.order
        assert game.order.target == 0
        assert game.poll_planning() is None
        assert len(game.announcements()) == 1
        assert game.announcements() == ()
        with pytest.raises(RuntimeError, match="in progress"):
            game.select(1)
    finally:
        release.set()


def test_pending_target_replacement_discards_old_solve(make_game):
    entered, release = threading.Event(), threading.Event()

    def solve(request):
        if request.target == 0:
            entered.set()
            assert release.wait(3.0)
        return initial_order(request)

    game = make_game(solve)
    try:
        game.select(0)
        assert entered.wait(3.0)
        selected = game.select(1)
        release.set()
        wait_for(lambda: game._planning_service.latest_result() is not None)
        assert game._planning_service.latest_result().request is selected
        game.poll_planning()
        assert game.order.target == 1
        assert len(game.announcements()) == 1
    finally:
        release.set()


def test_failed_initial_request_is_announced_once_and_headless_run_exits(make_game):
    def solve(request):
        raise ValueError("no solution")

    game = make_game(solve)
    game.select(0)
    wait_for(lambda: game._planning_service.latest_result() is not None)
    with pytest.raises(RuntimeError, match="no solution"):
        game.run_until_arrival(2.0)
    assert game.order is None
    assert game.announcements() == ("planning failed: ValueError: no solution",)
    assert game.poll_planning() is None and game.announcements() == ()


def test_epoch_change_discards_initial_completion(make_game):
    game = make_game(initial_order)
    game.craft.time_s = game.stations.time_s = 5.0
    game.select(0)
    wait_for(lambda: game._planning_service.latest_result() is not None)
    game.craft.time_s = game.stations.time_s = 0.0
    assert game.poll_planning() is None and game.order is None
    assert game.planning_error == "request changed"


@pytest.mark.parametrize("change, reason", [
    ("target", "target moved"), ("craft", "ON PLAN band"),
    ("expired", "first burn passed")])
def test_stale_initial_result_recaptures_visible_reason(make_game, change, reason):
    game = make_game(initial_order)
    request = game.select(0)
    wait_for(lambda: game._planning_service.latest_result() is not None)
    if change == "target":
        game.stations.offset[0] = 2e6
    elif change == "craft":
        game.craft.offset[0] = 2e6
    else:
        game.craft.time_s = game.stations.time_s = 21.0
    assert game.poll_planning() is None and game.order is None
    assert game._pending_plan.generation > request.generation
    assert reason in game.announcements()[0]


def test_immediate_start_replan_accepts_delayed_unchanged_assumptions(make_game):
    game = make_game(lambda request: CapturedPlan())
    previous = CapturedPlan()
    game.order = og.TransferOrder(0, 0.0, previous, og.Phasing(0, 0, 0), 0.0)
    game._request_replan(previous)
    request = game._pending_plan
    # Craft can be ahead of stations inside an existing frame.
    game.craft.time_s = 2.0
    wait_for(lambda: game._planning_service.latest_result() is not None)
    candidate = game._planning_service.latest_result().result
    candidate.t_start = 0.0
    assert game._poll_replan(previous) is candidate
    assert request.payload.target_time_s == 0.0
    assert len(game.announcements()) == 1


def test_disturbance_during_replan_invalidates_captured_assumptions(make_game):
    game = make_game(lambda request: CapturedPlan())
    previous = CapturedPlan()
    game.order = og.TransferOrder(0, 0.0, previous, og.Phasing(0, 0, 0), 0.0)
    game._request_replan(previous)
    request = game._pending_plan
    game.craft.offset[0] = 2e6
    wait_for(lambda: game._planning_service.latest_result() is not None)
    assert game._poll_replan(previous) is None
    assert game._pending_plan.generation > request.generation
    assert "ON PLAN band" in game.announcements()[0]


def test_changed_previous_plan_and_stop_suppress_publication(make_game):
    game = make_game(lambda request: CapturedPlan())
    previous = CapturedPlan()
    game.order = og.TransferOrder(0, 0.0, previous, og.Phasing(0, 0, 0), 0.0)
    game._request_replan(previous)
    wait_for(lambda: game._planning_service.latest_result() is not None)
    assert game._poll_replan(CapturedPlan()) is None
    assert "flown plan changed" in game.announcements()[0]
    game.close()
    with pytest.raises(RuntimeError, match="in progress"):
        game.select(0)


def test_failed_replan_keeps_previous_plan_without_automatic_error_loop(make_game):
    def fail(request):
        raise RuntimeError("replan failure")

    game = make_game(fail)
    previous = CapturedPlan()
    game.order = og.TransferOrder(0, 0.0, previous, og.Phasing(0, 0, 0), 0.0)
    game._request_replan(previous)
    wait_for(lambda: game._planning_service.latest_result() is not None)
    assert game._poll_replan(previous) is None
    assert game.flown_plan() is previous
    assert game._pending_plan is None
    assert game._poll_replan(previous) is None
    assert game._planning_service.solves == 1
    assert game.announcements() == (
        "re-plan failed: RuntimeError: replan failure; continuing previous plan",)


def test_cancelled_initial_request_terminates_headless_helper(make_game):
    game = make_game(initial_order)
    game.select(0)
    game.close()
    with pytest.raises(RuntimeError, match="cancelled"):
        game.run_until_arrival(2.0)


def test_candidate_declared_fuel_use_is_checked_without_a_tolerance(make_game):
    game = make_game(initial_order)
    request = game.select(0).payload
    request = replace(request, problem=SimpleNamespace(burns_propellant=True))
    candidate = CapturedPlan()
    candidate.fuel = 12.0
    assert game._fresh_plan(request, candidate) is None
    candidate.fuel = np.nextafter(12.0, math.inf)
    assert "propellant" in game._fresh_plan(request, candidate)


@pytest.mark.parametrize("new_arrival, expected_advanced, arrived", [
    (25.0, 325.0, True),
    (1300.0, 600.0, False),
])
def test_mid_frame_replan_respects_arrival_and_frame_end(
        make_game, monkeypatch, new_arrival, expected_advanced, arrived):
    """Real tracker loop/service publication; analytic craft, no native build."""
    import orbital_tracker as ot

    candidate = CapturedPlan()
    candidate.t_start, candidate.t_arrive = 0.0, new_arrival
    game = make_game(lambda request: candidate)
    previous = CapturedPlan()
    previous.t_start, previous.t_arrive = 0.0, 1000.0
    game.order = og.TransferOrder(0, 0.0, previous, og.Phasing(0, 0, 0), 0.0)
    craft = game.craft
    craft.mass_kg, craft.fuel_impulse_n_s = 1.0, 0.0
    craft.applied_delta_v_m_s = np.zeros(3)
    craft.applies_allocation = False
    craft.throttle = lambda values: None
    windows = []

    def advance(window):
        windows.append((craft.time_s, window))
        craft.time_s += window
        return window, window, None

    craft.advance = advance
    # Keep the real tracker round scheduling, adoption, and frame boundary.
    # Only force allocation is outside this lifecycle fixture's scope.
    monkeypatch.setattr(ot, "Actuation", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(ot, "_arm_burn", lambda *args: None)
    monkeypatch.setattr(ot, "_coast", lambda *args: (None, None, None))
    monkeypatch.setattr(ot, "_command", lambda *args: SimpleNamespace(
        phase="coast", throttles=np.zeros(1), gimbal_rad=None,
        wheel_torque_n_m=None, position_error_m=np.zeros(3),
        force_shortfall_n=np.zeros(3)))
    poll = game.replanner.poll
    monkeypatch.setattr(game.replanner, "poll", lambda plan, t, r, v:
                        None if t < 2.0 else poll(plan, t, r, v))
    game._request_replan(previous)
    wait_for(lambda: game._planning_service.latest_result() is not None)

    advanced = game.step(600.0)

    assert advanced == expected_advanced
    assert game.time_s == game.stations.time_s == expected_advanced
    assert len(game.rendezvous) == int(arrived)
    if arrived:
        assert game.rendezvous[0].time_s == candidate.t_arrive + og.ARRIVAL_SETTLE_S
    assert game.flown_plan() is candidate and game.replans() == 1
    assert windows[-1][0] + windows[-1][1] == expected_advanced
    assert game.announcements() == ("OFF PLAN: completed re-plan accepted",)


def test_replan_whose_arrival_window_has_passed_is_recaptured(make_game):
    candidate = CapturedPlan()
    candidate.t_start, candidate.t_arrive = -1000.0, -og.ARRIVAL_SETTLE_S
    game = make_game(lambda request: candidate)
    previous = CapturedPlan()
    game.order = og.TransferOrder(0, 0.0, previous, og.Phasing(0, 0, 0), 0.0)
    game._request_replan(previous)
    request = game._pending_plan
    wait_for(lambda: game._planning_service.latest_result() is not None)

    assert game._poll_replan(previous) is None
    assert game._pending_plan.generation > request.generation
    assert game.flown_plan() is previous
    assert "arrival passed during solve" in game.announcements()[0]


def test_failed_request_after_completed_order_terminates_headless_helper(make_game,
                                                                       monkeypatch):
    def fail(request):
        raise ValueError("replacement failed")

    game = make_game(fail)
    previous = CapturedPlan()
    game.order = og.TransferOrder(0, 0.0, previous, og.Phasing(0, 0, 0), 0.0)
    game.craft.time_s = game.stations.time_s = 500.0
    game.rendezvous.append(object())
    game.select(1)
    wait_for(lambda: game._planning_service.latest_result() is not None)
    # This unit gate focuses on the helper's termination after publication.
    # Native old-plan flight is not exercised by the analytic seam fixture.
    monkeypatch.setattr(game, "step", lambda window: game.poll_planning())
    with pytest.raises(RuntimeError, match="replacement failed"):
        game.run_until_arrival(2.0)
    assert len(game.rendezvous) == 1 and game.order.plan is previous


def test_main_exception_releases_worker_text_and_pygame(monkeypatch):
    closed = []

    class Game:
        specs = (og.TargetSpec("probe", 9e6, 0.0),)

        def __init__(self, **kwargs):
            pass

        def frame_window(self, window):
            return window

        def step(self, window):
            raise RuntimeError("frame failure")

        def close(self):
            closed.append("worker")

    class Text:
        def __init__(self, *args):
            pass

        def close(self):
            closed.append("text")

    pygame = SimpleNamespace(
        init=lambda: None, quit=lambda: closed.append("pygame"),
        OPENGL=1, DOUBLEBUF=2, HIDDEN=4,
        display=SimpleNamespace(set_mode=lambda *args: None,
                                set_caption=lambda *args: None),
        font=SimpleNamespace(SysFont=lambda *args: None),
        time=SimpleNamespace(Clock=lambda: None),
        event=SimpleNamespace(get=lambda: []))
    gl = SimpleNamespace(GL_COLOR_BUFFER_BIT=0, GL_RGB=0, GL_UNSIGNED_BYTE=0,
                         glClear=None, glClearColor=None, glReadPixels=None)
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=gl))
    monkeypatch.setitem(sys.modules, "OpenGL.GL", gl)
    monkeypatch.setitem(sys.modules, "gl_text", SimpleNamespace(TextLayer=Text))
    monkeypatch.setattr(og, "OrbitalGame", Game)
    with pytest.raises(RuntimeError, match="frame failure"):
        og.main(frames=1)
    assert closed == ["worker", "text", "pygame"]
