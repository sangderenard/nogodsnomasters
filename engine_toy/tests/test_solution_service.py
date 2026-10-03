"""Captured requests use the firing service's existing independent lifecycle."""
import threading
import time

from solution_service import Plan, SolutionService


def wait_for(predicate):
    deadline = time.monotonic() + 3.0
    while not predicate():
        assert time.monotonic() < deadline, "worker did not publish"
        time.sleep(0.001)


def test_captured_requests_supersede_and_publish_once():
    entered, release = threading.Event(), threading.Event()
    seen = []

    def solve(payload):
        seen.append(payload)
        if payload == (1,):
            entered.set()
            assert release.wait(3.0)
        return payload[0] * 2

    service = SolutionService(solver=solve, cycle_s=0.001).start()
    try:
        first = service.submit((1,))
        assert entered.wait(3.0)
        second = service.submit((2,))
        assert second.generation > first.generation
        assert service.latest_result() is None
        release.set()
        wait_for(lambda: service.latest_result() is not None)
        done = service.latest_result()
        assert done.request is second and done.result == 4
        time.sleep(0.01)
        assert service.latest_result() is done and service.solves == 1
        assert seen == [(1,), (2,)]
    finally:
        release.set()
        service.stop()


def test_worker_failure_is_a_completed_result_and_next_request_works():
    def solve(payload):
        if payload == "bad":
            raise ValueError("captured failure")
        return payload

    service = SolutionService(solver=solve, cycle_s=0.001).start()
    try:
        failed = service.submit("bad")
        wait_for(lambda: service.latest_result() is not None)
        assert service.latest_result().request is failed
        assert service.latest_result().error == "ValueError: captured failure"
        recovered = service.submit("good")
        wait_for(lambda: service.latest_result().request is recovered)
        assert service.latest_result().result == "good"
    finally:
        service.stop()


def test_stop_suppresses_late_result_and_duplicate_worker():
    entered, release = threading.Event(), threading.Event()

    def solve(payload):
        entered.set()
        assert release.wait(3.0)
        return payload

    service = SolutionService(solver=solve).start()
    service.submit("held")
    assert entered.wait(3.0)
    worker = service._thread
    try:
        service.stop()
        service.start()
        assert service._thread is worker
    finally:
        release.set()
        worker.join(3.0)
        service.stop()
    assert service.latest_result() is None


def test_firing_api_keeps_conditions_and_plan_publication(monkeypatch):
    service = SolutionService(table=object())
    service.update(valid=True, target_identity="original")
    captured = service.conditions
    plan = Plan(0.0, captured, (1, 2, 3), 1, 2, 3, 4, 5, True)
    monkeypatch.setattr(service, "_solve", lambda conditions: plan
                        if conditions is captured else None)
    assert service.solve_once() is plan
    assert service.latest() is plan and service.solves == 1
    service.update(valid=False)
    assert service.solve_once() is None
