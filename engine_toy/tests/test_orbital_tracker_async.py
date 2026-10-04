"""Async publication uses the tracker's existing round boundary/reset fields."""
from types import SimpleNamespace

import numpy as np

import orbital_tracker as ot


def test_pending_replan_keeps_old_plan_and_only_acceptance_counts(monkeypatch):
    old, replacement = object(), object()
    position, velocity = np.array((1., 0, 0)), np.array((0, 1., 0))
    craft = SimpleNamespace(r=lambda: (position, velocity), mass_kg=1.0,
                            design=object())
    mode = ot.TrackingMode(old, origin=old, anchor_s=0.0)
    actuation = SimpleNamespace(round_s=2.0)
    monkeypatch.setattr(ot, "plan_reference", lambda plan, t: (position, velocity))
    error = [0.1]
    monkeypatch.setattr(ot, "tracking_error", lambda *args, **kwargs: error[0])
    monkeypatch.setattr(ot, "_arm_burn", lambda *args: None)
    monkeypatch.setattr(ot, "_coast", lambda *args: (None, None, None))
    monkeypatch.setattr(ot, "_command", lambda *args: SimpleNamespace(phase="coast"))

    class Replanner:
        ready = None
        requested = 0

        def poll(self, *args):
            return self.ready

        def __call__(self, *args):
            self.requested += 1
            return None

    replanner = Replanner()
    ot._decide_round(craft, mode, ot.TrackingGains(), actuation, 0., 2., replanner)
    assert replanner.requested == 1
    assert mode.plan is old and not mode.on_plan and mode.replans == 0
    error[0] = 0.02
    ot._decide_round(craft, mode, ot.TrackingGains(), actuation, 2., 2., replanner)
    assert replanner.requested == 1 and mode.plan is old and mode.replans == 0
    replanner.ready, error[0] = replacement, 0.0
    ot._decide_round(craft, mode, ot.TrackingGains(), actuation, 4., 2., replanner)
    assert mode.plan is replacement and mode.replans == 1 and mode.on_plan
    assert mode.origin is old
