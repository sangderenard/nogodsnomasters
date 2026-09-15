from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine_cycle_sim import (  # noqa: E402
    DYNO_PULL_SETTLE_S,
    DYNO_PULL_SHIFT_RPM_FRAC,
    EngineCycleSim,
)
from engines import get  # noqa: E402


def test_wot_pull_traverses_every_real_gear_before_final_redline():
    sim = EngineCycleSim(get("ldt465-multifuel-deuce"))
    top_gear = len(sim.engine.transmission.gear_ratios)
    sim.start_wot_dyno_pull()
    sim._dyno_pull_timer_s = DYNO_PULL_SETTLE_S
    sim._update_dyno_pull(0.0)
    assert sim._quick_shift_target_gear == 1

    for gear in range(1, top_gear + 1):
        sim.gear_index = gear
        sim._quick_shift_state = None
        sim._update_dyno_pull(0.0)
        assert sim._dyno_pull_state == "pulling"
        assert sim.throttle == 1.0
        target = 0.98 if gear == top_gear else DYNO_PULL_SHIFT_RPM_FRAC
        sim.state.rpm = sim.engine.redline_rpm * target
        sim._update_dyno_pull(0.0)
        if gear < top_gear:
            assert sim._dyno_pull_state == "shift"
            assert sim._quick_shift_target_gear == gear + 1
            assert sim.throttle == 0.15

    assert sim._dyno_pull_state is None
    assert sim.gear_index == 0
    assert sim.throttle == 0.0
