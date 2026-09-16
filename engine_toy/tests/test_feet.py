"""What a machine stands on, and why it is not a vehicle.

The claims under test are the ones the module exists for: a levelling
screw does not equalise, three feet cannot get it wrong, four can, and a
foot that is not touching is a different problem from a pad that is
sinking.
"""
import math

import numpy as np
import pytest

import feet
from engine_mass_properties import RigidBodyProperties


def body(mass=800.0, cg=(0.0, 0.0, 0.0), i=None):
    if i is None:
        i = np.diag([60.0, 40.0, 50.0])
    return RigidBodyProperties(mass, tuple(cg), np.asarray(i, float))


def square(pad=0.16, y=0.0):
    return [feet.LevellingFoot(f"f.{n}", p, pad_diameter_m=pad)
            for n, p in (("fl", (-0.6, y, -0.5)), ("fr", (-0.6, y, 0.5)),
                         ("rl", (0.6, y, -0.5)), ("rr", (0.6, y, 0.5)))]


def tripod(pad=0.16):
    return [feet.LevellingFoot(f"f.{n}", p, pad_diameter_m=pad)
            for n, p in (("a", (-0.6, 0.0, -0.5)), ("b", (-0.6, 0.0, 0.5)),
                         ("c", (0.7, 0.0, 0.0)))]


LEVEL = dict(frame_underside_y=0.060)


# ---------------------------------------------------------------------
# the foot itself
# ---------------------------------------------------------------------

def test_the_compliance_is_the_grounds_and_not_the_screws():
    """The screw is two orders of magnitude stiffer than the dirt, so it
    contributes nothing to how far the foot moves."""
    g = feet.ground("firm-soil")
    f = feet.LevellingFoot("f", (0, 0, 0))
    assert f.screw_stiffness_n_per_m() > 100 * f.ground_stiffness_n_per_m(g)
    assert f.stiffness_n_per_m(g) == pytest.approx(
        f.ground_stiffness_n_per_m(g), rel=0.02)


def test_on_a_slab_the_screw_finally_matters():
    """Which is the honest reason concrete is in the table at all."""
    slab = feet.ground("concrete-slab")
    f = feet.LevellingFoot("f", (0, 0, 0))
    assert f.screw_stiffness_n_per_m() < f.ground_stiffness_n_per_m(slab)


def test_a_setting_is_stated_in_turns_because_that_is_how_it_is_set():
    f = feet.LevellingFoot("f", (0, 0, 0), thread_pitch_m=0.0025)
    assert f.turns_for(0.0025) == pytest.approx(1.0)
    assert f.turns_for(0.010) == pytest.approx(4.0)


# ---------------------------------------------------------------------
# three versus four
# ---------------------------------------------------------------------

def test_three_feet_are_determinate_and_stiffness_cannot_change_the_share():
    """The whole reason precision machines sit on three. Solve the same
    tripod on clay and on concrete -- six thousand times the modulus --
    and the load split must not move."""
    rb = body(cg=(0.1, 0.0, 0.05))
    soft = feet.stand(tripod(), rb, feet.ground("soft-clay"), **LEVEL)
    hard = feet.stand(tripod(), rb, feet.ground("concrete-slab"), **LEVEL)
    assert soft.determinate and hard.determinate
    for k in soft.forces:
        assert soft.forces[k] == pytest.approx(hard.forces[k], rel=1e-6)


def test_four_feet_are_indeterminate_and_the_screw_decides():
    """Same frame, same floor, same everything but a fifth of a turn on
    one foot -- and the share moves. That is the redundancy."""
    rb = body()
    g = feet.ground("firm-soil")
    even = feet.stand(square(), rb, g, **LEVEL)
    assert not even.determinate
    for f in even.forces.values():
        assert f == pytest.approx(even.forces["f.fl"], rel=0.05)

    tweaked = square()
    tweaked[0].screw_out_m -= 0.0005
    off = feet.stand(tweaked, rb, g, **LEVEL)
    assert off.forces["f.fl"] < even.forces["f.fl"] * 0.95
    # and it goes to the DIAGONAL, not to the neighbours
    assert off.forces["f.rr"] < off.forces["f.fr"]


def test_the_load_always_adds_up_to_the_weight():
    rb = body(mass=841.0)
    for fs in (square(), tripod()):
        s = feet.stand(fs, rb, feet.ground("firm-soil"), **LEVEL)
        assert sum(s.forces.values()) == pytest.approx(
            841.0 * feet.GRAVITY_M_S2, rel=1e-6)


# ---------------------------------------------------------------------
# unilateral contact
# ---------------------------------------------------------------------

def test_a_foot_cannot_pull():
    """Wind one foot far enough down and the opposite corner lifts. It
    must come back as zero, never as a negative force holding the frame
    down, which is what a plain linear solve would produce."""
    fs = square()
    fs[0].screw_out_m += 0.010
    s = feet.stand(fs, body(), feet.ground("firm-soil"), **LEVEL)
    assert all(f >= 0.0 for f in s.forces.values())
    assert s.lifted


def test_releasing_one_foot_does_not_shed_the_others():
    """The bug this solver had: dropping every foot that came out in
    tension in one pass, when releasing the first is usually enough to
    put the rest back down. A single short foot must leave three feet
    carrying, not two.

    The centre of gravity is deliberately off centre, because a machine
    whose mass is exactly in the middle of its feet is the one case
    where three is NOT the answer -- see the teetering test below."""
    fs = square()
    fs[0].screw_out_m -= 0.004
    s = feet.stand(fs, body(cg=(0.05, 0.0, 0.03)), feet.ground("firm-soil"),
                   **LEVEL)
    carrying = [i for i, f in s.forces.items() if f > 0.0]
    assert len(carrying) == 3, s.describe()
    assert "f.fl" not in carrying


def test_two_feet_is_reported_as_rocking_rather_than_solved():
    fs = [feet.LevellingFoot("f.a", (-0.5, 0, 0)),
          feet.LevellingFoot("f.b", (0.5, 0, 0))]
    s = feet.stand(fs, body(), feet.ground("firm-soil"), **LEVEL)
    assert s.rocks
    assert not s.ok


# ---------------------------------------------------------------------
# what the inertia is for
# ---------------------------------------------------------------------

def test_a_perfectly_centred_frame_teeters_on_the_diagonal():
    """The degenerate case, and it is real. Lift one corner of a frame
    whose centre of gravity is exactly in the middle and it balances on
    the remaining diagonal with no righting lever at all -- it does not
    fall back onto the fourth foot, because there is nothing to push it
    there. Two feet carry everything."""
    fs = square()
    fs[0].screw_out_m -= 0.004
    s = feet.stand(fs, body(cg=(0.0, 0.0, 0.0)), feet.ground("firm-soil"),
                   **LEVEL)
    carrying = [i for i, f in s.forces.items() if f > 0.0]
    assert len(carrying) == 2
    r = feet.rocking(body(), fs, s)
    assert r.lever_m < 1e-9 and r.tips


def test_rocking_uses_the_real_inertia_and_not_the_mass_alone():
    """Two frames of identical mass and identical feet, differing only
    in how their mass is spread, must rock at different frequencies."""
    fs = square()
    fs[0].screw_out_m -= 0.004
    g = feet.ground("firm-soil")
    off = (0.05, 0.0, 0.03)
    compact = body(cg=off, i=np.diag([20.0, 15.0, 20.0]))
    spread = body(cg=off, i=np.diag([200.0, 150.0, 200.0]))
    r1 = feet.rocking(compact, fs, feet.stand(fs, compact, g, **LEVEL))
    r2 = feet.rocking(spread, fs, feet.stand(fs, spread, g, **LEVEL))
    assert r1 is not None and r2 is not None
    assert r1.frequency_hz > r2.frequency_hz
    # and it is a real pendulum about that edge
    assert r1.inertia_kg_m2 > compact.total_mass_kg * r1.lever_m ** 2


def test_a_frame_on_all_its_feet_has_no_rocking_mode():
    fs = square()
    s = feet.stand(fs, body(), feet.ground("firm-soil"), **LEVEL)
    assert feet.rocking(body(), fs, s) is None


# ---------------------------------------------------------------------
# what it is owed
# ---------------------------------------------------------------------

def test_a_level_machine_owes_nothing():
    fs = square()
    g = feet.ground("firm-soil")
    s = feet.stand(fs, body(), g, **LEVEL)
    assert feet.levelling_need("m", s, fs, body(), g, **LEVEL) is None


def test_merely_uneven_is_below_due_so_it_is_idle_work():
    """The band the request asked for: real work, worth doing, and what
    somebody does when there is nothing better."""
    fs = square()
    fs[0].screw_out_m -= 0.0004
    g = feet.ground("firm-soil")
    s = feet.stand(fs, body(), g, **LEVEL)
    need = feet.levelling_need("m", s, fs, body(), g, **LEVEL)
    assert need is not None
    assert need.want == "levelling"
    assert 0.0 < need.urgency < 1.0
    assert not need.due


def test_a_foot_off_the_floor_is_due():
    fs = square()
    fs[0].screw_out_m -= 0.004
    g = feet.ground("firm-soil")
    s = feet.stand(fs, body(), g, **LEVEL)
    need = feet.levelling_need("m", s, fs, body(), g, **LEVEL)
    assert need.due and need.urgency >= 2.0


def test_a_sinking_pad_is_a_bearing_job_not_a_levelling_one():
    """Winding the screws would be work that undoes itself."""
    fs = square(pad=0.05)
    g = feet.ground("soft-clay")
    s = feet.stand(fs, body(), g, **LEVEL)
    need = feet.levelling_need("m", s, fs, body(), g, **LEVEL)
    assert need.want == "bearing"
    assert need.due
    assert "larger pad" in need.matches


def test_the_job_says_which_foot_which_way_and_how_far():
    """A directable instruction, not a complaint."""
    fs = square()
    fs[0].screw_out_m -= 0.004
    jobs = feet.levelling_jobs("m", fs, feet.ground("firm-soil"), **LEVEL)
    assert len(jobs) == 1
    job = jobs[0]
    assert "fl" in job.action and "down" in job.action and "1.6" in job.action
    assert job.position == tuple(fs[0].position)
    # winding a foot under a loaded frame moves the load path
    assert job.requires_shutdown and not job.interruptible


def test_the_duration_comes_from_the_turns_and_not_from_a_constant():
    g = feet.ground("firm-soil")
    near = square(); near[0].screw_out_m -= 0.001
    far = square(); far[0].screw_out_m -= 0.020
    jn = feet.levelling_jobs("m", near, g, **LEVEL)[0]
    jf = feet.levelling_jobs("m", far, g, **LEVEL)[0]
    assert jf.seconds > jn.seconds * 2


def test_a_foot_already_set_gets_no_job():
    jobs = feet.levelling_jobs("m", square(), feet.ground("firm-soil"), **LEVEL)
    assert jobs == []


def test_the_need_lands_in_the_same_queue_as_every_other_service_need():
    """It is a servicing.Need, so poll_needs and touch_job_for_need work
    on it with nothing added."""
    import servicing
    fs = square()
    fs[0].screw_out_m -= 0.004
    g = feet.ground("firm-soil")
    s = feet.stand(fs, body(), g, **LEVEL)
    need = feet.levelling_need("m", s, fs, body(), g, **LEVEL)
    assert isinstance(need, servicing.Need)
    job = servicing.touch_job_for_need(need)
    assert job.clears == "levelling"
    assert servicing.poll_needs([need])
