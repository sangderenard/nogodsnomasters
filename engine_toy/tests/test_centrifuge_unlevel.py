"""An unlevel centrifuge damages itself, at a rate commensurate with the
freedom one short foot gives it.

THE NUMBERS ARE SMALL BECAUSE THE FREEDOM IS SMALL. On firm soil a foot
carrying 250 N of this machine sinks a tenth of a millimetre, and the
other three redistribute as it is wound up, so the whole of "one foot
slightly short" happens inside half a millimetre: below that the foot
is partly unloaded and lifts sooner, at the point it lets go the frame
strikes it on every crossing, and past that the foot is out of reach
and the machine is a three-footed one with lower modes. On a slab the
same band is three microns wide. All of it is in the table.
"""
import pytest

import centrifuge_production as cp
import feet
import mode_table as mt

#: a sludge cake that has thrown unevenly: a gram at the bowl wall --
#: enough to lift a foot that has been unloaded, not enough to lift one
#: carrying its share, which is what makes levelling the variable
CAKE_KG_M = 0.001
UNDER = 0.06


@pytest.fixture(scope="module")
def doc():
    return cp.build().as_document()


def shortened(doc, foot_suffix: str, short_m: float, pad=0.08):
    return [feet.LevellingFoot(f.identity, f.position, pad_diameter_m=f.pad_diameter_m,
                               screw_out_m=f.screw_out_m
                               - (short_m if f.identity.endswith(foot_suffix) else 0.0))
            for f in feet.from_mounts(doc, pad_diameter_m=pad)]


def table(doc, short_m, ground="firm-soil", cake=CAKE_KG_M):
    ft = shortened(doc, "foot_fl", short_m)
    return mt.evaluate(doc, ft, feet.ground(ground), cp.cycle(sludge_unbalance_kg_m=cake),
                       frame_underside_y=feet.underside_y(doc) + UNDER)


def hammering(t):
    return sum(r.hammering_w for r in t.rows)


def test_the_machine_is_balanced_across_its_feet(doc):
    """The layout fault this test found: with the motor hung out at 220
    mm the front feet carried 47 N and lifted at the first gram. A real
    purifier puts bowl and motor either side of the centre."""
    t = table(doc, 0.0)
    forces = t.rows[0].stance.forces
    assert max(forces.values()) < 1.3 * min(forces.values())


def test_level_and_clean_it_neither_lifts_nor_strikes(doc):
    t = table(doc, 0.0, cake=0.0)
    assert hammering(t) == 0.0
    assert not any("leaves" in h or "struck" in h for r in t.rows for h in r.hazards)


def test_the_run_up_crosses_the_rocking_modes_and_the_table_says_where(doc):
    t = table(doc, 0.0, cake=0.0)
    crossings = [(r.sample, h) for r in t.rows if r.state == "spin-up"
                 for h in r.hazards if "crosses" in h]
    assert crossings
    assert any(0.1 < s < 0.9 for s, _ in crossings)
    assert all("rpm" in h and "Hz" in h for _, h in crossings)
    # and one of them is the motor at rated speed sitting on a rocking
    # mode -- a design finding the table makes, not a test artefact
    assert any(s == 1.0 and "motor" in h for s, h in crossings)


def test_a_short_foot_unloads_and_the_hammering_grows_with_the_shortfall(doc):
    """Commensurate: the short foot's static load falls linearly with the
    shortfall, the alternating force exceeds it sooner, and the energy of
    each landing grows continuously from nothing."""
    shorts_um = (0, 100, 150, 250, 350, 400)
    powers, statics = [], []
    for um in shorts_um:
        t = table(doc, um * 1e-6)
        fl = next(k for k in t.rows[0].stance.forces if k.endswith("foot_fl"))
        statics.append(t.rows[0].stance.forces[fl])
        powers.append(hammering(t))
    assert all(a > b for a, b in zip(statics, statics[1:]))
    assert powers[0] == 0.0
    assert all(b > a for a, b in zip(powers, powers[1:]))
    # four tenths of a millimetre is two orders of magnitude, not a flag
    assert powers[1] < 0.01 * powers[-1]


def test_the_worst_is_the_moment_the_foot_lets_go(doc):
    """Just short of letting go the foot carries almost nothing and
    lifts on nearly every row; just past it the foot is off, its
    clearance is a few tens of microns, and the frame's travel at that
    corner during the crossing STRIKES it; further off it is out of
    reach and the machine is a quieter three-footed one. The freedom is
    tiny and so is the band, and the damage peaks inside it."""
    level = table(doc, 0.0)
    nearly = table(doc, 440e-6)
    just_off = table(doc, 460e-6)
    well_off = table(doc, 1000e-6)
    assert not nearly.rows[0].stance.lifted
    assert len(just_off.rows[0].stance.lifted) == 1
    # three feet: the first rocking mode falls by a third
    assert just_off.rows[0].frequencies_hz[0] < 0.7 * level.rows[0].frequencies_hz[0]
    # the freed foot is struck while its clearance is inside the travel
    assert any("struck" in h for r in just_off.rows for h in r.hazards)
    assert not any("struck" in h for r in well_off.rows for h in r.hazards)
    # and the rate peaks at the point of letting go
    assert hammering(nearly) > hammering(just_off) > hammering(well_off) > 0.0
    assert hammering(nearly) > 4.0 * hammering(well_off)


def test_on_a_slab_a_millimetre_is_two_feet_and_the_table_says_it_rocks(doc):
    """Concrete cannot redistribute: the foot's compression is a micron,
    so one millimetre lifts a diagonal and the frame is on two feet. The
    model does not pretend to solve that; it reports it."""
    t = table(doc, 1e-3, ground="concrete-slab")
    assert t.rows[0].stance.rocks
    assert t.rows[0].whirl == []
    assert any("rocks" in h for h in t.rows[0].hazards)


def test_the_heavier_frame_hammers_less_for_the_same_cake():
    light = cp.build(frame="square-tube-welded").as_document()
    heavy = cp.build(frame="angle-bolted").as_document()
    pl = hammering(table(light, 250e-6))
    ph = hammering(table(heavy, 250e-6))
    assert ph < pl
