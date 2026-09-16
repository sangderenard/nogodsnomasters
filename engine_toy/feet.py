"""Levelling feet: what a frame stands on when it is not a vehicle.

A VEHICLE BALANCES ITSELF AND A MACHINE DOES NOT. That is the whole
reason this exists as its own part. A wheel hangs on a spring with real
travel, so a car put down on uneven ground settles until every corner
carries its share -- the suspension equalises without anyone asking it
to, and the validator that fits a chassis to a power unit can take that
for granted. A levelling screw has no travel. It is a rigid prop wound
out to a chosen length, and it carries exactly the share that its
length, the floor under it and the frame's own stiffness give it.
Nobody balances it. If it is wound short it carries nothing at all and
the machine stands on the other three, rocking on a diagonal.

So the load per foot has to be SOLVED, and outriggers.py is where the
alternative is visible: `OutriggerLeg.share_kg`, "the share of the
machine it holds up", set by its own set to machine_mass / number of
legs. For a hydraulic leg on a common circuit that is nearly true --
one pump, one pressure, and a cylinder that keeps extending until it
takes load, which is a suspension by another name. For a hand-set screw
it is not true at all, and assuming it hides the exact failure feet
have: one of them not touching.

THREE FEET CANNOT GET IT WRONG. Three points are statically
determinate: the load split follows from geometry alone and does not
depend on the screws, the floor, or how stiff anything is, and all
three are always in contact. Four is indeterminate -- one redundancy --
and that redundancy is exactly where the screw setting goes. This is
the wobbly-table problem, and it is why precision machines are so often
on three feet and why everything else needs levelling.

THE FOOT'S COMPLIANCE IS THE GROUND'S. A screw foot's steel column is
enormously stiffer than the dirt under its pad -- on firm soil, by more
than two orders of magnitude -- so the series stiffness is the ground's
and the screw contributes nothing to it. `foot_stiffness_n_per_m`
computes both and the difference is not close.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GRAVITY_M_S2 = 9.80665
STEEL_MODULUS_PA = 200e9


# ---------------------------------------------------------------------
# what it is standing on
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Ground:
    """What the pad is pressing into.

    Two numbers, and they answer two different questions. The allowable
    bearing pressure says whether the pad SINKS. The modulus says how
    far it settles before it stops, which is what decides the load share
    when there are more than three feet."""
    key: str
    label: str
    allowable_bearing_pa: float
    modulus_pa: float
    poisson: float
    why: str


#: Real, disclosed order-of-magnitude figures for allowable bearing
#: pressure and elastic modulus -- the same convention this project uses
#: elsewhere for published engineering ranges rather than one paper's
#: coefficients. The firm-soil anchor is outriggers.py's own existing
#: 250 kPa, kept identical so two parts of this codebase do not hold two
#: opinions about the same dirt.
GROUNDS: dict[str, Ground] = {
    "soft-clay": Ground(
        "soft-clay", "soft clay", 75_000.0, 5e6, 0.45,
        why="a pad that is fine on firm ground goes straight into this, and "
            "it keeps going: clay creeps under a sustained load long after "
            "it has stopped settling elastically"),
    "firm-soil": Ground(
        "firm-soil", "firm soil", 250_000.0, 25e6, 0.35,
        why="the reference, and outriggers.py's own default"),
    "compacted-gravel": Ground(
        "compacted-gravel", "compacted gravel", 400_000.0, 80e6, 0.30,
        why="the usual answer to soft ground under a machine: dig it out and "
            "put this back, because it is cheaper than a slab and stiffer "
            "than what was there"),
    "tarmac": Ground(
        "tarmac", "tarmac over base", 2_500_000.0, 150e6, 0.35,
        why="stiff until it is warm, and then a small pad left standing on "
            "it for a day leaves a dent -- the base carries it, the surface "
            "only spreads it"),
    "concrete-slab": Ground(
        "concrete-slab", "reinforced concrete slab", 5_000_000.0, 30e9, 0.20,
        why="stiff enough that the screw's own compliance finally matters, "
            "and the real limit stops being bearing and becomes punching "
            "through the slab"),
}


def ground(key: str) -> Ground:
    g = GROUNDS.get(str(key))
    if g is None:
        raise KeyError(f"unknown ground {key!r}; declared: "
                       f"{', '.join(sorted(GROUNDS))}")
    return g


# ---------------------------------------------------------------------
# the foot
# ---------------------------------------------------------------------

@dataclass
class LevellingFoot:
    """One screw foot under a frame.

    `screw_out_m` is the SETTING -- how far the pad hangs below the
    frame's underside. It is the only thing an operator changes and it
    is what decides this foot's share, which is why it is a field and
    not a constant."""
    identity: str
    position: tuple                  # where it meets the frame
    pad_diameter_m: float = 0.100
    thread_pitch_m: float = 0.0025   # M20 x 2.5, an ordinary levelling screw
    thread_core_diameter_m: float = 0.0169
    screw_out_m: float = 0.060
    screw_max_out_m: float = 0.120
    rated_load_n: float = 20_000.0

    @property
    def pad_area_m2(self) -> float:
        return math.pi * (self.pad_diameter_m / 2.0) ** 2

    def screw_stiffness_n_per_m(self) -> float:
        """The steel column, EA/L at the length it is wound out to."""
        area = math.pi * (self.thread_core_diameter_m / 2.0) ** 2
        return STEEL_MODULUS_PA * area / max(self.screw_out_m, 1e-4)

    def ground_stiffness_n_per_m(self, g: Ground) -> float:
        """A rigid circular pad pressed into an elastic half space.

        The classical rigid-punch result, k = 2.a.E / (1 - nu^2) with a
        the pad radius. It is the number that says how far this foot
        sinks per newton, and on anything but a slab it is the whole of
        the foot's compliance."""
        a = self.pad_diameter_m / 2.0
        return 2.0 * a * g.modulus_pa / (1.0 - g.poisson ** 2)

    def stiffness_n_per_m(self, g: Ground) -> float:
        ks = self.screw_stiffness_n_per_m()
        kg = self.ground_stiffness_n_per_m(g)
        return 1.0 / (1.0 / ks + 1.0 / kg)

    def pad_pressure_pa(self, load_n: float) -> float:
        return max(0.0, load_n) / self.pad_area_m2

    def sinks(self, load_n: float, g: Ground) -> bool:
        """A pad pushing harder than the ground can carry goes into it --
        outriggers.py's own test, on the same footing."""
        return self.pad_pressure_pa(load_n) > g.allowable_bearing_pa

    def overloaded(self, load_n: float) -> bool:
        return load_n > self.rated_load_n

    def turns_for(self, travel_m: float) -> float:
        """How much winding a change of setting is. A levelling screw is
        set by someone with a spanner, so the answer is in turns."""
        return travel_m / self.thread_pitch_m


# ---------------------------------------------------------------------
# standing on them
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Stance:
    """What happened when the frame was put down."""
    forces: dict                    # identity -> newtons, compression only
    settle_m: dict                  # identity -> how far that foot squashed
    gaps_m: dict                    # identity -> gap before it was put down
    lifted: tuple                   # feet carrying nothing at all
    sinking: tuple
    overloaded: tuple
    heave_m: float
    tilt_x: float                   # slope along x, radians (small)
    tilt_z: float
    determinate: bool
    rocks: bool
    reason: str

    @property
    def ok(self) -> bool:
        return (not self.rocks and not self.lifted and not self.sinking
                and not self.overloaded)

    def share(self) -> dict:
        total = sum(self.forces.values())
        if total <= 0.0:
            return {k: 0.0 for k in self.forces}
        return {k: v / total for k, v in self.forces.items()}

    def describe(self) -> str:
        carrying = sum(1 for f in self.forces.values() if f > 0.0)
        lines = [f"  {'determinate' if self.determinate else 'indeterminate'}"
                 f" on {carrying} of {len(self.forces)} feet -- {self.reason}"]
        for ident, f in sorted(self.forces.items()):
            tag = ""
            if ident in self.lifted:
                tag = "  NOT TOUCHING"
            elif ident in self.sinking:
                tag = "  SINKING"
            elif ident in self.overloaded:
                tag = "  OVER RATED LOAD"
            lines.append(f"    {ident.split('.')[-1]:14s} {f / 1000.0:7.2f} kN"
                         f"  {self.share()[ident] * 100:5.1f}%"
                         f"  settles {self.settle_m[ident] * 1000:5.2f} mm{tag}")
        if abs(self.tilt_x) > 1e-9 or abs(self.tilt_z) > 1e-9:
            lines.append(f"    frame tilts {self.tilt_x * 1000:+.3f} mm/m along x, "
                         f"{self.tilt_z * 1000:+.3f} mm/m along z")
        return "\n".join(lines)


def gaps_from_floor(feet, floor_height, frame_underside_y: float) -> dict:
    """How far short of the ground each foot is, before any load.

    Negative is not an error: a foot wound down past the floor preloads
    itself and unloads its neighbours, which is exactly what levelling a
    machine does."""
    if callable(floor_height):
        heights = {f.identity: float(floor_height(f)) for f in feet}
    elif isinstance(floor_height, dict):
        heights = {f.identity: float(floor_height.get(f.identity, 0.0))
                   for f in feet}
    else:
        heights = {f.identity: float(floor_height) for f in feet}
    return {f.identity: (frame_underside_y - f.screw_out_m) - heights[f.identity]
            for f in feet}


def level_settings(feet, floor_height, frame_underside_y: float) -> dict:
    """The screw setting per foot that puts every pad on the floor with
    the frame level.

    This is the levelling job stated as a number of turns, which is what
    it actually is for whoever has the spanner."""
    out = {}
    for f in feet:
        gap = gaps_from_floor([f], floor_height, frame_underside_y)[f.identity]
        out[f.identity] = f.screw_out_m + gap
    return out


def stand(feet, rigid_body, g: Ground, *,
          floor_height=0.0, frame_underside_y: float = 0.0,
          max_passes: int = 24) -> Stance:
    """Put a frame down on its feet and find what each one carries.

    A UNILATERAL CONTACT SOLVE, because a foot can push and cannot pull.
    Three unknowns -- how far the frame settles and the two slopes it
    settles at -- because vertical props constrain exactly three degrees
    of freedom and nothing else. Solve, drop every foot that came out in
    tension or never closed its gap, solve again, and stop when the set
    stops changing. That active set is the answer to "which feet is it
    actually standing on", and it is a question a declared even share
    cannot be asked.

    IT TAKES THE REAL RIGID BODY, not a mass and a guess at where it
    acts. An engine_mass_properties.RigidBodyProperties carries the
    centre of gravity the machine's own bodies put it at and the full
    inertia tensor about it, products of inertia included -- on this
    autoclave the yz product is -15 kg.m2, which is not a rounding error
    and is what makes it rock about a diagonal rather than about an axis
    somebody nominated. Statics needs the first two; `rocking` below
    needs the third."""
    feet = list(feet)
    mass_kg = float(rigid_body.total_mass_kg)
    cg = rigid_body.center_of_gravity
    if not feet:
        return Stance({}, {}, {}, (), (), (), 0.0, 0.0, 0.0, False, True,
                      "nothing to stand on")
    gaps = gaps_from_floor(feet, floor_height, frame_underside_y)
    weight = mass_kg * GRAVITY_M_S2
    cg = np.asarray(cg, float)
    k = {f.identity: f.stiffness_n_per_m(g) for f in feet}
    X = {f.identity: float(f.position[0]) - float(cg[0]) for f in feet}
    Z = {f.identity: float(f.position[2]) - float(cg[2]) for f in feet}

    active = [f.identity for f in feet]
    heave = tilt_x = tilt_z = 0.0
    forces = {f.identity: 0.0 for f in feet}
    reason = "standing"
    rocks = False
    for _ in range(max_passes):
        if len(active) < 3:
            rocks = True
            reason = (f"only {len(active)} foot/feet in contact -- three "
                      "non-collinear points are the least that will not rock")
            break
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in active:
            ki, xi, zi, gi = k[i], X[i], Z[i], gaps[i]
            row = np.array([1.0, xi, zi])
            A += ki * np.outer(row, row)
            b += ki * gi * row
        b[0] += weight
        if abs(np.linalg.det(A)) < 1e-6:
            rocks = True
            reason = ("the feet in contact are collinear -- there is no "
                      "polygon for the frame to sit on")
            break
        heave, tilt_x, tilt_z = np.linalg.solve(A, b)
        travel = {f.identity: heave + tilt_x * X[f.identity]
                  + tilt_z * Z[f.identity] for f in feet}
        compression = {i: travel[i] - gaps[i] for i in travel}
        forces = {i: (k[i] * compression[i] if compression[i] > 0.0 else 0.0)
                  for i in compression}
        # ONE AT A TIME. Dropping every foot that came out in tension in
        # the same pass is the obvious move and it is wrong: releasing
        # one changes the tilt, which is usually enough to put the others
        # back down. Done all at once, a frame with a single short foot
        # shed three of them and reported that it was standing on two.
        violating = [i for i in active if compression[i] <= 0.0]
        if violating:
            active = [i for i in active
                      if i != min(violating, key=lambda j: compression[j])]
            continue
        returning = [i for i in compression
                     if i not in active and compression[i] > 0.0]
        if not returning:
            break
        active = sorted(set(active) | {max(returning,
                                           key=lambda j: compression[j])})
    else:
        reason = "the set of feet in contact would not settle"

    settle = {i: (forces[i] / k[i] if k[i] else 0.0) for i in forces}
    lifted = tuple(sorted(i for i, f in forces.items() if f <= 0.0))
    by_id = {f.identity: f for f in feet}
    sinking = tuple(sorted(i for i, f in forces.items()
                           if f > 0.0 and by_id[i].sinks(f, g)))
    over = tuple(sorted(i for i, f in forces.items()
                        if by_id[i].overloaded(f)))
    determinate = len([f for f in forces.values() if f > 0.0]) == 3
    if not rocks and lifted:
        reason = (f"{len(lifted)} foot/feet wound short of the floor -- the "
                  "frame is standing on the others")
    return Stance(forces=forces, settle_m=settle, gaps_m=gaps, lifted=lifted,
                  sinking=sinking, overloaded=over, heave_m=float(heave),
                  tilt_x=float(tilt_x), tilt_z=float(tilt_z),
                  determinate=determinate, rocks=rocks, reason=reason)


# ---------------------------------------------------------------------
# what the inertia is for
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Rocking:
    """A frame standing on a foot that is not touching is a pendulum.

    It rocks about the line joining the two feet it is still on, and how
    fast is not a matter of opinion: gravity supplies the restoring
    moment through the offset from that line, and the inertia about that
    same line decides the period. This is the number a contact sim needs
    and the one a static load share can never produce."""
    axis: tuple                    # the tipping edge, as a direction
    through: tuple                 # a point on it
    lever_m: float                 # how far the cg stands out from the edge
    inertia_kg_m2: float           # about the edge, parallel axis included
    frequency_hz: float
    tips: bool                     # gravity does not bring it back at all

    def describe(self) -> str:
        if self.tips:
            return (f"    ROCKS AND DOES NOT COME BACK: the centre of gravity "
                    f"is {self.lever_m * 1000:.0f} mm past the edge it would "
                    "tip over")
        return (f"    rocks about that edge at {self.frequency_hz:.2f} Hz "
                f"({self.inertia_kg_m2:.0f} kg.m2 about it, "
                f"{self.lever_m * 1000:.0f} mm of righting lever)")


def rocking(rigid_body, feet, stance: Stance) -> "Rocking | None":
    """The rocking mode a lifted foot leaves behind.

    Returns None when every foot is carrying -- a frame standing on all
    of them has no edge to rock about, which is the whole point of
    levelling it."""
    carrying = [f for f in feet if stance.forces.get(f.identity, 0.0) > 0.0]
    if len(carrying) < 2 or not stance.lifted:
        return None
    cg = np.asarray(rigid_body.center_of_gravity, float)
    # the edge is the pair of loaded feet the centre of gravity is
    # nearest to being outside of -- the one it would go over first
    best = None
    for i, a in enumerate(carrying):
        for b in carrying[i + 1:]:
            pa = np.asarray(a.position, float)
            pb = np.asarray(b.position, float)
            d = pb - pa
            n = float(np.linalg.norm(d))
            if n < 1e-9:
                continue
            u = d / n
            r = cg - pa
            lever = float(np.linalg.norm(r - u * float(np.dot(r, u))))
            if best is None or lever < best[0]:
                best = (lever, u, pa)
    if best is None:
        return None
    lever, u, through = best
    # inertia about the edge: the tensor resolved onto it, plus the
    # parallel-axis term for standing a lever away from it
    I = np.asarray(rigid_body.inertia_tensor_kg_m2, float)
    about = float(u @ I @ u) + rigid_body.total_mass_kg * lever ** 2
    # gravity restores it while the centre of gravity is still inboard
    stiffness = rigid_body.total_mass_kg * GRAVITY_M_S2 * lever
    tips = not _inside_support(cg, carrying)
    freq = 0.0 if (tips or about <= 0.0) else math.sqrt(stiffness / about) / (2 * math.pi)
    return Rocking(axis=tuple(float(v) for v in u),
                   through=tuple(float(v) for v in through),
                   lever_m=lever, inertia_kg_m2=about,
                   frequency_hz=freq, tips=tips)


def _inside_support(cg, carrying) -> bool:
    """engine_mounts' own gate, on the feet that are actually loaded --
    which is the distinction that matters: a machine is stable on the
    feet carrying it, not on the feet it has."""
    from engine_mounts import check_stability
    return check_stability([tuple(float(v) for v in f.position)
                            for f in carrying],
                           tuple(float(v) for v in cg)).ok


# ---------------------------------------------------------------------
# what an unlevel machine is owed
# ---------------------------------------------------------------------

#: HOW BADLY THE LOAD MAY BE SHARED BEFORE ANYONE SHOULD CARE. A frame
#: on four screws is never exactly even, and chasing the last few per
#: cent is not work worth sending somebody to do. A foot down to a fifth
#: of its fair share is on its way to not touching at all, and that is.
LEVEL_TOLERANCE = 0.20

#: Winding a levelling screw is slow, fiddly and done with a spanner in
#: an awkward place, so the time is per turn with a fixed cost for
#: getting there, kneeling down and checking the level afterwards. The
#: point of deriving the duration from the actual turns is that a
#: machine one flat out of level is a two minute job and one sitting on
#: a diagonal is not.
SECONDS_PER_TURN = 12.0
LEVELLING_SETUP_S = 180.0


def levelling_need(identity: str, stance: "Stance", feet_, rigid_body,
                   g: Ground, *, floor_height=0.0,
                   frame_underside_y: float = 0.0, position=None,
                   label: str = ""):
    """What this machine is owed, as a servicing.Need.

    DECLARED INTO THE EXISTING QUEUE, because servicing.py already says
    that is how this works: "a service task is not a property of a
    centrifuge, or a filter, or a tank... every piece of equipment
    declares INTO it rather than inventing its own idea of maintenance".
    An unlevel machine is the same statement as a blocked filter --
    something is due, it is at a place, it takes a while -- so it goes
    into the same queue and an idle crew member finds it there.

    URGENCY IS THE WHOLE POINT. servicing.Need already treats 1.0 as the
    line between due and not, and a machine merely sharing its load
    unevenly sits BELOW it deliberately: real work, worth doing, and
    what somebody does when there is nothing better. A machine with a
    foot off the floor is above the line, because it is rocking. One
    standing on two is well above it.

    AND SOMETIMES LEVELLING IS THE WRONG JOB. A pad that is sinking will
    be out of level again tomorrow however carefully it is set today, so
    that comes back as a bearing problem rather than a levelling one --
    winding the screws would be work that undoes itself."""
    from servicing import Need
    feet_ = list(feet_)
    if position is None:
        pts = [f.position for f in feet_]
        position = tuple(float(np.mean([p[i] for p in pts])) for i in range(3))

    if stance.sinking:
        worst = max(stance.sinking, key=lambda i: stance.forces[i])
        foot = next(f for f in feet_ if f.identity == worst)
        pressure = foot.pad_pressure_pa(stance.forces[worst])
        ratio = pressure / g.allowable_bearing_pa
        return Need(
            identity=identity, want="bearing", position=tuple(position),
            quantity=round(ratio, 2), unit="x allowable",
            matches="larger pad or better ground", urgency=1.0 + ratio,
            minutes=90.0, skill="fitter", label=label or identity,
            why=f"{worst} is pressing {pressure / 1000:.0f} kPa into "
                f"{g.label}, which carries {g.allowable_bearing_pa / 1000:.0f}"
                " -- it is going in, and levelling it today means levelling "
                "it again tomorrow")

    wanted = level_settings(feet_, floor_height, frame_underside_y)
    turns = {f.identity: abs(f.turns_for(wanted[f.identity] - f.screw_out_m))
             for f in feet_}
    total_turns = sum(turns.values())
    minutes = (LEVELLING_SETUP_S + SECONDS_PER_TURN * total_turns) / 60.0

    if stance.rocks:
        urgency = 3.0
        why = (stance.reason + " -- it is not standing on a polygon, so it "
               "moves whenever anything on it does")
    elif stance.lifted:
        urgency = 2.0
        why = (", ".join(stance.lifted) + " not touching: the frame is on a "
               "diagonal and rocks")
    else:
        share = stance.share()
        even = 1.0 / max(len(share), 1)
        lightest = min(share.values()) if share else even
        worst = lightest / even
        if worst >= 1.0 - LEVEL_TOLERANCE:
            return None
        # DELIBERATELY BELOW DUE. Real work, no hurry.
        urgency = 0.3 + 0.6 * (1.0 - worst)
        why = (f"the lightest foot carries {lightest * 100:.0f}% where an even "
               f"share is {even * 100:.0f}% -- it is not loose yet and it is "
               "on its way to being")

    return Need(identity=identity, want="levelling", position=tuple(position),
                quantity=round(total_turns, 1), unit="turns",
                matches="spanner and a level", urgency=urgency,
                minutes=minutes, skill="fitter", label=label or identity,
                why=why)


def levelling_jobs(identity: str, feet_, g: Ground, *, floor_height=0.0,
                   frame_underside_y: float = 0.0) -> list:
    """One held interaction per foot that has to move, and which way.

    This is what makes the job directable rather than a complaint. The
    setting every foot should be at is already known -- level_settings
    computes it from the floor under each pad -- so the instruction is
    "this foot, this many turns, this way", at the foot's own position.
    That is the shape servicing.TouchJob already has, so an interaction
    system needs nothing new to run it."""
    from servicing import TouchJob
    feet_ = list(feet_)
    wanted = level_settings(feet_, floor_height, frame_underside_y)
    jobs = []
    for f in sorted(feet_,
                    key=lambda f: -abs(wanted[f.identity] - f.screw_out_m)):
        delta = wanted[f.identity] - f.screw_out_m
        if abs(delta) < f.thread_pitch_m / 8.0:
            continue                    # closer than an eighth of a turn
        turns = f.turns_for(abs(delta))
        way = "down" if delta > 0 else "up"
        jobs.append(TouchJob(
            identity=f"{f.identity}:levelling",
            action=f"wind {f.identity.split('.')[-1]} {way} {turns:.1f} turns",
            position=tuple(float(v) for v in f.position),
            seconds=LEVELLING_SETUP_S / max(len(feet_), 1)
                    + SECONDS_PER_TURN * turns,
            skill="fitter", consumes="",
            # THE MACHINE HAS TO BE OFF. Winding a foot under a running
            # machine moves the load path while it is loaded, and the
            # frame it is holding up is the thing that moves.
            requires_shutdown=True, interruptible=False,
            clears="levelling", label=identity))
    return jobs


def from_mounts(doc: dict, *, pad_diameter_m: float = 0.100, **foot_kw) -> list:
    """One levelling foot under every declared structural mount.

    A machine's feet are where its mounts are -- the same nodes
    machine_package collects and installation mates -- so they are read
    off the graph by what the node DECLARES and never by its name.
    `foot_kw` is the foot's own hardware (pad, thread, setting) and is
    the same for every foot, because a set of levelling feet is bought
    as a set."""
    from machine_package import MOUNT_PORT_ROLE
    out = []
    for n in doc["nodes"]:
        if n.get("port_role") != MOUNT_PORT_ROLE:
            continue
        out.append(LevellingFoot(n["identity"],
                                 tuple(float(v) for v in n["reference_position"]),
                                 pad_diameter_m=pad_diameter_m, **foot_kw))
    if not out:
        raise ValueError(f"{doc.get('identity', 'machine')}: no node declares "
                         f"port_role={MOUNT_PORT_ROLE!r}, so there is nothing "
                         "to stand it on")
    return out


def underside_y(doc: dict) -> float:
    """The frame's underside: the lowest declared mount."""
    from machine_package import MOUNT_PORT_ROLE
    ys = [float(n["reference_position"][1]) for n in doc["nodes"]
          if n.get("port_role") == MOUNT_PORT_ROLE]
    return min(ys) if ys else 0.0
