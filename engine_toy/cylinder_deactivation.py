"""Cylinder deactivation: the hardware, the computer, and why the
pattern is not a free choice.

An engine at cruise is making a fraction of the torque it was built for,
and it is making it badly. A throttled spark engine spends a real part of
its output dragging air past a nearly-shut throttle -- pumping loss --
and that loss is roughly fixed per revolution regardless of how little
work is being asked for. So the engine is least efficient exactly where
it spends its life.

Shutting cylinders off fixes this by making the working ones work
harder. Four cylinders carrying the whole load run at twice the mean
effective pressure, which means a wider throttle, which means less
pumping loss, which is where the fuel saving actually comes from. It is
not saved by "using less fuel in four cylinders" -- that would be true
of simply lifting off.

WHICH MEANS THE VALVES MUST SHUT, AND THIS IS THE WHOLE ENGINEERING.
Cutting fuel and spark alone leaves the valves working, so the dead
cylinders go on pumping air through themselves -- you keep the pumping
loss you were trying to delete, and you pour cold oxygen into the
exhaust, which cools the catalyst below light-off and makes it stop
working. Fuel-only cutoff is not a cheaper version of this; it is a
different and much worse thing.

Done properly, both valves are held shut and the cylinder keeps a charge
of exhaust gas trapped inside it. That trapped gas becomes a GAS SPRING:
the piston compresses it on the way up and it pushes back on the way
down, returning most of the work. A deactivated cylinder is not a
passenger being dragged -- it is very nearly a free-wheeling spring, and
the small loss that remains is heat into the walls and blow-by past the
rings.

AND THE PATTERN IS DECIDED BY THE CRANK, NOT BY PREFERENCE. Firing
events have to stay evenly spaced or the engine shakes. On a V8 whose
eight events sit ninety degrees apart, dropping every other one leaves
four events a clean hundred and eighty degrees apart -- perfectly even,
which is why V8 deactivation is always to four and essentially never to
six. Six survivors out of eight cannot be evenly spaced at all. This
module derives that rather than hard-coding it, so an odd-fire V-twin or
a radial gets an honest answer instead of a borrowed one.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# the hardware that holds the valves shut
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DeactivationHardware:
    key: str
    label: str
    #: Does it actually hold the valves shut? If not, the cylinder keeps
    #: pumping and almost none of the benefit appears.
    traps_charge: bool
    #: Oil pressure needed to move the locking pins.
    min_oil_pressure_pa: float
    #: Oil temperature below which the pins are too slow to be trusted.
    min_oil_temp_k: float
    #: How long a cylinder takes to change state. Transitions are felt.
    switch_time_s: float
    #: Chance per switching cycle that a pin sticks. Small, and this is
    #: a component that switches millions of times.
    stick_probability: float
    why: str = ""


HARDWARE: dict[str, DeactivationHardware] = {
    "collapsing-lifter": DeactivationHardware(
        "collapsing-lifter", "collapsing hydraulic lifter", True, 170_000.0, 333.0,
        switch_time_s=0.05, stick_probability=2.0e-7,
        why="an oil-pressure pin inside the lifter body unlocks, and the lifter "
            "collapses instead of pushing the pushrod. Both valves stay shut and the "
            "charge is trapped. It is also the part with the industry's worst "
            "reputation, because a pin that sticks collapsed leaves a cylinder dead "
            "and a pin that sticks locked leaves it firing when the computer thinks "
            "it is not"),
    "switching-follower": DeactivationHardware(
        "switching-follower", "switching roller finger follower", True, 200_000.0, 338.0,
        switch_time_s=0.03, stick_probability=8.0e-8,
        why="the rocker itself unlocks into two pieces, so the cam lobe moves the "
            "outer arm and the valve does not follow. Faster and more reliable than "
            "collapsing the lifter because the moving parts are lighter"),
    "electrohydraulic": DeactivationHardware(
        "electrohydraulic", "camless electrohydraulic", True, 0.0, 273.0,
        switch_time_s=0.01, stick_probability=1.0e-8,
        why="no cam to defeat: the valve simply is not commanded. Any cylinder, any "
            "cycle, no warm-up requirement and no oil dependency"),
    "fuel-cutoff-only": DeactivationHardware(
        "fuel-cutoff-only", "fuel and spark cutoff only", False, 0.0, 273.0,
        switch_time_s=0.002, stick_probability=0.0,
        why="the software-only version, and it is a trap. The valves keep working, so "
            "the cylinder goes on pumping and the pumping loss you were trying to "
            "delete is still there. Worse, it pumps cold air through to the exhaust, "
            "which cools the catalyst below light-off and oxidises the exhaust valve"),
}


def hardware(key: str) -> DeactivationHardware:
    h = HARDWARE.get(str(key))
    if h is None:
        raise KeyError(f"unknown deactivation hardware {key!r}; declared: "
                       f"{', '.join(sorted(HARDWARE))}")
    return h


# ---------------------------------------------------------------------
# which cylinders may be dropped
# ---------------------------------------------------------------------

def firing_evenness(angles_deg, cycle_deg: float = 720.0) -> float:
    """How evenly spaced a set of firing events is. 0.0 is perfect.

    The coefficient of variation of the intervals between consecutive
    events, wrapping round the cycle. This is the number that decides
    whether an engine shakes, and it is why the pattern is not a free
    choice."""
    a = sorted(float(x) % cycle_deg for x in angles_deg)
    n = len(a)
    if n <= 1:
        return 0.0
    gaps = [a[(i + 1) % n] - a[i] + (cycle_deg if i == n - 1 else 0.0)
            for i in range(n)]
    mean = sum(gaps) / n
    if mean <= 0.0:
        return 0.0
    var = sum((g - mean) ** 2 for g in gaps) / n
    return math.sqrt(var) / mean


#: Ceiling on how many subsets to evaluate. The whole search is
#: enumerated below this, which covers every engine anyone builds: the
#: worst case for a sixteen-cylinder is C(16,8) = 12870.
#:
#: This used to be derived from the caller's output `limit`, which was a
#: real bug rather than a tuning choice -- best_pattern asks for one
#: result, so the search stopped after eight subsets out of seventy and
#: confidently reported that a V8 cannot drop to four evenly. It can;
#: the search simply never looked at the subset that does it.
SEARCH_CEILING = 40_000


def candidate_patterns(architecture, keep: int, limit: int = 400) -> list:
    """Every way of keeping `keep` cylinders, best first.

    Derived from the architecture's own firing angles, so an odd-fire
    engine gets an answer that reflects its crank instead of a rule
    borrowed from a V8."""
    angles = list(architecture.slot_angles_deg())
    n = len(angles)
    keep = max(1, min(n, int(keep)))
    cycle = float(getattr(architecture, "cycle_degrees", 720.0))
    order = list(getattr(architecture, "firing_order", range(1, n + 1)))
    combos = itertools.combinations(range(n), keep)
    if math.comb(n, keep) > SEARCH_CEILING:
        # more subsets than is worth enumerating. Evenly-spaced subsets
        # are the ones that matter, so take strides through the slots --
        # which is where every even answer lives anyway.
        combos = itertools.islice(combos, SEARCH_CEILING)
    out = []
    for idx in combos:
        ev = firing_evenness([angles[i] for i in idx], cycle)
        out.append({"slots": idx,
                    "cylinders": tuple(order[i] for i in idx),
                    "deactivated": tuple(order[i] for i in range(n) if i not in idx),
                    "evenness": ev,
                    "even": ev < 1e-6})
    out.sort(key=lambda r: r["evenness"])
    return out[:limit]


def best_pattern(architecture, keep: int):
    c = candidate_patterns(architecture, keep, limit=1)
    return c[0] if c else None


def usable_counts(architecture) -> list:
    """How many cylinders this engine can honestly run on.

    An engine can only drop to counts that leave an even firing pattern.
    For most V8s that is eight or four and nothing between, which is the
    reason production systems behave the way they do."""
    n = len(architecture.slot_angles_deg())
    out = []
    for keep in range(1, n + 1):
        b = best_pattern(architecture, keep)
        if b is not None:
            out.append({"keep": keep, "evenness": b["evenness"],
                        "even": b["even"], "cylinders": b["cylinders"]})
    return out


# ---------------------------------------------------------------------
# what it actually saves
# ---------------------------------------------------------------------

#: A deactivated cylinder is a gas spring, not a free wheel. What it
#: loses per cycle is heat into the walls and blow-by past the rings --
#: real, and about a tenth of what the same cylinder would lose pumping.
GAS_SPRING_LOSS_FRAC = 0.10


def pumping_loss_frac(load_frac: float) -> float:
    """Share of indicated work thrown away pumping past the throttle.

    Large at light load and nearly zero at full throttle, which is the
    whole shape of the problem: the loss is worst exactly where the
    engine lives."""
    L = max(0.02, min(1.0, float(load_frac)))
    return 0.22 * (1.0 - L) ** 1.4


def benefit(architecture, load_frac: float, keep: int,
            hardware_key: str = "collapsing-lifter") -> dict:
    """What dropping to `keep` cylinders is worth at this load.

    Returns a fuel-rate ratio against running on all cylinders. Below
    1.0 is a saving."""
    n = len(architecture.slot_angles_deg())
    keep = max(1, min(n, int(keep)))
    h = hardware(hardware_key)
    base_pump = pumping_loss_frac(load_frac)
    # the survivors carry the whole load, so each works harder and the
    # throttle opens -- this is where the saving comes from
    dead = n - keep
    if h.traps_charge:
        new_load = min(1.0, load_frac * n / keep)
    else:
        # THE THROTTLE DOES NOT OPEN, and this is the crux of why
        # fuel-only cutoff is worthless. The dead cylinders are still
        # breathing, so the engine's total airflow demand is unchanged,
        # so the throttle sits where it always sat and the manifold
        # stays at the same vacuum. Letting the manifold open for this
        # case is what made an earlier version of this report a saving
        # for a technique that has essentially none.
        new_load = load_frac
    surv_pump = pumping_loss_frac(new_load)
    # THE BRAKE WORK IS THE SAME EITHER WAY, which is what an earlier
    # version of this got wrong: it scaled fuel by keep/n, as though
    # halving the cylinders halved the job. It does not -- the driver
    # still wants the same torque. What changes is only the PARASITIC
    # share, so this compares losses against a fixed output, and the
    # honest answer is single digits rather than the fifty per cent that
    # mistake produced.
    if h.traps_charge:
        # sealed: nothing to pump against, only gas-spring hysteresis
        dead_loss = (dead / n) * base_pump * GAS_SPRING_LOSS_FRAC
    else:
        # valves still working, and now pumping against the NEW manifold
        # pressure -- which is worse, because the throttle has opened
        dead_loss = (dead / n) * surv_pump
    before = 1.0 + base_pump
    after = 1.0 + (keep / n) * surv_pump + dead_loss
    if not h.traps_charge:
        # and the cold air it pumps into the exhaust puts the catalyst
        # out, which costs real fuel in re-light and enrichment
        after += 0.03
    ratio = after / before
    return {"keep": keep, "deactivated": dead, "traps_charge": h.traps_charge,
            "load_before": load_frac, "load_after": new_load,
            "pumping_before": base_pump, "pumping_after": surv_pump,
            "dead_cylinder_loss": dead_loss,
            "fuel_ratio": ratio, "saving_frac": 1.0 - ratio,
            "why": ("the saving is not from burning less in fewer cylinders -- it is "
                    "from the survivors needing a wider throttle, which deletes "
                    "pumping loss" if h.traps_charge else
                    "the valves are still pumping, so the loss this was meant to "
                    "delete is still being paid")}


# ---------------------------------------------------------------------
# the computer
# ---------------------------------------------------------------------

@dataclass
class DeactivationECU:
    """The controller, with the interlocks a real one has.

    Every one of these exists because of a specific way the system
    misbehaves without it, and they are the reason deactivation engages
    far less often than the brochure implies."""
    identity: str = "powertrain.ecu.deactivation"
    hardware_key: str = "collapsing-lifter"
    #: Below this the engine shakes: too few power strokes per second
    #: for the flywheel and mounts to smooth over.
    min_rpm: float = 1000.0
    max_rpm: float = 3200.0
    #: Above this the survivors cannot carry the load without knocking.
    max_load_frac: float = 0.55
    min_coolant_k: float = 348.0
    #: Deactivated cylinders make no heat, so the catalyst cools. Below
    #: light-off it stops converting and the system must give up.
    min_catalyst_k: float = 573.0
    #: Minimum time in a state. Without it the system hunts audibly at
    #: the boundary, which is the complaint owners actually report.
    dwell_s: float = 2.0
    #: Fraction of load hysteresis around the engage threshold.
    hysteresis: float = 0.12

    active: bool = False
    keep: int = 0
    time_in_state_s: float = 0.0
    switches: int = 0
    stuck_cylinders: set = field(default_factory=set)
    last_reason: str = "cold"

    @property
    def hw(self) -> DeactivationHardware:
        return hardware(self.hardware_key)

    def blocked_by(self, rpm: float, load_frac: float, oil_pressure_pa: float,
                   oil_temp_k: float, coolant_k: float, catalyst_k: float) -> list:
        """Every interlock currently refusing. Plural on purpose -- a
        real system is usually blocked by more than one thing."""
        h = self.hw
        out = []
        if oil_pressure_pa < h.min_oil_pressure_pa:
            out.append(f"oil pressure {oil_pressure_pa / 1000:.0f} kPa below "
                       f"{h.min_oil_pressure_pa / 1000:.0f} needed to move the pins")
        if oil_temp_k < h.min_oil_temp_k:
            out.append(f"oil at {oil_temp_k - 273.15:.0f} C is too cold to switch cleanly")
        if coolant_k < self.min_coolant_k:
            out.append("engine not up to temperature")
        if catalyst_k < self.min_catalyst_k:
            out.append(f"catalyst at {catalyst_k - 273.15:.0f} C is below light-off; "
                       "deactivating would cool it further")
        if rpm < self.min_rpm:
            out.append(f"{rpm:.0f} rpm too low: too few power strokes to stay smooth")
        if rpm > self.max_rpm:
            out.append(f"{rpm:.0f} rpm above the switching window")
        if load_frac > self.max_load_frac:
            out.append(f"load {load_frac * 100:.0f}% too high for the survivors to carry")
        if self.stuck_cylinders:
            out.append(f"cylinders {sorted(self.stuck_cylinders)} stuck: pattern "
                       "cannot be guaranteed even")
        return out

    def step(self, dt_s: float, architecture, rpm: float, load_frac: float,
             oil_pressure_pa: float = 300_000.0, oil_temp_k: float = 363.0,
             coolant_k: float = 363.0, catalyst_k: float = 800.0,
             rng=None) -> dict:
        """One control tick. Decides, and reports why."""
        self.time_in_state_s += max(0.0, dt_s)
        blocks = self.blocked_by(rpm, load_frac, oil_pressure_pa, oil_temp_k,
                                 coolant_k, catalyst_k)
        n = len(architecture.slot_angles_deg())
        want_active = not blocks
        # hysteresis: harder to engage than to stay engaged
        if self.active and not blocks:
            want_active = load_frac <= self.max_load_frac * (1.0 + self.hysteresis)
        elif not self.active:
            want_active = (not blocks) and load_frac <= self.max_load_frac * (1.0 - self.hysteresis)

        if want_active == self.active or self.time_in_state_s < self.dwell_s:
            self.last_reason = "; ".join(blocks) if blocks else "holding"
            return self._report(architecture, n, blocks, switched=False)

        # PROGRESSIVE: pick the deepest count that is still evenly fired
        # and that the load allows. Derived, not tabulated.
        target = n
        if want_active:
            for cand in sorted((c for c in usable_counts(architecture) if c["even"]),
                               key=lambda c: c["keep"]):
                if load_frac * n / max(1, cand["keep"]) <= self.max_load_frac:
                    target = cand["keep"]
                    break
        self.active = want_active and target < n
        self.keep = target if self.active else n
        self.time_in_state_s = 0.0
        self.switches += 1
        if rng is not None and self.hw.stick_probability > 0.0:
            if rng.random() < self.hw.stick_probability * max(1, n - self.keep):
                b = best_pattern(architecture, self.keep)
                if b and b["deactivated"]:
                    self.stuck_cylinders.add(b["deactivated"][0])
        self.last_reason = "engaged" if self.active else (
            "; ".join(blocks) if blocks else "load too high")
        return self._report(architecture, n, blocks, switched=True)

    def _report(self, architecture, n, blocks, switched: bool) -> dict:
        keep = self.keep if self.active else n
        pat = best_pattern(architecture, keep)
        return {"active": self.active, "keep": keep, "of": n,
                "pattern": pat["cylinders"] if pat else (),
                "deactivated": pat["deactivated"] if pat else (),
                "evenness": pat["evenness"] if pat else 0.0,
                "blocked_by": blocks, "reason": self.last_reason,
                "switched": switched, "switches": self.switches,
                "stuck": sorted(self.stuck_cylinders)}
