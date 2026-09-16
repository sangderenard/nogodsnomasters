"""What needs doing to this machine, when, and by whom.

Every filter, separator, tank and element in this plant expires. Some
clear themselves and some do not, most do BOTH -- a self-cleaning bowl
fires its sludge out on a timer for months and still has to be stripped
and scraped, because the discs coke up whether or not the discharge
works. Saying which is which is the difference between a plant that can
be run and a plant that is a diagram.

WHY THIS IS ITS OWN MODULE. A service task is not a property of a
centrifuge, or a filter, or a tank. It is the same statement every time:
something is due, it is at a place, it takes a while, it may need the
machine stopped, and it may consume a part. That is exactly the shape a
crew member's task queue wants, so it is declared once here and every
piece of equipment declares INTO it rather than inventing its own idea
of maintenance.

THE HANDOFF. `due_tasks` returns entries carrying a position and an
estimated duration, which is what a pathing and task system needs to
send somebody: go here, spend this long, the machine must/need not be
running. This module does not schedule anybody -- it says what is owed.

THREE WAYS A THING BECOMES DUE, and they are genuinely different:

    hours        a calendar or running-hours interval. An oil change,
                 an element swap, an annual strip-down.
    full         a vessel or bowl reaching its capacity. A sludge tank
                 does not care how long it took to fill.
    condition    a measured state crossing a limit -- blocked fraction,
                 soot percentage, wall thickness. The honest one,
                 because it is the actual reason the other two exist.

An auto-draining unit still carries its manual task; what the auto
drain buys is a much longer interval on it, not the absence of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HOURS = "hours"
FULL = "full"
CONDITION = "condition"


@dataclass(frozen=True)
class ServiceTask:
    """One thing that has to be done to one kind of equipment."""
    key: str
    label: str
    trigger: str                  # HOURS | FULL | CONDITION
    interval_hours: float = 0.0   # HOURS
    threshold_frac: float = 0.0   # FULL / CONDITION
    requires_shutdown: bool = False
    minutes: float = 15.0
    consumes: str = ""            # a part that has to be carried to it
    skill: str = "mechanic"
    why: str = ""


#: Every task the plant's fluid equipment can owe. Declared here so a
#: crew system reads one vocabulary instead of a dozen.
TASKS: dict[str, ServiceTask] = {
    "bowl-strip": ServiceTask(
        "bowl-strip", "strip and clean the separator bowl", HOURS,
        interval_hours=2000.0, requires_shutdown=True, minutes=180.0,
        skill="mechanic",
        why="a self-cleaning bowl ejects its sludge and still cokes up between the "
            "discs; the discharge mechanism buys a long interval, not exemption"),
    "bowl-scrape": ServiceTask(
        "bowl-scrape", "stop, open and scrape the bowl", FULL,
        threshold_frac=0.90, requires_shutdown=True, minutes=45.0,
        why="a solid bowl has no discharge at all: when it is full it stops separating "
            "and the only cure is somebody opening it"),
    "spinner-rotor": ServiceTask(
        "spinner-rotor", "pull and scrape the spinner rotor", FULL,
        threshold_frac=0.85, requires_shutdown=True, minutes=20.0,
        why="the truck-engine answer, and the classic thing nobody does until the "
            "rotor is a solid cake and has quietly stopped turning"),
    "sludge-pump-out": ServiceTask(
        "sludge-pump-out", "pump out the sludge receiver", FULL,
        threshold_frac=0.85, requires_shutdown=False, minutes=40.0, skill="operator",
        why="the bowls keep ejecting into it whether or not there is room, so a full "
            "tank is not a stopped process -- it is an overflowing one"),
    "element-swap": ServiceTask(
        "element-swap", "replace the filter element", CONDITION,
        threshold_frac=0.80, requires_shutdown=True, minutes=15.0,
        consumes="filter-element",
        why="past about eighty percent blocked the bypass is open and the element is "
            "filtering nothing at all"),
    "cyclone-apex": ServiceTask(
        "cyclone-apex", "rod the cyclone underflow apexes", HOURS,
        interval_hours=750.0, requires_shutdown=False, minutes=30.0,
        why="a cyclone has no moving parts to wear and one hole that blocks: the apex. "
            "A blocked apex sends the heavy phase out of the overflow and the bank "
            "goes on looking like it is working"),
    "auto-drain-check": ServiceTask(
        "auto-drain-check", "check the auto-drain actually fires", HOURS,
        interval_hours=500.0, requires_shutdown=False, minutes=10.0, skill="operator",
        why="an auto drain that has stopped draining is worse than none, because "
            "nobody is watching the thing it was fitted to make unnecessary"),
    "oil-change": ServiceTask(
        "oil-change", "drain and refill the oil charge", CONDITION,
        threshold_frac=0.05, requires_shutdown=True, minutes=60.0,
        consumes="oil-charge",
        why="soot past about five percent by mass condemns the charge whatever the "
            "wear metals say"),
}


@dataclass(frozen=True)
class ServiceProfile:
    """What a class of equipment owes, and whether it clears itself."""
    auto_drain: bool
    tasks: tuple
    why: str = ""


#: Equipment class -> what it owes. The `auto_drain` flag is the direct
#: answer to "does it clear itself or does somebody have to": a unit
#: with one still carries its manual tasks, on a longer interval.
PROFILES: dict[str, ServiceProfile] = {
    "disc-stack-self-cleaning": ServiceProfile(
        auto_drain=True, tasks=("bowl-strip", "auto-drain-check"),
        why="ejects on its own timer for months, and still has to be stripped"),
    "disc-stack-nozzle": ServiceProfile(
        auto_drain=True, tasks=("bowl-strip", "auto-drain-check"),
        why="discharges continuously through its nozzles, which are also what block"),
    "solid-bowl-batch": ServiceProfile(
        auto_drain=False, tasks=("bowl-scrape",),
        why="no discharge mechanism of any kind: it is entirely dependent on somebody"),
    "bypass-spinner": ServiceProfile(
        auto_drain=False, tasks=("spinner-rotor",),
        why="the same, on a smaller and more frequently forgotten scale"),
    "cyclone-bank": ServiceProfile(
        auto_drain=True, tasks=("cyclone-apex",),
        why="nothing moves and nothing wears; the apexes block"),
    "sludge-receiver": ServiceProfile(
        auto_drain=False, tasks=("sludge-pump-out",),
        why="the end of every discharge in the plant, and a hard limit"),
    "filter-element": ServiceProfile(
        auto_drain=False, tasks=("element-swap",),
        why="it blocks BECAUSE it is working"),
    "oil-charge": ServiceProfile(
        auto_drain=False, tasks=("oil-change",),
        why="the fluid itself expires"),
}


@dataclass
class ServiceItem:
    """One real piece of equipment, and how overdue it is.

    `hours` is its own running time since the last service, and
    `condition` is whatever fraction its trigger reads -- blocked
    fraction for an element, full fraction for a bowl or a tank, soot
    fraction for a charge."""
    identity: str
    profile: str
    position: tuple = (0.0, 0.0, 0.0)
    hours: float = 0.0
    condition: float = 0.0
    label: str = ""

    @property
    def spec(self) -> ServiceProfile:
        p = PROFILES.get(self.profile)
        if p is None:
            raise KeyError(
                f"no service profile for {self.profile!r}; declared: {', '.join(sorted(PROFILES))}")
        return p

    @property
    def auto_drain(self) -> bool:
        return self.spec.auto_drain

    def overdue_frac(self, task: ServiceTask) -> float:
        """How far past due, 1.0 being exactly due."""
        if task.trigger == HOURS:
            return 0.0 if task.interval_hours <= 0.0 else self.hours / task.interval_hours
        return 0.0 if task.threshold_frac <= 0.0 else self.condition / task.threshold_frac

    def due(self) -> list:
        out = []
        for key in self.spec.tasks:
            t = TASKS[key]
            f = self.overdue_frac(t)
            if f >= 1.0:
                out.append((t, f))
        return out

    def serviced(self, task_key: str) -> None:
        """Mark one task done: an hours task resets the clock, a full or
        condition task is cleared by whatever actually emptied it."""
        t = TASKS.get(task_key)
        if t is None:
            return
        if t.trigger == HOURS:
            self.hours = 0.0
        else:
            self.condition = 0.0


def due_tasks(items, now_hours: float = 0.0) -> list:
    """Everything the plant owes right now, worst first.

    Each entry is what a crew task queue needs and nothing it does not:
    where to go, what to do, roughly how long, whether the machine has
    to be stopped first, and what part to bring."""
    out = []
    for item in items:
        for task, frac in item.due():
            out.append({
                "identity": item.identity,
                "label": item.label or item.identity,
                "task": task.key,
                "action": task.label,
                "position": tuple(item.position),
                "minutes": task.minutes,
                "requires_shutdown": task.requires_shutdown,
                "consumes": task.consumes,
                "skill": task.skill,
                "overdue_frac": frac,
                "auto_drain": item.auto_drain,
            })
    out.sort(key=lambda e: -e["overdue_frac"])
    return out


def workload_minutes(entries) -> float:
    """Total time the outstanding list represents -- the number that
    says whether a plant is being maintained or merely inspected."""
    return sum(float(e["minutes"]) for e in entries)


# ---------------------------------------------------------------------
# WANTS: ONE LIST FOR EVERYTHING THAT NEEDS SOMETHING
# ---------------------------------------------------------------------
#
# A service task is a thing to DO. A need is a thing to BRING, and the
# plant is full of them: a barrel wants filling, and wants filling with
# a particular fuel; an air filter wants an element; a desiccant bed
# wants regenerating or replacing; a diffuser wants its media changed;
# an oil charge wants changing because it is fouled. Those are the same
# statement -- somewhere, something, this much of it -- and an AI crew
# should poll ONE list rather than knowing about barrels and desiccants
# and filters separately.
#
# MATCHING IS PART OF THE NEED. A barrel does not want "fuel", it wants
# the grade the plant is set up to burn, and a fuel control unit set to
# a different one is a real mismatch rather than a detail. So a need
# carries `matches`: what would actually satisfy it. A supply that does
# not match is not a supply.


@dataclass
class Need:
    """Something, somewhere, wants something brought to it."""
    identity: str
    want: str                     # "fuel" | "element" | "regeneration" | "oil-charge" | ...
    position: tuple = (0.0, 0.0, 0.0)
    quantity: float = 0.0
    unit: str = ""
    matches: str = ""             # the grade/spec that would satisfy it
    urgency: float = 1.0          # >= 1.0 is due; higher is worse
    minutes: float = 10.0
    skill: str = "operator"
    label: str = ""
    why: str = ""

    @property
    def due(self) -> bool:
        return self.urgency >= 1.0


def barrel_need(identity: str, position, fill_frac: float, capacity_l: float,
                grade: str, refill_below: float = 0.35) -> "Need | None":
    """A fuel barrel that wants filling, and says with what.

    The grade travels with the need so a fuel control unit set to
    something else is caught as a mismatch instead of being quietly
    filled with the wrong thing."""
    if fill_frac >= refill_below:
        return None
    return Need(identity=identity, want="fuel", position=tuple(position),
                quantity=capacity_l * (1.0 - fill_frac), unit="L",
                matches=grade, urgency=refill_below / max(1e-6, fill_frac),
                minutes=20.0, label=f"{identity} ({grade})",
                why="a barrel below about a third is a barrel that will run a plant dry "
                    "during the one shift nobody is watching it")


def element_need(identity: str, position, blocked_frac: float,
                 element: str = "filter-element", threshold: float = 0.80) -> "Need | None":
    """Any element that has stopped filtering: air, oil, fuel, or the
    diffuser media in an air-treatment stage, which blocks like anything
    else and is the one people forget is a consumable."""
    if blocked_frac < threshold:
        return None
    return Need(identity=identity, want="element", position=tuple(position),
                quantity=1.0, unit="each", matches=element,
                urgency=blocked_frac / threshold, minutes=15.0, skill="mechanic",
                label=identity,
                why="past about eighty percent blocked the bypass is open and the "
                    "element is filtering nothing at all")


def desiccant_need(identity: str, position, saturation_frac: float,
                   regenerable: bool = True, threshold: float = 0.85) -> "Need | None":
    """A desiccant bed that has taken all the water it can hold.

    A regenerable bed wants heat and a purge; a disposable one wants
    replacing. Either way it has stopped drying, and a bed that has
    stopped drying is passing wet air into everything downstream."""
    if saturation_frac < threshold:
        return None
    return Need(identity=identity, want="regeneration" if regenerable else "element",
                position=tuple(position), quantity=1.0, unit="each",
                matches="desiccant-bed", urgency=saturation_frac / threshold,
                minutes=45.0 if regenerable else 25.0, skill="mechanic", label=identity,
                why="a saturated bed does not fail loudly -- it simply stops taking "
                    "water out, and everything downstream starts getting it")


def fouled_charge_need(identity: str, position, fluid: str, fouled_frac: float,
                       capacity_l: float, grade: str = "", threshold: float = 0.05
                       ) -> "Need | None":
    """An oil, coolant or fuel charge that is too dirty to keep using."""
    if fouled_frac < threshold:
        return None
    return Need(identity=identity, want=f"{fluid}-charge", position=tuple(position),
                quantity=capacity_l, unit="L", matches=grade or fluid,
                urgency=fouled_frac / threshold, minutes=60.0, skill="mechanic",
                label=f"{identity} ({fluid})",
                why="fouling past the condemning limit is the fluid asking to be "
                    "replaced regardless of how long it has been in")


def poll_needs(*groups) -> list:
    """Gather every outstanding want into one list, worst first.

    Takes any number of iterables of `Need` (or None, which is what the
    helpers return when nothing is wanted), so a caller assembles the
    plant's wants by handing over whatever it has."""
    out: list = []
    for group in groups:
        for n in (group or ()):
            if n is not None and n.due:
                out.append(n)
    out.sort(key=lambda n: -n.urgency)
    return out


def unmatched(needs, available: dict) -> list:
    """Needs the stores cannot currently satisfy.

    `available` maps a spec to how much of it is on hand. This is where
    a fuel control unit set to one grade and barrels holding another
    shows up as a real problem rather than an assumption."""
    short = []
    for n in needs:
        have = float(available.get(n.matches, 0.0))
        if have < n.quantity:
            short.append((n, have))
    return short


# ---------------------------------------------------------------------
# TOUCH JOBS: somebody has to go and hold it
# ---------------------------------------------------------------------
#
# Emptying a condensate pot, pumping out a sludge tank, scraping a bowl,
# rodding a cyclone apex, laying a hose section -- every clearing task
# in this plant is the same interaction. Somebody walks to a place,
# takes hold of something, and STAYS THERE for a while; the thing clears
# when they finish, not when they arrive.
#
# That delay is the whole point and it is why this is a primitive rather
# than a flag. A job that completes on touch is a button; a job that
# takes forty minutes is a commitment -- the crew member is somewhere
# else for forty minutes, they can be interrupted, and whatever else
# needed them is waiting. It is also what makes a plant under attack
# feel different from a plant at rest: the work does not stop being
# owed, it just stops being done.
#
# Anything that can be cleared produces one of these, so an interaction
# system has exactly one kind of job to path to and run.


@dataclass
class TouchJob:
    """A held interaction at a place, that finishes after a delay."""
    identity: str
    action: str
    position: tuple = (0.0, 0.0, 0.0)
    seconds: float = 60.0
    elapsed: float = 0.0
    skill: str = "operator"
    consumes: str = ""
    requires_shutdown: bool = False
    interruptible: bool = True
    clears: str = ""              # what this job resolves when it finishes
    label: str = ""

    @property
    def progress(self) -> float:
        return 0.0 if self.seconds <= 0.0 else min(1.0, self.elapsed / self.seconds)

    @property
    def complete(self) -> bool:
        return self.progress >= 1.0

    def work(self, dt: float) -> bool:
        """Spend `dt` of somebody's time on it. True once it is done."""
        self.elapsed = min(self.seconds, self.elapsed + max(0.0, float(dt)))
        return self.complete

    def interrupt(self) -> float:
        """Walk away. An interruptible job keeps its progress -- a bowl
        half scraped is still half scraped -- and one that is not starts
        again, because a job you cannot leave part-done is exactly the
        kind you cannot leave part-done."""
        if self.interruptible:
            return self.progress
        self.elapsed = 0.0
        return 0.0


def touch_job_for_task(item: "ServiceItem", task: ServiceTask) -> TouchJob:
    """The held interaction that discharges one service task."""
    return TouchJob(
        identity=f"{item.identity}:{task.key}", action=task.label,
        position=tuple(item.position), seconds=task.minutes * 60.0,
        skill=task.skill, consumes=task.consumes,
        requires_shutdown=task.requires_shutdown,
        # a machine that had to be stopped for this should not be left
        # half-open with somebody wandering off
        interruptible=not task.requires_shutdown,
        clears=task.key, label=item.label or item.identity)


def touch_job_for_need(need: "Need") -> TouchJob:
    """The held interaction that satisfies one need."""
    return TouchJob(
        identity=f"{need.identity}:{need.want}", action=f"{need.want} {need.identity}",
        position=tuple(need.position), seconds=need.minutes * 60.0,
        skill=need.skill, consumes=need.matches,
        requires_shutdown=False, interruptible=True,
        clears=need.want, label=need.label or need.identity)


def touch_jobs(items=(), needs=()) -> list:
    """Every outstanding clearing task as one list of held interactions.

    This is the whole handoff to an interaction and pathing system: one
    kind of object, each with a place, a duration and what it resolves."""
    jobs: list = []
    for item in items or ():
        for task, _frac in item.due():
            jobs.append(touch_job_for_task(item, task))
    for need in needs or ():
        if need is not None and need.due:
            jobs.append(touch_job_for_need(need))
    jobs.sort(key=lambda j: -j.seconds)
    return jobs
