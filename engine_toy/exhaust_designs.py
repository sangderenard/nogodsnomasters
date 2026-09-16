"""Crossovers, and why a V8 needs one and a Ferrari does not.

An H-pipe or an X-pipe looks like an accessory and is actually a fix for
a problem the CRANKSHAFT creates. That is the whole story, and it is
derivable rather than assertable.

WHERE THE PROBLEM COMES FROM. A cross-plane V8 -- the American one, with
its counterweighted crank and its rumble -- fires evenly as an engine
and unevenly as two banks. Its eight events are ninety degrees apart,
but they do not alternate neatly between the sides. One bank ends up
firing at 0, 270, 450 and 540 degrees: gaps of 270, 180, 90 and 180.
That bank's exhaust pipe gets two pulses almost on top of each other and
then a long silence.

A flat-plane V8 -- the Ferrari one -- splits into two even four-cylinder
banks, 180 degrees apart each side, and has no such problem. It also has
no rumble, and those two facts are the same fact. The lumpy American V8
noise IS the uneven bank firing, heard directly.

WHAT A CROSSOVER DOES ABOUT IT. Connect the two banks' pipes and the
bank that is mid-silence lends its empty pipe to the bank that has two
pulses queued. A pulse arriving at a junction sees the pipe area
suddenly increase, and a sudden area increase reflects an EXPANSION
wave -- a negative pressure pulse -- back the way it came. Timed right
that suction arrives at a cylinder during valve overlap and helps pull
the last of the burnt gas out. That is scavenging, and it is free power.

  H-PIPE   a balance tube between the two pipes. Partial connection, so
           the banks stay mostly separate and the low-frequency beat of
           the uneven firing survives. This is the burble people fit
           them for, and it is a real pressure equalisation as well.

  X-PIPE   the pipes actually merge and cross. A far more complete
           connection: better scavenging, more top end, and a smoother
           higher-pitched note because the bank asymmetry is largely
           cancelled. People who want the rumble are disappointed by
           exactly the thing that makes it work better.

  Y-MERGE  both banks into one pipe and onward. Simple, cheap, and it
           gives up the dual path entirely -- fine on a small engine,
           a restriction on a big one.

AND ON AN INLINE ENGINE NONE OF IT APPLIES, because there is only one
bank. An inline six is already the smoothest firing arrangement there
is; there is nothing to balance against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Crossover:
    key: str
    label: str
    #: How completely the two banks are connected, 0..1. This is the one
    #: number that sets both the gain and the loss of character.
    coupling: float
    #: Fraction of the main pipe diameter.
    diameter_frac: float
    #: Power gain available IF the banks are uneven. Multiplied by the
    #: actual unevenness, so a flat-plane V8 gets nothing from it.
    peak_gain_frac: float
    #: How much of the low-frequency bank beat survives. Lower is
    #: smoother and higher-pitched.
    rumble_retained: float
    era: str = ""
    why: str = ""


CROSSOVERS: dict[str, Crossover] = {
    "none": Crossover(
        "none", "true duals, no crossover", 0.0, 0.0, 0.0, 1.0, "always",
        why="two entirely separate systems. Simplest, loudest at low frequency, and it "
            "leaves every bit of the bank asymmetry in place -- which is either the "
            "point or the problem depending on who fitted it"),
    "balance-tube": Crossover(
        "balance-tube", "small balance tube", 0.25, 0.45, 0.010, 0.92, "1950s",
        why="a modest pipe between the two systems, originally fitted to even out "
            "back-pressure rather than to find power. The oldest version of this idea "
            "and the least ambitious"),
    "h-pipe": Crossover(
        "h-pipe", "H-pipe", 0.55, 0.70, 0.022, 0.80, "1960s",
        why="the classic: a full-diameter crossover a third of the way back. Real "
            "scavenging help and the characteristic burble kept intact, which is why "
            "it never went away even after the X-pipe proved better on paper"),
    "x-pipe": Crossover(
        "x-pipe", "X-pipe", 0.90, 1.0, 0.035, 0.35, "1990s",
        why="the two pipes merge into a crossed junction and separate again. The most "
            "complete coupling short of a single pipe: the best scavenging of these, "
            "and it trades the rumble for a smoother, higher note"),
    "y-merge": Crossover(
        "y-merge", "single Y merge", 1.0, 1.0, 0.015, 0.25, "always",
        why="both banks into one pipe. Total coupling and a single path, so it is only "
            "as good as that one pipe is big -- on anything with real airflow it "
            "becomes the restriction itself"),
}


def crossover(key: str) -> Crossover:
    c = CROSSOVERS.get(str(key))
    if c is None:
        raise KeyError(f"unknown crossover {key!r}; declared: "
                       f"{', '.join(sorted(CROSSOVERS))}")
    return c


def bank_of(site_z: float) -> int:
    """Which bank a cylinder is on, from its own position. Sign of z."""
    return 0 if float(site_z) >= 0.0 else 1


def bank_firing_angles(architecture, site_zs) -> list:
    """Firing angles split into the two banks.

    Reads the architecture's own slot_angles_deg -- the single place
    firing timing is resolved -- so the exhaust cannot disagree with the
    crank, the combustion scheduler or the acoustics."""
    angles = list(architecture.slot_angles_deg())
    order = list(getattr(architecture, "firing_order", range(1, len(angles) + 1)))
    zs = list(site_zs)
    out = {0: [], 1: []}
    for slot, ang in enumerate(angles):
        cyl = int(order[slot])
        if 1 <= cyl <= len(zs):
            out[bank_of(zs[cyl - 1])].append(float(ang))
    return [sorted(out[0]), sorted(out[1])]


def bank_unevenness(architecture, site_zs) -> float:
    """How badly the banks fire unevenly, 0 = perfectly even.

    THE NUMBER THAT DECIDES WHETHER A CROSSOVER IS WORTH FITTING, and it
    comes from the crankshaft. Reuses the same evenness measure the
    cylinder-deactivation pattern search uses, because it is the same
    question asked of a different subset."""
    import cylinder_deactivation as cd
    cycle = float(getattr(architecture, "cycle_degrees", 720.0))
    banks = bank_firing_angles(architecture, site_zs)
    vals = [cd.firing_evenness(b, cycle) for b in banks if len(b) > 1]
    return max(vals) if vals else 0.0


@dataclass
class ExhaustDesign:
    """A crossover fitted to a particular engine, with real consequences."""
    crossover_key: str = "none"
    main_diameter_m: float = 0.063
    #: Distance back from the collectors, as a fraction of the system
    #: length. A crossover works at a pressure node and there is a best
    #: place for it -- roughly a third of the way back.
    position_frac: float = 0.33
    banks: int = 2

    @property
    def spec(self) -> Crossover:
        return crossover(self.crossover_key)

    @property
    def diameter_m(self) -> float:
        return self.main_diameter_m * self.spec.diameter_frac

    def placement_quality(self) -> float:
        """How well sited it is. Falls off either side of a third back."""
        d = abs(self.position_frac - 0.33)
        return max(0.25, 1.0 - (d / 0.33) ** 2)

    def power_gain_frac(self, unevenness: float) -> float:
        """What it is actually worth on THIS engine.

        Zero on an inline. Zero on a flat-plane V8. Real on a
        cross-plane V8, and that is the whole point -- the part does not
        carry a power figure, the engine does."""
        if self.banks < 2:
            return 0.0
        u = max(0.0, min(1.0, float(unevenness) / 0.45))
        return self.spec.peak_gain_frac * u * self.placement_quality()

    def sound(self, unevenness: float) -> dict:
        """What it does to the note.

        The rumble IS the bank asymmetry. A crossover that fixes the
        asymmetry necessarily removes the rumble, which is why the
        best-flowing option is the one enthusiasts argue about."""
        u = max(0.0, min(1.0, float(unevenness) / 0.45))
        beat = u * self.spec.rumble_retained
        return {"low_frequency_beat": beat,
                "character": ("flat and even" if u < 0.05 else
                              "rumble" if beat > 0.5 else
                              "smoothed, higher pitched" if beat > 0.2 else
                              "near-symmetric howl"),
                "why": ("the low beat is the two banks firing unevenly, heard directly. "
                        "Coupling them evens the pressure and quietens the beat in the "
                        "same stroke -- you cannot keep one and lose the other")}

    def report(self, architecture=None, site_zs=None,
               unevenness: float | None = None) -> dict:
        u = (unevenness if unevenness is not None
             else (bank_unevenness(architecture, site_zs)
                   if architecture is not None and site_zs is not None else 0.0))
        return {"crossover": self.spec.label, "era": self.spec.era,
                "coupling": self.spec.coupling,
                "diameter_mm": self.diameter_m * 1000.0,
                "bank_unevenness": u,
                "placement_quality": self.placement_quality(),
                "power_gain_frac": self.power_gain_frac(u),
                "sound": self.sound(u)}


def recommend(architecture, site_zs) -> dict:
    """What this engine actually wants, and why.

    Refuses to recommend hardware to an engine that cannot use it, which
    is most of the value -- an inline six with an X-pipe has bought a
    weld and nothing else."""
    banks = int(getattr(architecture, "banks", 1) or 1)
    # A RADIAL HAS NO BANKS. EngineArchitecture repurposes `banks` as
    # cylinders-per-row for radials and rotaries (it says so at its own
    # declaration), so reading it as a bank count here reported a
    # nine-cylinder Wasp as a nine-bank engine and cheerfully
    # recommended it an X-pipe. A radial's cylinders are arranged by
    # ANGLE around one crank throw, not split into two sides, and its
    # exhaust is a collector ring -- a different problem entirely.
    if getattr(architecture, "radial", False) or getattr(architecture, "rotary", False):
        return {"fit": "none", "bank_unevenness": 0.0,
                "why": "a radial has cylinders around a circle, not two banks facing "
                       "each other. There is no pair of pipes to cross over; what it "
                       "has is a collector ring, which is a different design question"}
    u = bank_unevenness(architecture, site_zs)
    if banks < 2:
        return {"fit": "none", "bank_unevenness": u,
                "why": "one bank. There is nothing to cross over TO, and an inline "
                       "six is already the most evenly firing arrangement there is"}
    if u < 0.05:
        return {"fit": "none", "bank_unevenness": u,
                "why": "the banks already fire evenly -- a flat-plane crank splits into "
                       "two even halves. A crossover would equalise pressures that are "
                       "already equal, and cost a restriction to do it"}
    best, best_gain = "none", 0.0
    for k in CROSSOVERS:
        g = ExhaustDesign(crossover_key=k, banks=banks).power_gain_frac(u)
        if g > best_gain:
            best, best_gain = k, g
    return {"fit": best, "bank_unevenness": u, "gain_frac": best_gain,
            "why": (f"the banks fire unevenly ({u:.2f}), which is what a crossover is "
                    "for. This is a consequence of the crankshaft, not of the exhaust")}


# ---------------------------------------------------------------------
# CUTOUTS: the straight pipe dump
# ---------------------------------------------------------------------
#
# A cutout is a hole in the pipe, close to the collector, with something
# over it. Open it and the gas takes the short way out; everything
# downstream -- catalyst, muffler, tailpipe, the whole tuned system --
# is simply bypassed.
#
# IT IS NOT FREE POWER, and the reason is the interesting part. A good
# exhaust system is not merely a hole for gas to leave through, it is a
# set of tuned pipes doing the same job the intake runners do: the
# collector's area step sends an expansion wave back up the primary and
# helps drag the last of the burnt charge out during valve overlap.
# Dump straight out of the collector and that tuning goes with it.
#
# So a cutout gives back the restriction and takes away the scavenging,
# and which wins depends entirely on engine speed. Low down, the tuned
# system is helping more than it is restricting and a dump LOSES
# torque -- the reason an open-piped car feels flat pulling away and
# then comes alive. High up, the restriction dominates and the dump
# wins. A restrictive stock system moves that crossover down; a good
# aftermarket one can push it past the redline, at which point a cutout
# is pure noise with a power penalty attached.
#
# THE NOISE IS THE POINT ANYWAY. Without a muffler an engine radiates
# most of its acoustic energy straight out, and it is not a matter of
# taste at 110 dB -- it is the difference between a vehicle that can
# move quietly and one whose position is audible for a mile. On this
# station that is not a styling decision.

@dataclass(frozen=True)
class CutoutKind:
    key: str
    label: str
    #: How much of the flow it can divert when fully open.
    max_open_frac: float
    #: Can it be worked from the seat, or does it need spanners?
    remote: bool
    #: Seconds to change state. A bolt-on cap is a job, not a switch.
    actuate_s: float
    leak_frac: float          # what escapes past it when shut
    era: str = ""
    why: str = ""


CUTOUTS: dict[str, CutoutKind] = {
    "none": CutoutKind("none", "no cutout", 0.0, False, 0.0, 0.0, "always",
                       why="a sealed system. Quiet, tuned, and whatever restriction it "
                           "has is the restriction you live with"),
    "bolt-on-cap": CutoutKind(
        "bolt-on-cap", "bolted cap over a collector dump", 0.95, False, 600.0, 0.02,
        "1930s",
        why="a flanged hole in the collector with a plate bolted over it. The original "
            "hot-rod answer: ten minutes with a spanner at the strip and ten minutes "
            "more before driving home. The plate warps and the gasket burns, so it "
            "never quite seals again after the first few times"),
    "cable": CutoutKind(
        "cable", "cable-operated cutout", 0.85, True, 2.0, 0.05, "1950s",
        why="a butterfly in the dump pipe on a pull cable to the cabin. Workable from "
            "the seat, which is the whole appeal, and the cable stretches and the "
            "butterfly cokes up in its bore until it will not shut fully"),
    "electric": CutoutKind(
        "electric", "electric cutout", 0.90, True, 1.5, 0.03, "1990s",
        why="a geared motor on the butterfly. Reliable until you remember the motor "
            "lives under the car in the heat and the salt, which is where this lives"),
    "vacuum": CutoutKind(
        "vacuum", "vacuum-actuated cutout", 0.80, True, 0.8, 0.04, "1970s",
        why="an actuator off manifold vacuum. Fast and simple, and it defaults to "
            "whatever position it fails to -- which on a split diaphragm is usually "
            "open, and loudly"),
    "dyno-bench": CutoutKind(
        "dyno-bench", "bench fitting (no cutout hardware)", 1.0, True, 0.0, 0.0,
        "always",
        why="not a part. A neutral fitting used to express how much of the system is "
            "present on a test bench, where the answer is set by what was bolted on "
            "rather than by a valve. It leaks nothing when shut, which every real "
            "cutout does -- routing a bench configuration through lake-pipes instead "
            "pinned it fully open, because lake pipes ARE fully open by definition"),
    "lake-pipes": CutoutKind(
        "lake-pipes", "open lake pipes", 1.0, False, 0.0, 1.0, "1940s",
        why="side pipes straight off the collectors with no system at all. Not a "
            "cutout so much as the absence of an exhaust, kept for how it looks and "
            "sounds and paid for in every other way"),
}


def cutout_kind(key: str) -> CutoutKind:
    c = CUTOUTS.get(str(key))
    if c is None:
        raise KeyError(f"unknown cutout {key!r}; declared: {', '.join(sorted(CUTOUTS))}")
    return c


#: How much of an engine's tuned scavenging comes from the system
#: DOWNSTREAM of the collector. Dumping at the collector keeps the
#: primaries and the collector's own area step, and loses the rest.
DOWNSTREAM_SCAVENGE_SHARE = 0.35


@dataclass
class ExhaustCutout:
    identity: str = "powertrain.exhaust_cutout"
    kind: str = "electric"
    open_frac: float = 0.0
    #: Back-pressure of the full system at rated flow. What the dump is
    #: bypassing, and therefore what it is worth.
    system_backpressure_pa: float = 12_000.0
    #: Where the tuned system stops helping and starts costing. Below
    #: this a dump loses torque; above it, it gains.
    crossover_rpm: float = 3800.0
    cap_warped: float = 0.0

    @property
    def spec(self) -> CutoutKind:
        return cutout_kind(self.kind)

    @property
    def effective_open(self) -> float:
        base = max(0.0, min(1.0, self.open_frac)) * self.spec.max_open_frac
        leak = self.spec.leak_frac + self.cap_warped * 0.25
        return max(0.0, min(1.0, max(base, leak)))

    def backpressure_pa(self) -> float:
        """What the engine still has to push against."""
        return self.system_backpressure_pa * (1.0 - self.effective_open) ** 1.5

    def power_frac(self, rpm: float) -> float:
        """Change in power at this speed. Below 1.0 it is costing you.

        Two terms pulling opposite ways: restriction removed (a gain,
        growing with rpm) against tuned scavenging lost (a loss, worst
        where the system was tuned to help). Where they cross is the
        whole character of an open pipe."""
        o = self.effective_open
        if o <= 0.0:
            return 1.0
        relief = (self.system_backpressure_pa / 101_325.0) * o * \
            min(1.6, (float(rpm) / max(1.0, self.crossover_rpm)) ** 1.5)
        lost = DOWNSTREAM_SCAVENGE_SHARE * o * 0.06 * \
            math.exp(-((float(rpm) - self.crossover_rpm * 0.55)
                       / max(1.0, self.crossover_rpm * 0.5)) ** 2)
        return 1.0 + relief - lost

    def noise_db(self, base_db: float = 78.0) -> float:
        """Straight-piped, an engine radiates almost everything."""
        return base_db + 32.0 * self.effective_open ** 0.6

    def report(self, rpm: float) -> dict:
        return {"kind": self.spec.label, "era": self.spec.era,
                "open": self.effective_open,
                "backpressure_kpa": self.backpressure_pa() / 1000.0,
                "power_frac": self.power_frac(rpm),
                "noise_db": self.noise_db(),
                "catalyst_bypassed": self.effective_open > 0.2,
                "why": ("below the crossover the tuned system was helping more than it "
                        "was restricting, so opening the dump costs torque; above it "
                        "the restriction dominates and the dump pays")}


# ---------------------------------------------------------------------
# MOUNTING: what holds it on, and why it breaks
# ---------------------------------------------------------------------
#
# An exhaust is bolted rigidly to an engine that is deliberately allowed
# to move on rubber mounts, and hung from a chassis that is not. Those
# two facts are irreconcilable, and every exhaust mounting decision is
# about managing the consequence.
#
# The FLEX JOINT is the admission. A braided bellows in the downpipe
# lets the engine rock on its mounts without tearing the system off the
# vehicle. Delete it -- as people do when they fabricate -- and the
# movement has to go somewhere, which is a cracked collector weld within
# months.
#
# The HANGERS are rubber for the same reason: a hard-mounted exhaust
# puts every pulse into the body as drumming. Rubber perishes, and a
# system with a broken hanger hangs on its remaining ones and on the
# flex joint, which then does a job it was not sized for.
#
# And the MANIFOLD STUDS are the head gasket problem again, in a smaller
# and much hotter place: a cast iron manifold bolted to an aluminium
# head grows at half the rate of what it is bolted to, and the studs
# take that difference every heat cycle until they fatigue and snap.
# Always the end ones, because that is where the accumulated difference
# is greatest.

@dataclass
class ExhaustMounting:
    identity: str = "powertrain.exhaust_mounting"
    hangers: int = 4
    hangers_perished: int = 0
    flex_joint: bool = True
    flex_cycles: float = 0.0
    manifold_studs: int = 12
    studs_broken: int = 0
    manifold_material: str = "cast-iron"
    head_material: str = "aluminium"
    manifold_span_m: float = 0.55
    heat_cycles: float = 0.0

    @property
    def supported(self) -> bool:
        return (self.hangers - self.hangers_perished) >= max(2, self.hangers // 2)

    def stud_load_per_cycle(self, delta_k: float = 380.0) -> float:
        """How far the manifold slides across the head when it heats.

        Straight from seals.differential_growth_m -- the same relation
        that scrubs a head gasket, applied to a hotter joint with fewer
        fasteners. An exhaust manifold sees a far bigger temperature
        swing than the head does, which is why its studs break and the
        head bolts do not."""
        import seals
        return seals.differential_growth_m(self.manifold_material,
                                           self.head_material,
                                           self.manifold_span_m, delta_k)

    def step(self, dt_s: float, heat_cycle: bool = False,
             engine_rocking_m: float = 0.004, delta_k: float = 380.0) -> dict:
        broke = []
        if heat_cycle:
            self.heat_cycles += 1.0
            slide = self.stud_load_per_cycle(delta_k)
            # CALIBRATED SO STUDS BREAK WHEN STUDS BREAK. At 2.2e-6 an
            # alloy head survived two hundred thousand heat cycles --
            # centuries of starting the engine twice a day. Real cast
            # manifolds on alloy heads start shedding their end studs in
            # a handful of years, so the first one goes around four
            # thousand cycles, and the constant follows from that.
            rate = (slide * 1000.0) ** 2 * 4.5e-5
            if self.studs_broken < self.manifold_studs and \
                    self.heat_cycles * rate > (self.studs_broken + 1):
                self.studs_broken += 1
                broke.append(f"manifold stud {self.studs_broken}")
        if self.flex_joint:
            self.flex_cycles += max(0.0, dt_s) * 0.5
        else:
            self.flex_cycles += max(0.0, dt_s) * 6.0 * (engine_rocking_m / 0.004)
        leaking = self.studs_broken >= max(2, self.manifold_studs // 5)
        return {"studs_broken": self.studs_broken, "of": self.manifold_studs,
                "manifold_leaking": leaking,
                "slide_mm": self.stud_load_per_cycle(delta_k) * 1000.0,
                "supported": self.supported, "flex_joint": self.flex_joint,
                "weld_fatigue": self.flex_cycles, "broke": broke,
                "why": ("no flex joint: the engine's own rocking is going into the "
                        "collector welds instead" if not self.flex_joint else
                        "the flex joint is taking the engine's movement, which is what "
                        "it is for")}

    def need(self):
        import servicing as sv
        if self.studs_broken == 0 and self.supported and self.flex_joint:
            return None
        what = []
        if self.studs_broken:
            what.append(f"{self.studs_broken} manifold studs broken")
        if not self.supported:
            what.append(f"{self.hangers_perished}/{self.hangers} hangers perished")
        if not self.flex_joint:
            what.append("no flex joint")
        return sv.Need(
            identity=self.identity, want="exhaust-mounting",
            position=(0.0, 0.0, 0.0), quantity=1.0, unit="each",
            matches="exhaust-hardware",
            urgency=0.9 if self.studs_broken else 0.5,
            minutes=180.0 + 90.0 * self.studs_broken, skill="mechanic",
            label=f"{self.identity}: " + ", ".join(what),
            why="a broken manifold stud leaks exhaust into the bay at the hottest "
                "point on the engine, and extracting the stump is most of the job")


# ---------------------------------------------------------------------
# QUEUE: MAGNESIUM HEADS AND CASTINGS
# ---------------------------------------------------------------------
#
# Magnesium is already half-declared -- seals.EXPANSION_PER_K carries it
# at 26.0e-6, the highest figure on that list -- and nothing uses it.
# It deserves to be a real option because it makes several models here
# say something they currently cannot.
#
# WHY IT IS WORTH HAVING. It is the lightest structural metal there is,
# about two-thirds the density of aluminium, and that is why it went
# into aircraft engines and racing crankcases and why Volkswagen built
# air-cooled crankcases out of it by the million. On this project it is
# the historically correct answer for a period aero engine.
#
# WHAT IT COSTS, AND EVERY ONE OF THESE LANDS ON SOMETHING ALREADY BUILT:
#
#   differential expansion   26.0e-6 against cast iron's 11.8e-6 is the
#                            WORST pairing in the table -- worse than
#                            aluminium on iron, which is already the
#                            combination responsible for most head
#                            gasket failures ever. A magnesium head on
#                            an iron block should be brutal on gaskets
#                            and on manifold studs, and seals.py will
#                            say so without changing a line.
#
#   galvanic corrosion       magnesium is the most anodic structural
#                            metal in use. Put it in a coolant circuit
#                            with steel or copper and it is the sacrificial
#                            anode whether you wanted one or not. This is
#                            the same electrochemistry seals.CorePlug and
#                            the head gasket's fire-ring erosion already
#                            model, at a far higher rate -- and it makes
#                            the coolant inhibitor a survival item rather
#                            than a maintenance one.
#
#   it burns                 and cannot be put out with water, which
#                            reacts with it to release hydrogen. A
#                            magnesium engine fire is a different event
#                            from a petrol fire, and the sim already has
#                            a fire model and an oxygen-service module
#                            that knows about metals burning.
#
#   creep                    it loses clamp under sustained heat faster
#                            than aluminium, which feeds straight into
#                            GasketJoint.clamp_lost_frac.
#
# WHAT IS NEEDED: a material declaration on the head and block carrying
# density, expansion, galvanic potential and ignition temperature, with
# seals, fouling and oxygen_service reading it. The expansion figure is
# already in place; the rest is not.


# ---------------------------------------------------------------------
# WHAT THE EXHAUST IS DOING WHILE THE ENGINE IS ON THE DYNO
# ---------------------------------------------------------------------
#
# Classically, an engine dyno runs OPEN HEADERS: primaries, collector,
# and nothing after it. Not because anyone thinks that is representative
# but because it is simple, repeatable, and the cell already has an
# extraction duct to catch the gas. Every published engine-dyno figure
# you have ever read was almost certainly made this way.
#
# AND THAT IS WHY ENGINE-DYNO NUMBERS DO NOT SURVIVE CONTACT WITH A
# VEHICLE. The same engine with its intended system on it gives up
# whatever that system costs in back-pressure -- five to fifteen per
# cent on a stock road exhaust -- and gains back whatever the system's
# tuning was worth low down. The gap is not measurement error and it is
# not the dyno being wrong. It is a different engine configuration, and
# the number is honest about the configuration it was measured in and
# silent about the one it will be sold in.
#
# THE EXTRACTION DUCT IS THE PART NOBODY MENTIONS. A cell's extraction
# fan pulls a real depression, and if the duct is SEALED to the collector
# the engine is running against less than atmospheric pressure. That is
# worth a few more per cent, entirely free, and entirely fictional.
# Proper practice is to leave an unsealed gap so the duct captures the
# gas without influencing the back-pressure -- and the reason it is
# "proper practice" rather than "obvious" is that the sealed version
# reads better and nothing on the printout says which was done.
#
# A CHASSIS DYNO HAS NONE OF THESE PROBLEMS and a different set: the
# whole vehicle is there, so the exhaust is by definition the real one,
# and what you lose instead is everything between the crank and the
# roller.
#
# The physics here is not new. An open collector IS a fully-open cutout,
# so this reuses ExhaustCutout rather than describing the same pressure
# relief a second time.

@dataclass(frozen=True)
class DynoExhaustConfig:
    key: str
    label: str
    #: How much of the system is bypassed, 0 = full system fitted.
    open_frac: float
    #: Depression the extraction duct imposes on the collector, Pa.
    #: Non-zero only when somebody sealed it to the pipe.
    extraction_depression_pa: float
    representative: bool
    why: str = ""


DYNO_EXHAUST: dict[str, DynoExhaustConfig] = {
    "open-collector": DynoExhaustConfig(
        "open-collector", "open headers, collector dumping to the duct", 1.0, 0.0,
        False,
        why="the classical engine-dyno setup and the least representative. Reads "
            "highest, and the number belongs to an engine configuration that will "
            "never be installed in anything"),
    "open-sealed-extraction": DynoExhaustConfig(
        "open-sealed-extraction", "open headers, duct sealed to the collector",
        1.0, 4_000.0, False,
        why="the same again with the extraction sealed on, so the engine breathes out "
            "against a partial vacuum. Reads highest of all and is the least honest "
            "configuration on this list -- nothing on the printout distinguishes it "
            "from the one above"),
    "dyno-header": DynoExhaustConfig(
        "dyno-header", "matched-length dyno header, no silencing", 0.75, 0.0, False,
        why="primaries and collector cut to the same lengths the vehicle system will "
            "use, so the tuning is right even though the silencing is absent. The "
            "honest compromise when the real system will not fit in the cell"),
    "full-system": DynoExhaustConfig(
        "full-system", "the vehicle's own system fitted", 0.0, 0.0, True,
        why="what the engine will actually have. Reads lowest and is the only engine-"
            "dyno configuration whose number means anything to a driver"),
    "chassis": DynoExhaustConfig(
        "chassis", "chassis dyno, whole vehicle", 0.0, 0.0, True,
        why="the real exhaust by definition, because the vehicle is around it. Trades "
            "the exhaust question for a driveline-loss question instead"),
}


def dyno_exhaust(key: str) -> DynoExhaustConfig:
    c = DYNO_EXHAUST.get(str(key))
    if c is None:
        raise KeyError(f"unknown dyno exhaust {key!r}; declared: "
                       f"{', '.join(sorted(DYNO_EXHAUST))}")
    return c


def dyno_reading(rpm: float, config: str = "open-collector",
                 system_backpressure_pa: float = 12_000.0,
                 crossover_rpm: float = 3800.0) -> dict:
    """What the dyno will read, and what the engine will actually make.

    Returns both, because the useful output of a dyno pull is not the
    number -- it is the number together with what would happen to it
    once the vehicle's own exhaust is bolted on."""
    c = dyno_exhaust(config)
    cut = ExhaustCutout(kind="dyno-bench", open_frac=c.open_frac,
                        system_backpressure_pa=system_backpressure_pa,
                        crossover_rpm=crossover_rpm)
    reading = cut.power_frac(rpm)
    # a sealed extraction duct is simply more pressure relief
    if c.extraction_depression_pa > 0.0:
        reading *= 1.0 + (c.extraction_depression_pa / 101_325.0) * 0.55
    installed = ExhaustCutout(kind="dyno-bench", open_frac=0.0,
                              system_backpressure_pa=system_backpressure_pa,
                              crossover_rpm=crossover_rpm).power_frac(rpm)
    return {"config": c.label, "representative": c.representative,
            "reading_frac": reading, "installed_frac": installed,
            "optimism_frac": reading / max(1e-9, installed) - 1.0,
            "why": c.why}


def recommend_dyno_exhaust(purpose: str = "development") -> str:
    """What to fit for what you are trying to find out.

    There is no single right answer and pretending otherwise is how
    numbers get quoted out of context."""
    return {
        "development": "dyno-header",
        "certification": "full-system",
        "comparison": "open-collector",
        "advertising": "open-sealed-extraction",
        "vehicle": "chassis",
    }.get(str(purpose), "dyno-header")
