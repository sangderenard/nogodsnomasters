"""Gaskets and seals: what actually holds them, and what actually kills
them.

A gasket is not a washer. It is a spring under a controlled preload, and
every failure mode on this list is a story about that preload being lost
or overcome. Nothing here fails on a timer.

THE FOUR THINGS THAT KILL A HEAD GASKET, in the order they matter:

  CLAMP LOSS.       Bolts relax. The gasket creeps. Every heat cycle
                    takes a little preload away and it never comes back,
                    which is why a gasket that has been on for years is
                    not the gasket that was fitted.

  LIFT-OFF.         Combustion pressure times bore area is a real force
                    trying to separate the head from the block, and it
                    is not small: eighteen megapascals on a hundred-
                    millimetre bore is a hundred and forty kilonewtons.
                    When that beats the local clamp the head lifts for a
                    fraction of a degree, the fire ring is hammered, and
                    it never seals as well again. This is why boosting an
                    engine lifts head gaskets and why diesels have more
                    head bolts than petrol engines of the same size.

  DIFFERENTIAL EXPANSION. An aluminium head on an iron block grows
                    almost twice as fast as what it is bolted to.
                    Across a long engine that is most of a millimetre of
                    relative movement every time it warms up, and the
                    gasket has to survive being scrubbed back and forth
                    for the life of the engine. This one combination is
                    responsible for most head gasket failures ever.

  CORROSION.        Coolant that has lost its inhibitor package
                    electrolytically eats the fire ring from the coolant
                    side. The gasket is then thinner where it matters
                    most and fails under a load it used to hold. This is
                    the one that is genuinely preventable by a bottle of
                    additive, which is why the additive exists.

AND THE REAR MAIN SEAL FAILS FOR A COMPLETELY DIFFERENT REASON, which
is the reason it is in this file. A lip seal is not clamped and is not
loaded by combustion. It is a thin elastomer lip riding on a spinning
journal, and what pushes it out is CRANKCASE PRESSURE.

That makes it the end of a chain the sim already has. Blow-by past worn
rings fills the crankcase with gas. A breather or PCV valve that has
fouled shut stops it leaving. Case pressure climbs. And the weakest hole
in the crankcase is the one with a rubber lip in it, so the seal is
extruded out of its bore and the engine empties itself through the back
of the block -- into the bellhousing, onto the clutch, which is exactly
where an engine keeps the one component least able to tolerate oil.

None of that is scripted. Foul the PCV and the rest follows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GasketClass:
    key: str
    label: str
    #: Seating stress it needs at the fire ring to seal at all.
    min_seating_pa: float
    #: Above this it is being crushed rather than sealed.
    max_seating_pa: float
    #: Fraction of preload lost per hundred full heat cycles. Composite
    #: creeps; steel barely does.
    creep_per_100_cycles: float
    #: Surface roughness it will tolerate. MLS is the fussy one -- it
    #: has no bulk to conform with, so it needs a finish that a hand
    #: scrape cannot produce.
    max_surface_ra_um: float
    max_temp_k: float
    #: How fast coolant electrolysis eats it with no inhibitor left.
    corrosion_sensitivity: float
    reusable: bool = False
    why: str = ""


GASKETS: dict[str, GasketClass] = {
    "composite-fibre": GasketClass(
        "composite-fibre", "composite fibre with fire ring", 12e6, 60e6,
        creep_per_100_cycles=0.030, max_surface_ra_um=3.2, max_temp_k=520.0,
        corrosion_sensitivity=1.0,
        why="a fibre body with a rolled steel fire ring. Forgiving of a rough or "
            "slightly untrue face, which is what makes it the field-repair gasket -- "
            "and it creeps, so it keeps losing clamp for as long as it is fitted"),
    "mls": GasketClass(
        "mls", "multi-layer steel", 25e6, 120e6,
        creep_per_100_cycles=0.004, max_surface_ra_um=0.8, max_temp_k=700.0,
        corrosion_sensitivity=0.35, reusable=False,
        why="three to five embossed steel layers, each a spring. It barely creeps and "
            "it tolerates far more clamp, which is why every modern engine uses one. "
            "The price is surface finish: it has no bulk to conform with, so a face "
            "that would have sealed with composite will weep with MLS"),
    "solid-copper": GasketClass(
        "solid-copper", "solid copper", 40e6, 200e6,
        creep_per_100_cycles=0.002, max_surface_ra_um=1.6, max_temp_k=900.0,
        corrosion_sensitivity=1.6, reusable=True,
        why="annealed copper sheet, and it does not seal on its own -- it needs a wire "
            "O-ring cut into the block and a receiver groove in the head. The racing "
            "answer for pressure nothing else holds, and it is galvanically greedy: "
            "copper against aluminium in coolant is a battery"),
    "elastomer-o-ring": GasketClass(
        "elastomer-o-ring", "elastomer O-ring", 2e6, 20e6,
        creep_per_100_cycles=0.012, max_surface_ra_um=1.6, max_temp_k=420.0,
        corrosion_sensitivity=0.2,
        why="a moulded ring in a groove, which is the right answer everywhere that is "
            "not a combustion chamber -- covers, housings, water pumps. It fails by "
            "the rubber hardening with age and heat rather than by anything mechanical"),
    "rtv-formed": GasketClass(
        "rtv-formed", "RTV formed in place", 0.5e6, 6e6,
        creep_per_100_cycles=0.020, max_surface_ra_um=6.3, max_temp_k=480.0,
        corrosion_sensitivity=0.3,
        why="silicone squeezed from a tube and assembled wet. It will seal a face that "
            "nothing else will, and too much of it ends up inside the engine blocking "
            "an oil pickup, which is a genuinely common way to kill a rebuild"),
    "paper-cork": GasketClass(
        "paper-cork", "paper or cork", 0.3e6, 3e6,
        creep_per_100_cycles=0.090, max_surface_ra_um=12.5, max_temp_k=400.0,
        corrosion_sensitivity=0.5,
        why="for covers and pans, where there is no pressure and the only job is to "
            "take up a wavy stamped flange. It relaxes enormously, which is why a rocker "
            "cover always weeps eventually"),
}


def gasket_class(key: str) -> GasketClass:
    g = GASKETS.get(str(key))
    if g is None:
        raise KeyError(f"unknown gasket {key!r}; declared: {', '.join(sorted(GASKETS))}")
    return g


#: Linear thermal expansion, per kelvin. The pair that matters is an
#: aluminium head on an iron block: nearly two to one.
EXPANSION_PER_K = {"aluminium": 23.1e-6, "cast-iron": 11.8e-6,
                   "steel": 12.0e-6, "magnesium": 26.0e-6}


def differential_growth_m(head_material: str, block_material: str,
                          span_m: float, delta_k: float) -> float:
    """How far the head slides across the block between cold and hot.

    This is what scrubs a head gasket to death, and it is the single
    number that explains why one material combination has a reputation
    and the other does not."""
    a = EXPANSION_PER_K.get(str(head_material), 12.0e-6)
    b = EXPANSION_PER_K.get(str(block_material), 12.0e-6)
    return abs(a - b) * max(0.0, span_m) * max(0.0, delta_k)


# ---------------------------------------------------------------------
# a joint
# ---------------------------------------------------------------------

@dataclass
class GasketJoint:
    """One clamped joint, with its own preload and its own history."""
    identity: str = "powertrain.head_gasket"
    gasket: str = "composite-fibre"
    bolts: int = 10
    #: Fallback when no bolt is declared. Prefer `bolt`, which derives
    #: this from a torque, a grade and a thread condition -- and knows
    #: how wrong it might be.
    bolt_preload_n: float = 40_000.0
    bolt: "HeadBolt | None" = None
    #: Area actually carrying the seal at the fire ring, per cylinder.
    fire_ring_area_m2: float = 0.0016
    #: Gasket land area the clamp is spread over.
    seal_area_m2: float = 0.045
    bore_m: float = 0.100
    cylinders: int = 4
    head_material: str = "aluminium"
    block_material: str = "cast-iron"
    span_m: float = 0.45
    surface_ra_um: float = 1.6
    #: Permanent flatness error. An overheat warps a head and the warp
    #: never comes back out without a skim.
    flatness_error_um: float = 10.0

    # ---- state ----
    heat_cycles: float = 0.0
    clamp_lost_frac: float = 0.0
    fire_ring_eroded_frac: float = 0.0
    lift_events: float = 0.0
    failed: bool = False
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def spec(self) -> GasketClass:
        return gasket_class(self.gasket)

    @property
    def per_bolt_preload_n(self) -> float:
        """One bolt's live preload, from the bolt itself when there is
        one. A declared bolt already carries its own drift, so the
        joint's creep and the bolt's loosening are two different losses
        and are not double-counted."""
        return self.bolt.preload_n if self.bolt is not None else self.bolt_preload_n

    @property
    def clamp_n(self) -> float:
        return self.bolts * self.per_bolt_preload_n * max(0.0, 1.0 - self.clamp_lost_frac)

    @property
    def clamp_band_n(self) -> tuple:
        """The clamp this joint might ACTUALLY have, given that a torque
        wrench does not measure preload. The low end is the one that
        decides whether it seals."""
        if self.bolt is None:
            return (self.clamp_n, self.clamp_n)
        lo, hi = self.bolt.preload_band_n
        k = self.bolts * max(0.0, 1.0 - self.clamp_lost_frac) * max(0.0, 1.0 - self.bolt.drift)
        return (lo * k, hi * k)

    @property
    def seating_pa(self) -> float:
        """Stress actually holding the gasket down where it SEALS.

        Against the fire ring area, not the whole gasket land -- which
        is the distinction that decides everything. A gasket does not
        seal over its full face; it seals on a narrow raised ring around
        each bore, and the clamp is concentrated there deliberately. A
        hundred kilonewtons over sixteen square centimetres is sixty
        megapascals at the ring, against eight if it were smeared over
        the whole land.

        Dividing by the land area instead is not a small error: it put
        every gasket here below its own seating requirement and made
        MLS -- which needs the most and creeps the least -- fail sooner
        than composite, which is backwards."""
        return self.clamp_n / max(1e-6, self.fire_ring_area_m2 * max(1, self.cylinders))

    @property
    def clamp_per_cylinder_n(self) -> float:
        """Clamp available to hold ONE cylinder down. The head does not
        share its bolts globally -- each bore is held by the bolts
        around it, which is why the end cylinders of a long engine let
        go first."""
        return self.clamp_n / max(1, self.cylinders)

    def lift_force_n(self, peak_cylinder_pa: float) -> float:
        """Combustion trying to separate the head from the block."""
        return float(peak_cylinder_pa) * math.pi * (self.bore_m * 0.5) ** 2

    def lifts_at(self, peak_cylinder_pa: float) -> bool:
        return self.lift_force_n(peak_cylinder_pa) > self.clamp_per_cylinder_n

    @property
    def surface_ok(self) -> bool:
        return self.surface_ra_um <= self.spec.max_surface_ra_um

    @property
    def integrity(self) -> float:
        """0..1. Seals fully at 1, weeps below about 0.6, gone at 0."""
        if self.failed:
            return 0.0
        need = self.spec.min_seating_pa
        have = self.seating_pa * (1.0 - self.fire_ring_eroded_frac)
        margin = min(1.5, have / max(1.0, need))
        finish = 1.0 if self.surface_ok else max(0.2, self.spec.max_surface_ra_um
                                                 / max(0.1, self.surface_ra_um))
        # a warped face is a leak path the clamp cannot close
        flat = max(0.0, 1.0 - self.flatness_error_um / 80.0)
        return max(0.0, min(1.0, margin * finish * flat))

    def step(self, dt_s: float, *, peak_cylinder_pa: float = 6e6,
             coolant_inhibitor_frac: float = 1.0, delta_k: float = 90.0,
             heat_cycle: bool = False, overheat_k: float = 0.0) -> dict:
        """Advance this joint. Every argument is something the sim knows."""
        s = self.spec
        if heat_cycle:
            self.heat_cycles += 1.0
            # creep: preload lost and never recovered
            self.clamp_lost_frac = min(0.85, self.clamp_lost_frac
                                       + s.creep_per_100_cycles / 100.0)
            # and the scrubbing from differential growth, which costs
            # clamp too because it works the gasket body
            slide = differential_growth_m(self.head_material, self.block_material,
                                          self.span_m, delta_k)
            self.clamp_lost_frac = min(0.85, self.clamp_lost_frac + slide * 0.02)
            # THE SAME SLIP UNWINDS THE BOLTS. Not a second mechanism --
            # the identical transverse movement, acting on the fastener
            # instead of on the gasket body.
            if self.bolt is not None:
                self.bolt.step(dt_s, transverse_slip_m=slide, heat_cycle=True)
        # AN OVERHEAT IS A STEP CHANGE, NOT AN ACCUMULATION. A head that
        # has been cooked is warped, permanently, and no amount of
        # careful running afterwards takes the warp out.
        if overheat_k > 0.0:
            over = max(0.0, overheat_k - 380.0)
            if over > 0.0:
                self.flatness_error_um += over * 0.8
        # corrosion of the fire ring, which is what the inhibitor is for
        bare = max(0.0, 1.0 - max(0.0, min(1.0, coolant_inhibitor_frac)))
        if bare > 0.0:
            self.fire_ring_eroded_frac = min(
                0.9, self.fire_ring_eroded_frac
                + bare * s.corrosion_sensitivity * dt_s * 2.0e-7)
        # lift-off: a real event with a real consequence
        lifted = self.lifts_at(peak_cylinder_pa)
        if lifted:
            self.lift_events += 1.0
            # every lift hammers the fire ring a little flatter
            self.fire_ring_eroded_frac = min(0.9, self.fire_ring_eroded_frac + 2.0e-4)
        if self.integrity <= 0.05 or self.seating_pa < s.min_seating_pa * 0.35:
            self.failed = True
        return {"integrity": self.integrity, "seating_mpa": self.seating_pa / 1e6,
                "needs_mpa": s.min_seating_pa / 1e6,
                "clamp_lost": self.clamp_lost_frac,
                "fire_ring_eroded": self.fire_ring_eroded_frac,
                "lifted": lifted, "lift_events": self.lift_events,
                "flatness_um": self.flatness_error_um,
                "heat_cycles": self.heat_cycles, "failed": self.failed}

    def need(self):
        import servicing as sv
        i = self.integrity
        if i > 0.55 and not self.failed:
            return None
        return sv.Need(
            identity=self.identity, want="gasket", position=tuple(self.position),
            quantity=1.0, unit="each", matches="head-gasket",
            urgency=1.0 if self.failed else 0.6, minutes=480.0, skill="mechanic",
            label=(f"{self.identity} integrity {i * 100:.0f}%, "
                   f"clamp down {self.clamp_lost_frac * 100:.0f}%, "
                   f"face out {self.flatness_error_um:.0f} um"),
            why="a head gasket is a preload, not a part: once the clamp is gone the "
                "gasket cannot be made to seal by anything except taking the head off, "
                "skimming the face true and torquing new bolts")


# ---------------------------------------------------------------------
# the rear main seal
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SealClass:
    key: str
    label: str
    #: Crankcase pressure that pushes it out of its bore. THE number.
    blowout_pa: float
    #: Surface speed it will tolerate on the journal before the lip
    #: cooks itself.
    max_surface_speed_m_s: float
    hardens_above_k: float
    #: Does it tolerate being run without oil for a moment on start-up?
    dry_start_tolerant: bool
    why: str = ""


SEALS: dict[str, SealClass] = {
    "ptfe-lip": SealClass(
        "ptfe-lip", "PTFE lip seal", 55_000.0, 25.0, 450.0, False,
        why="the modern one. It is fitted DRY -- oiling the lip on assembly ruins it, "
            "because the lip has to bed a microscopic wear track into the journal and "
            "oil stops that happening. It seals better than anything else here and it "
            "is unforgiving of being fitted by someone who assumed"),
    "nitrile-lip": SealClass(
        "nitrile-lip", "nitrile lip seal with garter spring", 40_000.0, 14.0, 393.0, True,
        why="a rubber lip held against the journal by a little coil spring. Ages by "
            "hardening: the rubber loses its elasticity, the lip stops following the "
            "journal, and it weeps long before it blows"),
    "rope-packing": SealClass(
        "rope-packing", "rope / braided packing", 20_000.0, 12.0, 520.0, True,
        why="graphited asbestos or its modern replacement, packed into a groove. It is "
            "MEANT to weep slightly -- the seepage is its lubrication -- so an engine "
            "with one is not faulty for marking its parking spot. It also tolerates "
            "heat and misalignment that would destroy a lip seal"),
    "labyrinth": SealClass(
        "labyrinth", "labyrinth / slinger", 8_000.0, 90.0, 800.0, True,
        why="no contact at all: a slinger throws oil back and a spiral groove pumps it "
            "inward. It cannot wear out and it cannot hold pressure, so it is only "
            "found where the crankcase is properly vented"),
}


def seal_class(key: str) -> SealClass:
    s = SEALS.get(str(key))
    if s is None:
        raise KeyError(f"unknown seal {key!r}; declared: {', '.join(sorted(SEALS))}")
    return s


#: Discharge coefficient for oil squeezing past a blown lip seal. It is
#: not a clean orifice -- it is a gap around a displaced ring.
BLOWN_SEAL_CD = 0.55
OIL_DENSITY_KG_M3 = 880.0


@dataclass
class MainSeal:
    """The crankshaft rear main seal, and the pressure that kills it."""
    identity: str = "powertrain.rear_main_seal"
    seal: str = "nitrile-lip"
    journal_diameter_m: float = 0.090
    #: Gap it opens to once extruded from its bore.
    blown_gap_m: float = 0.0012
    #: State
    hardening: float = 0.0          # 0 fresh, 1 no elasticity left
    journal_groove_um: float = 0.0  # the seal wears a track into the crank
    blown: bool = False
    dry_started: bool = False
    leaked_l: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)
    #: Where the oil goes. A rear main seal discharges into the
    #: bellhousing, which is where the clutch lives.
    discharges_into: str = "powertrain.bellhousing"

    @property
    def spec(self) -> SealClass:
        return seal_class(self.seal)

    @property
    def blowout_pa(self) -> float:
        """What it will hold TODAY. A hardened lip holds much less than
        a fresh one, which is why old engines blow seals at pressures
        they lived with for years."""
        return self.spec.blowout_pa * max(0.15, 1.0 - 0.7 * self.hardening) \
            * (0.6 if self.dry_started else 1.0)

    def surface_speed_m_s(self, rpm: float) -> float:
        return math.pi * self.journal_diameter_m * max(0.0, rpm) / 60.0

    @property
    def gap_area_m2(self) -> float:
        return math.pi * self.journal_diameter_m * self.blown_gap_m

    def leak_rate_l_s(self, crankcase_gauge_pa: float, oil_head_pa: float = 3_000.0) -> float:
        """How fast it empties once it is out.

        Orifice flow through the annulus, driven by case pressure plus
        the head of oil standing over it. A blown rear main is not a
        drip -- it is the crankcase being pushed out of a slot the whole
        way round the crankshaft."""
        if not self.blown:
            return 0.0
        dp = max(0.0, float(crankcase_gauge_pa)) + max(0.0, oil_head_pa)
        if dp <= 0.0:
            return 0.0
        v = math.sqrt(2.0 * dp / OIL_DENSITY_KG_M3)
        return BLOWN_SEAL_CD * self.gap_area_m2 * v * 1000.0

    def step(self, dt_s: float, *, crankcase_gauge_pa: float, rpm: float,
             oil_temp_k: float = 363.0, oil_present: bool = True) -> dict:
        """One tick of seal life. Blows when the case beats it.

        GAUGE PRESSURE, and the name says so deliberately. Seal ratings
        are quoted as gauge, crankcase_state.crankcase_pressure_pa is
        the pressure the accumulated blow-by contributes and is
        therefore also gauge, and an earlier version of this took one
        for the other and added an atmosphere to it. The engine then sat
        at what looked like sixty kilopascals of VACUUM while the seal
        reported that everything was fine -- the units disagreeing, not
        the physics."""
        dt = max(0.0, float(dt_s))
        s = self.spec
        # ageing: heat hardens the elastomer, permanently
        if oil_temp_k > s.hardens_above_k:
            self.hardening = min(1.0, self.hardening
                                 + (oil_temp_k - s.hardens_above_k) * dt * 2.0e-6)
        else:
            self.hardening = min(1.0, self.hardening + dt * 1.5e-8)
        # the lip cuts a groove in the journal, and a grooved journal
        # cannot be sealed by a new seal in the same position
        v = self.surface_speed_m_s(rpm)
        if v > 0.0:
            over = max(1.0, v / max(1.0, s.max_surface_speed_m_s))
            self.journal_groove_um += dt * 1.0e-4 * over * over
        if not oil_present and v > 1.0 and not s.dry_start_tolerant:
            self.dry_started = True
        blew = False
        if not self.blown and crankcase_gauge_pa >= self.blowout_pa:
            self.blown = True
            blew = True
        rate = self.leak_rate_l_s(crankcase_gauge_pa)
        self.leaked_l += rate * dt
        return {"blown": self.blown, "blew_this_step": blew,
                "holds_pa": self.blowout_pa,
                "case_gauge_pa": max(0.0, crankcase_gauge_pa),
                "hardening": self.hardening,
                "journal_groove_um": self.journal_groove_um,
                "leak_l_s": rate, "leaked_l": self.leaked_l,
                "surface_speed_m_s": v,
                "discharges_into": self.discharges_into,
                "why": ("extruded out of its bore by crankcase pressure. The oil is "
                        "going into the bellhousing and onto the clutch"
                        if self.blown else "holding")}

    def need(self):
        import servicing as sv
        if not self.blown and self.hardening < 0.7:
            return None
        return sv.Need(
            identity=self.identity, want="seal", position=tuple(self.position),
            quantity=1.0, unit="each", matches="rear-main-seal",
            urgency=1.0 if self.blown else 0.5, minutes=600.0, skill="mechanic",
            label=(f"{self.identity} "
                   + ("BLOWN OUT" if self.blown else
                      f"{self.hardening * 100:.0f}% hardened")
                   + (f", journal grooved {self.journal_groove_um:.0f} um"
                      if self.journal_groove_um > 20.0 else "")),
            why="the gearbox has to come out to reach it, which is why a rear main is "
                "the one leak people live with -- and why fixing the CAUSE, which is "
                "usually a blocked breather, matters more than fixing the seal")


def diagnose(case_gauge_pa: float, seal: MainSeal) -> str:
    """What a mechanic would say, given the case pressure.

    Included because the interesting failure here is a chain, and the
    end of the chain is not where the fault is."""
    if not seal.blown:
        if case_gauge_pa > seal.blowout_pa * 0.6:
            return ("crankcase pressure is most of the way to blowing the rear main. "
                    "The seal is not the problem -- find out why the case is not "
                    "venting")
        return "normal"
    return ("rear main blown OUT, not worn out. Replacing it without clearing the "
            "breather just buys another one of these")


# ---------------------------------------------------------------------
# CORE PLUGS, which almost nobody calls by the right name
# ---------------------------------------------------------------------
#
# THEY ARE NOT FREEZE PLUGS. A block is cast around sand cores that shape
# the water jacket, and when the sand is shaken out it leaves holes in
# the outside of the casting that were never meant to be there. A core
# plug is what closes them. That is the entire reason they exist.
#
# The folklore is that they pop out to save the block when coolant
# freezes. They sometimes do pop out, and it sometimes does help, but a
# block that freezes hard usually cracks somewhere else anyway -- ice
# does not politely choose the weakest designed point. Treating them as
# a safety device is a comfortable story rather than a design.
#
# WHAT ACTUALLY KILLS THEM IS THE SAME THING THAT KILLS A FIRE RING:
# coolant with no inhibitor left, eating a thin steel cup from the
# jacket side where nobody can see it. The plug looks perfect from
# outside until the day it weeps, because all the corrosion is happening
# on the face you cannot inspect. This is the second place in this file
# where a bottle of additive is the entire difference, and that is not a
# coincidence -- it is the same electrochemistry.
#
# AND THE REASON THEY ARE INTERESTING IS ACCESS. A block carries half a
# dozen of them. The ones down the side of the block are a twenty-minute
# job with a drift and a socket. The one behind the flywheel is an
# engine-out job. They corrode at the same rate and you do not get to
# choose which one goes first, so the question is never "can I fix it"
# but "which one was it".

@dataclass(frozen=True)
class PlugMaterial:
    key: str
    label: str
    #: Relative rate at which un-inhibited coolant eats it.
    corrosion_rate: float
    why: str = ""


PLUG_MATERIALS: dict[str, PlugMaterial] = {
    "steel-cup": PlugMaterial(
        "steel-cup", "pressed steel cup", 1.0,
        why="what almost every engine leaves the factory with: a thin mild steel cup "
            "pressed into a machined bore. Cheap, sealed by interference alone, and "
            "the first thing in the cooling system to be eaten"),
    "brass-cup": PlugMaterial(
        "brass-cup", "brass cup", 0.12,
        why="the rebuild upgrade, and a genuinely good one -- brass does not rust, so "
            "the plug outlives the coolant that would have killed a steel one. Fitted "
            "for the same money at the one point in an engine's life when the block is "
            "already apart"),
    "stainless-cup": PlugMaterial(
        "stainless-cup", "stainless cup", 0.06,
        why="better still and rarely worth it, because by the time corrosion is not "
            "the failure mode the plug is not the problem"),
    "expansion-rubber": PlugMaterial(
        "expansion-rubber", "rubber expansion plug", 0.02,
        why="a rubber slug squeezed by a wing nut. It does not corrode at all and it "
            "is not a repair, it is a way to get home -- the rubber ages, and it is the "
            "one plug that can be fitted without pulling anything apart"),
}


def plug_material(key: str) -> PlugMaterial:
    m = PLUG_MATERIALS.get(str(key))
    if m is None:
        raise KeyError(f"unknown plug material {key!r}; declared: "
                       f"{', '.join(sorted(PLUG_MATERIALS))}")
    return m


#: Wall thickness of a pressed cup. It is thinner than people picture --
#: this is why corrosion gets through it in years rather than decades.
PLUG_WALL_MM = 1.2


@dataclass
class CorePlug:
    """One core plug, with a corrosion state and an access cost."""
    identity: str = "powertrain.core_plug"
    material: str = "steel-cup"
    diameter_m: float = 0.040
    #: What has to come off to reach it. Empty means it is simply there
    #: on the side of the block.
    behind: str = ""
    access_minutes: float = 25.0
    #: Coolant system pressure it has to hold. A pressurised system
    #: pushes harder on a plug that is already thin.
    system_pressure_pa: float = 200_000.0
    #: LOCAL CONDITIONS, and the reason plugs do not fail together.
    #:
    #: Identical plugs in identical conditions pop on the same tick,
    #: which makes nonsense of the whole point -- there is no "which one
    #: was it" if the answer is always "all of them". Real jackets are
    #: not uniform: flow is brisk past some plugs and nearly stagnant
    #: behind others, and stagnant coolant is hotter, more concentrated
    #: and far more aggressive.
    #:
    #: The cruel part is that the stagnant, hot corner of the jacket is
    #: the BACK of the block, which is where the plug you cannot reach
    #: lives. So the expensive one is not merely as likely to fail as
    #: the others -- it is more likely, and that matches what anyone who
    #: has owned an old engine will tell you.
    exposure: float = 1.0
    wall_lost_mm: float = 0.0
    weeping: bool = False
    popped: bool = False
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def spec(self) -> PlugMaterial:
        return plug_material(self.material)

    @property
    def wall_left_mm(self) -> float:
        return max(0.0, PLUG_WALL_MM - self.wall_lost_mm)

    @property
    def remaining_frac(self) -> float:
        return self.wall_left_mm / PLUG_WALL_MM

    @property
    def holds_pa(self) -> float:
        """Pressure it can still take. A cup plug is held by interference
        around its rim, so what it holds falls with the wall that is
        left -- and the system it is in is pressurised, which is why
        plugs let go when the engine is hot rather than when it is
        parked."""
        return 900_000.0 * self.remaining_frac ** 2

    def step(self, dt_s: float, coolant_inhibitor_frac: float = 1.0,
             coolant_k: float = 363.0) -> dict:
        """Eat the plug from the inside, at the rate the coolant allows."""
        bare = max(0.0, 1.0 - max(0.0, min(1.0, coolant_inhibitor_frac)))
        if bare > 0.0 and not self.popped:
            # electrochemistry doubles roughly every ten kelvin, same as
            # every other rate process in the engine
            hot = 2.0 ** ((coolant_k - 353.0) / 10.0)
            self.wall_lost_mm = min(
                PLUG_WALL_MM,
                self.wall_lost_mm + bare * self.spec.corrosion_rate * hot
                * max(0.05, self.exposure) * dt_s * 3.0e-9)
        if not self.popped:
            if self.wall_left_mm <= 0.0:
                self.popped = True
            elif self.system_pressure_pa >= self.holds_pa:
                self.popped = True
            elif self.remaining_frac < 0.25:
                self.weeping = True
        return {"wall_left_mm": self.wall_left_mm,
                "remaining_frac": self.remaining_frac,
                "holds_pa": self.holds_pa, "weeping": self.weeping,
                "popped": self.popped, "behind": self.behind,
                "access_minutes": self.access_minutes}

    def coolant_loss_l_s(self) -> float:
        """How fast the system empties once it is gone.

        A popped plug is a hole the full diameter of the plug, low down
        on a pressurised system. It is not a leak to drive home on."""
        if not self.popped:
            return 0.02 if self.weeping else 0.0
        import math as _m
        area = _m.pi * (self.diameter_m * 0.5) ** 2
        dp = max(0.0, self.system_pressure_pa - 101_325.0) + 4_000.0
        v = _m.sqrt(2.0 * dp / 1030.0)          # glycol mix
        return 0.62 * area * v * 1000.0

    def need(self):
        import servicing as sv
        if not self.weeping and not self.popped:
            return None
        return sv.Need(
            identity=self.identity, want="core-plug", position=tuple(self.position),
            quantity=1.0, unit="each", matches="core-plug",
            urgency=1.0 if self.popped else 0.5,
            minutes=self.access_minutes, skill="mechanic",
            label=(f"{self.identity} "
                   + ("POPPED" if self.popped else "weeping")
                   + (f", behind {self.behind}" if self.behind else "")
                   + f" -- {self.access_minutes:.0f} min access"),
            why=("a core plug is a twenty minute job or an engine-out job depending "
                 "entirely on which one it is, and corrosion does not take that into "
                 "account. The plugs all rot at the same rate; only the bill differs"))


def block_plug_set(n_side: int = 4, rear: bool = True,
                   material: str = "steel-cup",
                   prefix: str = "powertrain.core_plug",
                   seed: int | None = None, jitter: float = 0.15) -> list:
    """A block's worth of plugs, with honest access costs.

    The asymmetry is the whole point of modelling these: they corrode on
    their own schedules and they cost wildly different amounts to reach.

    ON THE JITTER, which is small and does one specific job. The
    exposure gradient below is already enough to separate the failures
    -- measured, it spreads them across 1.88x, well over a year apart --
    so jitter is NOT needed to stop them failing together. What it stops
    is something else: with no variation at all the rear plug is first
    every single time, and an outcome that never varies is not a risk,
    it is a schedule. A modest casting-and-fit variation leaves the rear
    plug first about nine times in ten, which keeps it the thing to
    worry about while leaving room to be wrong.

    Deterministic unless a seed is given, so tests stay reproducible."""
    import random as _random
    rng = _random.Random(seed) if seed is not None else None

    def _vary(x: float) -> float:
        return x if rng is None else x * rng.uniform(1.0 - jitter, 1.0 + jitter)

    out = []
    for i in range(max(0, n_side)):
        # coolant leaves the pump at the front and is hottest by the
        # time it reaches the back, so exposure climbs down the block
        frac = i / max(1, n_side - 1) if n_side > 1 else 0.0
        out.append(CorePlug(identity=f"{prefix}.side_{i + 1}", material=material,
                            access_minutes=25.0,
                            exposure=_vary(0.85 + 0.45 * frac)))
    if rear:
        # hot, stagnant, and behind the gearbox. The worst combination
        # of conditions on the block sits at the most expensive address.
        out.append(CorePlug(identity=f"{prefix}.rear", material=material,
                            behind="powertrain.bellhousing", access_minutes=600.0,
                            exposure=_vary(1.6)))
    return out


# ---------------------------------------------------------------------
# THE BOLTS, and why a torque wrench is a blunt instrument
# ---------------------------------------------------------------------
#
# Everything above treats clamp as a number. It is not a number, it is
# an ESTIMATE, and the size of the error is the single most useful thing
# to know about a bolted joint.
#
# A torque wrench does not measure preload. It measures the torque
# needed to overcome friction -- under the head, and in the threads --
# plus the small part that actually stretches the bolt. On an ordinary
# fastener something like ninety per cent of the torque is spent on
# friction and only ten per cent becomes clamp. So the reading is mostly
# a measurement of how greasy the bolt is.
#
#     T = K . D . F
#
# K is the "nut factor", and it is not a constant: it ranges from about
# 0.10 with anti-seize to about 0.20 as-received, and it scatters by a
# quarter even within one box of bolts. Which means a correctly applied
# torque gives a preload good to roughly plus or minus twenty-five per
# cent, and the low end of that band is the one that leaks.
#
# THAT ERROR IS THE REASON TORQUE-TO-YIELD EXISTS. Tighten to a small
# snug torque and then a measured ANGLE, and you are no longer measuring
# friction at all -- you are measuring stretch directly, because past
# yield the bolt's extension is set by geometry rather than by force.
# The scatter collapses from twenty-five per cent to under ten. The
# price is that the bolt is permanently stretched and cannot be reused,
# which is why modern head bolts are single-use and people who reuse
# them get away with it until they do not.
#
# AND BOLTS LOOSEN FROM SIDEWAYS MOVEMENT, NOT FROM PULLING. This is
# the counter-intuitive one, and it is the Junker effect: axial load
# barely loosens a fastener at all, but TRANSVERSE slip does, and it
# does it quickly. A head joint that is scrubbing because an aluminium
# head is growing across an iron block is exactly that transverse
# movement -- so the same differential expansion that wears the gasket
# is also unwinding the bolts holding it down. What resists it is
# thread retention: an anaerobic locker, a proper flange, or a bolt
# stretched far enough that it cannot slip.

#: Proof stress by metric grade, Pa. Preload targets are a fraction of
#: proof load, never of ultimate -- proof is the point past which the
#: bolt takes a permanent set, and a bolt that has yielded once does not
#: hold what it used to.
BOLT_GRADES: dict[str, float] = {
    "8.8": 580e6,
    "10.9": 830e6,
    "12.9": 970e6,
}

#: Tensile stress area by nominal size, m^2. Not the shank area -- the
#: threads take a bite out of it, and using the plain diameter
#: overstates a bolt's capacity by about twenty per cent.
BOLT_STRESS_AREA_M2: dict[float, float] = {
    0.008: 36.6e-6, 0.010: 58.0e-6, 0.011: 72.3e-6,
    0.012: 84.3e-6, 0.014: 115e-6, 0.016: 157e-6,
}


@dataclass(frozen=True)
class ThreadCondition:
    key: str
    label: str
    nut_factor: float          # K in T = K.D.F
    scatter: float             # fractional, one sigma-ish
    #: How well the threads resist transverse-slip loosening, 0..1.
    lock_frac: float
    why: str = ""


THREAD_CONDITIONS: dict[str, ThreadCondition] = {
    "as-received": ThreadCondition(
        "as-received", "dry, as received", 0.20, 0.25, 0.35,
        why="straight out of the box with whatever oil the factory left on it. The "
            "widest scatter of any of these and the default anyone reaches for"),
    "oiled": ThreadCondition(
        "oiled", "light engine oil", 0.16, 0.18, 0.30,
        why="what a manual usually specifies, because it is repeatable. Oiling a bolt "
            "and using the DRY torque figure overloads it by about a quarter, which is "
            "one of the commonest ways to pull a thread out of a block"),
    "moly": ThreadCondition(
        "moly", "moly grease", 0.13, 0.15, 0.25,
        why="what performance head studs ship with. Low friction means more of the "
            "torque becomes clamp, which is the point -- and means the torque figure "
            "must come down with it"),
    "anti-seize": ThreadCondition(
        "anti-seize", "anti-seize compound", 0.12, 0.22, 0.20,
        why="for anything going into aluminium or anything that has to come out again. "
            "Slippery and inconsistent: low K and poor retention at once"),
    "thread-locker": ThreadCondition(
        "thread-locker", "anaerobic thread locker", 0.18, 0.20, 0.95,
        why="cures in the absence of air into a solid filling the thread gaps, so there "
            "is nowhere for transverse slip to go. It is not glue holding the bolt in "
            "tension -- it works by stopping the sideways movement that unwinds it"),
    "galled": ThreadCondition(
        "galled", "galled or damaged thread", 0.34, 0.45, 0.15,
        why="a thread that has picked up and torn. The torque reading goes UP while the "
            "clamp goes DOWN, so the wrench says it is tight and the joint is not. "
            "This is the failure that looks like success"),
}


def thread_condition(key: str) -> ThreadCondition:
    t = THREAD_CONDITIONS.get(str(key))
    if t is None:
        raise KeyError(f"unknown thread condition {key!r}; declared: "
                       f"{', '.join(sorted(THREAD_CONDITIONS))}")
    return t


def stress_area_m2(diameter_m: float) -> float:
    d = float(diameter_m)
    if d in BOLT_STRESS_AREA_M2:
        return BOLT_STRESS_AREA_M2[d]
    # 0.75 of the plain area is the standard approximation for the bite
    # the threads take, and it is within a couple of per cent
    return math.pi * (d * 0.5) ** 2 * 0.75


@dataclass
class HeadBolt:
    """A head bolt or stud: what it is rated for, and what it achieved.

    Preload is DERIVED from torque and the thread's condition rather
    than declared, so a joint's clamp is only ever as good as the story
    of how it was tightened."""
    identity: str = "powertrain.head_bolt"
    diameter_m: float = 0.012
    grade: str = "10.9"
    torque_nm: float = 120.0
    thread: str = "oiled"
    #: Tightened past yield by angle instead of by torque. Far more
    #: consistent, and single-use.
    torque_to_yield: bool = False
    #: Angle past snug, for a TTY bolt. What actually sets the stretch.
    angle_deg: float = 90.0
    #: Accumulated loss of preload from transverse slip, 0..1.
    drift: float = 0.0
    #: Loosening events noticed.
    slipped_cycles: float = 0.0

    @property
    def spec(self) -> ThreadCondition:
        return thread_condition(self.thread)

    @property
    def proof_load_n(self) -> float:
        return BOLT_GRADES.get(self.grade, 830e6) * stress_area_m2(self.diameter_m)

    @property
    def nominal_preload_n(self) -> float:
        """What it is MEANT to be holding, before drift."""
        if self.torque_to_yield:
            # past yield the stretch is set by the angle, so preload
            # lands near proof regardless of friction. That is the whole
            # reason for doing it.
            return self.proof_load_n * 0.95
        k = self.spec.nut_factor
        return self.torque_nm / max(1e-6, k * self.diameter_m)

    @property
    def preload_band_n(self) -> tuple:
        """The honest range this tightening method actually delivers.

        The number people quote is the middle of this. The bottom of it
        is what the joint has to survive on."""
        s = 0.08 if self.torque_to_yield else self.spec.scatter
        p = self.nominal_preload_n
        return (p * (1.0 - s), p * (1.0 + s))

    @property
    def preload_n(self) -> float:
        """What it is holding now, after whatever it has lost."""
        return self.nominal_preload_n * max(0.0, 1.0 - self.drift)

    @property
    def proof_frac(self) -> float:
        """Share of proof load used. Below ~0.6 the joint can separate
        under load; above ~0.9 the bolt is close to taking a set."""
        return self.preload_n / max(1.0, self.proof_load_n)

    @property
    def overloaded(self) -> bool:
        return self.preload_band_n[1] > self.proof_load_n

    @property
    def underloaded(self) -> bool:
        return self.preload_band_n[0] < self.proof_load_n * 0.55

    def step(self, dt_s: float, transverse_slip_m: float = 0.0,
             heat_cycle: bool = False) -> dict:
        """Let it drift. Loosening is driven by SIDEWAYS movement.

        `transverse_slip_m` is the relative sliding at the joint face --
        seals.differential_growth_m is exactly this number for a head on
        a block, so the same expansion that scrubs the gasket unwinds
        the bolts holding it."""
        retention = max(0.0, min(1.0, self.spec.lock_frac))
        # A TIGHT JOINT CANNOT SLIP, AND A JOINT THAT CANNOT SLIP CANNOT
        # LOOSEN. This is the gate an earlier version of this was
        # missing, and without it every head bolt in the sim quietly
        # unwound itself to nothing over a couple of thousand heat
        # cycles -- which is not what head bolts do.
        #
        # The Junker mechanism needs the clamped faces to actually move
        # relative to each other, and that only happens when the
        # transverse force beats friction at the joint face. A bolt near
        # its proof load is holding the faces together hard enough that
        # they never break loose; a bolt at half proof is not. So the
        # susceptibility goes with how far BELOW a proper preload the
        # bolt is sitting, which is also why an under-torqued bolt
        # loosens and a correctly torqued one does not -- the error
        # compounds itself.
        slip_prone = max(0.0, 1.0 - self.proof_frac / 0.75) ** 1.5
        if heat_cycle and transverse_slip_m > 0.0 and slip_prone > 0.0:
            loss = (transverse_slip_m * 1000.0 * (1.0 - retention)
                    * slip_prone * 0.004)
            if loss > 0.0:
                self.drift = min(0.60, self.drift + loss)
                self.slipped_cycles += 1.0
        return {"preload_n": self.preload_n,
                "nominal_n": self.nominal_preload_n,
                "band_n": self.preload_band_n,
                "proof_load_n": self.proof_load_n,
                "proof_frac": self.proof_frac,
                "drift": self.drift, "retention": retention,
                "overloaded": self.overloaded, "underloaded": self.underloaded,
                "slipped_cycles": self.slipped_cycles}

    def report(self) -> str:
        lo, hi = self.preload_band_n
        how = (f"angle {self.angle_deg:.0f} deg past snug" if self.torque_to_yield
               else f"{self.torque_nm:.0f} Nm, {self.spec.label}")
        return (f"{self.identity}: M{self.diameter_m * 1000:.0f} grade {self.grade}, {how}"
                f" -> {self.nominal_preload_n / 1000:.1f} kN "
                f"(could be {lo / 1000:.1f}-{hi / 1000:.1f}), "
                f"{self.proof_frac * 100:.0f}% of proof"
                + (", OVER PROOF" if self.overloaded else "")
                + (f", drifted {self.drift * 100:.0f}%" if self.drift > 0.01 else ""))


def torque_for_preload(target_n: float, diameter_m: float = 0.012,
                       thread: str = "oiled") -> float:
    """What to set the wrench to. The inverse of the same relation, and
    it moves with the thread condition -- which is why using a dry
    torque figure on an oiled bolt overloads it."""
    k = thread_condition(thread).nut_factor
    return k * float(diameter_m) * max(0.0, float(target_n))
