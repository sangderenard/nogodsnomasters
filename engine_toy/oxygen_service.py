"""Oxygen service: why the compressor is a different machine, and what
happens when it is not.

This module exists to make one accident EMERGE rather than be scripted.
Nothing here decides that a compressor catches fire. What it does is
carry three quantities the rest of the sim already produces --
contaminant mass from fouling, oxygen fraction from the separation
plant, and gas temperature from the compressor's own heat of compression
-- and let them meet. When they meet in the wrong place the ignition is
a consequence, and the yield is computed from what was actually there.

THE OXYGEN FIRE TRIANGLE IS THE ORDINARY ONE WITH THE NUMBERS MOVED.
Fuel, oxidiser, ignition. What changes in oxygen service is that all
three get dramatically easier:

  the fuel      is anything. Not "fuel" -- ANYTHING carbonaceous. A
                fingerprint, a thread of cotton, the residue of the
                solvent somebody cleaned the part with. Oxygen systems
                are degreased not because grease is dirty but because
                grease is a fuel.

  the oxidiser  is not 21% any more. Autoignition temperature falls by
                roughly a hundred and fifty kelvin going from air to
                pure oxygen, and minimum ignition energy falls by about
                two orders of magnitude.

  the ignition  is free, because compression makes heat. A compressor
                does not need an ignition source; it IS one.

AND THE PRESSURE TERM IS THE ONE THAT CATCHES PEOPLE. Autoignition
temperature falls further as pressure rises -- so a hydrocarbon that
needs 620 K in air at one bar can go at something near 370 K in oxygen
at two hundred. That is BELOW the normal discharge temperature of an
ordinary compressor stage. Which is the whole reason oxygen compressors
are diaphragm or ionic-liquid machines with no lubricant in the gas
path: not as a precaution against a rare event, but because a
conventional compressor in oxygen service is operating above the
autoignition temperature of its own lubricant as a matter of routine.

THE KINDLING CHAIN IS WHAT MAKES IT AN EXPLOSION RATHER THAN A FIRE.
A few grams of oil burning is a few hundred kilojoules and would be
survivable. But metals burn in high-pressure oxygen -- aluminium at
31 MJ/kg, carbon steel at 6.7 -- and the oil fire is the kindling that
lights them. Once the housing itself is burning there is no extinguishing
it, because the oxidiser is the process fluid and shutting the valve
traps it inside. The energy release is then set by how much METAL is
involved, and that is three orders of magnitude above the contaminant
that started it.

THE THIRD TERM IS NOT CHEMICAL AT ALL. A vessel at two hundred bar is
already a bomb before anything burns. That stored pneumatic energy adds
to the chemical release and is computed the same way burst.py computes
it for any other pressure vessel, so the two agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ATM_PA = 101_325.0
R_AIR_J_PER_KG_K = 287.0
TNT_J_PER_KG = 4.184e6
#: Pure oxygen's oxidising power on the registry's air-relative scale.
#: Derived, not asserted: 1 / 0.232, because air is 23.2% oxygen by mass.
OXYGEN_POWER = 1.0 / 0.232
#: Kilograms of oxygen to burn a kilogram of ordinary hydrocarbon.
#: CH2 + 1.5 O2, so 48 g of oxygen per 14 g of fuel.
STOICH_O2_PER_FUEL = 48.0 / 14.0


@dataclass
class Atmosphere:
    """A declared mixture of oxidisers, by mass fraction.

    Vector, not scalar, because that is what the quantity is: a space
    can hold air and oxygen and nitrous at once, and the ignition
    behaviour depends on all of them together. The properties come from
    the fluid registry so there is exactly one place that says how
    strong an oxidiser each fluid is."""
    fractions: dict = field(default_factory=lambda: {"gas": 1.0})

    def _rows(self):
        import fluids as fl
        by = {f.key: f for f in fl.FLUIDS}
        out = []
        for key, frac in self.fractions.items():
            f = by.get(key)
            if f is None:
                raise KeyError(f"undeclared fluid {key!r} in an atmosphere")
            out.append((f, max(0.0, float(frac))))
        return out

    @property
    def total(self) -> float:
        return sum(w for _, w in self._rows()) or 1.0

    @property
    def oxidising_power(self) -> float:
        """Mass-weighted, with AIR as 1.0.

        Air itself is not declared an oxidiser in the registry -- it is
        the baseline everything else is measured against -- so it
        contributes 1.0 per unit mass here rather than 0."""
        tot = 0.0
        for f, w in self._rows():
            tot += w * (f.oxidising_power if getattr(f, "oxidiser", False) else 1.0)
        return tot / self.total

    @property
    def oxygen_equivalence(self) -> float:
        """Kilograms of usable oxygen per kilogram of this atmosphere."""
        tot = 0.0
        for f, w in self._rows():
            tot += w * (f.oxygen_equivalence if getattr(f, "oxidiser", False) else 0.232)
        return tot / self.total

    @property
    def decomposition_j_kg(self) -> float:
        """Energy the atmosphere can release WITH NO FUEL AT ALL.

        Zero for air and oxygen. Not zero for nitrous oxide, and that is
        the whole reason an N2O bottle is a different hazard class from
        an O2 bottle -- it does not need anything to burn."""
        tot = 0.0
        for f, w in self._rows():
            if getattr(f, "oxidiser", False) and f.self_sustaining:
                tot += w * f.decomposition_j_kg
        return tot / self.total

    @property
    def self_sustaining(self) -> bool:
        return any(getattr(f, "oxidiser", False) and f.self_sustaining
                   for f, w in self._rows() if w > 0.0)

    @property
    def label(self) -> str:
        return ", ".join(f"{k} {v * 100 / self.total:.0f}%"
                         for k, v in self.fractions.items() if v > 0.0)


# ---------------------------------------------------------------------
# what can burn, and how easily
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Combustible:
    key: str
    label: str
    #: Autoignition temperature IN AIR at one atmosphere. The datasheet
    #: number, and the one that is misleading in oxygen service.
    ait_air_k: float
    heat_of_combustion_j_kg: float
    #: How far the AIT falls going from air to pure oxygen.
    oxygen_sensitivity_k: float
    #: How far it falls per e-fold of pressure. This is the term that
    #: turns a safe margin into a negative one on a high-pressure stage.
    pressure_sensitivity_k: float
    kindling: bool = False        # can it light something harder to light
    why: str = ""

    def autoignition_k(self, atmosphere=None, pressure_pa: float = ATM_PA) -> float:
        """Where this actually goes off, in THIS atmosphere at THIS
        pressure. Not the datasheet number.

        `atmosphere` is an Atmosphere -- a declared mixture of oxidisers
        from the fluid registry -- not an oxygen fraction. A scalar
        cannot express a space that is sixty per cent nitrous oxide, and
        nitrous is not merely a weaker oxygen: it brings its own energy
        and so pushes ignition harder than its oxygen content alone
        would suggest."""
        power = 1.0 if atmosphere is None else atmosphere.oxidising_power
        # 1.0 is air by definition; pure oxygen is 4.31; nitrous is above
        # that. Normalised so "1.0 of enrichment" means "as far as air to
        # pure oxygen", and values past 1.0 are allowed and meaningful.
        enrich = max(0.0, (power - 1.0) / (OXYGEN_POWER - 1.0))
        p = max(ATM_PA, float(pressure_pa))
        drop = (self.oxygen_sensitivity_k * enrich
                + self.pressure_sensitivity_k * math.log(p / ATM_PA))
        return max(250.0, self.ait_air_k - drop)


COMBUSTIBLES: dict[str, Combustible] = {
    "hydrocarbon-oil": Combustible(
        "hydrocarbon-oil", "mineral oil or grease", 620.0, 46.0e6, 150.0, 20.0,
        kindling=True,
        why="the classic contaminant and the reason for degreasing. In air it needs "
            "620 K, which no compressor reaches; in oxygen at 200 bar it needs about "
            "370 K, which every compressor reaches on every stroke"),
    "ptfe": Combustible(
        "ptfe", "PTFE seal material", 850.0, 5.0e6, 180.0, 35.0,
        why="chosen for oxygen service because it is hard to light, and it is -- but "
            "it is not impossible to light, and at high pressure the margin is much "
            "smaller than the air figure suggests"),
    "elastomer": Combustible(
        "elastomer", "nitrile or viton seal", 620.0, 35.0e6, 160.0, 25.0,
        kindling=True,
        why="an O-ring is a fuel sitting in the gas path by design. Oxygen-service "
            "seals are specified by ignition behaviour, not by sealing ability"),
    "cotton-lint": Combustible(
        "cotton-lint", "lint, cloth, paper", 530.0, 17.0e6, 190.0, 15.0,
        kindling=True,
        why="a rag fibre left in a fitting. Enormous surface area for its mass, so it "
            "lights at the bottom of the range and lights everything else"),
    "aluminium": Combustible(
        "aluminium", "aluminium housing", 1200.0, 31.0e6, 400.0, 60.0,
        why="metals burn in oxygen and aluminium burns well -- more energy per "
            "kilogram than TNT by a factor of seven. It will not start on its own; it "
            "needs kindling, and a few grams of oil is enough kindling"),
    "carbon-steel": Combustible(
        "carbon-steel", "carbon steel housing", 1600.0, 6.7e6, 500.0, 90.0,
        why="iron to magnetite. Steel genuinely burns in high-pressure oxygen -- this "
            "is what oxy-fuel cutting IS, and a cutting torch is exactly this reaction "
            "run deliberately"),
    "stainless": Combustible(
        "stainless", "stainless steel", 1750.0, 4.5e6, 520.0, 95.0,
        why="better than carbon steel and not immune. Chromium oxide is protective "
            "until it is not, and at high oxygen pressure it is not"),
}


def combustible(key: str) -> Combustible:
    c = COMBUSTIBLES.get(str(key))
    if c is None:
        raise KeyError(f"unknown combustible {key!r}; declared: "
                       f"{', '.join(sorted(COMBUSTIBLES))}")
    return c


def adiabatic_shock_k(initial_k: float, from_pa: float, to_pa: float,
                      gamma: float = 1.395) -> float:
    """Gas temperature after a fast pressurisation into a dead end.

    THE CLASSIC OXYGEN ACCIDENT, and it needs no compressor at all.
    Crack a valve quickly into a closed line and the gas already in that
    line is compressed adiabatically by the gas arriving. From one bar
    to two hundred that is over twelve hundred kelvin -- hotter than the
    autoignition temperature of anything on the list above, reached in
    milliseconds, with no moving machinery involved.

    It is why oxygen valves are opened slowly, and why "opened it too
    fast" is a sufficient explanation for a destroyed regulator."""
    r = max(1.0, float(to_pa) / max(1.0, float(from_pa)))
    return float(initial_k) * r ** ((gamma - 1.0) / gamma)


# ---------------------------------------------------------------------
# the compressor, as a thing that can be contaminated
# ---------------------------------------------------------------------

@dataclass
class OxygenCompressor:
    """A compressor in oxygen service, carrying its own contamination.

    Not a new compressor model -- it wraps whatever construction is
    declared in compressors.py and adds the two things oxygen service
    cares about: what is in the gas path that should not be, and how hot
    each stage gets."""
    identity: str = "asu.o2_compressor"
    construction: str = "diaphragm"
    stages: int = 3
    per_stage_ratio: float = 4.0
    inlet_k: float = 293.15
    #: Intercooling between stages. Without it stage temperatures stack
    #: and the machine cooks itself; with it each stage starts fresh.
    intercooled: bool = True
    intercooler_outlet_k: float = 308.15
    intercooler_fouled_frac: float = 0.0
    atmosphere: Atmosphere = field(
        default_factory=lambda: Atmosphere({"oxygen": 1.0}))
    #: Contaminant actually present in the gas path, in grams, by kind.
    #: Fed by fouling.py, by a failed diaphragm, or by assembly.
    contaminant_g: dict = field(default_factory=dict)
    #: A diaphragm compressor's real failure mode. The gas side and the
    #: hydraulic side are separated by a stack of thin metal diaphragms
    #: with leak detection BETWEEN the layers, precisely because this is
    #: the hazard. A ruptured stack puts hydraulic oil into oxygen.
    diaphragm_layers: int = 3
    diaphragm_breached: int = 0
    housing_material: str = "carbon-steel"
    housing_kg: float = 90.0
    #: Margin the housing is built to over working pressure. Real
    #: pressure vessels carry three to four.
    burst_safety_factor: float = 3.0
    swept_volume_m3: float = 0.004
    ignited: bool = False
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def polytropic_index(self) -> float:
        try:
            import compressors as cp
            return float(cp.construction(self.construction).polytropic_index)
        except Exception:
            return 1.30

    @property
    def discharge_pressure_pa(self) -> float:
        return ATM_PA * self.per_stage_ratio ** self.stages

    def stage_outlet_k(self, stage: int) -> float:
        """Gas temperature leaving a given stage.

        A fouled intercooler is not a small efficiency loss here: the
        stage inlet rises, and because compression is multiplicative the
        outlet rises with it. An intercooler that has stopped working is
        a route to ignition, not to a higher power bill."""
        n = self.polytropic_index
        if self.intercooled:
            base = self.intercooler_outlet_k + (
                (self.inlet_k - self.intercooler_outlet_k)
                if stage == 0 else 0.0)
            # fouling lets the previous stage's heat through
            if stage > 0 and self.intercooler_fouled_frac > 0.0:
                unc = self.stage_outlet_k(stage - 1)
                base = base + (unc - base) * min(1.0, self.intercooler_fouled_frac)
        else:
            base = self.inlet_k if stage == 0 else self.stage_outlet_k(stage - 1)
        return base * self.per_stage_ratio ** ((n - 1.0) / n)

    @property
    def hottest_k(self) -> float:
        return max(self.stage_outlet_k(s) for s in range(max(1, self.stages)))

    @property
    def hottest_stage(self) -> int:
        temps = [self.stage_outlet_k(s) for s in range(max(1, self.stages))]
        return int(max(range(len(temps)), key=lambda i: temps[i]))

    def stage_pressure_pa(self, stage: int) -> float:
        return ATM_PA * self.per_stage_ratio ** (stage + 1)

    def contaminate(self, kind: str, grams: float) -> float:
        combustible(kind)                      # refuse unknown species
        self.contaminant_g[kind] = self.contaminant_g.get(kind, 0.0) + max(0.0, float(grams))
        return self.contaminant_g[kind]

    def breach_diaphragm(self, layers: int = 1, oil_g: float = 4.0) -> dict:
        """A layer lets go. The interlayer detector is what should catch
        this before the last layer goes."""
        self.diaphragm_breached = min(self.diaphragm_layers,
                                      self.diaphragm_breached + max(1, int(layers)))
        through = self.diaphragm_breached >= self.diaphragm_layers
        if through:
            self.contaminate("hydrocarbon-oil", oil_g)
        return {"breached_layers": self.diaphragm_breached,
                "of": self.diaphragm_layers, "oil_through": through,
                "detected": self.diaphragm_breached < self.diaphragm_layers,
                "why": ("interlayer leak detection exists to catch exactly this while "
                        "there is still a layer left. Through the last layer there is "
                        "hydraulic oil in the oxygen stream and no warning was missed "
                        "-- the warning was the earlier layer")}

    def ignition_check(self) -> dict:
        """Does anything in here go off, at the conditions it is at?

        Nothing is decided in advance: this compares each contaminant's
        autoignition temperature -- adjusted for the oxygen fraction and
        the stage pressure it is sitting in -- against the temperature
        that stage actually reaches."""
        findings = []
        worst_margin = float("inf")
        igniter = None
        for stage in range(max(1, self.stages)):
            t = self.stage_outlet_k(stage)
            p = self.stage_pressure_pa(stage)
            for kind, grams in self.contaminant_g.items():
                if grams <= 0.0:
                    continue
                c = combustible(kind)
                ait = c.autoignition_k(self.atmosphere, p)
                margin = ait - t
                findings.append({"stage": stage, "kind": kind, "grams": grams,
                                 "gas_k": t, "pressure_bar": p / 1e5,
                                 "ait_k": ait, "ait_air_k": c.ait_air_k,
                                 "margin_k": margin, "ignites": margin <= 0.0})
                if margin < worst_margin:
                    worst_margin, igniter = margin, findings[-1]
        return {"findings": findings, "worst_margin_k": worst_margin,
                "igniter": igniter,
                "ignites": igniter is not None and igniter["margin_k"] <= 0.0}

    def step(self, dt_s: float) -> dict:
        """Run it. Returns an ignition record if one happened."""
        r = self.ignition_check()
        if r["ignites"] and not self.ignited:
            self.ignited = True
            r["yield"] = self.yield_j()
        return r

    # -----------------------------------------------------------------
    # AN ENCLOSED IGNITION IS A COMBUSTION CHAMBER EVENT
    # -----------------------------------------------------------------
    #
    # This is the distinction that decides whether anything explodes at
    # all, and it is not about how much energy is released -- it is
    # about whether the energy is CONFINED.
    #
    # A spray that lights in open air is a fire. It burns at
    # atmospheric pressure, the hot gas expands away freely, and the
    # only output is heat. Nothing bursts because nothing was holding
    # anything in.
    #
    # The same reaction inside a sealed housing is a constant-volume
    # combustion, which is exactly what a cylinder does at top dead
    # centre: the charge cannot expand, so the heat release goes
    # entirely into raising temperature, and pressure follows
    # temperature directly. That is the engine's own ignition
    # arithmetic, applied to a machine never meant to be an engine.
    #
    # SO THE GATE IS A PRESSURE COMPARISON, not an energy one. Compute
    # what the housing reaches; compare it with what the housing holds.
    # Under it, the fire is contained and burns through the casing
    # slowly -- bad, but not a blast. Over it, the housing ruptures and
    # everything stored goes at once.
    #
    # And the numbers are brutal: a few grams of oil is enough. Not
    # because grams of oil are much energy, but because there is so
    # little gas in the chamber to absorb it.

    @property
    def free_volume_m3(self) -> float:
        """Gas space inside the housing: swept volume plus passages."""
        return self.swept_volume_m3 * 2.5

    @property
    def burst_pressure_pa(self) -> float:
        """What the housing holds. A real pressure vessel carries a
        genuine margin over working pressure -- but it is a margin
        against PRESSURE, and nobody designs one against its own
        contents catching fire."""
        return self.discharge_pressure_pa * self.burst_safety_factor

    def charge_kg(self) -> float:
        """Oxidiser actually in the housing at discharge conditions."""
        r_specific = 260.0          # oxygen
        return (self.discharge_pressure_pa * self.free_volume_m3
                / (r_specific * max(1.0, self.hottest_k)))

    def constant_volume_burn(self) -> dict:
        """The chamber calculation. Same shape as a cylinder, different
        oxidiser.

        Limited by whichever runs out first -- and in an oxygen
        compressor it is never the oxidiser, which is precisely what
        makes it dangerous. In air a few grams of oil would find only a
        few grams of oxygen nearby and self-limit; here the fuel burns
        to completion every time."""
        charge = self.charge_kg()
        fuel_kg = sum(self.contaminant_g.values()) / 1000.0
        o2_available = charge * self.atmosphere.oxygen_equivalence
        o2_needed = fuel_kg * STOICH_O2_PER_FUEL
        burned_kg = fuel_kg if o2_available >= o2_needed else \
            o2_available / STOICH_O2_PER_FUEL
        total_g = sum(self.contaminant_g.values()) or 1.0
        lhv = sum(combustible(k).heat_of_combustion_j_kg * g
                  for k, g in self.contaminant_g.items()) / total_g
        chem_j = burned_kg * lhv
        # AND THE OXIDISER'S OWN ENERGY, if it is a monopropellant. This
        # term has no equivalent in an air engine: nitrous decomposing
        # releases energy whether or not there is any fuel at all.
        decomp_j = charge * self.atmosphere.decomposition_j_kg
        total_j = chem_j + decomp_j
        cv = 660.0                  # oxygen, constant volume
        dt_k = total_j / max(1e-6, charge * cv)
        t2 = self.hottest_k + dt_k
        p2 = self.discharge_pressure_pa * (t2 / max(1.0, self.hottest_k))
        return {"charge_kg": charge, "fuel_kg": fuel_kg, "burned_kg": burned_kg,
                "fuel_limited": o2_available >= o2_needed,
                "chemical_j": chem_j, "decomposition_j": decomp_j,
                "total_j": total_j, "peak_k": t2, "peak_pa": p2,
                "peak_bar": p2 / 1e5, "burst_bar": self.burst_pressure_pa / 1e5,
                "ruptures": p2 >= self.burst_pressure_pa,
                "pressure_ratio": p2 / max(1.0, self.discharge_pressure_pa)}

    def yield_j(self) -> dict:
        """What is released, and whether it gets out.

        Gated by the chamber calculation: if the housing holds there is
        no blast at all, only a contained fire that burns its way out,
        which is a different and slower failure."""
        burn = self.constant_volume_burn()
        kindling_j = sum(
            g / 1000.0 * combustible(k).heat_of_combustion_j_kg
            for k, g in self.contaminant_g.items() if combustible(k).kindling)
        housing = combustible(self.housing_material)
        housing_ait = housing.autoignition_k(self.atmosphere,
                                             self.discharge_pressure_pa)
        lit = kindling_j > 50_000.0 and burn["peak_k"] > housing_ait
        involved_frac = min(0.35, kindling_j / 2.0e6) if lit else 0.0
        metal_j = involved_frac * self.housing_kg * housing.heat_of_combustion_j_kg
        p = self.discharge_pressure_pa
        pneumatic = (self.charge_kg() * 260.0 * 293.15
                     * math.log(max(1.0001, p / ATM_PA)))
        if not burn["ruptures"]:
            return {"ruptures": False, "burn": burn, "metal_lit": lit,
                    "metal_j": metal_j, "total_j": 0.0, "tnt_kg": 0.0,
                    "volume_m3": max(0.01, self.free_volume_m3 * 4.0),
                    "why": ("the housing held at {:.0f} bar against a {:.0f} bar "
                            "burst. This is a CONTAINED fire, not a blast -- and "
                            "containing an oxygen fire is not a reprieve, because "
                            "the metal goes on burning until it opens a hole of its "
                            "own").format(burn["peak_bar"], burn["burst_bar"])}
        total = burn["total_j"] + metal_j + pneumatic
        return {"ruptures": True, "burn": burn,
                "chemical_j": burn["chemical_j"],
                "decomposition_j": burn["decomposition_j"],
                "metal_j": metal_j, "metal_lit": lit,
                "housing_involved_frac": involved_frac,
                "pneumatic_j": pneumatic, "total_j": total,
                "tnt_kg": total / TNT_J_PER_KG,
                "volume_m3": max(0.01, self.free_volume_m3 * 4.0),
                "why": ("the contaminant is the fuse, not the bomb. What decides "
                        "the yield is whether the housing lit -- metal fire is "
                        "orders of magnitude above the oil that started it")}

    def hand_off(self, sim, part: str = "") -> dict:
        """Give the yield to the damage engine.

        Uses the sim's own raise_blast, so this failure arrives through
        exactly the same path as a burst vessel or a demolition charge
        and gets the same fragments, sound and overpressure."""
        y = self.yield_j()
        if not y.get("ruptures", False):
            # a contained fire raises no blast, and saying so is the
            # point: the handoff is conditional on the chamber result.
            return y
        sim.raise_blast(y["total_j"], part or self.identity, y["volume_m3"])
        return y

    def service_need(self):
        import servicing as sv
        r = self.ignition_check()
        if self.diaphragm_breached <= 0 and r["worst_margin_k"] > 60.0:
            return None
        urgent = self.diaphragm_breached >= self.diaphragm_layers or r["ignites"]
        return sv.Need(
            identity=self.identity, want="oxygen-service", position=tuple(self.position),
            quantity=1.0, unit="each", matches="degrease",
            urgency=1.0 if urgent else 0.7, minutes=180.0, skill="mechanic",
            label=(f"{self.identity}: {self.diaphragm_breached}/{self.diaphragm_layers} "
                   f"diaphragm layers breached, ignition margin "
                   f"{r['worst_margin_k']:.0f} K"),
            why="an oxygen compressor with hydrocarbon in the gas path is operating "
                "near or above the autoignition temperature of its own contamination; "
                "there is no safe running interval, only a shutdown")
