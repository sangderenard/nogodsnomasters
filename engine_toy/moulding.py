"""Silicone, wax, and the shell: making a mould instead of ramming one.

foundry.py rams sand around a pattern and pours. This is the other
route, and it exists because sand cannot do three things: it cannot hold
an undercut, it cannot be drawn off a shape with no draft, and it cannot
give a surface better than its own grain. Silicone and wax get around
all three by making the mould DESTRUCTIBLE or FLEXIBLE, which turns out
to be the whole idea.

THE CHAIN, AND WHY IT IS FOUR STEPS FOR ONE PART

    master pattern -> silicone mould -> wax copies -> ceramic shell
    -> burn the wax out -> pour -> break the shell off

    Every step exists to solve the step before it. Silicone flexes, so
    it comes off an undercut a rigid mould would lock onto -- but
    silicone cannot take molten metal. So it makes WAX copies instead,
    which are cheap and identical. Wax cannot take metal either, but it
    MELTS, so a ceramic shell built around it can be emptied without
    ever opening it -- and a mould with no parting line has no flash and
    no draft anywhere on the part.

    The price is that the shell is destroyed to get the casting out. One
    shell, one part, every time. That is why investment casting is
    expensive per piece and why the silicone mould matters: it is the
    only reusable thing in the chain.

COMPOUNDING SHRINKAGE, WHICH IS THE PROBLEM SAND DOES NOT HAVE

    In sand, the metal shrinks and you allow for it once. Here the WAX
    shrinks when it freezes in the silicone, and then the METAL shrinks
    when it freezes in the shell, and the errors multiply:

        final / master  =  (1 - wax_shrink) . (1 - metal_shrink)

    Wax loses around 4% and aluminium 6.5%, so a part comes out about
    10% small unless the master pattern is made oversize by exactly that
    compounded amount. Getting it wrong by allowing for only one of them
    is the classic beginner's error and produces a part that is visibly,
    uselessly undersized.

BURNOUT, WHICH BREAKS SHELLS

    Wax EXPANDS when it melts -- it is a normal material in that
    respect, unlike water. A shell heated slowly and evenly has molten
    wax pushing outward against ceramic from the inside, and ceramic is
    strong in compression and weak in tension, so it cracks. The real
    answer is counterintuitive: heat it FAST, so the wax touching the
    shell melts and runs first and leaves the rest somewhere to expand
    INTO. That is what a steam autoclave or a flash-fire burnout is
    doing, and it is why 'gently warm it through' is precisely wrong.

    And any wax left behind becomes gas the moment metal arrives.

SILICONE HAS A LIFE, AND IT IS COUNTED IN PULLS

    Not in time. RTV silicone tears at its thin sections, so a mould
    with fine detail and deep undercuts dies faster than a simple one --
    the same property that lets it release an undercut is what wears it
    out. It also has a real service temperature, and pouring a wax that
    is too hot cooks it: tin-cure gives up around 450 K, platinum-cure a
    little higher, and neither will ever take metal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- silicone, real properties ----------------------------------------
#: Service temperature, K. Above this the rubber degrades: it goes
#: sticky, loses tear strength, and the next pull destroys it.
SILICONE_SERVICE_K = {"tin-cure": 450.0, "platinum-cure": 500.0}
#: Linear shrinkage on curing. Tin-cure condensation silicone gives off
#: alcohol as it cures and shrinks measurably; platinum-cure addition
#: silicone gives off nothing and barely moves. That difference is the
#: whole reason platinum costs more.
SILICONE_CURE_SHRINK = {"tin-cure": 0.004, "platinum-cure": 0.0005}
#: Pulls before the mould tears, for a simple shape with no undercuts.
SILICONE_BASE_LIFE_PULLS = {"tin-cure": 25, "platinum-cure": 60}

# --- wax, real properties ---------------------------------------------
WAX_MELT_K = 336.0                 # ~63 C, real pattern wax
WAX_POUR_K = 350.0                 # poured a little above melting
WAX_DENSITY_KG_M3 = 900.0
WAX_LATENT_J_PER_KG = 180_000.0
WAX_SPECIFIC_HEAT_J_PER_KGK = 2100.0
#: Volumetric shrinkage when the wax freezes. Large, and the first half
#: of the compounding problem above.
WAX_SOLIDIFICATION_SHRINK = 0.04
#: And it EXPANDS by about this much on melting again -- the thing that
#: cracks shells during burnout.
WAX_MELT_EXPANSION = 0.042

# --- the ceramic shell ------------------------------------------------
#: Shell strength is in compression; tension is where it fails, and it
#: is roughly a tenth as strong that way. This ratio is why expanding
#: wax cracks shells and why the shell survives the metal's own head.
SHELL_TENSILE_PA = 3.5e6
SHELL_COMPRESSIVE_PA = 35e6
#: Young's modulus of the fired shell. With the tensile strength above
#: this sets the failure STRAIN, which is what actually decides whether
#: expanding wax cracks it.
SHELL_MODULUS_PA = 2.0e10
#: THE SHELL IS NOT A CLOSED VESSEL, and that is the whole reason
#: investment casting works at all. It has an open pouring cup, so any
#: wax that is BOTH molten AND connected to that cup leaves instead of
#: pushing. A first version of this modelled relief as a function of
#: heating rate alone and concluded that even a flash burnout cracks
#: every shell -- which would make the entire process impossible, and
#: is how the missing mechanism was found.
#:
#: What actually decides it is the TEMPERATURE GRADIENT. Heat from
#: outside and the wax against the ceramic melts first, opening a path
#: to the cup before anything deeper has expanded. Soak the whole thing
#: uniformly and the bulk reaches melting point together, with no path
#: out and nowhere to go. This is exactly why "into a hot furnace" and
#: "steam autoclave" work while "bring it up gently from cold" is the
#: advice that cracks shells.
UNIFORM_SOAK_GRADIENT_K_PER_M = 200.0
STEEP_GRADIENT_K_PER_M = 20_000.0

#: AN HONEST GAP, LEFT OPEN RATHER THAN TUNED SHUT.
#:
#: The mechanism above is right: wax expands on melting, ceramic is weak
#: in tension, and relief comes from molten wax draining out of the open
#: cup. What is NOT established is where the shell actually fails.
#:
#: A purely elastic comparison -- imposed strain against sigma/E --
#: convicts every real technique including the steam autoclave, which
#: plainly works in practice. Three separate formulations of the relief
#: term were tried and all three cracked every shell, so the error is
#: not in the relief: it is that a shell full of MOLTEN wax is not an
#: elastic containment problem at all. Liquid wax under pressure
#: extrudes; only wax that is still solid can impose a strain, and
#: modelling that properly needs a melt-front position, not a
#: temperature.
#:
#: Unlike engine_harm's skirt clearance (anchored to a published
#: machining spec) and materials_handling's shear work (anchored to
#: published kWh/tonne), there is no figure here to solve against.
#: Rather than pick a margin that makes the demo behave, the margin is
#: named as unjustified and the raw strain is reported alongside every
#: verdict. A reader can then see exactly how far the model is from
#: convicting, and the next person with a real dewax dataset can delete
#: this constant and fit the thing properly.
CRACK_MARGIN_UNCALIBRATED = 40.0
CALIBRATION_GAP = ("burnout cracking is mechanistically right and numerically "
                   "unanchored; CRACK_MARGIN_UNCALIBRATED is a placeholder, not "
                   "a measurement")
#: Coats of slurry and stucco. Each adds thickness and strength and
#: needs its own drying time, which is why a shell takes days.
SHELL_COAT_THICKNESS_M = 0.0008
SHELL_DRY_HOURS_PER_COAT = 4.0


def compounded_shrink(metal: str, wax_shrink: float = WAX_SOLIDIFICATION_SHRINK) -> float:
    """Total linear-equivalent shrinkage from master to finished part.

    Both stages multiply. Volumetric figures are converted to linear by
    the cube root, because a pattern is scaled in three dimensions and a
    6.5% volume loss is only a 2.2% linear one -- confusing the two is
    the other classic error here, in the opposite direction."""
    import phase_table as pt
    m = pt.phases(metal)
    freeze = m.get(pt.FREEZE) if m else None
    metal_vol = abs(freeze.volume_change_frac) if freeze else 0.0
    wax_lin = 1.0 - (1.0 - wax_shrink) ** (1.0 / 3.0)
    metal_lin = 1.0 - (1.0 - metal_vol) ** (1.0 / 3.0)
    return 1.0 - (1.0 - wax_lin) * (1.0 - metal_lin)


def master_oversize_factor(metal: str) -> float:
    """How much bigger the MASTER must be cut so the part comes out right."""
    return 1.0 / max(1e-6, 1.0 - compounded_shrink(metal))


@dataclass
class SiliconeMould:
    """A flexible mould that can release an undercut, and wears out doing it."""
    identity: str = "moulding.silicone"
    kind: str = "tin-cure"
    cavity_volume_m3: float = 0.0004
    #: 0..1, how much undercut and fine detail the shape has. The same
    #: property that makes silicone the right choice also kills it.
    detail_severity: float = 0.3
    pulls_used: int = 0
    torn: bool = False

    @property
    def service_k(self) -> float:
        return SILICONE_SERVICE_K.get(self.kind, 450.0)

    @property
    def life_pulls(self) -> int:
        base = SILICONE_BASE_LIFE_PULLS.get(self.kind, 25)
        return max(3, int(base * (1.0 - 0.7 * max(0.0, min(1.0, self.detail_severity)))))

    def cure_shrink(self) -> float:
        return SILICONE_CURE_SHRINK.get(self.kind, 0.004)

    def pour_wax(self, wax_temp_k: float = WAX_POUR_K) -> tuple:
        """One wax copy. Returns (ok, note)."""
        if self.torn:
            return False, "mould is torn -- it will flash and the copy is scrap"
        if wax_temp_k > self.service_k:
            self.torn = True
            return False, (f"wax at {wax_temp_k:.0f} K against a "
                           f"{self.service_k:.0f} K service limit: the silicone is "
                           "cooked. It goes sticky and tears on this pull")
        self.pulls_used += 1
        if self.pulls_used > self.life_pulls:
            self.torn = True
            return False, (f"torn after {self.pulls_used} pulls (rated "
                           f"{self.life_pulls} at this detail level). Deep undercuts "
                           "are what kill a silicone mould, and they are also why "
                           "you chose it")
        left = self.life_pulls - self.pulls_used
        return True, f"wax copy {self.pulls_used}, {left} pulls left in this mould"


@dataclass
class InvestmentShell:
    """A ceramic shell built around a wax pattern, then emptied of it."""
    identity: str = "moulding.shell"
    coats: int = 7
    pattern_volume_m3: float = 0.0004
    pattern_area_m2: float = 0.04
    wax_removed_frac: float = 0.0
    cracked: bool = False
    preheat_k: float = 293.15

    @property
    def thickness_m(self) -> float:
        return self.coats * SHELL_COAT_THICKNESS_M

    @property
    def build_hours(self) -> float:
        return self.coats * SHELL_DRY_HOURS_PER_COAT

    def imposed_strain(self, expansion_frac: float) -> float:
        """Linear strain the expanding wax forces on the shell.

        A volume trying to grow by expansion_frac stretches its
        container by the cube root of that -- the shell has to get
        linearly bigger in every direction, and 4.2% by volume is only
        1.4% linearly."""
        return (1.0 + max(0.0, expansion_frac)) ** (1.0 / 3.0) - 1.0

    def failure_strain(self) -> float:
        """How much the ceramic can stretch before it cracks: sigma/E.

        NO FITTED FACTOR. An earlier version of this computed a stress
        and multiplied it by a bare 0.001 to make the numbers behave,
        which is the thing this project keeps catching itself doing.
        Strain against failure strain needs no such term: both sides are
        dimensionless and both are real properties.

        The answer is brutal -- 3.5 MPa over 20 GPa is 0.0175%, against
        1.4% imposed. The shell can take barely a HUNDREDTH of what
        melting wax does to it, which is precisely why gentle burnout
        does not work and why steam autoclaves and flash furnaces exist
        at all."""
        return SHELL_TENSILE_PA / SHELL_MODULUS_PA

    def burnout(self, heating_rate_k_per_s: float,
                gradient_k_per_m: float | None = None) -> tuple:
        """Melt the wax out, and find out whether the shell survives.

        FAST IS SAFE, which is the counterintuitive part. Heat quickly
        and the wax against the ceramic melts first and runs out,
        leaving the bulk somewhere to expand into. Heat slowly and the
        whole mass softens and expands together with nowhere to go, and
        the shell fails in tension."""
        # how much of the expansion is relieved by wax escaping first
        # A steep gradient means the interface melts well before the
        # core does, so a drain path to the open cup exists while the
        # bulk is still solid. A shallow one means everything softens at
        # once, sealed in. Gradient defaults to what the given heating
        # rate implies: fast external heating IS a steep gradient.
        if gradient_k_per_m is None:
            gradient_k_per_m = (UNIFORM_SOAK_GRADIENT_K_PER_M
                                + heating_rate_k_per_s * 40_000.0)
        relieved = gradient_k_per_m / (gradient_k_per_m + UNIFORM_SOAK_GRADIENT_K_PER_M * 5.0)
        effective = WAX_MELT_EXPANSION * (1.0 - relieved)
        strain = self.imposed_strain(effective)
        limit = self.failure_strain()
        # UNCALIBRATED, AND SAID SO. See CALIBRATION_GAP below: the
        # elastic comparison convicts every real technique, so it is
        # reported and not used as the verdict.
        if strain > limit * CRACK_MARGIN_UNCALIBRATED:
            self.cracked = True
            return False, (f"SHELL CRACKED: {strain * 100:.3f}% strain imposed against "
                           f"a {limit * 100:.4f}% failure strain -- ceramic can take "
                           f"barely a hundredth of what melting wax does to it. At "
                           f"{heating_rate_k_per_s:.2f} K/s the gradient is only "
                           f"{gradient_k_per_m:.0f} K/m, so just "
                           f"{relieved * 100:.1f}% of the wax reaches the open cup "
                           "before the rest expands. Go FASTER, not gentler: melt the "
                           "interface first and give the bulk a way out")
        self.wax_removed_frac = min(1.0, 0.85 + 0.15 * relieved)
        residue = 1.0 - self.wax_removed_frac
        if residue > 0.05:
            return True, (f"shell intact but {residue * 100:.0f}% of the wax is still "
                          "in there, and it becomes gas the moment metal arrives -- "
                          "hold it hotter for longer")
        return True, (f"clean burnout, {self.wax_removed_frac * 100:.0f}% of the wax "
                      f"out, shell intact (strain {strain * 100:.3f}% against a "
                      f"{limit * 100:.4f}% elastic limit -- see CALIBRATION_GAP: the "
                      "margin between them is not yet anchored to anything)")

    def pour_advantage(self) -> str:
        """Why a hot shell beats cold sand.

        foundry.Crucible.fluidity_m is linear in superheat, and superheat
        is the difference between the metal and what it is running into.
        A shell preheated to 1000 K takes far less superheat to fill than
        cold sand does, which is exactly why investment casting can do
        thin walls that sand cannot."""
        if self.preheat_k < 700.0:
            return (f"shell at only {self.preheat_k:.0f} K: the metal will chill in "
                    "the thin sections. Preheat it -- that is half the reason this "
                    "process can cast thin walls")
        return (f"shell preheated to {self.preheat_k:.0f} K: metal stays fluid into "
                "thin sections that sand would cold-shut")


def plan(metal: str = "aluminium", mould: SiliconeMould | None = None,
         shell: InvestmentShell | None = None) -> str:
    mould = mould or SiliconeMould()
    shell = shell or InvestmentShell(preheat_k=1000.0)
    shrink = compounded_shrink(metal)
    over = master_oversize_factor(metal)
    L = [f"investment chain for {metal}",
         "",
         f"  COMPOUNDING SHRINKAGE: wax then metal, multiplied",
         f"    total linear {shrink * 100:.2f}%  ->  cut the master "
         f"{(over - 1) * 100:.2f}% OVERSIZE",
         f"    allowing for only one of the two leaves the part visibly undersized",
         "",
         f"  SILICONE ({mould.kind}): service {mould.service_k:.0f} K, "
         f"cure shrink {mould.cure_shrink() * 100:.2f}%, "
         f"{mould.life_pulls} pulls at this detail level"]
    ok, why = mould.pour_wax()
    L.append(f"    {why}")
    ok2, why2 = mould.pour_wax(wax_temp_k=470.0)
    L.append(f"    pouring wax at 470 K: {why2}")
    L.append("")
    L.append(f"  SHELL: {shell.coats} coats, {shell.thickness_m * 1000:.1f} mm, "
             f"{shell.build_hours:.0f} h to build and dry")
    for rate, lbl in ((0.02, "slow soak from cold"), (0.35, "steam autoclave"),
                      (0.8, "straight into a hot furnace")):
        s = InvestmentShell(coats=shell.coats,
                            pattern_volume_m3=shell.pattern_volume_m3,
                            pattern_area_m2=shell.pattern_area_m2)
        good, note = s.burnout(rate)
        L.append(f"    {lbl:18s} ({rate:.2f} K/s): {note}")
    L.append("")
    L.append(f"  {shell.pour_advantage()}")
    L.append("")
    L.append("  and the shell is destroyed getting the part out: one shell, one part.")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    print(plan(sys.argv[1] if len(sys.argv) > 1 else "aluminium"))
