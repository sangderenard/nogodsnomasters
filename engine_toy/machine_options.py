"""Optional assemblies: what a machine comes with, and what that costs.

loadouts.py already answers "what equipment does this machine carry, and
can its plant run all of it at once". That is the DUTY question. This is
the other one, which it does not ask:

    Is this assembly fitted at all -- and if it is not, what has to
    happen instead?

An autoclave is the clean example. It can be supplied with its own
boiler, in which case it needs nothing but fuel and water, and the
boiler takes up a large piece of the machine's own volume. Or it can be
supplied without one, in which case that volume is free for something
else and the machine now REQUIRES a steam connection from outside --
which means a flanged inlet on the casing that the integral version
does not need at all.

Both are the same machine. Neither is a variant of the other. And the
difference is not a flag, it is three real consequences:

  1. PARTS. Fitted, the assembly contributes real bodies with real mass
     and real plumbing.

  2. SPACE. Fitted, it CLAIMS a volume -- and turret_production already
     enforces claims: clear_volumes are checked against every member,
     with the observation that "a clearance that is not checked is a
     comment". An optional assembly's claim is exactly that mechanism,
     asserted only when the option is taken. Unfitted, the volume is
     available and something else may use it.

  3. REQUIREMENTS. This is the part that makes options interesting
     rather than bookkeeping. An assembly PROVIDES something, and if it
     is absent that thing must arrive another way -- through a port that
     only exists in the other configuration. So removing the boiler does
     not simply delete a box: it obliges the casing to grow a steam
     inlet, and a configuration that removes the boiler WITHOUT adding
     the inlet is incomplete, not merely lighter.

WHY THIS IS NOT A SUBCLASS PER VARIANT. Two options can both claim the
same space, and then they are mutually exclusive by geometry rather than
by anyone declaring them so -- which is the honest reason a machine
cannot have both an integral boiler and a bigger chamber. Conflicts fall
out of the volumes, and a combination nobody thought to forbid is still
caught.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Claim:
    """A volume an assembly occupies when it is fitted.

    Same record shape turret_production.clear_volumes already uses --
    identity, owner, min, max -- so a claim asserted by a fitted option
    IS a clear volume and gets that module's existing enforcement
    without translation."""
    identity: str
    min: tuple
    max: tuple
    note: str = ""

    def volume_m3(self) -> float:
        return abs((self.max[0] - self.min[0])
                   * (self.max[1] - self.min[1])
                   * (self.max[2] - self.min[2]))

    def overlaps(self, other: "Claim") -> bool:
        return all(self.min[i] < other.max[i] and other.min[i] < self.max[i]
                   for i in range(3))

    def as_clear_volume(self, owner: str) -> dict:
        return {"identity": self.identity, "owner": owner,
                "min": self.min, "max": self.max, "note": self.note}


@dataclass(frozen=True)
class OptionalAssembly:
    """One thing a machine may or may not be supplied with.

    `provides` and `requires` are what turn a parts list into a
    configuration problem. An assembly that provides "steam" satisfies
    any requirement for steam; one that requires "fuel" adds a
    requirement of its own, because an integral boiler has to be fed."""
    identity: str
    label: str
    provides: tuple = ()
    requires: tuple = ()
    claims: tuple = ()               # Claim records
    mass_kg: float = 0.0
    #: Ports the CASING must carry when this option is fitted. An
    #: integral boiler needs a flue and a feedwater connection; it does
    #: not need a steam inlet, because it makes its own.
    casing_ports: tuple = ()
    note: str = ""

    def claimed_m3(self) -> float:
        return sum(c.volume_m3() for c in self.claims)


@dataclass
class Configuration:
    """One buildable combination, and whether it actually works."""
    machine: str
    fitted: tuple = ()               # OptionalAssembly records
    #: What the machine needs no matter which options are taken.
    base_requires: tuple = ()
    #: Requirements a site can satisfy from outside, through a port --
    #: the shore supplies. Declaring these is what makes "without a
    #: boiler" a real configuration rather than a broken one.
    shore_supplies: tuple = ()

    def provided(self) -> set:
        out = set()
        for a in self.fitted:
            out.update(a.provides)
        return out

    def required(self) -> set:
        out = set(self.base_requires)
        for a in self.fitted:
            out.update(a.requires)
        return out

    def unsatisfied(self) -> list:
        """What nothing on this machine provides.

        Each is either met from shore -- in which case the casing needs
        a port for it -- or it is a hole in the configuration."""
        return sorted(self.required() - self.provided())

    def from_shore(self) -> list:
        return [r for r in self.unsatisfied() if r in self.shore_supplies]

    def missing(self) -> list:
        return [r for r in self.unsatisfied() if r not in self.shore_supplies]

    def space_conflicts(self) -> list:
        """Options that cannot coexist, found by geometry.

        Nobody has to declare two assemblies mutually exclusive: if they
        want the same cubic metres they are, and a combination nobody
        thought to forbid is caught anyway."""
        out = []
        claims = [(a, c) for a in self.fitted for c in a.claims]
        for i, (a1, c1) in enumerate(claims):
            for a2, c2 in claims[i + 1:]:
                if a1.identity == a2.identity:
                    continue
                if c1.overlaps(c2):
                    out.append(f"{a1.label} and {a2.label} both want "
                               f"{c1.identity} / {c2.identity}: "
                               f"{c1.volume_m3() * 1000:.0f} L of the same space")
        return out

    def required_casing_ports(self) -> list:
        """Every port this configuration obliges the casing to carry.

        Two sources, and the second is the interesting one: the ports
        the fitted options need, PLUS a port for every requirement met
        from shore. Take the boiler out and the casing grows a steam
        inlet it did not previously need -- that is the 'alternate
        housing' an absent assembly forces."""
        ports = []
        for a in self.fitted:
            ports.extend(a.casing_ports)
        for r in self.from_shore():
            ports.append(f"{r}-inlet")
        return sorted(set(ports))

    def freed_space_m3(self, catalogue) -> float:
        """Volume left available by the options NOT taken."""
        fitted = {a.identity for a in self.fitted}
        return sum(a.claimed_m3() for a in catalogue if a.identity not in fitted)

    def mass_kg(self) -> float:
        return sum(a.mass_kg for a in self.fitted)

    def check(self, catalogue=()) -> dict:
        conflicts = self.space_conflicts()
        missing = self.missing()
        return {"machine": self.machine,
                "fitted": [a.label for a in self.fitted],
                "buildable": not conflicts and not missing,
                "missing": missing,
                "from_shore": self.from_shore(),
                "space_conflicts": conflicts,
                "casing_ports": self.required_casing_ports(),
                "claimed_m3": sum(a.claimed_m3() for a in self.fitted),
                "freed_m3": self.freed_space_m3(catalogue),
                "mass_kg": self.mass_kg()}

    def apply_to(self, graph) -> None:
        """Assert every fitted option's claim on a production graph.

        This is where the space reservation stops being a description:
        turret_production checks clear_volumes against every member, so
        a claim asserted here is enforced by machinery that already
        exists."""
        for a in self.fitted:
            for c in a.claims:
                graph.clear_volumes.append(c.as_clear_volume(a.identity))


# ---------------------------------------------------------------------
# the autoclave's own options, as a worked example
# ---------------------------------------------------------------------

INTEGRAL_BOILER = OptionalAssembly(
    identity="autoclave.integral_boiler",
    label="integral boiler",
    provides=("steam",),
    # AN INTEGRAL BOILER IS NOT FREE OF REQUIREMENTS -- it simply moves
    # them. It has to be fed fuel and water, and its flue has to go
    # somewhere, so taking this option trades one shore connection for
    # three.
    requires=("fuel", "feedwater"),
    claims=(Claim(identity="autoclave.boiler_bay",
                  min=(-0.75, -0.55, -0.60), max=(-0.34, 0.55, 0.60),
                  note="the boiler, its firebox and the space to pull tubes"),),
    mass_kg=310.0,
    casing_ports=("flue", "fuel-fill", "feedwater-fill", "ashpan-door"),
    note="self-contained: needs only fuel and water, and carries its own "
         "pressure source")

SHORE_STEAM = OptionalAssembly(
    identity="autoclave.shore_steam_set",
    label="shore steam connection",
    provides=("steam",),
    requires=(),
    # A CONNECTION CLAIMS ALMOST NOTHING, which is the entire trade:
    # the boiler bay is free for something else.
    claims=(Claim(identity="autoclave.shore_manifold",
                  min=(-0.75, 0.30, -0.15), max=(-0.60, 0.50, 0.15),
                  note="reducing valve, strainer and trap set"),),
    mass_kg=24.0,
    casing_ports=("steam-inlet-flange", "condensate-return"),
    note="needs steam from somewhere else, and the site has to have it")

VACUUM_SET = OptionalAssembly(
    identity="autoclave.vacuum_set",
    label="pre-vacuum pump set",
    provides=("vacuum",),
    requires=("electricity",),
    claims=(Claim(identity="autoclave.pump_bay",
                  min=(0.34, -0.40, 0.05), max=(0.72, -0.05, 0.42),
                  note="pump, motor and the room to change its oil"),),
    mass_kg=48.0,
    casing_ports=("vacuum-line", "pump-drain"),
    note="without it the purge is downward-displacement only, which cannot "
         "reach air trapped in a porous load")

DIAPHRAGM_VACUUM_SET = OptionalAssembly(
    identity="autoclave.diaphragm_vacuum_set",
    label="PTFE diaphragm pump set",
    # THE SECOND THING IT PROVIDES IS THE WHOLE REASON IT EXISTS. Both
    # pumps pull a vacuum. Only this one may pull one through solvent
    # vapour, and saying so as a separate provision is what lets a
    # solvent duty REQUIRE it and be refused the other.
    provides=("vacuum", "solvent-safe-vacuum"),
    requires=("electricity",),
    # SAME BAY AS THE OIL-SEALED SET, which makes them alternatives
    # without anyone declaring them alternatives -- they want the same
    # space and space_conflicts says so.
    claims=(Claim(identity="autoclave.pump_bay",
                  min=(0.34, -0.40, 0.05), max=(0.72, -0.05, 0.42),
                  note="pump, motor and the room to change diaphragms"),),
    mass_kg=31.0,
    casing_ports=("vacuum-line", "pump-exhaust"),
    # NO DRAIN. An oil-sealed pump has one because it has oil to drop;
    # this has none, and it grows an EXHAUST instead, because what it
    # pulls out of the chamber has to be taken somewhere rather than
    # blown across the room.
    note="oil-free: the only wetted parts are the PTFE diaphragm and its "
         "valves, so there is nothing for a solvent to ruin")

#: WHY THE OIL-SEALED PUMP IS NOT MERELY WORSE HERE, in the order the
#: problems actually bite:
#:
#:   IT STOPS WORKING. Solvent vapour drawn into an oil-sealed pump
#:   condenses and dissolves into the sealing oil. Oil is what fills
#:   that pump's clearance and is the entire reason its ultimate is
#:   three decades below a diaphragm pump's; contaminated oil has a
#:   vapour pressure of its own and the pump can no longer reach the
#:   vacuum it was chosen for. Gas ballast exists to fight this and
#:   costs ultimate vacuum to do it -- compressors.VacuumPump already
#:   models that trade.
#:
#:   IT IS AN IGNITION SOURCE. Flammable vapour, hot oil and a motor
#:   in one housing is the arrangement nobody wants.
#:
#:   IT CORRODES. Chlorinated solvent and water make acid, and the
#:   pump internals are what it gets to.
#:
#: AND WHAT IT COSTS TO SWAP: ultimate vacuum, by three decades. That
#: sounds decisive and is not, because the DUTY here is a pre-vacuum
#: purge -- air out of a porous load before the steam goes in, which
#: wants something like 50 to 100 mbar absolute. A three-stage
#: diaphragm pump reaches about 1 mbar and is comfortably enough. The
#: oil-sealed pump's extra three decades buy nothing this machine
#: needs, and cost it every cycle that has a solvent in it.

LARGE_CHAMBER = OptionalAssembly(
    identity="autoclave.long_chamber",
    label="extended chamber",
    provides=(),
    requires=("steam",),
    # DELIBERATELY OVERLAPS THE BOILER BAY. Nobody declares these two
    # incompatible; they simply want the same metal, and
    # space_conflicts finds it.
    claims=(Claim(identity="autoclave.chamber_extension",
                  min=(-0.80, -0.35, -0.35), max=(-0.40, 0.35, 0.35),
                  note="the chamber runs further aft"),),
    mass_kg=95.0,
    casing_ports=(),
    note="a longer vessel, which has to grow into whatever is behind it")

CATALOGUE = (INTEGRAL_BOILER, SHORE_STEAM, VACUUM_SET,
             DIAPHRAGM_VACUUM_SET, LARGE_CHAMBER)

#: What a site can hand a machine through its casing. Declaring this is
#: what makes "no boiler" a configuration rather than a fault.
#: A site can pipe steam, wire power and run a fuel line. It cannot
#: hand a machine a pump that is safe to put solvent through -- that is
#: hardware the machine either carries or does not, which is why
#: "solvent-safe-vacuum" is deliberately absent from this list.
SITE_SUPPLIES = ("steam", "electricity", "fuel", "feedwater", "vacuum")


#: WHAT THE MACHINE IS BEING ASKED TO DO, as requirements. A duty is
#: not a mode flag: it is a thing the configuration has to be able to
#: supply, checked by exactly the mechanism that checks everything else.
#: Declaring the solvent duty this way is what makes fitting the wrong
#: pump an UNBUILDABLE machine rather than a machine that quietly ruins
#: its pump on the first cycle.
DUTIES = {
    "steam-sterilising": ("steam",),
    "porous-load": ("steam", "vacuum"),
    "solvent-dewax": ("steam", "vacuum", "solvent-safe-vacuum"),
}


def configuration(*options, base_requires=("steam",), duty: str | None = None,
                  shore=SITE_SUPPLIES) -> Configuration:
    needs = tuple(DUTIES[duty]) if duty else tuple(base_requires)
    return Configuration(machine="plant.autoclave", fitted=tuple(options),
                         base_requires=needs,
                         shore_supplies=tuple(shore))


def report(cfg: Configuration) -> str:
    r = cfg.check(CATALOGUE)
    L = [f"{r['machine']}: {', '.join(r['fitted']) or 'no options'}",
         f"  {'BUILDABLE' if r['buildable'] else 'NOT BUILDABLE'}   "
         f"{r['mass_kg']:.0f} kg of options, "
         f"{r['claimed_m3'] * 1000:.0f} L claimed, "
         f"{r['freed_m3'] * 1000:.0f} L left free"]
    if r["space_conflicts"]:
        L.append("  SPACE CONFLICTS (found by geometry, not by a rule):")
        for c in r["space_conflicts"]:
            L.append(f"    ! {c}")
    if r["from_shore"]:
        L.append(f"  met from shore: {', '.join(r['from_shore'])}")
    if r["missing"]:
        L.append(f"  UNMET: {', '.join(r['missing'])} -- nothing provides these "
                 "and the site cannot either")
    L.append(f"  casing must carry: {', '.join(r['casing_ports']) or 'nothing extra'}")
    return "\n".join(L)


if __name__ == "__main__":
    print(report(configuration(INTEGRAL_BOILER, VACUUM_SET)))
    print()
    print(report(configuration(SHORE_STEAM, VACUUM_SET)))
    print()
    print(report(configuration(INTEGRAL_BOILER, LARGE_CHAMBER)))
    print()
    print(report(configuration()))
