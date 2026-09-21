"""The engine's chemical boundary into the master solver.

WHAT THIS IS, AND WHAT IT REFUSES TO BE

An engine is not a reaction. It is a thing that injects species amounts
at a boundary, and this module is that injection and nothing else:
`combustion_products.ProductRates` in, canonical chemical-state amounts
out, packed through the exchange layout the chemistry deployment already
defines (`EngineSpeciesExchangeLayout`).

It authors no equilibrium constant, no rate law and no reaction row.
Combustion chemistry belongs to the compendium, which balances every
reaction it accepts over elements and charge and integrates them with
its own laws. A second reaction solver living on the engine side would
have to be reconciled with that one forever, and would lose. What the
engine side legitimately owns is a SOURCE FLUX -- how much of each
species crosses the boundary per second -- and the arithmetic that
proves that flux conserves every element, which
`combustion_products.burn` already does before anything gets here.

Where the flux cannot be made consistent, this module does not repair
it. A rich charge whose products need more oxygen than the air held is
reported by `ProductRates.closure()` as `oxygen_overdrawn_kg_s` and
passed through as a finding. The physical resolution -- more carbon
staying as carbon monoxide -- is equilibrium speciation, which is the
solver's reaction rows, not a correction to apply here.

THE CONCORDANCE IS DECLARED, NOT INFERRED

`species_state()` reads one explicit table. No name is derived from
another by lowercasing, by stripping, by prefix or by any other rule,
and the fact that `soot` happens to be spelled the same on both sides
is a coincidence this module does not rely on and does not generalise
from. The canonical side of the mapping lives in
`src/common/chemistry/organics.ENGINE_CONCORDANCE`, which
`ChemicalIdentityRegistry.verify_external_composition` proves is
formula-identical to the engine rows -- so a concordance row that is
merely plausible fails a test rather than quietly mis-routing a species.

What this module adds on top of that species concordance is the PHASE,
because a chemical state is a species and a phase together and an
engine's exhaust carries the same element in three of them: carbon
dioxide as a gas, sulphate as a liquid droplet, soot as a solid. A
species map alone cannot name a state.
"""
from __future__ import annotations

from dataclasses import dataclass

import organic_species as chem


#: The phase each product leaves the engine in, declared per species.
#:
#: Not inferred from `Molecule.ambient_phase`: what matters is the phase
#: the species is in AS IT CROSSES THE BOUNDARY, in a hot exhaust
#: stream, which is not the phase it would sit in at room temperature.
#: Water is the clearest case -- liquid at ambient, and unambiguously a
#: gas leaving a running engine -- and getting it wrong would route the
#: whole water flux into the wrong state of the solver's axis.
PRODUCT_PHASE: dict[str, str] = {
    # the bulk gases
    "carbon-dioxide": "gas",
    "water": "gas",
    "nitrogen": "gas",
    "oxygen": "gas",
    "argon": "gas",
    # the incomplete products, all gases
    "carbon-monoxide": "gas",
    "nitric-oxide": "gas",
    "nitrogen-dioxide": "gas",
    "sulfur-dioxide": "gas",
    # the condensed phase: this is what makes smoke visible
    "soot": "solid",
    "elemental-carbon": "solid",
    # a sulphate aerosol is a real liquid droplet, which is why it shows
    # up in a particulate number as well as in a sulphur one
    "sulfuric-acid": "liquid",
    # lubricating oil past a ring leaves as droplets, not vapour. This
    # is the blue smoke, and it is a different state from unburnt fuel
    # in the same way it is a different colour.
    "lubricating-oil": "liquid",
    # unburnt fuel escapes as vapour
    "gasoline-blend": "gas",
    "diesel-blend": "gas",
    "kerosene-blend": "gas",
    "heavy-fuel-oil-blend": "gas",
    "crude-blend": "gas",
    "methane": "gas",
    "propane": "gas",
    "hydrogen-gas": "gas",
    "methanol": "gas",
    "ethanol": "gas",
    "nitromethane": "gas",
}


class UnmappedSpecies(KeyError):
    """A product with no declared phase or no canonical counterpart.

    Raised rather than skipped. A species dropped silently at the
    boundary is mass that vanishes between two models that each believe
    they conserve it, and it would show as a slow drift with no obvious
    cause -- so the boundary refuses to carry what it cannot name."""


@dataclass(frozen=True)
class BoundaryPlan:
    """The declared exchange for one engine endpoint.

    `states` is the canonical chemical-state identity for each engine
    species this endpoint can emit. It is computed once, at deployment,
    so a running tick does no lookup that could fail."""
    endpoint: str
    states: dict           # engine species key -> canonical state key

    def bindings(self):
        """`ExternalStateBinding` rows for `EngineSpeciesExchangeLayout`.

        Imported lazily so `engine_toy` keeps working without the
        chemistry package on the path -- the engine-side balance is
        useful on its own, and only the boundary needs the solver."""
        from src.common.chemistry import ExternalStateBinding

        return tuple(
            ExternalStateBinding(self.endpoint, species, state)
            for species, state in sorted(self.states.items()))


def species_state(registry, species: str) -> str:
    """Canonical chemical state for one engine species, or raise.

    Two declared lookups, in order: the species concordance, then the
    phase table above. Neither is inferred."""
    phase = PRODUCT_PHASE.get(species)
    if phase is None:
        raise UnmappedSpecies(
            f"{species!r} has no declared boundary phase. Add it to "
            f"PRODUCT_PHASE with the phase it is in AS IT LEAVES THE "
            f"ENGINE, which is not necessarily its ambient phase.")
    try:
        return registry.state(species, phase)
    except KeyError as error:
        raise UnmappedSpecies(
            f"{species!r} as {phase} is not a declared chemical state. "
            f"Either the species is absent from the compendium (see "
            f"organics.UNMAPPED_ENGINE_SPECIES) or that phase is not "
            f"allowed for it.") from error


def plan_boundary(registry, endpoint: str = "engine.exhaust",
                  species=None) -> BoundaryPlan:
    """Resolve every species this endpoint can emit, once, up front.

    Resolving at deployment rather than per tick is the point: an
    unmappable species is a configuration error and should stop the
    build, not appear on the thousandth step as a quiet shortfall."""
    names = sorted(PRODUCT_PHASE) if species is None else sorted(set(species))
    return BoundaryPlan(endpoint, {n: species_state(registry, n) for n in names})


def exchange_mixture(rates, plan: BoundaryPlan) -> dict:
    """`ProductRates` as the mixture `pack_kilograms` expects.

    Keyed by ENGINE identity, which the layout's bindings route to the
    canonical state; the amounts are kilograms per second and the layout
    divides by each species' own molar mass. For a one-carbon-basis
    pseudo-species that conversion yields moles of fuel CARBON, which is
    what the compendium's row means and what its formula is written per."""
    mixture = {}
    for species, kg in rates.kg_s.items():
        if kg <= 0.0:
            continue
        if species not in plan.states:
            raise UnmappedSpecies(
                f"{species!r} is leaving the engine but is not in this "
                f"endpoint's plan, so it has nowhere to go. Extend the "
                f"plan rather than dropping it: silently skipping a "
                f"product is mass lost between two conserving models.")
        mixture[species] = float(kg)
    return {plan.endpoint: mixture}


def inject(rates, layout, plan: BoundaryPlan):
    """Pack one interval's source flux onto the solver's state axis.

    Returns the layout's own packed tensor, indexed
    `[endpoint][state][limb]`, in moles. This is the whole engine/solver
    interface: a source term, on the manifest's exact state axis, with
    every element already proven balanced on the way in."""
    return layout.pack_kilograms(exchange_mixture(rates, plan))


def audit(rates, plan: BoundaryPlan) -> str:
    """What crosses the boundary, and what the balance had to say."""
    lines = [f"endpoint {plan.endpoint}", ""]
    for species, kg in sorted(rates.kg_s.items(), key=lambda kv: -kv[1]):
        if kg <= 0.0:
            continue
        state = plan.states.get(species, "UNMAPPED")
        lines.append(f"  {species:<22} -> {state:<22} {kg * 1e3:>12.6f} g/s")
    closure = rates.closure()
    lines.append("")
    lines.append("  LEDGER (residual per element, plus charge and mass)")
    for key in sorted(closure):
        if key.endswith("_residual_kg_s"):
            lines.append(f"    {key:<28} {closure[key]:+.3e}")
    if rates.note:
        lines += ["", f"  {rates.note}"]
    return "\n".join(lines)
