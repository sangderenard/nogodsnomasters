"""The engine's chemical boundary: a source flux, on the solver's axis."""
import sys
from pathlib import Path

import pytest

import chemistry_boundary as cb
import combustion_products as cp
import emissions
import organic_species as chem
import working_fluids


_TURING = Path(__file__).resolve().parents[2] / "turing"


@pytest.fixture(scope="module")
def registry():
    if str(_TURING) not in sys.path:
        sys.path.insert(0, str(_TURING))
    try:
        from src.common.chemistry.compendium import atmospheric_aqueous_nucleus
        from src.common.chemistry.organics import (
            combustion_identity_registry, with_organic_combustion)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"chemistry compendium not importable: {error}")
    return combustion_identity_registry(
        with_organic_combustion(atmospheric_aqueous_nucleus()))


@pytest.fixture(scope="module")
def plan(registry):
    return cb.plan_boundary(registry)


def _case(fuel, fuel_kg_s=0.002, phi=0.7, load=1.0, oil_kg_s=4.0e-6):
    fluid = working_fluids.working_fluid(fuel)
    ci = fluid.ignition == "compression"
    mol = chem.molecule(chem.FUEL_COMPOSITION[fuel])
    air = fuel_kg_s * mol.stoichiometric_afr() / phi
    eo = emissions.engine_out(air + fuel_kg_s, fuel_kg_s, phi, load, ci)
    return cp.for_engine_out(fuel, fuel_kg_s, phi, load, eo, oil_kg_s=oil_kg_s)


def test_every_product_resolves_to_a_canonical_state(plan):
    """Resolved once at deployment, so an unmappable species stops the
    build rather than appearing as a quiet shortfall on step one
    thousand."""
    assert plan.states["water"] == "H2O@gas"
    assert plan.states["soot"] == "soot@solid"
    assert plan.states["sulfuric-acid"] == "H2SO4@liquid"
    assert plan.states["lubricating-oil"] == "lube-oil@liquid"
    assert plan.states["heavy-fuel-oil-blend"] == "residual-fuel-oil@gas"


def test_phase_is_where_the_species_crosses_the_boundary(registry, plan):
    """Not its ambient phase. Water is liquid at room temperature and
    unambiguously a gas leaving a running engine, and routing the whole
    water flux into the wrong state is what the distinction prevents."""
    assert chem.molecule("water").ambient_phase == "liquid"
    assert plan.states["water"].endswith("@gas")


def test_an_unmapped_species_is_refused_not_dropped(registry, plan):
    """Mass that vanishes between two conserving models is the worst
    possible failure here: it drifts with no obvious cause."""
    with pytest.raises(cb.UnmappedSpecies):
        cb.species_state(registry, "n-heptane")      # no compendium row
    with pytest.raises(cb.UnmappedSpecies):
        cb.species_state(registry, "benzene")        # no declared phase

    narrow = cb.plan_boundary(registry, species=["water", "carbon-dioxide"])
    rates = _case("ultra-low-sulfur-diesel")
    with pytest.raises(cb.UnmappedSpecies):
        cb.exchange_mixture(rates, narrow)


@pytest.mark.parametrize("fuel,phi", [
    ("ultra-low-sulfur-diesel", 0.70),
    ("heavy-fuel-oil", 0.80),
    ("pump-gasoline-87", 1.00),
    ("hydrogen", 0.60),
])
def test_it_packs_and_round_trips_through_the_exchange_layout(plan, fuel, phi):
    """The whole engine/solver interface, exercised end to end.

    Kilograms per second in, moles on the manifest's exact state axis
    out, and back to kilograms without loss."""
    from src.common.chemistry import (
        ComponentChemistry, EngineSpeciesExchangeLayout, deployment_manifest)
    from src.common.chemistry.compendium import atmospheric_aqueous_nucleus
    from src.common.chemistry.organics import with_organic_combustion

    compendium = with_organic_combustion(atmospheric_aqueous_nucleus())
    manifest = deployment_manifest(compendium, [ComponentChemistry(
        "engine-bay", tuple(sorted(set(plan.states.values()))))])
    layout = EngineSpeciesExchangeLayout(
        manifest, (plan.endpoint,), 2, plan.bindings())

    rates = _case(fuel, fuel_kg_s=0.002 if fuel != "hydrogen" else 0.0005,
                  phi=phi)
    packed = cb.inject(rates, layout, plan)
    assert int(packed.numel()) == layout.physical_size

    back = layout.unpack_moles(packed)[plan.endpoint]
    master = manifest.reduced.master
    for species, kg in rates.kg_s.items():
        if kg <= 0.0:
            continue
        state = plan.states[species]
        molar = master.species[master.states[state].species].molar_mass_kg_mol
        assert back[state] * molar == pytest.approx(kg, rel=1e-12)


def test_the_boundary_authors_no_reaction():
    """It is a source flux. Combustion chemistry belongs to the
    compendium, which balances and integrates it with its own laws; a
    second reaction solver on the engine side would have to be
    reconciled with that one forever."""
    source = Path(cb.__file__).read_text(encoding="utf-8")
    for forbidden in ("EquilibriumLaw", "KineticLaw", "Reaction.make",
                      "log10_k", "arrhenius"):
        assert forbidden not in source, forbidden


def test_what_crosses_the_boundary_still_balances(plan):
    """The flux is proven element-closed before it is packed, so the
    solver is never handed a source that violates conservation."""
    rates = _case("heavy-fuel-oil", phi=0.8)
    closure = rates.closure()
    for element in cp.ELEMENTS_TRACKED:
        assert abs(closure[f"{element}_residual_kg_s"]) < 1e-9 * rates.fuel_kg_s
    assert closure["charge_residual_kg_s"] == 0.0
    mixture = cb.exchange_mixture(rates, plan)[plan.endpoint]
    assert sum(mixture.values()) == pytest.approx(rates.total_kg_s)
