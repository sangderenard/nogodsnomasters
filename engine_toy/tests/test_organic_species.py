"""The organic chemistry layer, checked against what the project already
declares by hand -- and against conservation, which nothing can fudge."""
import pytest

import combustion_products as cp
import emissions
import organic_species as chem
import working_fluids


# ---------------------------------------------------------------------
# the identity layer against the existing catalogues
# ---------------------------------------------------------------------

def test_stoichiometric_ratios_derive_from_formulas():
    """Air/fuel ratios computed from elemental formulas reproduce the
    ones working_fluids.py states by hand.

    Nothing in the formula table was fitted to these. Two rows disagree
    by more than a few percent and they are named in
    `test_the_two_disagreements_are_real_and_named` rather than excused
    by a loose tolerance here."""
    known_disagreements = {"biodiesel-b100", "engine-oil-waste"}
    checked = 0
    for row in chem.verify_against_working_fluids():
        if row["fuel"] in known_disagreements:
            continue
        assert abs(row["error_frac"]) < 0.02, (
            f"{row['fuel']} as {row['as']}: derived {row['derived_afr']:.2f} "
            f"vs declared {row['declared_afr']:.2f}")
        checked += 1
    assert checked >= 15, "the check should cover most of the catalogue"


def test_the_two_disagreements_are_real_and_named():
    """Where the derivation and the catalogue disagree, the derivation is
    the one with a reason behind it.

    Methyl oleate is C19H36O2 and its stoichiometric ratio is arithmetic
    -- about 12.6. The catalogue says 13.8 for B100. A lubricating base
    oil at CH1.87 works out near 14.5; the catalogue says 13.5 for waste
    oil. Both catalogue figures look eyeballed low, and this test exists
    so that changing either of them is a deliberate act rather than a
    silent one."""
    rows = {r["fuel"]: r for r in chem.verify_against_working_fluids()}
    assert rows["biodiesel-b100"]["derived_afr"] == pytest.approx(12.58, abs=0.1)
    assert rows["biodiesel-b100"]["declared_afr"] == 13.8
    assert rows["engine-oil-waste"]["derived_afr"] == pytest.approx(14.54, abs=0.1)
    assert rows["engine-oil-waste"]["declared_afr"] == 13.5


def test_carbon_fractions_match_gas_works():
    """The four solid fuels' carbon fractions fall out of their ultimate
    analyses rather than being typed beside them."""
    for row in chem.verify_carbon_fractions():
        assert abs(row["error_frac"]) < 0.02, row


def test_air_derives_its_own_molar_mass_and_oxygen_fraction():
    assert chem.AIR_MOLAR_MASS_G_MOL == pytest.approx(28.95, abs=0.05)
    # gas_works.py cites 0.233; the composition implies 0.2316
    assert chem.AIR_O2_MASS_FRACTION == pytest.approx(0.233, abs=0.002)
    # working_fluids gives compressed air R = 287.0
    assert chem.AIR_GAS_CONSTANT_J_KGK == pytest.approx(287.0, abs=0.5)


def test_derivation_reproduces_the_two_constants_cryogenics_hardcodes():
    """`DewarAtmosphere.pressure_pa` spells these 296.8 and 4124.0.

    Both come straight out of the molar masses declared here, which is
    what makes `mixture_gas_constant_j_kgk` an offer to that method
    rather than a competing opinion."""
    assert chem.molecule("nitrogen").gas_constant_j_kgk == pytest.approx(296.8, abs=0.1)
    assert chem.molecule("hydrogen-gas").gas_constant_j_kgk == pytest.approx(4124.0, abs=1.0)


def test_a_mixture_prices_itself_by_mass_not_by_assumption():
    """The case the hardcoded 296.8 gets wrong: an ullage that is mostly
    carbon dioxide is nowhere near air, and mixing is linear in moles."""
    air_like = chem.mixture_gas_constant_j_kgk({"nitrogen": 0.78, "oxygen": 0.21,
                                                "argon": 0.01})
    assert air_like == pytest.approx(chem.AIR_GAS_CONSTANT_J_KGK, rel=0.01)

    co2_ullage = chem.mixture_gas_constant_j_kgk({"carbon-dioxide": 1.0})
    assert co2_ullage == pytest.approx(188.9, abs=1.0)
    # priced as air it would be over-read by more than half
    assert chem.AIR_GAS_CONSTANT_J_KGK / co2_ullage > 1.5

    # The MOLAR MASS of a mixture is not the mass average of its parts:
    # half a kilogram of hydrogen brings fourteen times the moles that
    # half a kilogram of nitrogen does, so the mixture's molar mass sits
    # near the light end, nowhere near halfway. This is the step where a
    # mass-tracked gas space has to go through moles, and the reason
    # `mixture_molar_mass_g_mol` exists rather than an average.
    m_mix = chem.mixture_molar_mass_g_mol({"hydrogen-gas": 0.5, "nitrogen": 0.5})
    mass_average_m = 0.5 * 2.016 + 0.5 * 28.014
    assert m_mix < mass_average_m * 0.3

    # Specific gas constant, by contrast, IS exactly mass-linear, since
    # R = Ru/M and 1/M_mix is the mass-weighted sum of 1/M_i. Worth
    # asserting because it is the non-obvious half of the same identity
    # and it is what makes the hydrogen branch in
    # `DewarAtmosphere.pressure_pa` wrong rather than merely coarse: the
    # real answer for a part-hydrogen ullage is between the two
    # constants, and that branch can only ever return one of them.
    half_and_half = chem.mixture_gas_constant_j_kgk({"hydrogen-gas": 0.5,
                                                    "nitrogen": 0.5})
    assert half_and_half == pytest.approx(0.5 * 4124.0 + 0.5 * 296.8, rel=1e-3)
    assert 296.8 < half_and_half < 4124.0


def test_complete_combustion_conserves_mass():
    for key in ("methane", "gasoline-blend", "diesel-blend", "methanol",
                "nitromethane", "heavy-fuel-oil-blend", "wood-fuel",
                "triolein", "toluene"):
        assert abs(chem.products_conserve_mass(key, 1.0)) < 1e-12, key


def test_mixtures_are_refused_a_formula_rather_than_given_a_fiction():
    """Coal gas and wood gas are mixtures of carbon monoxide, hydrogen
    and methane. Giving either one an average formula would be inventing
    a substance that does not exist."""
    assert set(chem.unmapped_fuels()) == {"coal-gas", "wood-gas", "ethanol-e85"}
    with pytest.raises(KeyError):
        cp.for_engine_out("wood-gas", 0.001, 1.0, 1.0,
                          emissions.EngineOutRates(0, 0, 0, 0, 0.2))


def test_structure_not_carbon_count_predicts_sooting():
    """The whole reason the sooting index is a declared property: an
    aromatic soots far harder than an alkane of similar size, and a
    heavier alkane does not catch up."""
    benzene = chem.molecule("benzene")          # C6
    n_decane = chem.molecule("n-decane")        # C10, bigger
    assert benzene.sooting_index > n_decane.sooting_index * 2
    # and H/C is the axis it runs on
    assert benzene.hydrogen_to_carbon < n_decane.hydrogen_to_carbon
    assert chem.molecule("methane").hydrogen_to_carbon == 4.0


def test_smog_and_soot_are_different_questions():
    """Benzene soots hard and forms little ozone; a xylene does the
    reverse. No single 'hydrocarbon' number can express that."""
    benzene = chem.molecule("benzene")
    xylene = chem.molecule("xylene")
    assert benzene.sooting_index < xylene.sooting_index * 0.6
    assert benzene.ozone_reactivity_mir < xylene.ozone_reactivity_mir * 0.2
    # and methane is effectively not a smog precursor at all, which is
    # why regulation counts non-methane hydrocarbon separately
    assert chem.molecule("methane").ozone_reactivity_mir < 0.05


# ---------------------------------------------------------------------
# the balance
# ---------------------------------------------------------------------

def _case(fuel, fuel_kg_s=0.002, phi=0.7, load=1.0, oil_kg_s=0.0):
    fluid = working_fluids.working_fluid(fuel)
    ci = fluid.ignition == "compression"
    mol = chem.molecule(chem.FUEL_COMPOSITION[fuel])
    air = fuel_kg_s * mol.stoichiometric_afr() / phi
    eo = emissions.engine_out(air + fuel_kg_s, fuel_kg_s, phi, load, ci)
    return cp.for_engine_out(fuel, fuel_kg_s, phi, load, eo, oil_kg_s=oil_kg_s)


@pytest.mark.parametrize("fuel,phi,load", [
    ("ultra-low-sulfur-diesel", 0.70, 1.0),
    ("ultra-low-sulfur-diesel", 0.70, 0.3),
    ("heavy-fuel-oil", 0.80, 1.0),
    ("pump-gasoline-87", 1.00, 0.8),
    ("pump-gasoline-87", 1.45, 0.8),
    ("methanol-race", 1.10, 0.9),
    ("hydrogen", 0.60, 1.0),
])
def test_every_atom_is_accounted_for(fuel, phi, load):
    """Every element on the axis, plus charge, plus total mass.

    The name used to overpromise: it asserted carbon and hydrogen only,
    and adding nitrogen, oxygen and sulphur to the ledger immediately
    found two real faults the two-element version had been hiding -- the
    nitrogen in NOx arrived from nowhere, and the oxygen bound in CO and
    NOx was spent twice. Both are fixed by deriving molecular nitrogen
    and leftover oxygen as residuals the same way carbon dioxide and
    water already were."""
    r = _case(fuel, phi=phi, load=load, oil_kg_s=4.0e-6)
    c = r.closure()
    scale = max(r.fuel_kg_s, 1e-12)
    for element in cp.ELEMENTS_TRACKED:
        residual = c[f"{element}_residual_kg_s"]
        if element == "O" and c["oxygen_overdrawn_kg_s"] > 0.0:
            # a rich charge: the products asked for more oxygen than the
            # air held, which is reported rather than resolved here
            assert residual < 0.0
            assert residual == pytest.approx(-c["oxygen_overdrawn_kg_s"])
            continue
        assert abs(residual) < 1e-9 * scale, (element, c)
    assert c["charge_residual_kg_s"] == 0.0
    if c["oxygen_overdrawn_kg_s"] == 0.0:
        assert abs(c["mass_residual_kg_s"]) < 1e-9 * scale, c


def test_rich_mixtures_report_an_oxygen_they_cannot_pay_for():
    """The oxygen analogue of the carbon overdraw, and the same policy.

    Above about phi 1.4 the CO and HC trends in `emissions.py`, plus
    complete conversion of everything they do not claim, need more
    oxygen than the charge holds. The real resolution is that more
    carbon stays as CO -- equilibrium speciation, which is the master
    solver's reaction rows, not a boundary flux's business. So this
    module reports it and does not invent a shift reaction."""
    lean = _case("pump-gasoline-87", phi=1.0, load=0.8)
    rich = _case("pump-gasoline-87", phi=1.45, load=0.8)
    assert lean.closure()["oxygen_overdrawn_kg_s"] == 0.0
    assert rich.closure()["oxygen_overdrawn_kg_s"] > 0.0
    assert "OXYGEN OVERDRAWN" in rich.note
    # and it never emits a negative amount of anything
    assert all(v >= 0.0 for v in rich.kg_s.values())


def test_the_balance_catches_emissions_py_emitting_carbon_from_hydrogen():
    """A real defect, found by conservation rather than by inspection.

    `emissions.engine_out` derives carbon monoxide and hydrocarbon from
    equivalence ratio alone and never asks what the fuel is made of, so
    it returns about a tenth of a gram per second of carbon monoxide for
    a hydrogen engine. The balance cannot let that through: the fuel has
    no carbon atom to make it from."""
    eo = emissions.engine_out(0.029, 0.0005, 0.6, 1.0, False)
    assert eo.co_kg_s > 0.0, "if this ever becomes zero, emissions.py was fixed"

    r = _case("hydrogen", fuel_kg_s=0.0005, phi=0.6, load=1.0)
    assert r.closure()["carbon_overdrawn_kg_s"] > 0.0
    assert "CARBON OVERDRAWN" in r.note
    # and what comes out is physical regardless: no carbon species can
    # exceed what the fuel could supply
    assert r.get("soot") == 0.0
    assert r.get("carbon-monoxide") == 0.0
    assert r.get("hydrocarbon") == 0.0
    # the slip is real fuel, kept under the fuel's own key
    assert r.get("hydrogen-gas") > 0.0


def test_sulfur_reaches_the_exhaust_at_all():
    """The connection that did not exist: working_fluids declares sulphur
    on nine fuels, air_treatment has had an so2 slot the whole time."""
    ulsd = _case("ultra-low-sulfur-diesel")
    hfo = _case("heavy-fuel-oil", phi=0.8)
    assert ulsd.get("sulfur-dioxide") > 0.0
    # 2.5 % sulphur against 15 ppm: three orders of magnitude apart, and
    # that is the real difference between bunker fuel and road diesel
    assert hfo.get("sulfur-dioxide") / ulsd.get("sulfur-dioxide") > 100.0
    # and part of it is a droplet, not a gas -- why low-sulphur fuel cut
    # measured particulate before any filter was fitted
    assert hfo.get("sulfuric-acid") > 0.0
    assert hfo.particulate_kg_s > hfo.get("soot")


def test_a_diesel_smokes_at_a_ratio_a_petrol_engine_would_call_lean():
    """The structural point of the module: global phi is the wrong
    variable for a diffusion flame."""
    diesel = _case("ultra-low-sulfur-diesel", phi=0.7, load=1.0)
    petrol = _case("pump-gasoline-87", phi=0.7, load=1.0)
    assert diesel.get("soot") > petrol.get("soot") * 20


def test_smoke_falls_off_with_load_not_with_mixture_on_a_diesel():
    full = _case("ultra-low-sulfur-diesel", phi=0.7, load=1.0)
    part = _case("ultra-low-sulfur-diesel", phi=0.7, load=0.3)
    assert part.get("soot") < full.get("soot") * 0.2


def test_a_rich_petrol_engine_makes_far_more_smog_than_a_stoich_one():
    stoich = _case("pump-gasoline-87", phi=1.0, load=0.8)
    rich = _case("pump-gasoline-87", phi=1.45, load=0.8)
    species = chem.FUEL_COMPOSITION["pump-gasoline-87"]
    assert (rich.ozone_potential_kg_s(species)
            > stoich.ozone_potential_kg_s(species) * 5)


def test_premixed_soot_only_ever_rises_with_richness():
    """No step in the middle of the curve where the rich mechanism takes
    over from the floor."""
    g = chem.molecule("gasoline-blend")
    previous = -1.0
    for i in range(80):
        phi = 0.6 + i * 0.03
        value = cp.soot_fraction_of_fuel(g, phi, 0.8, False)
        assert value >= previous - 1e-18, f"soot dropped at phi {phi:.2f}"
        previous = value


def test_blue_smoke_is_a_different_substance_from_black():
    """Oil past the rings leaves as droplets of oil, not as soot."""
    clean = _case("pump-gasoline-87", phi=1.0, load=0.8, oil_kg_s=0.0)
    worn = _case("pump-gasoline-87", phi=1.0, load=0.8, oil_kg_s=2.0e-5)
    assert clean.get("lubricating-oil") == 0.0
    assert worn.get("lubricating-oil") > 0.0
    assert worn.get("soot") == clean.get("soot")
    assert worn.particulate_kg_s > clean.particulate_kg_s * 3


def test_it_hands_air_treatment_a_loaded_contaminant():
    """The join to the gas-cleaning train that already exists."""
    r = _case("heavy-fuel-oil", phi=0.8, oil_kg_s=4.0e-6)
    c = cp.to_contaminant(r)
    for species in ("soot", "so2", "nox", "co", "hc", "oil"):
        assert c.get(species) > 0.0, species
    # the stage vocabulary air_treatment already declares still applies
    assert "soot" in c.SOLIDS and "so2" in c.SOLUBLE_GASES


def test_carbon_dioxide_and_water_are_most_of_what_leaves():
    """The mass nothing in this project was counting."""
    r = _case("ultra-low-sulfur-diesel")
    combustion = r.get("carbon-dioxide") + r.get("water")
    assert combustion > r.particulate_kg_s * 1000
    # roughly three kilograms of CO2 per kilogram of diesel, which is
    # what the carbon fraction requires and what a fuel-burn figure is
    # converted with in the real world
    assert r.get("carbon-dioxide") / r.fuel_kg_s == pytest.approx(3.17, abs=0.1)
