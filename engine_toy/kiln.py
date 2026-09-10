"""Real kiln/furnace physics for a high-temperature industrial
attachment -- smelting ore, firing ceramics, stress-relief annealing --
anything needing sustained heat well past what an open firebox reaches.
Built on the SAME real fuel/air machinery gas_works.py already has
(GasGenerator/Boiler-class fuel sources, Blower for draft) rather than
inventing a parallel combustion system: a kiln is that same real
burner, pointed into a real insulated chamber, with a real chamber
thermal balance layered on top.

One correction on the way in: fire is NOT plasma. A flame is a real,
ordinary exothermic gas-phase chemical reaction -- hot and light-
emitting, but its ionization fraction is negligible even at a hot oxy-
fuel flame's ~3400 K. Genuine plasma (a gas with a real, significant
fraction of its molecules stripped into ions/free electrons) needs
either much higher real temperatures (tens of thousands of K --
lightning, welding-arc, stellar/fusion territory) or an external energy
source doing the ionizing directly. That real arc IS modeled here,
honestly: not as the kiln's actual heat source (an arc hot enough to
smelt a whole charge directly needs a genuinely enormous real
electrical supply, well past a vehicle's own 12/24 V accessory bus),
but as the real IGNITER -- a small, genuinely ionized kernel that
kindles the surrounding fuel/air mixture into an ordinary flame. That's
the same real role a spark plug or glow plug already plays elsewhere
in this toy, just with a hotter, more energetic real ignition source
for a burner that has to light reliably against a full forced-draft
blast instead of a quiescent cylinder charge.

Two real, physically DIFFERENT atmospheres, not a difficulty setting:
  - "ceramics-firing"/"annealing" want OXIDIZING conditions (excess
    air) -- a real reducing atmosphere causes real glaze/body defects
    and incomplete, sooty combustion.
  - "iron-smelting" needs a real REDUCING (fuel-rich, CO-rich)
    atmosphere -- 2Fe2O3 + 3CO -> 4Fe + 3CO2 is the real bloomery/
    blast-furnace chemistry, and CO is exactly what gas_works.py's own
    producer-gas generator already makes (see PROCESS_ATMOSPHERE and
    Kiln.step's own real combustion-efficiency penalty for the wrong
    atmosphere).

The blower's own real drive (see gas_works.BlowerSpec) is deliberately
open here, not fixed -- but "pto-clutch" is the real, representative
choice for a kiln-scale attachment: real fan power scales with flow x
pressure rise, and at real industrial air demand that's a genuine
fraction of a whole engine's output, not an accessory-belt afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gas_works import Blower, BlowerSpec, COMBUSTION_AIR_PER_FUEL_KG, FUEL_HHV_J_PER_KG

# Real physical constant -- the Stefan-Boltzmann constant, W/(m^2 K^4).
STEFAN_BOLTZMANN_W_PER_M2_K4 = 5.670374419e-8


# Real, disclosed representative process temperatures -- order-of-
# magnitude figures for each real process, not one specific kiln's
# logbook.
PROCESS_TARGET_TEMP_K = {
    "ceramics-firing": 1_573.15,   # ~1300 C, real stoneware/porcelain firing range
    "iron-smelting": 1_803.15,     # ~1530 C, real bloomery/blast-furnace range (pure iron melts at 1538 C)
    "annealing": 1_073.15,         # ~800 C, real steel stress-relief range
}

PROCESS_ATMOSPHERE = {
    "ceramics-firing": "oxidizing",
    "iron-smelting": "reducing",
    "annealing": "oxidizing",
}

# Real, disclosed combustion-efficiency penalty for running a process
# in the WRONG atmosphere (a lean burn on a smelt, or a rich/sooty burn
# on a ceramics firing) -- doesn't zero it out (some real heat is still
# released either way), but a real mismatched atmosphere both wastes
# fuel and, for smelting specifically, genuinely fails to reduce the
# ore at all regardless of temperature reached (see Kiln.reducing_ok).
ATMOSPHERE_MISMATCH_PENALTY_FRAC = 0.5


@dataclass(frozen=True)
class KilnSpec:
    """A real refractory-lined chamber, a real burner (fuel/air ratio
    below), and a real forced-draft Blower feeding it."""
    process: str                    # "ceramics-firing" | "iron-smelting" | "annealing"
    chamber_mass_kg: float = 300.0  # real refractory brick/insulation thermal mass
    chamber_specific_heat_j_per_kg_k: float = 900.0  # real, typical firebrick/refractory figure
    # Real, disclosed lumped wall-loss coefficient (W/K): how fast the
    # chamber sheds heat to ambient per degree above it. Scales with
    # insulation quality; not modeled in more real detail than one
    # disclosed number (a full radiation+conduction solve is out of
    # scope for this toy, same convention the boiler's own steam-table
    # interpolation already uses instead of full IAPWS). This is the
    # real CONDUCTIVE/convective term -- dominant at modest temperature
    # rise, but genuinely NOT what caps a real kiln's high-end
    # equilibrium; see chamber_surface_area_m2/wall_emissivity below
    # for the real radiative term that actually does.
    wall_loss_w_per_k: float = 25.0
    # Real radiative loss surface -- at real kiln temperatures
    # (hundreds to ~1500+ C), radiative loss (Stefan-Boltzmann, scaling
    # with T^4) genuinely dominates conduction and is the real reason a
    # kiln finds a stable equilibrium instead of climbing indefinitely
    # under a burner that outweighs its conductive loss alone. 6 m^2 is
    # a disclosed, representative surface for a modest chamber (roughly
    # a 1 m cube); 0.8 is a real, typical refractory-brick emissivity.
    chamber_surface_area_m2: float = 6.0
    wall_emissivity: float = 0.8
    raw_fuel: str = "coal"
    combustion_efficiency: float = 0.80  # real, disclosed: a lined kiln runs hotter/cleaner than an open firebox
    # equivalence ratio the operator is actually running: 1.0 =
    # stoichiometric, <1.0 = lean/oxidizing (excess air), >1.0 =
    # rich/reducing (excess fuel) -- this, not the process name alone,
    # is what Kiln.step actually checks against PROCESS_ATMOSPHERE.
    equivalence_ratio: float = 1.0
    blower: BlowerSpec = field(default_factory=lambda: BlowerSpec(max_flow_kg_s=0.3, drive="pto-clutch"))
    ambient_temp_k: float = 293.15

    @property
    def target_temp_k(self) -> float:
        return PROCESS_TARGET_TEMP_K.get(self.process, 1_273.15)

    @property
    def required_atmosphere(self) -> str:
        return PROCESS_ATMOSPHERE.get(self.process, "oxidizing")

    def build(self) -> "Kiln":
        return Kiln(spec=self, chamber_temp_k=self.ambient_temp_k, _blower=self.blower.build())


@dataclass
class Kiln:
    """One real, running kiln instance. Needs BOTH a lit pilot and a
    real air supply to do anything at all -- an unlit kiln, or one with
    no draft (blower off/declutched/starved), just cools toward ambient
    like any other real thermal mass."""
    spec: KilnSpec
    chamber_temp_k: float = 293.15
    pilot_lit: bool = False
    _blower: Blower = None
    fuel_valve_frac: float = 1.0   # the real operator's own fuel-side lever, independent of the blower's damper

    def light_pilot(self) -> None:
        """Real electric-arc (or glow-element) pilot -- see this
        module's own docstring on why the arc's real localized
        ionization is the igniter, not the kiln's actual heat source."""
        self.pilot_lit = True

    def extinguish(self) -> None:
        self.pilot_lit = False

    def step(self, dt: float, engine_rpm: float = 0.0, pto_engaged: bool = False) -> float:
        """Advances the real chamber thermal state by dt, returns the
        new chamber temperature."""
        spec = self.spec
        if spec.blower.drive == "pto-clutch":
            self._blower.pto_engaged = pto_engaged
        air_kg_s = self._blower.air_output_kg_s(dt, engine_rpm) if self.pilot_lit else 0.0
        # real: fuel follows the declared equivalence ratio off
        # whatever air is actually being delivered, not the other way
        # around -- a starved blower starves the burn, same as every
        # other real forced-draft device in this toy
        stoich_fuel_kg_s = air_kg_s / COMBUSTION_AIR_PER_FUEL_KG
        fuel_kg_s = stoich_fuel_kg_s * max(0.0, spec.equivalence_ratio) * max(0.0, min(1.0, self.fuel_valve_frac))
        hhv = FUEL_HHV_J_PER_KG.get(spec.raw_fuel, 25.0e6)
        efficiency = spec.combustion_efficiency
        atmosphere_matched = (self.reducing_ok if spec.required_atmosphere == "reducing"
                               else self.oxidizing_ok)
        if not atmosphere_matched:
            efficiency *= ATMOSPHERE_MISMATCH_PENALTY_FRAC
        heat_in_w = fuel_kg_s * hhv * efficiency if self.pilot_lit else 0.0
        conductive_loss_w = spec.wall_loss_w_per_k * (self.chamber_temp_k - spec.ambient_temp_k)
        # real Stefan-Boltzmann radiative loss -- the T^4 term that
        # actually caps a real kiln's high-end equilibrium; see
        # KilnSpec.chamber_surface_area_m2's own docstring
        radiative_loss_w = (spec.wall_emissivity * STEFAN_BOLTZMANN_W_PER_M2_K4
                             * spec.chamber_surface_area_m2
                             * (self.chamber_temp_k ** 4 - spec.ambient_temp_k ** 4))
        heat_loss_w = conductive_loss_w + radiative_loss_w
        thermal_mass_j_per_k = spec.chamber_mass_kg * spec.chamber_specific_heat_j_per_kg_k
        d_temp = (heat_in_w - heat_loss_w) / max(thermal_mass_j_per_k, 1.0) * dt
        self.chamber_temp_k = max(spec.ambient_temp_k, self.chamber_temp_k + d_temp)
        return self.chamber_temp_k

    @property
    def reducing_ok(self) -> bool:
        """Real: an equivalence ratio meaningfully above 1.0 (fuel-
        rich) is what actually produces the CO a smelt needs -- reached
        temperature alone never reduces an oxide without it."""
        return self.spec.equivalence_ratio > 1.1

    @property
    def oxidizing_ok(self) -> bool:
        return self.spec.equivalence_ratio < 0.95

    @property
    def at_process_temp(self) -> bool:
        return self.chamber_temp_k >= self.spec.target_temp_k

    @property
    def ready_for_process(self) -> bool:
        """Real: reaching temperature isn't enough for a smelt -- the
        atmosphere has to actually be reducing too, or the ore just
        sits there hot and unreduced."""
        if not self.at_process_temp:
            return False
        if self.spec.required_atmosphere == "reducing":
            return self.reducing_ok
        return self.oxidizing_ok
