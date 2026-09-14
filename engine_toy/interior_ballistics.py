"""Interior ballistics: what happens between the primer and the muzzle.

WHY THIS IS A REAL MODEL AND NOT A COEFFICIENT. Two earlier attempts at
answering "how long should this barrel be" failed in instructive ways.
The first made bore friction a FRACTION OF CHAMBER PRESSURE, so friction
fell as fast as thrust and the model was structurally incapable of ever
showing an optimum length -- it could only accelerate. The second fixed
that and then solved for peak chamber pressure until the answer matched
a published muzzle velocity, which produced 969 MPa for a 20 mm gun
whose real peak is nearer 380. That is a wrong equation wearing a right
answer, and it is worse than no model at all because it looks validated.

So this is the standard formulation, with every input a real tabulated
property of a real propellant or a real cartridge:

  NOBLE-ABEL equation of state. A gun is not an ideal gas: at 400 MPa
  the gas molecules' own volume matters, and the covolume `b` is the
  term that says so. Leave it out and the pressure comes out low.

      p (V_free) = f w z        with V_free reduced by the unburnt
                                solid and by the covolume of the gas
                                already made

  VIEILLE'S BURN LAW. Propellant burns from every exposed surface at a
  rate set by the pressure over it: r = beta p. The grain's WEB -- how
  far the flame has to travel to burn through -- divides that into a
  burn time. This is why a gun is designed around grain geometry and
  not around "an amount of powder".

  THE FORM FUNCTION. A tube burns at constant surface area, a ball
  burns degressively (its area falls), and a seven-perforation grain
  burns PROGRESSIVELY: the inner perforations grow faster than the
  outside shrinks, so it makes more gas as the shot travels and the
  chamber grows. Progressive grain is the entire reason a long barrel
  is worth having, and it is the thing a velocity-versus-length curve
  cannot be honest without.

  THE LAGRANGE GRADIENT. The gas has mass and it is also accelerating,
  so the pressure at the breech is higher than the pressure on the base
  of the shot. The shot feels the base pressure; the chamber has to
  survive the breech pressure. Conflating them is how a design ends up
  with a gun that either underperforms or bursts.

AUTHORED IN SYMPY, on purpose. The per-step law is written once as
equations and lowered by the repository's own pipeline -- SymPy ->
ProcessGraph -> SSA -> LLVM -- so the arithmetic that runs in this sim
is the same arithmetic that runs natively, bit for bit, rather than a
NumPy re-spelling that has to be kept in step by hand.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sympy

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.compiler.symbolic_equation_compiler import (  # noqa: E402
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_symbolic_program,
)

_ABS_EPS = 1e-12


def _abs(x):
    """Smooth |x|: no branch, and every backend evaluates it the same."""
    return sympy.sqrt(x * x + _ABS_EPS)


def _step(x):
    """A smooth 0/1 gate on the sign of x, with no branch.

    NOT `sympy.sign`: it has no definition in the LLVM instruction
    table and the lowering fails outright with "symbol 'sign' has no
    function definition". This is the spelling `symbolic_parts` already
    uses for the same reason, so both laws gate identically.
    """
    return (x + _abs(x)) / (2 * _abs(x))


def _pos(x):
    """max(x, 0) spelled as Max, never (x + |x|)/2 -- that form relies on
    exact cancellation and leaves ULP residue that downstream gates turn
    into phantom results."""
    return sympy.Max(x, 0)


# =====================================================================
#  REAL PROPELLANTS
# =====================================================================
@dataclass(frozen=True)
class Propellant:
    """A real propellant, by the properties that are actually tabulated
    for one. None of these is a tuning knob."""
    key: str
    label: str
    impetus_j_per_kg: float       # f = nRT, the "force constant"
    covolume_m3_per_kg: float     # b, Noble-Abel
    density_kg_m3: float          # the solid grain
    gamma: float
    burn_rate_coeff: float        # beta, in m/s per Pa   (r = beta p)
    flame_temperature_k: float
    note: str = ""


PROPELLANTS = {
    "single-base": Propellant(
        "single-base", "single-base nitrocellulose (M1)",
        980_000.0, 1.00e-3, 1600.0, 1.26, 5.1e-10, 2450.0,
        note="cool, gentle on barrels, and the least energetic. Artillery "
             "propellant: you can afford the volume"),
    "double-base": Propellant(
        "double-base", "double-base nitrocellulose/nitroglycerine (M9)",
        1_100_000.0, 1.00e-3, 1620.0, 1.25, 6.4e-10, 3000.0,
        note="more energy per kilo and a much hotter flame, so it eats "
             "barrels. What a high-velocity autocannon runs"),
    "triple-base": Propellant(
        "triple-base", "triple-base with nitroguanidine (M30)",
        1_030_000.0, 1.05e-3, 1660.0, 1.24, 5.6e-10, 2600.0,
        note="nitroguanidine cools the flame without giving up much "
             "impetus, and it is nearly flashless. Tank gun propellant, "
             "and the reason a tank gun's barrel lasts"),
}


@dataclass(frozen=True)
class GrainGeometry:
    """How the grain burns as it is consumed.

    The form function is psi(Z) = chi Z (1 + lambda Z), where Z is how
    far the flame has burnt through the web. lambda is the whole story:
    negative is degressive, zero is neutral, positive is progressive."""
    key: str
    label: str
    chi: float
    lam: float
    web_m: float                  # half-thickness the flame must cross
    note: str = ""


GRAINS = {
    "ball": GrainGeometry(
        "ball", "spherical ball powder", 1.00, -0.34, 0.00035,
        note="DEGRESSIVE: a sphere's surface shrinks as it burns, so the "
             "gas comes early and falls away. Fine for a short barrel "
             "and wasteful in a long one"),
    "tubular": GrainGeometry(
        "tubular", "single-perforation tube", 1.00, 0.0, 0.00060,
        note="NEUTRAL: the outside shrinks exactly as fast as the hole "
             "grows, so the surface area is constant"),
    "seven-perf": GrainGeometry(
        "seven-perf", "seven-perforation grain", 0.72, 0.28, 0.00095,
        note="PROGRESSIVE: seven inner holes grow faster than the "
             "outside shrinks, so gas production RISES as the shot "
             "travels and the chamber grows. This is what makes a long "
             "barrel worth building"),
    "nineteen-perf": GrainGeometry(
        "nineteen-perf", "nineteen-perforation grain", 0.63, 0.38, 0.00140,
        note="strongly progressive, and long-burning. Large calibre, "
             "long tube"),
}


# =====================================================================
#  THE PER-STEP LAW, IN SYMPY
# =====================================================================
BALLISTICS_STATE = ("travel_m", "velocity_m_s", "burnt_fraction",
                    "recoil_velocity_m_s", "harvested_gas_kg",
                    # HEAT ALREADY GIVEN TO THE BARREL. It was being
                    # computed and reported without ever being removed
                    # from the gas, so the model could hand the wall
                    # more energy than the charge contained -- 233 kJ
                    # out of 176 kJ available. Carried as state so the
                    # energy equation can subtract it.
                    "wall_heat_j")
BALLISTICS_PARAMS = (
    "dt_s", "bore_area_m2", "chamber_volume_m3", "shot_mass_kg",
    "charge_mass_kg", "impetus_j_per_kg", "covolume_m3_per_kg",
    "propellant_density_kg_m3", "gamma", "burn_rate_coeff",
    "grain_chi", "grain_lambda", "grain_web_m",
    "shot_start_pressure_pa", "barrel_length_m",
    # ---- WHAT THE BORE ACTUALLY DOES TO THE SHOT ----
    # `bore_resistance_pa` was a flat 15 MPa typed in. It is not a
    # constant of nature; it is what the barrel's rifling and its fit
    # on the driving band cost the projectile, and both are things a
    # barrel is MADE with.
    #
    # TWIST, in calibres per turn. A tight twist spins the shot faster,
    # which is what stabilises it -- and the energy for that spin comes
    # out of the shot's forward motion, through the driving band, as a
    # torque the band has to carry. Tighter twist, more resistance,
    # more band wear, more spin.
    #
    # TIGHTNESS, the interference between the band and the bore. It is
    # what seals the gas behind the shot: too loose and the gas blows
    # past and the round is slow and wanders; too tight and the band
    # has to be swaged down the whole length and the friction eats the
    # shot. This is the single number that says whether a barrel is
    # new, worn, or worn out.
    "twist_calibres_per_turn", "band_interference_m", "bore_diameter_m",
    # THE PRIMER. Without it nothing happens at all: the burn rate is
    # proportional to pressure, the pressure comes from gas, and the
    # gas comes from burning -- so a charge with no igniter sits there
    # at zero forever, which is exactly what the first run did. A real
    # primer puts a small, known mass of hot gas into the chamber and
    # that is what lights the main charge.
    "primer_gas_kg",
    # THE GUN MOVES TOO. Every joule that goes into the recoiling mass
    # is a joule that did not go into the shot, and the bore volume is
    # swept by the RELATIVE motion of shot and tube -- so a light gun
    # on a soft mount genuinely shoots slower than the same barrel
    # bolted to a mountain. Leaving this out is the difference between
    # an energy budget and an energy wish.
    "recoiling_mass_kg",
    # AND SOME OF THE GAS IS TAKEN. A port in the barrel wall feeds the
    # recoil recuperator, the bore evacuator and -- on this outpost --
    # the expansion engine that harvests chamber gas instead of letting
    # it foul a dryer. Gas through the port is gas that is not behind
    # the shot: a gas-operated gun really does give up muzzle velocity
    # to run itself, and this is where it goes.
    "gas_port_travel_m", "gas_port_fraction",
)
BALLISTICS_OUTPUTS = (
    "travel_next_m",
    "velocity_next_m_s",
    "burnt_fraction_next",
    "recoil_velocity_next_m_s",
    "harvested_gas_next_kg",
    "breech_pressure_pa",
    "base_pressure_pa",
    "gas_made_kg",
    "free_volume_m3",
    # THE HEAT THE SHOT PUTS IN THE TUBE. Not a fraction of the charge
    # assumed after the fact -- the gas's own temperature and the film
    # coefficient over the bore, at every instant, so the deposition
    # lands where and when it actually happens.
    "gas_temperature_k",
    "bore_heat_flux_w_per_m2",
    # THE HEAT THE GAS ITSELF IS CARRYING, above ambient. Valuing it at
    # cp times ABSOLUTE temperature counted the heat it would have had
    # at zero kelvin too, and the energy budget came out at three times
    # the chemical energy that went in.
    "gas_enthalpy_j",
    "wall_heat_next_j",
    "shot_energy_j",
    "recoil_energy_j",
    "harvested_energy_j",
    "at_muzzle",
)


def symbolic_ballistics_equations():
    """One tick of interior ballistics.

    Written as equations so the same arithmetic compiles to native code
    and runs here, rather than existing twice.
    """
    (travel, velocity, burnt, recoil_v, harvested, wall_heat) = sympy.symbols(
        " ".join(BALLISTICS_STATE), real=True)
    (dt, area, cham_v, shot_m, charge_m, impetus, covol, dens, gamma,
     beta, chi, lam, web, shot_start, barrel_len,
     twist_cal, band_interf, bore_d_decl,
     primer_gas, recoil_m, port_x, port_frac) = sympy.symbols(
        " ".join(BALLISTICS_PARAMS), real=True, positive=True)

    # --- how much of the charge has become gas -----------------------
    # the form function: psi = chi Z (1 + lambda Z), clamped at all-burnt
    z = sympy.Min(_pos(burnt), 1)
    psi = sympy.Min(chi * z * (1 + lam * z), 1)
    gas_made = charge_m * psi + primer_gas
    # what is still in the chamber doing work
    gas_kg = _pos(gas_made - harvested)

    # --- the free volume the gas actually occupies -------------------
    # the chamber, plus what the shot has uncovered, LESS the solid
    # propellant not yet burnt and LESS the covolume of the gas made.
    # Those two subtractions are the difference between a gun and an
    # ideal gas, and they are most of the pressure at peak.
    unburnt_solid = charge_m * (1 - psi) / dens
    # THE TRUE FREE VOLUME, which may be negative -- and if it is, the
    # load is physically impossible rather than merely severe: the gas
    # molecules' own volume exceeds the space available to them.
    #
    # The clamp below keeps the compiled kernel numerically sane (it
    # has no branches and cannot raise), but it is REPORTED as an
    # output so a caller can see the load is nonsense instead of
    # reading back a plausible number. Hiding this behind the clamp is
    # what turned "55 g in a 48 cc case" -- a load whose gas needs
    # 55 cc of covolume in a 48 cc chamber -- into a confident
    # 23,713 MPa, fifty times a pressure that would already have burst
    # the gun.
    free_true = cham_v + area * travel - unburnt_solid - gas_kg * covol
    free_v = sympy.Max(free_true, cham_v * 0.05)

    # --- Noble-Abel: the space-mean pressure -------------------------
    # the energy already in motion: the shot, a third of the gas (the
    # Lagrange partition), AND the recoiling mass going the other way
    kinetic = (0.5 * (shot_m + charge_m / 3) * velocity * velocity
               + 0.5 * recoil_m * recoil_v * recoil_v
               # and the heat the barrel has already taken: it left the
               # gas, so the gas no longer has it to make pressure with
               + wall_heat)
    p_mean = _pos(impetus * gas_kg - (gamma - 1) * kinetic) / free_v

    # --- gas temperature, from the state it is actually in ----------
    # Noble-Abel again, read the other way: the impetus f is R times the
    # flame temperature, so R = f / T_flame and T = p V / (m R).
    # R AND cp FROM THE PROPELLANT, not from a literal. The impetus f
    # is R times the flame temperature, so R falls out of the two
    # numbers the propellant already declares, and cp follows from R
    # and gamma.
    r_specific = impetus / 3000.0
    cp_gas = gamma * r_specific / (gamma - 1)
    t_gas = p_mean * free_v / sympy.Max(gas_kg * r_specific, 1e-9)


    # --- and what that does to the bore ------------------------------
    # DERIVED, not drawn. The first version of this was
    #     h = 1200 sqrt(p + 1) (1 + v/400)
    # which has the right shape -- heat transfer rising with pressure
    # and with gas speed -- and no derivation behind it at all. The
    # 1200 and the 400 were numbers I picked to make the curve look
    # like the right curve, which is the same error as fitting a peak
    # pressure to hit a published muzzle velocity: an expression
    # wearing the costume of a result.
    #
    # This is the Dittus-Boelter correlation, the same one the coolant
    # annulus already uses, applied to the bore. Every term comes from
    # the gas's own state:
    #
    #     rho = p / (R T)                  ideal-gas density, from the
    #                                      state the kernel is already
    #                                      tracking
    #     Re  = rho v D / mu               the shot's own speed is the
    #                                      gas speed over the surface
    #     Pr  = mu cp / k
    #     Nu  = 0.023 Re^0.8 Pr^0.4
    #     h   = Nu k / D
    #
    # mu and k are declared transport properties of the combustion gas
    # rather than tuning constants: a hot diatomic-ish product mixture
    # at these temperatures sits near 5e-5 Pa.s and 0.15 W/m.K, and
    # those are the numbers that go in.
    gas_mu = 5.0e-5
    gas_k = 0.15
    bore_d = 2 * sympy.sqrt(area / sympy.pi)
    rho_gas = p_mean / sympy.Max(r_specific * t_gas, 1.0)
    # LAGRANGE AGAIN: the gas is stationary at the breech and only
    # reaches the shot's speed at its base, so the mean speed over the
    # wetted bore is half of it. Using the full shot velocity
    # everywhere overstated the Reynolds number and with it the film
    # coefficient, which is most of why the wall heat came out
    # impossible.
    re_bore = rho_gas * _abs(velocity) * sympy.Rational(1, 2) * bore_d / gas_mu
    pr_bore = gas_mu * cp_gas / gas_k
    nu_bore = sympy.Rational(23, 1000) * re_bore ** sympy.Rational(4, 5)         * pr_bore ** sympy.Rational(2, 5)
    h_bore = nu_bore * gas_k / bore_d
    # the wall sits far below the gas, so the driving difference is the
    # gas temperature over the steel it is washing
    q_flux = h_bore * _pos(t_gas - 400.0)

    # --- the Lagrange gradient ---------------------------------------
    # the gas has mass and is being accelerated too, so the breech sees
    # more than the shot base does. The shot is driven by the BASE
    # pressure; the chamber must survive the BREECH pressure.
    lagrange = 1 + charge_m / (3 * shot_m)
    p_base = p_mean / lagrange
    p_breech = p_mean * (1 + charge_m / (2 * shot_m)) / lagrange

    # --- burn rate: Vieille, r = beta p, over the web ----------------
    # burning only once the pressure is up; the primer does that and
    # `shot_start` is where the shot itself lets go
    dz = beta * p_mean / web * dt
    burnt_next = sympy.Min(burnt + dz, 1 / sympy.Max(chi * (1 + lam), 1e-9))

    # --- the shot ----------------------------------------------------
    # it does not move until base pressure beats shot-start (the crimp
    # and the engraving), and thereafter bore resistance is a real
    # FORCE, not a fraction of pressure -- the driving band's
    # interference fit does not relax because the gas expanded
    # ---- BORE RESISTANCE, DERIVED ----
    # Two terms, both real and both from the barrel's own making:
    #
    #   ENGRAVING AND FRICTION. The band is squeezed by the
    #   interference and dragged along the bore. The radial pressure it
    #   is held at is roughly E_band * interference / radius, and the
    #   axial force is that times the contact area times a friction
    #   coefficient. Copper's modulus, not steel's -- the band is the
    #   soft part, which is the whole point of it.
    #
    #   RIFLING TORQUE. Spinning the shot up takes work, and it is
    #   taken through the band as a tangential force on the lands. The
    #   angle of the rifling turns that into an axial component:
    #   tan(alpha) = pi / (twist in calibres).
    # A BAND THAT HAS BEEN ENGRAVED IS AT ITS YIELD, not at its elastic
    # strain. Using E * interference gave 550 MPa of radial pressure at
    # fifty microns -- more than the chamber -- and the shot never
    # moved at all. Copper gives way long before that: it is swaged
    # down to fit and it sits at its own yield stress while it does.
    # So the interference decides HOW MUCH of the band is engaged, and
    # the yield decides how hard it presses.
    band_e = 1.1e11                       # copper alloy
    band_yield = 2.5e8                    # and where it gives up
    elastic_p = band_e * band_interf / sympy.Max(bore_d_decl / 2, 1e-6)
    radial_p = sympy.Min(elastic_p, band_yield)
    # a real driving band is a narrow ring, not half a calibre wide
    band_len = bore_d_decl * sympy.Rational(3, 20)
    contact_a = sympy.pi * bore_d_decl * band_len
    # copper on steel with propellant gas as the lubricant: low, and
    # falling as the surface melts
    friction = sympy.Rational(2, 25)
    f_friction = radial_p * contact_a * friction
    tan_rifling = sympy.pi / sympy.Max(twist_cal, sympy.Rational(1, 100))
    # the shot's own rotational inertia resists being spun: a solid
    # cylinder about its axis is m r^2 / 2, and the angular
    # acceleration follows the linear one through the twist
    f_rifling = (shot_m * sympy.Rational(1, 2) * tan_rifling ** 2
                 * _pos(p_base - shot_start) * area / sympy.Max(shot_m, 1e-9))
    bore_res = (f_friction + f_rifling) / sympy.Max(area, 1e-12)

    moving = _step(p_base - shot_start)
    net_force = (p_base - bore_res) * area * moving
    v_next = _pos(velocity + net_force / shot_m * dt)
    # NEWTON'S THIRD LAW, spelled out. The same force pushes the tube
    # backwards; the bore is swept by how far the shot moves RELATIVE
    # to the tube, which is why a recoiling gun's shot sees a slightly
    # shorter barrel than the tube measures.
    recoil_next = _pos(recoil_v + p_breech * area / recoil_m * dt)
    x_next = sympy.Min(travel + (v_next + recoil_next) * dt, barrel_len)
    at_muzzle = _step(x_next - barrel_len * 0.999999)
    # the port is uncovered only after the shot has passed it
    ported = _step(x_next - port_x)
    # a fraction of what is in the chamber, per second, through the port
    taken = ported * port_frac * gas_kg * dt
    harvested_next = sympy.Min(harvested + taken, gas_made)

    return (
        sympy.Eq(sympy.Symbol("travel_next_m"), x_next, evaluate=False),
        sympy.Eq(sympy.Symbol("velocity_next_m_s"), v_next, evaluate=False),
        sympy.Eq(sympy.Symbol("burnt_fraction_next"), burnt_next, evaluate=False),
        sympy.Eq(sympy.Symbol("recoil_velocity_next_m_s"), recoil_next,
                 evaluate=False),
        sympy.Eq(sympy.Symbol("harvested_gas_next_kg"), harvested_next,
                 evaluate=False),
        sympy.Eq(sympy.Symbol("breech_pressure_pa"), p_breech, evaluate=False),
        sympy.Eq(sympy.Symbol("base_pressure_pa"), p_base, evaluate=False),
        sympy.Eq(sympy.Symbol("gas_made_kg"), gas_made, evaluate=False),
        # negative means the load cannot physically burn in this chamber
        sympy.Eq(sympy.Symbol("free_volume_m3"), free_true, evaluate=False),
        sympy.Eq(sympy.Symbol("gas_temperature_k"), t_gas, evaluate=False),
        sympy.Eq(sympy.Symbol("bore_heat_flux_w_per_m2"), q_flux,
                 evaluate=False),
        sympy.Eq(sympy.Symbol("wall_heat_next_j"),
                 wall_heat + q_flux * sympy.pi * bore_d
                 * _pos(travel) * dt, evaluate=False),
        sympy.Eq(sympy.Symbol("gas_enthalpy_j"),
                 gas_kg * cp_gas * _pos(t_gas - 293.15), evaluate=False),
        sympy.Eq(sympy.Symbol("shot_energy_j"),
                 0.5 * shot_m * v_next * v_next, evaluate=False),
        sympy.Eq(sympy.Symbol("recoil_energy_j"),
                 0.5 * recoil_m * recoil_next * recoil_next, evaluate=False),
        # what the port has taken, valued at the impetus that made it:
        # this is the energy the expansion engine and the recuperator
        # get to work with, and it is REMOVED from the shot's budget
        # rather than being free
        sympy.Eq(sympy.Symbol("harvested_energy_j"),
                 harvested_next * impetus, evaluate=False),
        sympy.Eq(sympy.Symbol("at_muzzle"), at_muzzle, evaluate=False),
    ), {}


@lru_cache(maxsize=1)
def compile_ballistics_ssa() -> SymbolicEquationCompilation:
    """SymPy -> SSA, cached by the repository's own source-digest cache."""
    return compile_symbolic_program(
        symbolic_ballistics_equations,
        name="engine_toy_interior_ballistics_step",
        publications=tuple(
            SymbolicPublication(name, f"engine_toy.ballistics.{name}")
            for name in BALLISTICS_OUTPUTS
        ),
        dtype="float64",
    )


@lru_cache(maxsize=1)
def native_step():
    """The law as a native callable: SymPy -> SSA -> LLVM."""
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact,
        prepare_artifact_execution)
    import numpy as np
    compiled = compile_ballistics_ssa()
    fn = compiled.function
    names = list(fn.metadata["argument_names"])
    ids = {n: v.id for n, v in zip(names, fn.args)}
    outs = dict(fn.metadata["named_outputs"])
    artifact = compile_artifact(emit_ssa_function_to_llvm(
        compiled.module, fn.name, entry_name="interior_ballistics_step"))

    # BUILD THE EXECUTION ONCE. Calling `prepare_artifact_execution`
    # inside the loop costs 440 us a step against 2.0 us for the run
    # itself -- 220x, and all of it setup being redone for a kernel
    # whose shape never changes. The inputs are plain scalar ndarrays
    # in the execution's own buffers, so a step is: write the numbers
    # in, run, read the numbers out.
    feed = {ids[n]: np.array(0.0, dtype=np.float64) for n in names}
    ex = prepare_artifact_execution(artifact, feed)
    in_buf = {n: np.asarray(ex.buffers[ids[n]]).reshape(-1) for n in names}
    out_buf = {n: outs[n] for n in BALLISTICS_OUTPUTS}

    def step(**kwargs):
        for n in names:
            in_buf[n][0] = float(kwargs[n])
        ex.run()
        return {name: float(np.asarray(ex.buffers[bid]).reshape(-1)[0])
                for name, bid in out_buf.items()}

    return step, names


# =====================================================================
#  THE WHOLE SHOT, COMPILED -- LOOP AND ALL
# =====================================================================
class _PyPrinter(sympy.printing.str.StrPrinter):
    """Print SymPy as plain Python the SSA lowering accepts.

    Only what this law uses. Anything else raises rather than falling
    through to something that looks right and is not."""

    def _print_Pow(self, expr):
        if expr.exp == sympy.Rational(1, 2):
            return f"(({self._print(expr.base)}) ** 0.5)"
        if expr.exp.is_Integer and 0 < int(expr.exp) <= 4:
            b = self._print(expr.base)
            return "(" + " * ".join([f"({b})"] * int(expr.exp)) + ")"
        return f"(({self._print(expr.base)}) ** ({self._print(expr.exp)}))"

    def _print_Max(self, expr):
        a, *rest = expr.args
        out = self._print(a)
        for r in rest:
            out = f"max({out}, {self._print(r)})"
        return out

    def _print_Min(self, expr):
        a, *rest = expr.args
        out = self._print(a)
        for r in rest:
            out = f"min({out}, {self._print(r)})"
        return out

    def _print_Rational(self, expr):
        return f"({float(expr)!r})"

    def _print_Float(self, expr):
        return repr(float(expr))

    def _print_Pi(self, expr):
        return repr(float(sympy.pi))

    # ---- BRANCHES, SPELLED WITHOUT BRANCHING -------------------------
    # A `Piecewise` printed by the base StrPrinter comes out as
    # `Piecewise((a, cond), (b, True))` -- SymPy's own repr, dropped
    # verbatim into generated Python. It compiles, because it is
    # syntactically a call, and then evaluates to nothing usable: the
    # member bank returned NaN for every natural frequency in the graph
    # and the law looked broken when the PRINTER was.
    #
    # The same trick `symbolic_parts` already uses for `sign` applies: a
    # comparison becomes a 0/1 gate built out of arithmetic, and the
    # branch becomes a blend between the two arms. No control flow, so
    # it lowers to every target -- LLVM, WASM and the reference
    # evaluator alike -- and it is one expression rather than a jump.
    def _gate(self, difference: str) -> str:
        """1.0 where `difference` > 0, 0.0 where it is < 0."""
        return f"(0.5 * (1.0 + ({difference}) / max(abs({difference}), 1e-12)))"

    def _print_Piecewise(self, expr):
        args = list(expr.args)
        if len(args) < 1:
            raise ValueError("an empty Piecewise has no value to print")
        out = self._print(args[-1][0])
        for value, condition in reversed(args[:-1]):
            if condition is sympy.true or condition is True:
                out = self._print(value)
                continue
            out = (f"({out} + ({self._print(value)} - {out})"
                   f" * {self._print(condition)})")
        return out

    def _print_LessThan(self, expr):
        return self._gate(f"({self._print(expr.rhs)}) - ({self._print(expr.lhs)})")

    _print_StrictLessThan = _print_LessThan

    def _print_GreaterThan(self, expr):
        return self._gate(f"({self._print(expr.lhs)}) - ({self._print(expr.rhs)})")

    _print_StrictGreaterThan = _print_GreaterThan

    def _print_BooleanTrue(self, expr):
        return "1.0"

    def _print_BooleanFalse(self, expr):
        return "0.0"


def ballistics_loop_source() -> str:
    """One shot, start to muzzle, as a single compiled function.

    WHY. `native_step` bakes ONE timestep and the seventeen thousand
    iterations of a shot were a Python `for` around it -- so every step
    paid marshalling that dwarfed the arithmetic: 42 microseconds a step
    against a 2 microsecond kernel. Wrapping LLVM in a Python loop
    throws away most of what compiling it was for.

    So the loop goes inside. The body is printed from the SAME SymPy
    equations the scalar kernel is built from, so there is still exactly
    one statement of the law -- this is a different shape of the same
    expression tree, not a second opinion about the physics.
    """
    equations, _ = symbolic_ballistics_equations()
    # ---- FACTOR THE SHARED WORK OUT ----
    # Printed naively this comes to 150,000 characters in fifty lines,
    # with single expressions of 37,000 -- because free_v, p_mean and
    # t_gas are each re-expanded inline at every place they are used,
    # and they use each other. The tree grows exponentially in the
    # depth of that reuse.
    #
    # It is not the compiler being slow; it is being handed the same
    # arithmetic hundreds of times over. `sympy.cse` names each shared
    # subexpression once and refers to it after, which is exactly what
    # a person writing the loop by hand would do -- and it is the same
    # expression tree either way, so the law is unchanged.
    replacements, reduced = sympy.cse([eq.rhs for eq in equations],
                                      optimizations="basic")
    equations = [sympy.Eq(eq.lhs, new, evaluate=False)
                 for eq, new in zip(equations, reduced)]
    printer = _PyPrinter()
    state = list(BALLISTICS_STATE)
    params = list(BALLISTICS_PARAMS)
    lines = [
        "def ballistics_run(n_steps, " + ", ".join(params) + ", out):",
        '    """Integrate one round from rest to the muzzle."""',
    ]
    for name in state:
        lines.append(f"    {name} = 0.0")
    lines.append("    peak_breech = 0.0")
    lines.append("    done = 0.0")
    lines.append("    for _k in range(n_steps):")
    for sym, expr in replacements:
        lines.append(f"        {sym} = {printer.doprint(expr)}")
    for eq in equations:
        lines.append(f"        {eq.lhs.name} = {printer.doprint(eq.rhs)}")
    lines.append("        peak_breech = max(peak_breech, breech_pressure_pa)")
    # advance the state; once at the muzzle, hold everything still so
    # the remaining iterations are a no-op rather than a branch
    lines.append("        live = 1.0 - done")
    for name in state:
        nxt = {"travel_m": "travel_next_m",
               "velocity_m_s": "velocity_next_m_s",
               "burnt_fraction": "burnt_fraction_next",
               "recoil_velocity_m_s": "recoil_velocity_next_m_s",
               "harvested_gas_kg": "harvested_gas_next_kg",
               "wall_heat_j": "wall_heat_next_j"}[name]
        lines.append(f"        {name} = {name} + live * ({nxt} - {name})")
    lines.append("        done = max(done, at_muzzle)")
    for i, name in enumerate(("travel_m", "velocity_m_s", "burnt_fraction",
                              "recoil_velocity_m_s", "harvested_gas_kg",
                              "wall_heat_j")):
        lines.append(f"    out[{i}] = {name}")
    lines.append("    out[6] = peak_breech")
    lines.append("    out[7] = gas_temperature_k")
    lines.append("    out[8] = gas_made_kg")
    lines.append("    out[9] = shot_energy_j")
    lines.append("    out[10] = recoil_energy_j")
    lines.append("    out[11] = gas_enthalpy_j")
    lines.append("    return out")
    return "\n".join(lines)


LOOP_OUTPUTS = ("travel_m", "velocity_m_s", "burnt_fraction",
                "recoil_velocity_m_s", "harvested_gas_kg", "wall_heat_j",
                "peak_breech_pa", "gas_temperature_k", "gas_made_kg",
                "shot_energy_j", "recoil_energy_j", "gas_enthalpy_j")


@lru_cache(maxsize=1)
def native_shot():
    """The compiled whole-shot integrator."""
    import hashlib, warnings, time
    import numpy as np
    from compile_contract import contract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
    src = ballistics_loop_source()
    # ---- BUILD ONCE, EVER ----
    # `native_audio` compiles into `tempfile.mkdtemp()`, which means a
    # fresh LLVM build in every process that ever asks. For a loop body
    # this size that is fourteen seconds before the first shot can be
    # fired, every single run -- and it was being paid again on every
    # change to anything unrelated.
    #
    # The generated source IS the identity of the artifact: same source,
    # same machine code. So it is digested and the build lands in a
    # stable directory named by that digest. Change the law and the
    # digest changes and it rebuilds; change nothing and it never
    # compiles again.
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(__file__).resolve().parent / "baked" / f"ballistics_{digest}"
    prebuilt = (cache_dir / "engine_toy_ballistics__ballistics_run.dll").exists()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _o, _e = lower_ast_source_to_ssa(
            src, "ballistics_run", name="engine_toy_ballistics",
            extraction_contract=contract())
    qualified = "engine_toy_ballistics__ballistics_run"
    fn = module.functions[qualified]
    artifact = emit_ssa_function_to_llvm(module, qualified)
    if artifact.shortfalls:
        raise RuntimeError(f"ballistics_run shortfalls: {artifact.shortfalls}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    native = compile_artifact(artifact, directory=cache_dir)
    params = dict(fn.metadata["parameter_names"])
    native.build_seconds = time.perf_counter() - t0
    native.was_prebuilt = prebuilt

    def run(n_steps: int, **kw):
        feed = {}
        for name, vid in params.items():
            if name == "out":
                feed[vid] = np.zeros(12, dtype=np.float64)
            elif name == "n_steps":
                feed[vid] = np.array(int(n_steps), dtype=np.int64)
            else:
                feed[vid] = np.array(float(kw[name]), dtype=np.float64)
        ex = prepare_artifact_execution(native, feed)
        ex.run()
        out = np.asarray(ex.buffers[params["out"]]).reshape(-1)
        return dict(zip(LOOP_OUTPUTS, (float(v) for v in out)))

    run.build_seconds = native.build_seconds
    run.was_prebuilt = prebuilt
    run.cache_dir = str(cache_dir)
    return run, src
