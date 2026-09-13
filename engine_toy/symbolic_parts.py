"""engine_toy parts authored as symbolic programs.

This is the lane that makes a part real to the game rather than only
real to this simulation. A law written here is authored ONCE as SymPy
equations and then lowered by the repository's own pipeline -- SymPy ->
ProcessGraph -> SSA -> C / WASM / LLVM -- so the same arithmetic runs
in the Python sim, in a native build, and in whatever the game runs on,
bit for bit. Nothing is re-implemented per target, which is the only
way two implementations of one law can be kept from drifting apart.

It also happens to be the efficiency answer. A hand-vectorised NumPy
version of a per-tick scalar law is a rewrite that has to be kept in
step with the original; a compiled one is the original. The existing
precedent in this repository is stark -- an authored Python eigh went
from 521 s to 0.26 s through this path -- so the first question about a
slow inner loop here should be whether it can be authored as equations,
not whether it can be re-spelled in array operations.

CONVENTIONS THAT MATTER, learned from the laws already in the tree:

  Use `sympy.Max` / `sympy.Min`, never `(x + Abs(x)) / 2`. The latter is
  the same function in real arithmetic but relies on exact
  cancellation, and SymPy re-spells the operands so the compiled
  schedule keeps a few ULP of residue -- which a downstream gate then
  turns into a phantom result. (vehicle_mechanical_material.py records
  measuring exactly that: 2**-14 of residue scaling every stress by
  2**-13.)

  Spell absolute value as sqrt(x*x + eps). It is smooth, it has no
  branch, and every backend evaluates it identically.

  Every equation is `Eq(Symbol(output_name), expression,
  evaluate=False)`, and the output order must match the declared tuple.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import sympy
from sympy.printing.str import StrPrinter

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.compiler.symbolic_equation_compiler import (  # noqa: E402
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_symbolic_program,
)

# a smooth |x|: no branch, identical in every backend
_ABS_EPS = 1e-12


def _abs(x):
    return sympy.sqrt(x * x + _ABS_EPS)


def _step(x):
    """A smooth 0/1 gate on the sign of x, with no branch.

    SymPy's own sign function is NOT usable here: it has no definition
    in the LLVM instruction table and the lowering fails outright with
    "symbol 'sign' has no function definition". This spelling is covered
    everywhere, gives 1 well above zero and 0 well below, and returns
    exactly 0.5 at zero rather than an undefined jump."""
    return (x + _abs(x)) / (2 * _abs(x))


# =====================================================================
#  THE OLEO-PNEUMATIC STRUT  (actuators.GasOverOilStrut)
# =====================================================================
OLEO_STRUT_OUTPUTS = (
    "compression_next_m",
    "gas_pressure_pa",
    "spring_force_n",
    "damping_force_n",
    "total_force_n",
)


def symbolic_oleo_strut_equations() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """One tick of a gas-spring / oil-damper strut.

    The same law as `actuators.GasOverOilStrut`, and the same physics it
    documents: the gas is a polytropic spring whose rate climbs steeply
    as it is compressed, and the oil is forced through an orifice so its
    resistance goes with the SQUARE of the velocity.

    The gas volume is floored rather than allowed to reach zero. Past
    that floor a real strut is on its mechanical stop and the structure
    is carrying the load, not the gas; letting the expression run to
    infinity instead would be arithmetic rather than physics.
    """
    names = ("dt compression_m velocity_m_s "
             "piston_area_m2 gas_volume_m3 charge_pressure_pa polytropic_n "
             "orifice_area_m2 oil_density_kg_m3 stroke_m volume_floor_frac")
    s = {name: sympy.Symbol(name, real=True) for name in names.split()}

    # Where it ends up FIRST, because the spring is evaluated at the new
    # compression, not the old one. That is semi-implicit (symplectic)
    # Euler and it is deliberate: evaluating a stiff spring at the
    # position you are leaving rather than the one you are arriving at
    # is the explicit scheme, and a gas spring whose rate climbs as it
    # is compressed is exactly the case where that goes unstable. The
    # Python law (actuators.GasOverOilStrut.step) already does this;
    # authoring it the other way here put a 1.5e-3 relative disagreement
    # between two implementations of one law, which is the kind of drift
    # this whole lane exists to prevent.
    advanced = s["compression_m"] + s["velocity_m_s"] * s["dt"]
    compression_next = sympy.Min(sympy.Max(advanced, 0.0), s["stroke_m"])

    # --- the gas spring, at that new compression ---
    swept = s["piston_area_m2"] * compression_next
    floor = s["volume_floor_frac"] * s["gas_volume_m3"]
    gas_volume = sympy.Max(s["gas_volume_m3"] - swept, floor)
    gas_pressure = s["charge_pressure_pa"] * (s["gas_volume_m3"] / gas_volume) ** s["polytropic_n"]
    spring_force = gas_pressure * s["piston_area_m2"]

    # --- the oil damper ---
    # dp = rho/2 * (A v / Ao)^2, force = dp * A, signed by the velocity.
    # Written as v*|v| so the sign comes out without a branch.
    flow_speed = s["piston_area_m2"] / s["orifice_area_m2"]
    damping_force = (s["oil_density_kg_m3"] / 2.0 * flow_speed * flow_speed
                     * s["velocity_m_s"] * _abs(s["velocity_m_s"]) * s["piston_area_m2"])

    values = {
        "compression_next_m": compression_next,
        "gas_pressure_pa": gas_pressure,
        "spring_force_n": spring_force,
        "damping_force_n": damping_force,
        "total_force_n": spring_force + damping_force,
    }
    assert tuple(values) == OLEO_STRUT_OUTPUTS
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expr, evaluate=False)
                      for name, expr in values.items())
    return equations, s


@lru_cache(maxsize=1)
def compile_oleo_strut_ssa() -> SymbolicEquationCompilation:
    """SymPy -> SSA. Cached, and the repository's own source-digest cache
    keeps it warm across processes."""
    return compile_symbolic_program(
        symbolic_oleo_strut_equations,
        name="engine_toy_oleo_strut_step",
        publications=tuple(
            SymbolicPublication(name, f"engine_toy.strut.{name}")
            for name in OLEO_STRUT_OUTPUTS
        ),
        dtype="float64",
    )


# =====================================================================
#  THE HYDRAULIC CYLINDER  (actuators.LinearActuator)
# =====================================================================
CYLINDER_OUTPUTS = (
    "position_next_m",
    "velocity_m_s",
    "delivered_force_n",
    "flow_l_min",
    "stalled",
)


def symbolic_cylinder_equations() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """One tick of a double-acting cylinder.

    The three facts this law exists to state, all of which the Python
    version already computes and all of which the game needs:

      The rod exerts pressure times area, less seal drag. That is what a
      catalogue quotes and what a load actually feels.

      Speed comes from FLOW and only from flow. A cylinder is a
      volumetric device: halve the supply and it halves the speed at
      exactly the same force.

      The areas are NOT equal. The rod takes its own area off the
      retract side, so the same cylinder pulls with less force and
      retracts faster, and the direction term here carries that.
    """
    names = ("dt position_m command bore_m rod_m stroke_m "
             "supply_pressure_pa available_flow_l_min load_n seal_friction_frac")
    s = {name: sympy.Symbol(name, real=True) for name in names.split()}

    pi = sympy.pi
    area_extend = pi * s["bore_m"] ** 2 / 4
    area_retract = area_extend - pi * s["rod_m"] ** 2 / 4
    # a smooth selector on the command's sign: +1 extending, 0 retracting
    extending = _step(s["command"])
    area = extending * area_extend + (1 - extending) * area_retract

    gross = s["supply_pressure_pa"] * area
    friction = s["seal_friction_frac"] * s["supply_pressure_pa"] * area_extend
    delivered = sympy.Max(gross - friction, 0.0)

    # it moves only if what it delivers beats what is holding it back
    opposing = _abs(s["load_n"])
    moving = _step(delivered - opposing)
    speed = (_abs(s["command"]) * s["available_flow_l_min"] / 60000) / area
    velocity = moving * speed * (2 * extending - 1)

    advanced = s["position_m"] + velocity * s["dt"]
    position_next = sympy.Min(sympy.Max(advanced, 0.0), s["stroke_m"])
    # the flow it really consumed is the distance it really moved
    velocity_real = (position_next - s["position_m"]) / sympy.Max(s["dt"], 1e-9)
    flow = _abs(velocity_real) * area * 60000

    values = {
        "position_next_m": position_next,
        "velocity_m_s": velocity_real,
        "delivered_force_n": delivered * (2 * extending - 1),
        "flow_l_min": flow,
        "stalled": 1 - moving,
    }
    assert tuple(values) == CYLINDER_OUTPUTS
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expr, evaluate=False)
                      for name, expr in values.items())
    return equations, s


@lru_cache(maxsize=1)
def compile_cylinder_ssa() -> SymbolicEquationCompilation:
    return compile_symbolic_program(
        symbolic_cylinder_equations,
        name="engine_toy_cylinder_step",
        publications=tuple(
            SymbolicPublication(name, f"engine_toy.cylinder.{name}")
            for name in CYLINDER_OUTPUTS
        ),
        dtype="float64",
    )


LAWS = {
    "oleo-strut": (symbolic_oleo_strut_equations, compile_oleo_strut_ssa, OLEO_STRUT_OUTPUTS),
    "cylinder": (symbolic_cylinder_equations, compile_cylinder_ssa, CYLINDER_OUTPUTS),
}


# =====================================================================
#  EXTERIOR BALLISTICS  (paired with engine_rays / ballistics.py)
# =====================================================================
TRAJECTORY_OUTPUTS = (
    "position_next_x_m", "position_next_y_m", "position_next_z_m",
    "velocity_next_x_m_s", "velocity_next_y_m_s", "velocity_next_z_m_s",
    "speed_m_s", "kinetic_energy_j",
)


def symbolic_trajectory_equations() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """One step of a point-mass trajectory with drag and wind.

    This is the half of shooting that the penetration engine does not
    do. `engine_rays` answers what happens when a projectile meets a
    wall; this answers what the projectile is still DOING by the time it
    gets there, which at any real distance is a different question:

      DROP. Gravity acts for the whole flight, so a bullet is falling
      from the moment it leaves the muzzle. Nothing about the aim
      changes that; it is time in the air.

      DRAG. The retarding force goes with the SQUARE of speed, so a
      projectile sheds energy fastest when it is fastest. This is why
      the difference between two cartridges at the muzzle is smaller
      than the difference at four hundred metres, and why a heavy slow
      round can beat a light fast one downrange.

      WINDAGE. Drag acts on velocity RELATIVE TO THE AIR, not relative
      to the ground. That single substitution is the whole of wind
      deflection: a crosswind does not push the bullet sideways so much
      as make the air arrive at an angle, and the drag force then has a
      sideways component for the entire flight.

    Integrated semi-implicitly -- velocity first, then position from the
    NEW velocity. The same choice as the strut above, and for the same
    reason: it is stable where explicit Euler is not, and it is what
    the reference implementation does.
    """
    names = ("dt position_x_m position_y_m position_z_m "
             "velocity_x_m_s velocity_y_m_s velocity_z_m_s "
             "wind_x_m_s wind_y_m_s wind_z_m_s "
             "mass_kg diameter_m drag_coefficient air_density_kg_m3 gravity_m_s2")
    s = {name: sympy.Symbol(name, real=True) for name in names.split()}

    # velocity relative to the AIR: this is what drag acts on
    rel_x = s["velocity_x_m_s"] - s["wind_x_m_s"]
    rel_y = s["velocity_y_m_s"] - s["wind_y_m_s"]
    rel_z = s["velocity_z_m_s"] - s["wind_z_m_s"]
    rel_speed = sympy.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z + _ABS_EPS)

    area = sympy.pi * s["diameter_m"] ** 2 / 4
    # F = 1/2 rho Cd A |v_rel| v_rel, so the acceleration coefficient is
    k = s["air_density_kg_m3"] * s["drag_coefficient"] * area / (2 * s["mass_kg"])

    ax = -k * rel_speed * rel_x
    ay = -k * rel_speed * rel_y - s["gravity_m_s2"]
    az = -k * rel_speed * rel_z

    vx = s["velocity_x_m_s"] + ax * s["dt"]
    vy = s["velocity_y_m_s"] + ay * s["dt"]
    vz = s["velocity_z_m_s"] + az * s["dt"]

    speed = sympy.sqrt(vx * vx + vy * vy + vz * vz + _ABS_EPS)
    values = {
        "position_next_x_m": s["position_x_m"] + vx * s["dt"],
        "position_next_y_m": s["position_y_m"] + vy * s["dt"],
        "position_next_z_m": s["position_z_m"] + vz * s["dt"],
        "velocity_next_x_m_s": vx,
        "velocity_next_y_m_s": vy,
        "velocity_next_z_m_s": vz,
        "speed_m_s": speed,
        "kinetic_energy_j": s["mass_kg"] * speed * speed / 2,
    }
    assert tuple(values) == TRAJECTORY_OUTPUTS
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expr, evaluate=False)
                      for name, expr in values.items())
    return equations, s


@lru_cache(maxsize=1)
def compile_trajectory_ssa() -> SymbolicEquationCompilation:
    return compile_symbolic_program(
        symbolic_trajectory_equations,
        name="engine_toy_trajectory_step",
        publications=tuple(
            SymbolicPublication(name, f"engine_toy.trajectory.{name}")
            for name in TRAJECTORY_OUTPUTS
        ),
        dtype="float64",
    )


LAWS["trajectory"] = (symbolic_trajectory_equations, compile_trajectory_ssa, TRAJECTORY_OUTPUTS)


# =====================================================================
#  THE SAME LAW, OVER A BATCH AXIS
# =====================================================================
TRAJECTORY_STATE = ("position_x_m", "position_y_m", "position_z_m",
                    "velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s")
TRAJECTORY_PARAMS = ("dt", "mass_kg", "diameter_m", "drag_coefficient",
                     "air_density_kg_m3", "gravity_m_s2",
                     "wind_x_m_s", "wind_y_m_s", "wind_z_m_s")


@lru_cache(maxsize=1)
def trajectory_batch_source() -> str:
    """The trajectory law, printed from SymPy as AbstractTensor code.

    WHY THIS EXISTS. `compile_trajectory_ssa` above is a SCALAR kernel --
    one round, one step. That is what a symbolic emitter is for and it is
    the right shape for proving the law, but it is the wrong shape for a
    runtime core: measured, it costs about 0.17 ms a call, so a frame of
    eighty-four ballistic substeps for sixty-four rounds in the air would
    be the better part of a second. Handing it a batch does not help
    either -- it silently takes the first element and returns one answer.

    So the law is printed over a BATCH AXIS instead, and the printing is
    done by SymPy rather than by hand. That matters: a hand-written
    vectorised copy is a second opinion about the physics, and the two
    can drift. Generated from the same expression tree, there is only
    ever one law, and `check_batch_parity` proves the generated form and
    the compiled scalar kernel agree to machine precision.

    Nothing about the FIDELITY changes here. Same drag on air-relative
    velocity, same gravity, same epsilon, same semi-implicit order --
    velocity first, then position from the new velocity. Only the width.
    """
    equations, symbols = symbolic_trajectory_equations()
    printer = _AbstractTensorPrinter()
    lines = [
        "def trajectory_step_batch(" + ", ".join(TRAJECTORY_STATE + TRAJECTORY_PARAMS) + "):",
        '    """One trajectory step for every round at once."""',
    ]
    for eq in equations:
        lines.append(f"    {eq.lhs.name} = {printer.doprint(eq.rhs)}")
    lines.append("    return (" + ", ".join(name for name in TRAJECTORY_OUTPUTS) + ",)")
    return "\n".join(lines)


class _AbstractTensorPrinter(StrPrinter):
    """Print SymPy as AbstractTensor-compatible expressions.

    Only what this law actually uses: arithmetic, powers and sqrt. A
    printer that silently emitted something else would be the bug this
    whole approach exists to avoid, so anything unhandled raises rather
    than falling through to a scalar builtin."""

    def _print_Pow(self, expr):
        if expr.exp == sympy.Rational(1, 2):
            return f"({self._print(expr.base)}).sqrt()"
        if expr.exp == -sympy.Rational(1, 2):
            return f"(1.0 / ({self._print(expr.base)}).sqrt())"
        if expr.exp.is_Integer:
            return f"({self._print(expr.base)}) ** {int(expr.exp)}"
        raise NotImplementedError(
            f"trajectory law printed a power this backend does not lower: {expr}")

    def _print_Float(self, expr):
        return repr(float(expr))

    def _print_Pi(self, expr):
        return repr(float(sympy.pi))


@lru_cache(maxsize=1)
def trajectory_batch_step():
    """The generated batched step, as a callable."""
    namespace: dict = {}
    exec(compile(trajectory_batch_source(), "<trajectory-batch>", "exec"), namespace)
    return namespace["trajectory_step_batch"]


def check_batch_parity(steps: int = 200) -> float:
    """Worst disagreement between the batched law and the compiled
    scalar kernel over a real flight.

    This is the proof that widening the law cost no fidelity. Without
    it the batched form is an assertion; with it, it is a measurement."""
    import numpy as np
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution
    from src.common.tensors.abstraction import AbstractTensor

    compiled = compile_trajectory_ssa()
    fn = compiled.function
    from world_gun import _trajectory_artifact
    art = _trajectory_artifact(compiled)
    ids = {n: v.id for n, v in zip(fn.metadata["argument_names"], fn.args)}
    outs = dict(fn.metadata["named_outputs"])
    step = trajectory_batch_step()

    params = {"dt": 2.0e-4, "mass_kg": 0.87, "diameter_m": 0.040,
              "drag_coefficient": 0.295, "air_density_kg_m3": 1.225,
              "gravity_m_s2": 9.80665,
              "wind_x_m_s": 1.5, "wind_y_m_s": 0.0, "wind_z_m_s": -0.7}
    # a batch of four rounds, all starting differently, so the batch is
    # genuinely doing four different flights rather than one repeated
    state_b = {
        "position_x_m": AbstractTensor.tensor([0.0, 0.0, 1.0, -1.0]),
        "position_y_m": AbstractTensor.tensor([2.6, 2.6, 3.0, 2.0]),
        "position_z_m": AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0]),
        "velocity_x_m_s": AbstractTensor.tensor([0.0, 3.0, -2.0, 0.5]),
        "velocity_y_m_s": AbstractTensor.tensor([12.0, 5.0, 20.0, 0.0]),
        "velocity_z_m_s": AbstractTensor.tensor([1010.0, 900.0, 1010.0, 800.0]),
    }
    lane = 0
    state_k = {k: float(np.asarray(v.tolist())[lane]) for k, v in state_b.items()}
    batch_params = {k: AbstractTensor.tensor([v] * 4) for k, v in params.items()}

    worst = 0.0
    for _ in range(steps):
        out = step(**state_b, **batch_params)
        produced = dict(zip(TRAJECTORY_OUTPUTS, out))
        state_b = {
            "position_x_m": produced["position_next_x_m"],
            "position_y_m": produced["position_next_y_m"],
            "position_z_m": produced["position_next_z_m"],
            "velocity_x_m_s": produced["velocity_next_x_m_s"],
            "velocity_y_m_s": produced["velocity_next_y_m_s"],
            "velocity_z_m_s": produced["velocity_next_z_m_s"],
        }
        vals = {**params, **state_k}
        ex = prepare_artifact_execution(
            art, {ids[n]: np.array(x, dtype=np.float64) for n, x in vals.items()})
        ex.run()

        def g(name):
            return float(np.asarray(ex.buffers[outs[name]]).reshape(-1)[0])

        state_k = {
            "position_x_m": g("position_next_x_m"),
            "position_y_m": g("position_next_y_m"),
            "position_z_m": g("position_next_z_m"),
            "velocity_x_m_s": g("velocity_next_x_m_s"),
            "velocity_y_m_s": g("velocity_next_y_m_s"),
            "velocity_z_m_s": g("velocity_next_z_m_s"),
        }
        for key, scalar in state_k.items():
            lane_value = float(np.asarray(state_b[key].tolist())[lane])
            worst = max(worst, abs(lane_value - scalar))
    return worst
