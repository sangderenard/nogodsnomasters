"""THE WHOLE MEMBER BANK, COMPILED, WITH THE LOOPS INSIDE.

    python structure_native.py

One statement of the beam law -- `beam_theory.symbolic_beam_dynamics_
equations` -- reaches three targets already: the scalar LLVM kernel, the
WebAssembly component, and SymPy itself. All three step ONE member. A
structure is sixteen hundred of them and the step has to happen sixty
times a second, so the shape that matters is the bank.

    a Python loop around a compiled kernel is not a compiled program.

The ballistics lane proved it: 42 microseconds of marshalling around a
2 microsecond kernel, which threw away most of what compiling was for.
So the loops go INSIDE -- over substeps and over members -- and the body
is printed from the same equations, through the same `sympy.cse`, into
the same `lower_ast_source_to_ssa` -> `emit_ssa_function_to_llvm` path
that every other compiled law in this project uses.

NOT BESPOKE NATIVE CODE. Nothing here is hand-written machine code or a
hand-written C shim. The law is SymPy, the loop is printed from the law,
and the compiler does the rest; the only thing authored by hand is the
LOOP STRUCTURE -- which member, which substep, what is read and what is
written -- and that is a scheduling statement, not arithmetic.

BATCHED, BECAUSE THE CORE IS A BANK. Not a scalar core called in a
loop: one function whose inner extent is the member index, so the
arrays are the interface and the kernel never returns between members.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy

_TURING = Path(__file__).resolve().parent.parent / "turing"
if str(_TURING) not in sys.path:
    sys.path.insert(0, str(_TURING))

#: Per-member inputs, each an array of length n_members.
BANK_ARRAYS = ("length_m", "outer_radius_m", "wall_m", "density_kg_m3",
               "youngs_cold_pa", "temperature_k", "damping_ratio",
               "added_mass_per_m_kg", "force_n")
#: Per-member state, updated in place.
BANK_STATE = ("modal_q_m", "modal_qdot_m_s")
#: Per-member readings written out each call.
BANK_OUT = ("tip_deflection_m", "tip_slope_rad", "natural_frequency_hz")
#: the most of its own stroke a joint may travel in one substep
STEP_OF_STROKE = 0.02


@lru_cache(maxsize=1)
def _substep_rate():
    """Derive the scheduling rate from the generated velocity update.

    For acceleration = drive - c*v - k*q, semi-implicit Euler requires
    k*h*h + 2*c*h < 4. Choosing h <= 1/(sqrt(k) + c) keeps that sum at
    most 2, inside the boundary even for an undamped or overdamped mode.
    This uses the law's actual coefficients, including its hot modulus
    and modal-mass floor, without restating the beam mechanics here.
    """
    import beam_theory
    from interior_ballistics import _PyPrinter
    equations, _ = beam_theory.symbolic_beam_dynamics_equations()
    velocity = next(eq.rhs for eq in equations
                    if eq.lhs.name == "modal_qdot_next_m_s")
    symbols = {s.name: s for s in velocity.free_symbols}
    stiffness = -sympy.diff(velocity, symbols["modal_q_m"]).subs(
        symbols["dt_s"], 1)
    damping = (1 - sympy.diff(velocity, symbols["modal_qdot_m_s"])).subs(
        symbols["dt_s"], 1)
    names = tuple(n for n in BANK_ARRAYS if n != "force_n")
    evaluate = sympy.lambdify(
        [symbols[n] for n in names], sympy.sqrt(stiffness) + damping,
        modules=[{"max": np.maximum, "min": np.minimum, "abs": np.abs}],
        printer=_PyPrinter(), cse=True)
    return names, evaluate


class _BankSchedule:
    def substep_plan(self, dt_s: float, n_sub: int = 1) -> tuple[int, float]:
        """Cover the full outer dt; n_sub is a minimum, not a safety cap.

        Parameters are live arrays, so recalculate before every outer step.
        A zero-duration step still refreshes the bank's readings.
        """
        dt = float(dt_s)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("beam-bank dt_s must be finite and nonnegative")
        count = max(int(n_sub), 1)
        if self.n:
            if np.any(self.arrays["damping_ratio"] < 0.0):
                raise ValueError("beam-bank damping_ratio must be nonnegative")
            names, evaluate = _substep_rate()
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                rates = evaluate(*[self.arrays[n] for n in names])
            if not np.isfinite(rates).all() or np.any(rates < 0.0):
                raise ValueError("beam-bank parameters give invalid substep rates")
            count = max(count, math.ceil(dt * float(np.max(rates))))
        return count, dt / count


def print_bank(equations, *, name: str, arrays, state, out,
               carried, index: str = "_m", dt_symbol: str = "dt_s") -> str:
    """Print a bank loop around ANY symbolic law.

    THE BEAM LAW WAS NOT SPECIAL, it was only first. This is the body
    of what `bank_source` used to be with the beam's own names lifted
    out: give it equations, the per-item arrays, the state it carries
    and the readings it writes, and it prints the same shape of loop.
    The joint law then reaches the compiler through THIS door rather
    than through a second one written beside it, which is the whole
    point -- two printers for two laws is two things to keep in step,
    and they do not stay in step.

    `carried` maps a state name to the equation that advances it, so
    the printer knows which results are stored back and which are only
    reported.

    Everything the original said about the shape still holds:

      SHARED WORK IS FACTORED with `sympy.cse`, because terms used
      several times re-expand inline at every use and the printed tree
      grows in the depth of that reuse rather than in the size of the
      law.

      THE ARRAY AND THE SCALAR MUST NOT SHARE A NAME. Written
      `length_m = length_m[_m]`, the first iteration rebinds the name
      to a float and the second indexes a scalar. Python says so
      loudly; the compiled target did not -- it computed item zero,
      returned NaN for item one and zeros for the rest, which looked
      exactly like a loop that would not iterate.

      ONE LOOP, OVER ITEMS, and the substep count stays outside it. An
      array stored in an INNER loop and carried across an OUTER one
      does not lower -- "carried update value has no producer inside
      the loop body" -- and that defect sits in the store-chain
      identity area under active work, so this takes the other road
      rather than widening it. The cost is nothing: the expensive loop
      is the item loop and it stays inside.
    """
    from interior_ballistics import _PyPrinter
    by_name = {eq.lhs.name: eq.rhs for eq in equations}
    order = list(carried.values()) + list(out)
    exprs = [by_name[n] for n in order]
    replacements, reduced = sympy.cse(exprs, optimizations="basic")
    printer = _PyPrinter()
    in_arrays = [f"in_{a}" for a in arrays]
    args = ["n_items", "n_sub", "dt_s"] + in_arrays + list(state) + ["out"]
    lines = [f"def {name}({', '.join(args)}):",
             '    """Step every item of the bank once."""']
    # THE LAW NAMES ITS OWN TIMESTEP. The beam law calls it `dt_s` and
    # the strut law calls it `dt`; one signature cannot be both, so the
    # law's spelling is bound to the signature's rather than either one
    # being renamed to suit the other.
    if dt_symbol != "dt_s":
        lines.append(f"    {dt_symbol} = dt_s")
    lines.append(f"    for {index} in range(n_items):")
    for a in arrays:
        lines.append(f"        {a} = in_{a}[{index}]")
    for held, store in zip(state, carried):
        lines.append(f"        {store} = {held}[{index}]")
    for sym, expr in replacements:
        lines.append(f"        {sym} = {printer.doprint(expr)}")
    for nm, expr in zip(order, reduced):
        lines.append(f"        {nm} = {printer.doprint(expr)}")
    for held, advanced in zip(state, carried.values()):
        lines.append(f"        {held}[{index}] = {advanced}")
    for i, nm in enumerate(out):
        lines.append(f"        out[{index} * {len(out)} + {i}] = {nm}")
    lines.append("    return out")
    return "\n".join(lines)


def bank_source() -> str:
    """The beam bank step, printed from the law itself."""
    import beam_theory
    equations, _ = beam_theory.symbolic_beam_dynamics_equations()
    return print_bank(
        equations, name="beam_bank_run", arrays=BANK_ARRAYS,
        state=("q", "qdot"), out=BANK_OUT,
        carried={"modal_q_m": "modal_q_next_m",
                 "modal_qdot_m_s": "modal_qdot_next_m_s"})


# =====================================================================
#  THE JOINT LAW, THROUGH THE SAME DOOR
# =====================================================================
#: what every travelling joint declares, in the order the law wants.
#: `velocity_m_s` is an INPUT, not state: the joint is told how fast it
#: is being worked and answers with a force.
JOINT_ARRAYS = ("velocity_m_s", "piston_area_m2", "gas_volume_m3",
                "charge_pressure_pa", "polytropic_n", "orifice_area_m2",
                "oil_density_kg_m3", "stroke_m", "volume_floor_frac",
                "spring_scale", "damping_scale")
#: the only thing a joint carries between ticks is where it is along
#: its own travel
JOINT_STATE = ("compression",)
JOINT_OUT = ("gas_pressure_pa", "spring_force_n", "damping_force_n",
             "total_force_n")


def joint_bank_source() -> str:
    """The travelling joints, stepped by the law that already exists.

    `symbolic_parts.symbolic_oleo_strut_equations` IS the joint law --
    a polytropic gas spring and an orifice damper, authored once, with
    its semi-implicit ordering spelled out in its own docstring. It was
    already compiled to SSA on its own; what was missing was a BANK, so
    it could be run over every travelling joint in a graph at once
    instead of one strut at a time. That is all this adds.

    A JOINT DOES NOT KNOW A MASS AND MUST NOT BE GIVEN ONE. It is a
    FORCE ELEMENT: told how fast it is being worked, it answers with
    the force it makes -- gas spring plus orifice damper -- and where
    that force goes, and what it accelerates, is the business of
    whatever owns the bodies. The first version of this bank carried a
    `carried_mass_kg` and advanced a velocity, which is an integrator
    hidden inside a constitutive law: it made the joint a little
    simulator of its own, disagreeing with the one around it, and it
    was why the substep plan came out at four thousand steps a frame
    for joints that were all silently defaulted to one kilogram.

    So the state is one number -- how far along its travel the joint
    is -- and it advances from the velocity it is handed, which is
    kinematics rather than dynamics. Force out, velocity in.
    """
    import symbolic_parts as sp_parts
    import sympy
    equations, _sym = sp_parts.symbolic_oleo_strut_equations()
    spring_scale = sympy.Symbol("spring_scale", real=True)
    damping_scale = sympy.Symbol("damping_scale", real=True)
    rhs = {str(eq.lhs): eq.rhs for eq in equations}
    equations = tuple(
        sympy.Eq(eq.lhs,
                 rhs["spring_force_n"] * spring_scale
                 if str(eq.lhs) == "spring_force_n" else
                 rhs["damping_force_n"] * damping_scale
                 if str(eq.lhs) == "damping_force_n" else
                 (rhs["spring_force_n"] * spring_scale
                  + rhs["damping_force_n"] * damping_scale)
                 if str(eq.lhs) == "total_force_n" else eq.rhs,
                 evaluate=False)
        for eq in equations)
    return print_bank(
        tuple(equations), name="joint_bank_run", arrays=JOINT_ARRAYS,
        state=JOINT_STATE, out=JOINT_OUT,
        carried={"compression_m": "compression_next_m"},
        index="_j", dt_symbol="dt")



class JointBank(_BankSchedule):
    """Every travelling joint in a graph, under the one joint law.

    Hand it the velocities its joints are being worked at, step it, and
    read the forces back. It holds no mass and integrates nothing but
    its own travel."""

    def __init__(self, n_joints: int):
        self.n = int(n_joints)
        self.source = joint_bank_source()
        indices = np.arange(self.n)
        scope = {"max": np.maximum, "abs": np.abs, "min": np.minimum,
                 "range": lambda _n: (indices,)}
        exec(compile(self.source, "<joint-bank>", "exec"), scope)
        self._fn = scope["joint_bank_run"]
        self.arrays = {name: np.zeros(self.n, np.float64)
                       for name in JOINT_ARRAYS}
        self.compression = np.zeros(self.n, np.float64)
        self.out = np.zeros(self.n * len(JOINT_OUT), np.float64)
        self.source_digest = hashlib.sha256(
            self.source.encode("utf-8")).hexdigest()[:16]
        self.identities = []

    def substep_plan(self, dt_s: float, n_sub: int = 1):
        """THE TRAVEL IS WHAT LIMITS THE STEP, and nothing else.

        With no mass here there is no stiffness-over-mass to bound, and
        `_BankSchedule`'s plan reads the BEAM law's coefficients, which
        are not this law's. What can go wrong in one step is a joint
        stepping clean past its own stroke while the gas volume is near
        its floor, so the bound is the travel: no substep may move a
        joint more than a fraction of its stroke.
        """
        dt = float(dt_s)
        count = max(int(n_sub), 1)
        if self.n:
            travel = np.abs(self.arrays["velocity_m_s"]) * dt
            allowed = np.maximum(self.arrays["stroke_m"], 1e-6) * STEP_OF_STROKE
            need = float(np.max(travel / allowed)) if travel.size else 0.0
            if math.isfinite(need) and need > 0.0:
                count = max(count, math.ceil(need))
        return count, dt / max(count, 1)

    def coupled_rate_components(self, effective_mass_kg
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Natural and velocity-gradient rates behind coupled scheduling."""
        if not self.n:
            empty = np.zeros(0, dtype=float)
            return empty, empty
        mass = np.maximum(np.asarray(effective_mass_kg, float), 1.0e-6)
        a = self.arrays
        area = np.maximum(a["piston_area_m2"], 1.0e-12)
        volume0 = np.maximum(a["gas_volume_m3"], 1.0e-12)
        volume = np.maximum(volume0 - area * self.compression,
                            a["volume_floor_frac"] * volume0)
        pressure = (a["charge_pressure_pa"]
                    * (volume0 / volume) ** a["polytropic_n"])
        gas_k = (a["spring_scale"] * a["polytropic_n"] * pressure
                 * area * area / volume)
        orifice = np.maximum(a["orifice_area_m2"], 1.0e-12)
        quadratic = (a["damping_scale"] * a["oil_density_kg_m3"] / 2.0
                     * (area / orifice) ** 2 * area)
        damping_slope = 2.0 * quadratic * np.abs(a["velocity_m_s"])
        return (np.sqrt(np.maximum(gas_k, 0.0) / mass),
                damping_slope / mass)

    def coupled_substep_plan(self, dt_s: float,
                             effective_mass_kg) -> tuple[int, float]:
        """Bound travel and the actual nonlinear force gradients.

        The body owner supplies the reduced mass seen along each joint.  The
        gas spring contributes ``sqrt(k/m)`` and the quadratic orifice
        contributes ``dF/dv / m``.  Limiting the explicit exchange by those
        rates prevents the force/body ping-pong that a travel-only bound
        cannot see, without clipping force, velocity, stroke, or energy.
        """
        dt = float(dt_s)
        count, _h = self.substep_plan(dt)
        if not self.n or dt <= 0.0:
            return count, dt / max(count, 1)
        gas_rate, damping_rate = self.coupled_rate_components(
            effective_mass_kg)
        rate = np.maximum(gas_rate, damping_rate)
        finite = rate[np.isfinite(rate)]
        if finite.size:
            # Four exchanges per fastest local time constant leaves a wide
            # margin inside the explicit force-coupling stability boundary.
            count = max(count, math.ceil(dt * float(np.max(finite)) * 4.0))
        return count, dt / max(count, 1)

    def step(self, dt_s: float, n_sub: int = 1):
        count, h = self.substep_plan(dt_s, n_sub)
        arrays = [self.arrays[k] for k in JOINT_ARRAYS]
        for _ in range(count):
            self._fn(self.n, 1, h, *arrays, self.compression, self.out)
        return self.out.reshape(self.n, len(JOINT_OUT))

    def step_coupled(self, dt_s: float):
        """Advance once because the owning body solver chose this ``dt``.

        Repeating the constitutive law while holding body velocity fixed does
        not couple a stiff damper to its mass.  LiveStructure therefore asks
        ``substep_plan`` for a safe interval, advances the joint once, and
        advances every beam/rigid coordinate over that same interval before
        asking again.
        """
        arrays = [self.arrays[k] for k in JOINT_ARRAYS]
        self._fn(self.n, 1, float(dt_s), *arrays, self.compression, self.out)
        return self.out.reshape(self.n, len(JOINT_OUT))

    def force_on(self, identity: str) -> float:
        """What this joint is making right now, along its own axis."""
        i = self.identities.index(identity)
        return float(self.out.reshape(self.n, len(JOINT_OUT))[i, -1])


def travelling_joints(document: dict) -> list:
    """The edges that TRAVEL, because they said so.

    `joints.MemberConstraint.travels` is the declaration -- it is what
    separates a slider, a strut and a spring-damper from a weld, and it
    is already set on every constraint that has a stroke. Reading it is
    the whole selection: no name matching, no guessing from an edge's
    identity, no list of prefixes to keep in step with the builders.
    """
    from joints import MEMBER_CONSTRAINTS, constraint_of
    out = []
    for e in document["edges"]:
        tok = e.get("constraint_token")
        mc = (constraint_of(tok) if tok is not None
              else MEMBER_CONSTRAINTS.get(e.get("constraint")))
        if mc is not None and getattr(mc, "travels", False):
            out.append(e)
    return out


def force_joints(document: dict) -> list:
    """The travelling edges governed by this bank's oleo law.

    Travel alone does not declare a constitutive law.  A rail slider and a
    direct-drive lockup travel too, but treating either as a gas-over-oil
    strut invents a spring that is absent from the graph.  The constraint is
    the graph's declaration: this bank owns only spring-dampers and explicit
    oleo recoil slides.  Commanded hydraulic actuators remain owned by their
    actuator engine.
    """
    from joints import constraint_of
    owned = {"spring-damper", "oleo-recoil-slide",
             "belleville-preload-stack"}
    out = []
    for edge in travelling_joints(document):
        token = edge.get("constraint_token")
        constraint = (constraint_of(token).key if token is not None
                      else edge.get("constraint"))
        if (constraint in owned
                or (constraint == "linear-hydraulic-actuator"
                    and edge.get("part_role") == "platform-actuator")):
            out.append(edge)
    return out


def linear_spring_joints(document: dict) -> list:
    """Spring/dampers whose graph declaration is the complete law.

    A plain structural spring/damper declares a rate and linear damping but
    has no piston area.  It must not be assigned the pneumatic/orifice law:
    doing so invents both a piston and a quadratic damper.  Spring-dampers
    that *do* declare piston geometry remain in the oleo bank (for example
    the recoil recuperator and the adaptive platform dampers).
    """
    return [edge for edge in force_joints(document)
            if (edge.get("constraint") == "belleville-preload-stack"
                or edge.get("constraint") == "chain-winch-hoist"
                or edge.get("part_role") == "platform-actuator"
                or (edge.get("constraint") == "spring-damper"
                    and (float(edge.get("piston_area_m2", 0.0)) <= 0.0
                         or "linear_damping_n_s_per_m" in edge
                         or set(edge.get("force_components") or ()) ==
                         {"spring-recuperator"})))]


def oleo_force_joints(document: dict) -> list:
    """Force joints that genuinely declare pneumatic/orifice hardware."""
    linear_ids = {edge["identity"] for edge in linear_spring_joints(document)}
    return [edge for edge in force_joints(document)
            if edge["identity"] not in linear_ids]


def joint_bank_for(document: dict, *, oil_density_kg_m3: float = 870.0,
                   polytropic_n: float = 1.35,
                   volume_floor_frac: float = 0.03) -> "JointBank":
    """Load every travelling joint of a graph into the bank.

    A JOINT MAY DECLARE ITS GAS DIRECTLY, and a lot of them do not --
    they declare a linear spring rate and a preload, because that is
    what a spring-damper is in the vocabulary the builders use. Those
    two are enough, and the conversion is the gas law rather than a
    fitting:

        a gas spring's rate at its charge volume is  k = n p0 A^2 / V0
        and the force it stands at is                F = p0 A

    so a declared preload gives p0 = F / A, and a declared rate then
    gives V0 = n p0 A^2 / k. Nothing is chosen; the two numbers the
    edge already carries name one gas spring and this is it.
    """
    from actuators import _area
    joints = oleo_force_joints(document)
    bank = JointBank(len(joints))
    a = bank.arrays
    for i, e in enumerate(joints):
        area = float(e.get("piston_area_m2", 0.0))
        if area <= 0.0:
            area = _area(float(e.get("bore_m", 0.0))) or math.pi * 0.05 ** 2
        stroke = float(e.get("stroke_m", 0.0)
                       or e.get("recoil_stroke_m", 0.0)
                       or e.get("travel_m", 0.0) or 0.2)
        preload = float(e.get("spring_preload_n", 0.0)
                        or e.get("preload_n", 0.0))
        rate = float(e.get("spring_rate_n_per_m", 0.0))
        charge = float(e.get("charge_pressure_pa", 0.0))
        volume = float(e.get("gas_volume_m3", 0.0))
        if charge <= 0.0:
            charge = (preload / area) if preload > 0.0 else 2.0e6
        if volume <= 0.0:
            volume = (polytropic_n * charge * area * area / rate
                      if rate > 0.0 else area * stroke * 4.0)
        orifice = float(e.get("orifice_area_m2", 0.0))
        if orifice <= 0.0:
            orifice = area * float(e.get("aperture_frac", 0.06))
        a["piston_area_m2"][i] = area
        a["gas_volume_m3"][i] = volume
        a["charge_pressure_pa"][i] = charge
        a["polytropic_n"][i] = polytropic_n
        a["orifice_area_m2"][i] = orifice
        a["oil_density_kg_m3"][i] = oil_density_kg_m3
        a["stroke_m"][i] = stroke
        a["volume_floor_frac"][i] = volume_floor_frac
        components = set(e.get("force_components") or ())
        a["spring_scale"][i] = 0.0 if components == {"oil-orifice"} else 1.0
        a["damping_scale"][i] = (
            0.0 if components in ({"pneumatic-recuperator"},
                                  {"spring-recuperator"}) else 1.0)
    bank.identities = [e["identity"] for e in joints]
    return bank


@lru_cache(maxsize=1)
def native_bank():
    """Compile the bank, once ever, keyed by the source's own digest.

    RESTORED 2026-09-12 after I deleted it. A patch of mine replaced a
    span bounded by "the joint arrays" and "class JointBank", and by
    then the JointBank had been appended to the END of the file -- so
    the span swallowed `native_bank`, `PythonBank`, `bank_for` and
    `bank_for_target` on its way past. The lesson is not about care: it
    is that a patch bounded by two markers must assert what it is about
    to remove, because index-to-index replacement will happily eat
    anything that drifted between them.

    Rebuilt from the artifacts it had already baked -- `baked/beam_bank_
    <digest>/engine_toy_structure__beam_bank_run.dll` names the cache
    layout, the module and the entry -- and from the identical idiom in
    `interior_ballistics.native_shot`. Parity against the Python target
    is what proves the rebuild, and both targets run the same printed
    source, so a disagreement is a rebuild fault rather than a law
    fault.
    """
    import time
    import warnings
    from compile_contract import contract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact,
        prepare_artifact_execution)

    src = bank_source()
    # THE GENERATED SOURCE IS THE IDENTITY OF THE ARTIFACT: same source,
    # same machine code. Digest it, build into a directory named by that
    # digest, and a process that changes nothing never compiles again.
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(__file__).resolve().parent / "baked" / f"beam_bank_{digest}"
    qualified = "engine_toy_structure__beam_bank_run"
    prebuilt = (cache_dir / f"{qualified}.dll").exists()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _o, _e = lower_ast_source_to_ssa(
            src, "beam_bank_run", name="engine_toy_structure",
            extraction_contract=contract())
    fn = module.functions[qualified]
    artifact = emit_ssa_function_to_llvm(module, qualified)
    if artifact.shortfalls:
        raise RuntimeError(f"beam_bank_run shortfalls: {artifact.shortfalls}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    native = compile_artifact(artifact, directory=cache_dir)
    params = dict(fn.metadata["parameter_names"])
    build_seconds = time.perf_counter() - t0

    class NativeBank(_BankSchedule):
        """The compiled bank. The arrays ARE the kernel's own buffers.

        Nothing is marshalled per step: `self.arrays`, `q`, `qdot` and
        `out` are numpy views onto the execution's buffers, so a step is
        the kernel running over memory the caller has already written.
        That was the whole argument for banking the law in the first
        place -- forty-two microseconds of marshalling around a two
        microsecond kernel is not a compiled program.
        """

        def __init__(self, n_members: int):
            self.n = int(n_members)
            self.source = src
            self.source_digest = digest
            self.build_seconds = build_seconds
            self.was_prebuilt = prebuilt
            self.identities = []
            feed = {}
            for name, vid in params.items():
                if name == "n_items" or name == "n_sub":
                    feed[vid] = np.array(0, dtype=np.int64)
                elif name == "dt_s":
                    feed[vid] = np.array(0.0, dtype=np.float64)
                elif name == "out":
                    feed[vid] = np.zeros(self.n * len(BANK_OUT), np.float64)
                else:
                    feed[vid] = np.zeros(self.n, np.float64)
            self._ex = prepare_artifact_execution(native, feed)
            self._params = params

            # A PARAMETER THE BODY NEVER READS IS NOT A PARAMETER by
            # the time the lowering is done -- `n_sub` is carried in the
            # signature for symmetry with the Python target and the
            # substep loop is outside the kernel, so it is dropped and
            # asking for its buffer raises. Views are taken for what the
            # compiled function actually has.
            def view(name):
                vid = params.get(name)
                if vid is None:
                    return None
                return np.asarray(self._ex.buffers[vid]).reshape(-1)

            self.arrays = {name: view(f"in_{name}") for name in BANK_ARRAYS}
            self.q = view("q")
            self.qdot = view("qdot")
            self.out = view("out")
            self._n_items = view("n_items")
            self._n_sub = view("n_sub")
            self._dt = view("dt_s")

        def step(self, dt_s: float, n_sub: int = 1):
            count, h = self.substep_plan(dt_s, n_sub)
            self._n_items[0] = self.n
            if self._n_sub is not None:
                self._n_sub[0] = 1
            self._dt[0] = h
            for _ in range(count):
                self._ex.run()
            return self.out.reshape(self.n, len(BANK_OUT))

    return NativeBank


def bank_for(document: dict, *, damping: float = 0.015,
             temperature_k: float = 293.15):
    """A bank loaded from every member of a graph that declares a beam."""
    return bank_for_target(document, target="native", damping=damping,
                           temperature_k=temperature_k)


class PythonBank(_BankSchedule):
    """The printed source, executed as Python instead of compiled.

    NOT A SECOND SOLVER. It is the SAME generated source -- the same
    equations through the same `sympy.cse` through the same printer --
    handed to a different target. `compile_sympy_equations` already
    treats the law as the authority and the target as a choice; this is
    that choice taken all the way back to the interpreter.

    Kept because a law that agrees between an interpreter and a
    compiler is a law that has actually been compiled rather than
    merely transcribed -- which is the only way to tell the two apart,
    and this bank has already been wrong once in a way only a second
    target could have caught.
    """

    def __init__(self, n_members: int):
        self.n = int(n_members)
        self.source = bank_source()
        # The same source also evaluates a vector of member indices. All
        # members are independent, and advanced indexing snapshots q/qdot
        # before the stores. Keep the scalar source (and its digest) shared
        # with native; only Python's array operations and loop dispatch differ.
        indices = np.arange(self.n)
        scope = {"max": np.maximum, "abs": np.abs, "min": np.minimum,
                 "range": lambda _n: (indices,)}
        exec(compile(self.source, "<beam-bank>", "exec"), scope)
        self._fn = scope["beam_bank_run"]
        self.arrays = {name: np.zeros(self.n, np.float64)
                       for name in BANK_ARRAYS}
        self.q = np.zeros(self.n, np.float64)
        self.qdot = np.zeros(self.n, np.float64)
        self.out = np.zeros(self.n * 3, np.float64)
        self.source_digest = hashlib.sha256(
            self.source.encode("utf-8")).hexdigest()[:16]
        self.build_seconds = 0.0
        self.was_prebuilt = True
        self.identities = []

    def step(self, dt_s: float, n_sub: int = 1):
        count, h = self.substep_plan(dt_s, n_sub)
        arrays = [self.arrays[k] for k in BANK_ARRAYS]
        for _ in range(count):
            self._fn(self.n, 1, h, *arrays,
                     self.q, self.qdot, self.out)
        return self.out.reshape(self.n, 3)


def bank_for_target(document: dict, *, target: str = "native", **kw):
    """`bank_for`, with the target named rather than assumed."""
    import beam_theory
    members = beam_theory.beam_members(document)
    bank = (PythonBank(len(members)) if target == "python"
            else native_bank()(len(members)))
    damping = kw.get("damping", 0.015)
    temperature_k = kw.get("temperature_k", 293.15)
    for i, m in enumerate(members):
        bank.arrays["length_m"][i] = m["length_m"]
        bank.arrays["outer_radius_m"][i] = m["outer_radius_m"]
        bank.arrays["wall_m"][i] = m["wall_m"]
        bank.arrays["density_kg_m3"][i] = m["density_kg_m3"]
        bank.arrays["youngs_cold_pa"][i] = m["youngs_cold_pa"]
        bank.arrays["temperature_k"][i] = temperature_k
        bank.arrays["damping_ratio"][i] = m.get("damping_ratio", damping)
        bank.arrays["added_mass_per_m_kg"][i] = m.get("added_mass_per_m_kg", 0.0)
    bank.identities = [m["identity"] for m in members]
    return bank
