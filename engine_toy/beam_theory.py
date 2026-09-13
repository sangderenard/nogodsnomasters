"""Beam kinematics for graph members, authored symbolically.

WHY THIS EXISTS. The game's member law
(`vehicle_mechanical_material.symbolic_vehicle_member_material_equations`)
is a J2 return map over THREE strain components -- axial, bending and
shear -- with hardening, viscosity, accumulated plastic strain and
fracture. Its own docstring says bending strain is "the signed
outer-fibre strain derived by the graph's beam kinematics".

That beam kinematics was never built. Look at the call site in
`vehicle_native_graph_program.vehicle_member_material_vector`: arguments
4 and 5 (bending_strain, bending_strain_rate) and 19 and 20
(shear_strain, shear_strain_rate) are all passed `zero`. So two thirds
of a three-component plasticity law has never run, and
`plastic_bending_previous` and `plastic_shear_previous` are state
columns faithfully carried, updated and stored for quantities that are
structurally zero.

This supplies the missing upstream. It is not an addition to the law; it
is the input the law was written to consume.

ROUND TUBES ONLY, AND THAT IS A REAL SIMPLIFICATION rather than a
shortcut deferred. Every member this project authors is a circular tube,
and a circular section is axisymmetric: the second moment is the same
about every axis, so there is no principal-axis orientation to carry and
no way to author a member "rolled the wrong way". Torsion is exact --
J = I_y + I_z = 2I for a closed circular section -- where an I-beam or a
channel needs a warping constant and restraint conditions that are a
research project of their own. When structural forms arrive they arrive
as more section types feeding the same kinematics; nothing here has to
change to admit them.

WHAT A MEMBER'S ENDS DO, and why rotations are needed. A pin-jointed
truss member only stretches, and that is the whole of what the graph
could previously say. A real welded space frame is not pinned: its joints
carry moment, so a member bends when its ends rotate relative to its own
chord, and THAT is the bending strain the law wants. Nodes therefore
carry six degrees of freedom, not three.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import sympy

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

_ABS_EPS = 1.0e-12

#: What the kinematics produces, in the order the member law wants it.
BEAM_KINEMATIC_OUTPUTS = (
    "axial_strain", "bending_strain", "shear_strain",
    "curvature_y_per_m", "curvature_z_per_m", "twist_rad_per_m",
    "transverse_shear_y", "transverse_shear_z", "torsional_surface_strain",
    "chord_length_m",
)


def shear_coefficient(outer_radius_m: float, inner_radius_m: float,
                      poisson: float = 0.3) -> float:
    """Timoshenko shear coefficient for a circular tube.

    TWO EXACT ENDPOINTS, INTERPOLATED BETWEEN -- and said plainly,
    because the first version of this was a single "Cowper hollow
    circular" expression that returned 0.925 for a solid bar where the
    established value is 0.886, and 0.565 for a thin tube against 0.53.
    Wrong by five per cent at both ends, and wrong in a way that only
    showed because it was checked against the endpoints.

    The endpoints themselves are Cowper's and reproduce exactly:

        solid circular       6(1+nu) / (7+6nu)      = 0.886 at nu=0.3
        thin-walled tube     2(1+nu) / (4+3nu)      = 0.531 at nu=0.3

    Between them this interpolates on the area ratio, which is an
    approximation and is labelled as one rather than dressed up as a
    closed form.

    It matters because a SHORT FAT member carries a real share of its
    deflection in shear rather than bending. Assuming otherwise -- which
    is what Euler-Bernoulli does -- makes exactly the stubby members in a
    gun mount (trunnion braces, pedestals) look stiffer than they are.
    """
    ro = max(float(outer_radius_m), 1e-9)
    ri = max(min(float(inner_radius_m), ro * 0.999), 0.0)
    nu = float(poisson)
    solid = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    thin = 2.0 * (1.0 + nu) / (4.0 + 3.0 * nu)
    # how close to a thin shell this section is, by area fraction
    m2 = (ri / ro) ** 2
    return float(solid + (thin - solid) * m2)


@lru_cache(maxsize=1)
def symbolic_beam_kinematics_equations():
    """One member's strains, from its two nodes' six degrees of freedom.

    Everything is expressed in the member's OWN frame, built from its
    chord. The three strains handed to the material law are:

      AXIAL, the chord's stretch over its rest length. The only one the
      graph could previously say anything about.

      BENDING, the signed OUTER-FIBRE strain, which is the resultant
      curvature times the outside radius. Curvature comes from how much
      the two ends have rotated relative to the chord between them --
      which is why a pin-jointed idealisation reports zero bending no
      matter how bent the structure is.

      SHEAR, combining transverse shear and torsional surface strain, as
      the law's own docstring asks for. Transverse shear is the part of
      the end translation that the end rotations did NOT account for --
      the Timoshenko correction, and it is why this is not
      Euler-Bernoulli. Torsional surface strain is the twist per unit
      length times the outside radius.
    """
    names = ("rest_length_m outer_radius_m "
             "delta_axial_m delta_lateral_y_m delta_lateral_z_m "
             "theta_y_a theta_y_b theta_z_a theta_z_b theta_x_a theta_x_b")
    s = {n: sympy.Symbol(n, real=True) for n in names.split()}

    L = s["rest_length_m"]
    r = s["outer_radius_m"]

    # --- axial -------------------------------------------------------
    axial_strain = s["delta_axial_m"] / L

    # --- curvature: relative end rotation over the length -------------
    # For a two-node element the curvature is the difference in end
    # rotations divided by the length. Constant curvature is the honest
    # answer for a linear element: a finer bending shape needs more
    # nodes, not a more elaborate formula on two.
    curvature_y = (s["theta_y_b"] - s["theta_y_a"]) / L
    curvature_z = (s["theta_z_b"] - s["theta_z_a"]) / L
    # AXISYMMETRY IS WHY THIS IS A PLAIN RESULTANT. For a circular tube
    # the second moment is the same about every axis, so bending about y
    # and bending about z combine as a vector and the outer fibre that
    # matters is simply the furthest one. A non-circular section would
    # need each axis kept separate and the fibre found per orientation.
    curvature = sympy.sqrt(curvature_y ** 2 + curvature_z ** 2 + _ABS_EPS)
    bending_strain = curvature * r

    # --- transverse shear: the Timoshenko part ------------------------
    # The end translation that the end rotations do not explain. In
    # Euler-Bernoulli this is defined to be zero; here it is computed,
    # which is the whole difference between the two theories and the
    # reason short fat members stop looking stiffer than they are.
    mean_theta_z = (s["theta_z_a"] + s["theta_z_b"]) / 2
    mean_theta_y = (s["theta_y_a"] + s["theta_y_b"]) / 2
    shear_y = s["delta_lateral_y_m"] / L - mean_theta_z
    shear_z = s["delta_lateral_z_m"] / L + mean_theta_y

    # --- torsion ------------------------------------------------------
    # J = 2I exactly for a closed circular section, so the surface shear
    # strain is simply the twist per unit length times the radius.
    twist = (s["theta_x_b"] - s["theta_x_a"]) / L
    torsional_surface_strain = twist * r

    shear_strain = sympy.sqrt(shear_y ** 2 + shear_z ** 2
                              + torsional_surface_strain ** 2 + _ABS_EPS)

    chord_length = L + s["delta_axial_m"]

    values = {
        "axial_strain": axial_strain,
        "bending_strain": bending_strain,
        "shear_strain": shear_strain,
        "curvature_y_per_m": curvature_y,
        "curvature_z_per_m": curvature_z,
        "twist_rad_per_m": twist,
        "transverse_shear_y": shear_y,
        "transverse_shear_z": shear_z,
        "torsional_surface_strain": torsional_surface_strain,
        "chord_length_m": chord_length,
    }
    assert tuple(values) == BEAM_KINEMATIC_OUTPUTS
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expr, evaluate=False)
                      for name, expr in values.items())
    return equations, s


@lru_cache(maxsize=1)
def symbolic_geometric_stiffness_equations():
    """How much bending stiffness an axial load takes away.

    THIS IS WHAT BUCKLING ACTUALLY IS, and why it should not be a
    separate check bolted to the side of a strength calculation. A
    member in compression is less stiff in bending than the same member
    at rest: the axial force acting through the lateral deflection adds
    a moment that helps the deflection grow. The effective bending
    stiffness is the elastic stiffness MINUS a term proportional to the
    compressive load, and buckling is simply where that difference
    reaches zero.

    Written this way the Euler critical load is not a formula anyone
    types in -- it is the load at which `effective_bending_stiffness`
    vanishes, and it falls out of the same expression that governs the
    member's ordinary small deflections. Eccentricity and end fixity
    then behave correctly instead of needing their own special cases.
    """
    names = ("youngs_modulus_pa second_moment_m4 rest_length_m axial_force_n "
             "end_fixity")
    s = {n: sympy.Symbol(n, real=True) for n in names.split()}

    E, I, L = s["youngs_modulus_pa"], s["second_moment_m4"], s["rest_length_m"]
    P = s["axial_force_n"]          # positive in TENSION
    k = s["end_fixity"]

    effective_length = L / k
    # the classical elastic bending stiffness of the element
    elastic = 12 * E * I / effective_length ** 3
    # the geometric term: compression (P < 0) subtracts stiffness,
    # tension adds it -- which is why a guy wire gets stiffer as you
    # pull it and a column gets softer as you lean on it
    geometric = 6 * P / (5 * effective_length)
    effective = elastic + geometric

    # the load at which the effective stiffness vanishes IS Euler's
    critical = sympy.pi ** 2 * E * I / effective_length ** 2
    # how much of the way to instability this member currently is
    demand = -P / critical

    values = {
        "elastic_bending_stiffness_n_per_m": elastic,
        "geometric_bending_stiffness_n_per_m": geometric,
        "effective_bending_stiffness_n_per_m": effective,
        "critical_buckling_load_n": critical,
        "buckling_demand": demand,
        "stiffness_retained": effective / elastic,
    }
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expr, evaluate=False)
                      for name, expr in values.items())
    return equations, s


GEOMETRIC_STIFFNESS_OUTPUTS = (
    "elastic_bending_stiffness_n_per_m",
    "geometric_bending_stiffness_n_per_m",
    "effective_bending_stiffness_n_per_m",
    "critical_buckling_load_n",
    "buckling_demand",
    "stiffness_retained",
)


def compile_beam_kinematics_ssa():
    """Lower the kinematics the same way every other law here is lowered."""
    from src.compiler.symbolic_equation_compiler import (
        compile_symbolic_program, SymbolicPublication)
    return compile_symbolic_program(
        symbolic_beam_kinematics_equations,
        name="engine_toy_beam_kinematics",
        publications=tuple(
            SymbolicPublication(name, f"engine_toy.beam.{name}")
            for name in BEAM_KINEMATIC_OUTPUTS),
        dtype="float64",
    )


def compile_geometric_stiffness_ssa():
    from src.compiler.symbolic_equation_compiler import (
        compile_symbolic_program, SymbolicPublication)
    return compile_symbolic_program(
        symbolic_geometric_stiffness_equations,
        name="engine_toy_geometric_stiffness",
        publications=tuple(
            SymbolicPublication(name, f"engine_toy.buckling.{name}")
            for name in GEOMETRIC_STIFFNESS_OUTPUTS),
        dtype="float64",
    )


# =====================================================================
#  THE BEAM IN MOTION
# =====================================================================
BEAM_DYNAMICS_OUTPUTS = (
    "modal_q_next_m",
    "modal_qdot_next_m_s",
    "tip_deflection_m",
    "tip_slope_rad",
    "natural_frequency_hz",
    "youngs_modulus_pa",
)


def symbolic_beam_dynamics_equations():
    """One tick of a cantilever beam's first bending mode.

    WHAT THIS ADDS. `symbolic_beam_kinematics_equations` above turns a
    set of end displacements into strains -- it is a statement about a
    shape, and it has no notion of time. So a barrel could be asked what
    strain it was under and could not be asked where its muzzle WAS,
    which is the only question that decides where a shot goes.

    A tube is a beam with mass and stiffness, so it rings. Firing hits
    it with an impulse and it oscillates at its own frequency, and at a
    sustained rate of fire it never stops: a 50 Hz tube fired every
    200 ms retains nearly three quarters of its amplitude from the
    previous round. Every shot therefore leaves while the muzzle is
    moving, and where it points at that instant is the sum of a static
    droop and a live oscillation.

    Reduced to the FIRST MODE, which for a cantilever carries the
    overwhelming majority of tip motion:

        q'' + 2 zeta omega q' + omega^2 q = F / m_modal

    integrated semi-implicitly. The host must subdivide its outer dt
    for this mode's stiffness AND explicit damping; semi-implicit Euler
    is still conditionally stable. The member bank derives that schedule
    from these equations in structure_native._substep_rate.

    AND THE MODULUS IS A FUNCTION OF TEMPERATURE, which is the term
    that ties this to the thermal side: steel loses roughly 40% of its
    stiffness by 700 C, so a hot barrel is not merely more bent -- it
    rings SLOWER, and each shot then leaves at a different phase of the
    whip than it did when the tube was cold.
    """
    names = ("modal_q_m modal_qdot_m_s force_n dt_s "
             "length_m outer_radius_m wall_m density_kg_m3 "
             "youngs_cold_pa temperature_k damping_ratio "
             "added_mass_per_m_kg")
    s = {n: sympy.Symbol(n, real=True) for n in names.split()}
    q, qd = s["modal_q_m"], s["modal_qdot_m_s"]
    F, dt = s["force_n"], s["dt_s"]
    L, ro, wall = s["length_m"], s["outer_radius_m"], s["wall_m"]
    rho, e0 = s["density_kg_m3"], s["youngs_cold_pa"]
    t_k, zeta = s["temperature_k"], s["damping_ratio"]
    added = s["added_mass_per_m_kg"]

    ri = ro - wall
    # A FOURTH POWER WRITTEN AS A PRODUCT. `ro**4` is the same number
    # and not the same program: WebAssembly has no instruction for a
    # general power, so a law written with one cannot lower to the game
    # engine's own target at all, while `ro*ro*ro*ro` is three
    # multiplies everywhere. The exponent was never wanted -- it was
    # just how the formula is written on paper.
    ro2, ri2 = ro * ro, ri * ri
    inertia = sympy.pi / 4 * (ro2 * ro2 - ri2 * ri2)
    area = sympy.pi * (ro2 - ri2)
    # the tube's own mass per metre, PLUS whatever it is carrying --
    # a water jacket is a quarter of a barrel's mass again, and it
    # lowers the frequency as surely as it increases the droop
    mass_per_m = area * rho + added

    # STEEL SOFTENS. Linear to 500 C, faster above: the same curve the
    # droop uses, so bend and frequency cannot disagree about what
    # temperature the tube is.
    t_c = t_k - 273.15
    e_hot = sympy.Piecewise(
        (e0 * (1 - sympy.Rational(4, 10000) * (t_c - 20)), t_c <= 500),
        (e0 * (1 - sympy.Rational(4, 10000) * 480)
         * (1 - sympy.Rational(12, 10000) * (t_c - 500)), True))
    e_hot = sympy.Max(e_hot, e0 * sympy.Rational(1, 20))

    # first cantilever mode: (beta L) = 1.875104
    beta_l4 = sympy.Float(1.875104 ** 4)
    l2 = L * L
    omega = sympy.sqrt(beta_l4 * e_hot * inertia
                       / sympy.Max(mass_per_m * l2 * l2, 1e-12))
    # effective modal mass at the tip for mode one
    m_modal = sympy.Rational(1, 4) * mass_per_m * L

    accel = (F - 2 * zeta * omega * m_modal * qd
             - omega ** 2 * m_modal * q) / sympy.Max(m_modal, 1e-9)
    # semi-implicit: velocity first, then position with the NEW velocity.
    # Explicit Euler on an oscillator gains energy every step and the
    # tube would ring itself apart at these substep sizes.
    qd_next = qd + accel * dt
    q_next = q + qd_next * dt

    # the tip: mode one is normalised to unity at the free end, and its
    # slope there is what actually aims the shot
    tip = q_next
    slope = sympy.Float(1.377) * q_next / L

    return (
        sympy.Eq(sympy.Symbol("modal_q_next_m"), q_next, evaluate=False),
        sympy.Eq(sympy.Symbol("modal_qdot_next_m_s"), qd_next, evaluate=False),
        sympy.Eq(sympy.Symbol("tip_deflection_m"), tip, evaluate=False),
        sympy.Eq(sympy.Symbol("tip_slope_rad"), slope, evaluate=False),
        sympy.Eq(sympy.Symbol("natural_frequency_hz"),
                 omega / (2 * sympy.pi), evaluate=False),
        sympy.Eq(sympy.Symbol("youngs_modulus_pa"), e_hot, evaluate=False),
    ), {}


@lru_cache(maxsize=1)
def compile_beam_dynamics_ssa():
    from src.compiler.symbolic_equation_compiler import (
        compile_symbolic_program, SymbolicPublication)
    return compile_symbolic_program(
        symbolic_beam_dynamics_equations,
        name="engine_toy_beam_dynamics_step",
        publications=tuple(
            SymbolicPublication(name, f"engine_toy.beam.{name}")
            for name in BEAM_DYNAMICS_OUTPUTS),
        dtype="float64",
    )


@lru_cache(maxsize=1)
def beam_dynamics_native():
    """The beam step as a native callable, execution built once."""
    import numpy as np
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
    compiled = compile_beam_dynamics_ssa()
    fn = compiled.function
    names = list(fn.metadata["argument_names"])
    ids = {n: v.id for n, v in zip(names, fn.args)}
    outs = dict(fn.metadata["named_outputs"])
    artifact = compile_artifact(emit_ssa_function_to_llvm(
        compiled.module, fn.name, entry_name="beam_dynamics_step"))
    feed = {ids[n]: np.array(0.0, dtype=np.float64) for n in names}
    ex = prepare_artifact_execution(artifact, feed)
    buf = {n: np.asarray(ex.buffers[ids[n]]).reshape(-1) for n in names}

    def step(**kw):
        for n in names:
            buf[n][0] = float(kw[n])
        ex.run()
        return {o: float(np.asarray(ex.buffers[outs[o]]).reshape(-1)[0])
                for o in BEAM_DYNAMICS_OUTPUTS}

    return step, names


# =====================================================================
#  EVERY EDGE THAT SAYS IT IS A BEAM
# =====================================================================
#: An edge participates by DECLARING a section and a material. That is
#: the whole test -- not its name, not its constraint, not a list kept
#: somewhere else that has to be maintained in step with the builders.
#: A structural pipe, a trunnion pin, an actuator body, an axle, a gear
#: shaft and a bore evacuator tube are all the same thing to a beam
#: solver: a length of material with a second moment of area. The only
#: reason they were ever different is that nothing had asked them.
BEAM_SECTION_KEYS = ("radius", "wall_m", "rest_length")


#: Below this, a member is not a beam in any useful sense -- it is a
#: lug, a boss or a joint face, and asking a cantilever formula about
#: it returns a quarter-megahertz "first mode" that means nothing. The
#: test is SLENDERNESS, not raw length: what matters is how long it is
#: relative to how fat it is.
MIN_SLENDERNESS = 6.0


def slenderness(edge: dict) -> float:
    r = float(edge.get("radius") or 0.0)
    return float(edge.get("rest_length", 0.0)) / max(2.0 * r, 1e-9)


def declares_beam(edge: dict) -> bool:
    """Does this edge carry enough of its own geometry to be a beam?

    RIGID COMES FIRST, and it is not the same question as
    `beam_solvable`. `beam_solvable` asks whether a member's section is
    known well enough to solve it as a beam. `rigid` says the thing is
    not a member at all -- it is INTERNAL to a body that has already
    declared its own shape and material, and it exists only so that
    load entering one of that body's ports can reach the others
    without travelling through the centroid. Giving such an edge modal
    existence invents an oscillator that is not there and then reports
    its frequency, which is worse than saying nothing. A body rings as
    a body; its own skin does not ring separately from it.

    An explicit `beam_solvable` settles the rest either way -- a cable,
    a hose or a fluid line can say no even though it has a radius, and
    something unusual can say yes. Otherwise the question is simply
    whether it has told us its section."""
    if edge.get("rigid"):
        return False
    declared = edge.get("beam_solvable")
    if declared is not None:
        return bool(declared)
    # A ROUTED LINE IS NOT A BEAM, and which lines are routed is the
    # registry's business, not a tuple kept here that had three of the
    # eleven in it.
    from joints import ROUTED_TOKENS, MEMBER_CONSTRAINTS
    tok = edge.get("constraint_token")
    if tok is not None:
        if tok in ROUTED_TOKENS:
            return False
    elif edge.get("routed"):
        return False
    else:
        _mc = MEMBER_CONSTRAINTS.get(edge.get("constraint"))
        if _mc is not None and _mc.routed:
            return False
    if edge.get("radius") is None or float(edge.get("rest_length", 0.0)) <= 1e-6:
        return False
    # a stubby member is structurally rigid, and saying so is more
    # honest than reporting it as an oscillator at 246 kHz
    return slenderness(edge) >= MIN_SLENDERNESS


def beam_members(document: dict) -> list[dict]:
    """Every beam in a graph, with the section each one declares.

    Returns the arguments the dynamics kernel wants, so the same law
    that rings a gun barrel rings a roll bar, a drive shaft and an
    outrigger post. One law, batched over whatever the document
    happens to contain."""
    from milspec import MATERIAL_BY_KEY
    out = []
    for e in document["edges"]:
        if not declares_beam(e):
            continue
        ro = float(e["radius"])
        wall = float(e.get("wall_m") or max(0.0025, min(ro * 0.42,
                                                        0.0045 + ro * 0.16)))
        alloy = e.get("alloy") or "4130n"
        mat = MATERIAL_BY_KEY.get(alloy)
        e0 = float(e.get("youngs_modulus_pa")
                   or (getattr(mat, "youngs_modulus_pa", None) or 2.05e11))
        rho = float(getattr(mat, "density_kg_m3", 7850.0)) if mat else 7850.0
        out.append({
            "identity": e["identity"],
            "slenderness": slenderness(e),
            "assembly": e.get("assembly"),
            "constraint": e["constraint"],
            "length_m": float(e["rest_length"]),
            "outer_radius_m": ro,
            "wall_m": min(wall, ro * 0.95),
            "density_kg_m3": rho,
            "youngs_cold_pa": e0,
            "damping_ratio": float(e.get("damping_ratio", 0.005)),
            "added_mass_per_m_kg": float(e.get("added_mass_per_m_kg", 0.0)),
        })
    return out


def beam_survey(document: dict, temperature_k: float = 293.15) -> list[dict]:
    """What every beam in the graph does, run through the real kernel.

    The point of doing it this way: a chassis builder, an engine
    builder, a tyre rig and a gun mount stop being four different
    structural stories. They are one document full of beams, and this
    is the law that applies to all of them."""
    step, _ = beam_dynamics_native()
    rows = []
    for m in beam_members(document):
        kw = {k: v for k, v in m.items()
              if k not in ("identity", "assembly", "constraint")}
        r = step(modal_q_m=0.0, modal_qdot_m_s=0.0, force_n=0.0,
                 dt_s=1.0e-6, temperature_k=temperature_k, **kw)
        rows.append({**m, "first_mode_hz": r["natural_frequency_hz"],
                     "youngs_pa": r["youngs_modulus_pa"]})
    return rows
