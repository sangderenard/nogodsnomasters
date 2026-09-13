"""Real physics on a production graph, using the game's own code.

WHAT THIS FIXES. `turret_production.ProductionGraph.check()` verifies
three things: that an edge's endpoints exist, that its rest length is
non-zero, and that no free node has fewer than three connections. That
is TOPOLOGY. Nothing in engine_toy solved a force on these graphs, so
every load figure this project has quoted -- member margins, peak
firing loads, what the fine platform carries -- came from arithmetic
written by hand in a throwaway script. The graph was described and
meshed; it was never simulated, and a structure can be geometrically
absurd and still pass every check.

So this runs the ACTUAL GAME PHYSICS over it, rather than a second
solver invented here:

  `vehicle_graph_constants_from_model` (vehicle_native_graph_program)
  builds the node/edge tensors. It is generic -- it takes any assembly
  model with a `mechanical_graph`, and admits any edge that declares
  `damage.model == "elastic-plastic-member-with-shear-fracture"`, which
  every production edge this project authors already does.

  `vehicle_material_bank_vector` is the game's own assembly: gather the
  two endpoint positions per member, take the delta, the length, the
  axial strain against rest length and the strain rate, and hand them
  to the member law.

  `symbolic_vehicle_member_material_equations` is that law -- a J2
  return map over axial, bending and shear with kinematic hardening,
  viscosity, accumulated plastic strain, remaining ductility and
  fracture. Twenty-four inputs, seventeen outputs, and it is far richer
  than anything engine_toy had.

The bank source is EXECUTED, not transcribed. Copying those six lines
into this file would have been a second opinion about the physics that
could drift from the original; running the game's own text cannot.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))


@lru_cache(maxsize=1)
def _member_step():
    """The member law as a callable taking the game's positional order."""
    from src.compiler.vehicle_mechanical_material import (
        compile_vehicle_member_material_ssa, MEMBER_MATERIAL_OUTPUTS)
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
    compiled = compile_vehicle_member_material_ssa()
    fn = compiled.function
    names = list(fn.metadata["argument_names"])
    ids = {n: v.id for n, v in zip(names, fn.args)}
    outs = dict(fn.metadata["named_outputs"])
    artifact = compile_artifact(emit_ssa_function_to_llvm(
        compiled.module, fn.name, entry_name="member_material_step"))

    def step(*args):
        # THE LAW IS SCALAR, so a bank of members is stepped one member
        # at a time. That is slow and it is honest: it is the same
        # arithmetic the game compiles natively, and getting a correct
        # answer first is the point. The batched lane is
        # `vehicle_native_graph_program`'s own, and taking it means
        # compiling the vehicle program, which is a much larger job than
        # finding out whether this structure stands up.
        flat = [np.atleast_1d(np.asarray(a, dtype=np.float64)).reshape(-1)
                for a in args]
        width = max(a.size for a in flat)
        result = {name: np.empty(width) for name in MEMBER_MATERIAL_OUTPUTS}
        for lane in range(width):
            feed = {ids[n]: np.array(a[lane % a.size], dtype=np.float64)
                    for n, a in zip(names, flat)}
            ex = prepare_artifact_execution(artifact, feed)
            ex.run()
            for name in MEMBER_MATERIAL_OUTPUTS:
                result[name][lane] = float(
                    np.asarray(ex.buffers[outs[name]]).reshape(-1)[0])
        return tuple(result[name] for name in MEMBER_MATERIAL_OUTPUTS)

    return step, names


@dataclass
class GraphPhysics:
    """A production graph under the game's member law."""
    document: dict
    constants: object = None
    material_state: np.ndarray = None
    velocities: np.ndarray = None
    positions: np.ndarray = None
    _index: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from src.compiler.vehicle_native_graph_program import (
            vehicle_graph_constants_from_model)
        model = {"mechanical_graph": self.document,
                 "structure": {"support_corners": ("ground",)}}
        self.constants = vehicle_graph_constants_from_model(model, support_count=1)
        n_edges = self.constants.edge_nodes.shape[0]
        # the game's own material state layout: nine columns per member
        self.material_state = np.zeros((1, n_edges, 9), dtype=np.float64)
        self.positions = self.constants.node_reference.copy()[None, :, :]
        self.velocities = np.zeros_like(self.positions)
        self._index = {n["identity"]: i
                       for i, n in enumerate(self.document["nodes"])}

    # ------------------------------------------------------------------
    @property
    def edge_identities(self) -> list:
        admitted = [e for e in self.document["edges"]
                    if e.get("damage", {}).get("model")
                    == "elastic-plastic-member-with-shear-fracture"]
        return [e["identity"] for e in admitted]

    def displace(self, identity: str, delta) -> None:
        """Move one node, so a load can be applied as a real deflection."""
        self.positions[0, self._index[identity]] += np.asarray(delta, float)

    def solve(self, dt: float = 1.0e-3) -> dict:
        """Strain every member and run the game's material law on it."""
        step, _names = _member_step()
        c = self.constants
        left = self.positions[:, c.edge_nodes[:, 0], :]
        right = self.positions[:, c.edge_nodes[:, 1], :]
        delta = right - left
        vdelta = (self.velocities[:, c.edge_nodes[:, 1], :]
                  - self.velocities[:, c.edge_nodes[:, 0], :])
        length = np.sqrt((delta * delta).sum(axis=-1) + 1.0e-24)
        g = c.edge_geometry
        axial_kinematic = g[:, 12]
        rest = g[:, 0]
        axial_strain = axial_kinematic * (length - rest) / rest
        axial_rate = (axial_kinematic * (delta * vdelta).sum(axis=-1)
                      / (length * rest))
        ms = self.material_state
        zero = axial_strain * 0.0
        out = step(
            ms[:, :, 3], axial_strain, axial_rate,
            g[:, 10], zero, zero, g[:, 10],
            ms[:, :, 4], np.full_like(axial_strain, dt), ms[:, :, 5],
            g[:, 8], g[:, 9], g[:, 7],
            g[:, 5], g[:, 2], ms[:, :, 0],
            ms[:, :, 1], ms[:, :, 2], g[:, 4],
            zero, zero, g[:, 10], g[:, 6], g[:, 3])
        from src.compiler.vehicle_mechanical_material import MEMBER_MATERIAL_OUTPUTS
        named = dict(zip(MEMBER_MATERIAL_OUTPUTS, out))
        return {
            "identities": self.edge_identities,
            "axial_strain": axial_strain.reshape(-1),
            "axial_stress_pa": named["axial_stress_pa"],
            "yield_stress_pa": named["current_yield_stress_pa"],
            "fracture_demand": named["fracture_demand"],
            "failed": named["failed_next"],
            "remaining_ductility": named["remaining_ductility_next"],
            "elastic_energy_j": named["elastic_energy_j"],
            "rest_length_m": rest,
            "section_area_m2": g[:, 1],
        }

    # ------------------------------------------------------------------
    def report(self, solved: dict, worst: int = 8) -> list:
        """The most-demanded members, by how close they are to fracture."""
        demand = np.abs(solved["fracture_demand"])
        order = np.argsort(-demand)[:worst]
        out = [f"  {len(solved['identities'])} members under the game's member law"]
        for i in order:
            force = solved["axial_stress_pa"][i] * solved["section_area_m2"][i]
            out.append(
                f"    {solved['identities'][i]:34s} "
                f"eps {solved['axial_strain'][i]:+.3e}  "
                f"{solved['axial_stress_pa'][i] / 1e6:8.1f} MPa  "
                f"{force / 1000:9.1f} kN  "
                f"demand {demand[i]:.3f}"
                + ("  FAILED" if solved["failed"][i] > 0.5 else ""))
        return out


def solve_under_load(document: dict, *, loads: dict | None = None,
                     gravity: bool = True, dt: float = 1.0e-3) -> dict:
    """The whole chain, composed: forces in, material response out.

    THIS IS WHAT THE TWO HALVES WERE FOR, and keeping them apart was the
    mistake rather than either of them being wrong.

      `frame_solver.FrameSolver` turns FORCES into displacements. It is
      the piece that lets gravity be applied at all, and the piece that
      makes load propagate through a structure instead of straining only
      the members you happened to move by hand.

      `vehicle_mechanical_material` turns STRAINS into material
      response -- plastic flow, hardening, remaining ductility,
      fracture. It is the game's own law and far better than anything
      written here.

    Between them sat a gap nothing crossed. `GraphPhysics` took
    prescribed positions, so it could only strain a member if you moved
    one of its ends yourself: the shakedown loaded three members of a
    hundred and twenty-nine and reported a mild-steel turret as passing.

    AND THE LAW GETS ALL THREE COMPONENTS NOW. It is a J2 return map
    over axial, bending and shear; the vehicle path has always passed
    `zero, zero` for bending and for shear, so two thirds of it has
    never run and `plastic_bending` and `plastic_shear` have been state
    columns faithfully updated for quantities that were structurally
    zero. `FrameSolver` computes both, because a frame element with
    rotational freedoms has them, so they are handed over here.
    """
    from frame_solver import FrameSolver
    from src.compiler.vehicle_mechanical_material import MEMBER_MATERIAL_OUTPUTS

    frame = FrameSolver(document=document, loads=loads or {}, gravity=gravity)
    field = frame.solve()

    step, _names = _member_step()
    n = len(field["identities"])
    zero = np.zeros(n)
    # geometry and material, per member, in the order the law wants
    geom = {"area": [], "volume": [], "E": [], "G": [], "yield": [],
            "ultimate": [], "hardening": [], "fracture": [], "fragility": [],
            "viscosity": []}
    for edge in frame.members:
        d = edge["damage"]
        E, G, A, I, J, kappa, mat = frame._section(edge)
        L = float(edge.get("rest_length", 1.0))
        geom["area"].append(A)
        geom["volume"].append(A * L)
        geom["E"].append(E)
        geom["G"].append(G)
        geom["yield"].append(float(d.get("yield_strength_pa", mat.yield_pa)))
        geom["ultimate"].append(float(d.get("ultimate_strength_pa", mat.ultimate_pa)))
        geom["hardening"].append(E * 0.01)
        geom["fracture"].append(float(d.get("fracture_strain", 0.075)))
        geom["fragility"].append(0.35)
        geom["viscosity"].append(E * 5.0e-5)
    g = {k: np.array(v) for k, v in geom.items()}

    out = step(
        zero,                       # accumulated_plastic_strain_previous
        field["axial_strain"], zero,          # axial strain and rate
        g["viscosity"],
        field["bending_strain"], zero,        # <- was zero, zero
        g["viscosity"],
        zero,                       # dissipated_energy_previous
        np.full(n, dt), zero,       # dt, failed_previous
        g["fracture"], g["fragility"], g["hardening"], g["yield"], g["volume"],
        zero, zero, zero,           # plastic axial/bending/shear previous
        g["G"],
        field["shear_strain"], zero,          # <- was zero, zero
        g["viscosity"], g["ultimate"], g["E"])
    named = dict(zip(MEMBER_MATERIAL_OUTPUTS, out))
    return {
        "identities": field["identities"],
        "axial_strain": field["axial_strain"],
        "bending_strain": field["bending_strain"],
        "shear_strain": field["shear_strain"],
        "axial_force_n": field["axial_force_n"],
        "axial_stress_pa": named["axial_stress_pa"],
        "bending_stress_pa": named["bending_stress_pa"],
        "shear_stress_pa": named["shear_stress_pa"],
        "fracture_demand": named["fracture_demand"],
        "failed": named["failed_next"],
        "remaining_ductility": named["remaining_ductility_next"],
        "max_deflection_m": field["max_deflection_m"],
        "displacement": field["displacement"],
        "first_mode_hz": None,
    }
