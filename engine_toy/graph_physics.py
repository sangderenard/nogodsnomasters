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

import math
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))


class GraphJointForces:
    """Evaluate graph-declared recoil elements and scatter their forces.

    This owns no body state and performs no integration.  It is the shared
    boundary between constitutive elements and whichever integrator owns the
    graph's masses.  An element may couple to several guide edges; that is how
    a recoil equalizer loads both slide carriages without concentrating the
    whole reaction at the breech node.
    """

    def __init__(self, document: dict, *, linear_embedded: bool = False):
        from structure_native import (linear_spring_joints, oleo_force_joints,
                                      joint_bank_for)

        self.document = document
        self.linear_embedded = bool(linear_embedded)
        self.index = {node["identity"]: i
                      for i, node in enumerate(document["nodes"])}
        self.edge_by_id = {edge["identity"]: edge
                           for edge in document["edges"]}
        self.passive_edges = oleo_force_joints(document)
        self.linear_edges = linear_spring_joints(document)
        self.mr_edges = [edge for edge in document["edges"]
                         if edge.get("kind") == "magnetorheological"]
        self.bump_stop_edges = [edge for edge in document["edges"]
                                if edge.get("kind") == "bump-stop"]
        self.edges = [*self.passive_edges, *self.linear_edges, *self.mr_edges,
                      *self.bump_stop_edges]
        self.bank = joint_bank_for(document)
        self.rest_force = self.bank.step(0.0)[:, -1].copy()
        self.passive_effective_mass_kg = np.ones(self.bank.n, dtype=float)
        self.bump_effective_mass_kg = np.ones(len(self.bump_stop_edges),
                                              dtype=float)
        self.last_element_force = {
            edge["identity"]: 0.0 for edge in self.edges}
        self.last_step_limiter = None

    def _position_forward_stop(self, stage: str, command: float) -> None:
        """Place an adjustable stop for the requested stage preload.

        The stop reaches its full-forward set position at the stage's normal
        installed preload, not only at the ram's emergency maximum.  Below
        that setting it withdraws proportionally so a zero-force command
        genuinely releases the swing to hang.
        """
        for edge in self.bump_stop_edges:
            if (edge.get("part_role") != "adjustable-forward-preload-stop"
                    or edge.get("stage") != stage):
                continue
            nominal = float(edge["nominal_clearance_m"])
            release = float(edge.get("release_adjustment_m", 0.0))
            installed = max(float(edge.get(
                "installed_preload_command_frac", 0.0)), 1.0e-12)
            closure = min(1.0, max(0.0, command) / installed)
            edge["clearance_m"] = nominal - (1.0 - closure) * release
            edge["preload_command_frac"] = command

    def command_platform_preload(self, fraction: float) -> None:
        """Command both swing stages as a fraction of each ram's capacity."""
        command = min(1.0, max(0.0, float(fraction)))
        stages = set()
        for edge in self.linear_edges:
            if edge.get("part_role") != "platform-actuator":
                continue
            low = float(edge.get("minimum_preload_n", 0.0))
            high = float(edge.get("maximum_preload_n",
                                  edge.get("holding_force_n", low)))
            edge["commanded_preload_n"] = low + command * (high - low)
            edge["preload_command_frac"] = command
            stages.add(str(edge.get("stage")))
        for stage in stages:
            self._position_forward_stop(stage, command)

    def command_platform_preload_force(self, force_n: float,
                                       stage: str | None = None) -> dict[str, float]:
        """Press selected swing stage(s) into their forward stop in newtons.

        ``force_n`` is the total force across the stage.  It is divided evenly
        between that stage's parallel hydraulic rams and clamped only by
        their declared combined capacity.  The returned mapping reports the
        force actually commanded for each selected stage.
        """
        requested = max(0.0, float(force_n))
        grouped: dict[str, list[dict]] = {}
        for edge in self.linear_edges:
            if edge.get("part_role") != "platform-actuator":
                continue
            identity = str(edge.get("stage"))
            if stage is None or identity == stage:
                grouped.setdefault(identity, []).append(edge)
        if stage is not None and stage not in grouped:
            raise KeyError(f"no platform actuators for stage {stage!r}")
        applied = {}
        for identity, actuators in grouped.items():
            capacity = sum(float(edge.get(
                "maximum_preload_n", edge.get("holding_force_n", 0.0)))
                           for edge in actuators)
            total = min(requested, capacity)
            share = total / len(actuators)
            for edge in actuators:
                high = float(edge.get(
                    "maximum_preload_n", edge.get("holding_force_n", 0.0)))
                commanded = min(share, high)
                edge["commanded_preload_n"] = commanded
                edge["preload_command_frac"] = (
                    commanded / high if high > 0.0 else 0.0)
            # Equal rams currently have equal capacities.  Use the achieved
            # total so this remains correct if a future graph derates one.
            achieved = sum(float(edge["commanded_preload_n"])
                           for edge in actuators)
            fraction = achieved / capacity if capacity > 0.0 else 0.0
            self._position_forward_stop(identity, fraction)
            applied[identity] = achieved
        return applied

    @staticmethod
    def _stop_penetration(edge: dict, separation: float) -> float:
        gap = float(edge["clearance_m"])
        if edge.get("contact_side") == "maximum-separation":
            return separation - gap
        return gap - separation

    def _paths(self, edge: dict) -> list[tuple[int, int, float, object]]:
        coupled = tuple(edge.get("coupled_slide_edges") or ())
        if not coupled:
            return [(self.index[edge["a"]], self.index[edge["b"]], 1.0,
                     edge.get("slide_axis"))]
        shares = tuple(float(x) for x in
                       (edge.get("slide_load_share") or ()))
        if len(shares) != len(coupled) or not np.isclose(sum(shares), 1.0):
            raise ValueError(
                f"{edge['identity']}: slide_load_share must sum to one")
        return [(self.index[self.edge_by_id[name]["a"]],
                 self.index[self.edge_by_id[name]["b"]], shares[i],
                 self.edge_by_id[name].get("slide_axis"))
                for i, name in enumerate(coupled)]

    @staticmethod
    def _path_axis(position: np.ndarray, ia: int, ib: int,
                   declared_axis=None) -> np.ndarray:
        if declared_axis is not None:
            axis = np.asarray(declared_axis, float)
            length = float(np.linalg.norm(axis))
            if length > 1.0e-12:
                return axis / length
        axis = position[ib] - position[ia]
        length = float(np.linalg.norm(axis))
        return axis / length if length > 1.0e-12 else np.zeros(3)

    def _compression_velocity(self, edge: dict, position: np.ndarray,
                              velocity: np.ndarray) -> float:
        # Every coupled guide sees the same ideal slide coordinate.  Average
        # their measured rates so tiny elastic differences do not make one
        # carriage dictate the damper command.
        values = []
        for ia, ib, _share, declared_axis in self._paths(edge):
            axis = self._path_axis(position, ia, ib, declared_axis)
            values.append(-float(axis @ (velocity[ib] - velocity[ia])))
        return float(np.mean(values)) if values else 0.0

    def _load_passive_velocities(self, position: np.ndarray,
                                 velocity: np.ndarray) -> None:
        for i, edge in enumerate(self.passive_edges):
            self.bank.arrays["velocity_m_s"][i] = \
                self._compression_velocity(edge, position, velocity)

    def coupled_step_size(self, maximum_dt: float, positions,
                          velocities) -> float:
        """Travel-limited interval shared by joints and their owning bodies."""
        position = np.asarray(positions, float).reshape(len(self.index), 3)
        velocity = np.asarray(velocities, float).reshape(len(self.index), 3)
        self._load_passive_velocities(position, velocity)
        _count, h = self.bank.coupled_substep_plan(
            float(maximum_dt), self.passive_effective_mass_kg)
        count = max(1, int(math.ceil(float(maximum_dt) / max(float(h), 1e-30))))
        self.last_step_limiter = None
        if self.bank.n:
            from structure_native import STEP_OF_STROKE
            a = self.bank.arrays
            travel_count = np.ceil(
                np.abs(a["velocity_m_s"]) * float(maximum_dt)
                / (np.maximum(a["stroke_m"], 1.0e-6)
                   * STEP_OF_STROKE))
            gas_rate, damping_rate = self.bank.coupled_rate_components(
                self.passive_effective_mass_kg)
            gas_count = np.ceil(float(maximum_dt) * gas_rate * 4.0)
            damping_count = np.ceil(float(maximum_dt) * damping_rate * 4.0)
            requirements = np.vstack((travel_count, gas_count,
                                      damping_count))
            kind_i, joint_i = np.unravel_index(
                int(np.argmax(requirements)), requirements.shape)
            kinds = ("travel", "gas-gradient", "orifice-gradient")
            self.last_step_limiter = {
                "identity": self.passive_edges[int(joint_i)]["identity"],
                "kind": kinds[int(kind_i)],
                "required_substeps": int(requirements[kind_i, joint_i]),
                "velocity_m_s": float(a["velocity_m_s"][joint_i]),
                "effective_mass_kg": float(
                    self.passive_effective_mass_kg[joint_i]),
            }
        for edge, mass in zip(self.bump_stop_edges,
                              self.bump_effective_mass_kg):
            coordinate, coordinate_rate = self._stop_coordinate(
                edge, position, velocity)
            penetration = max(0.0, self._stop_penetration(
                edge, coordinate))
            if penetration <= 0.0:
                continue
            tangent = (float(edge.get("linear_stiffness_n_per_m", 0.0))
                       + 3.0 * float(edge.get(
                           "cubic_stiffness_n_per_m3", 0.0))
                       * penetration ** 2)
            mean_closing = -coordinate_rate
            if edge.get("contact_side") == "maximum-separation":
                mean_closing = coordinate_rate
            damping = (float(edge.get("compression_damping_n_s_per_m", 0.0))
                       if mean_closing > 0.0 else 0.0)
            rate = math.sqrt(max(tangent, 0.0) / max(mass, 1e-6)) \
                + damping / max(mass, 1e-6)
            if math.isfinite(rate):
                # Contact is the sharpest law in the graph. Sixteen force/
                # body exchanges per local contact time constant preserve the
                # progressive impact without force clipping or restitution
                # tuning.
                bump_count = math.ceil(float(maximum_dt) * rate * 16.0)
                if bump_count > count:
                    self.last_step_limiter = {
                        "identity": edge["identity"],
                        "kind": "bump-contact-gradient",
                        "required_substeps": int(bump_count),
                        "velocity_m_s": float(coordinate_rate),
                        "effective_mass_kg": float(mass),
                    }
                count = max(count, bump_count)
        return float(maximum_dt) / count

    def configure_coupled_mass(self, nodal_mass, fixed_nodes=()) -> None:
        """Set the reduced endpoint mass seen by each banked joint."""
        mass = np.asarray(nodal_mass, float).reshape(len(self.index), -1)[:, :3]
        fixed = set(fixed_nodes)
        def reduced_mass(edge):
            inverse_mass = 0.0
            for ia, ib, share, _axis in self._paths(edge):
                a_name = self.document["nodes"][ia]["identity"]
                b_name = self.document["nodes"][ib]["identity"]
                if a_name not in fixed:
                    inverse_mass += share * share / max(float(np.mean(mass[ia])), 1.0e-6)
                if b_name not in fixed:
                    inverse_mass += share * share / max(float(np.mean(mass[ib])), 1.0e-6)
            return 1.0 / max(inverse_mass, 1.0e-12)

        self.passive_effective_mass_kg = np.asarray(
            [reduced_mass(edge) for edge in self.passive_edges], dtype=float)
        self.bump_effective_mass_kg = np.asarray(
            [reduced_mass(edge) for edge in self.bump_stop_edges], dtype=float)

    def _scatter(self, nodal: np.ndarray, edge: dict, force: float,
                 position: np.ndarray) -> None:
        paths = self._paths(edge)
        signs = tuple(float(x) for x in
                      (edge.get("coupled_motion_sign") or ()))
        for path_i, (ia, ib, share, declared_axis) in enumerate(paths):
            axis = self._path_axis(position, ia, ib, declared_axis)
            sign = signs[path_i] if len(signs) == len(paths) else 1.0
            reaction = axis * (force * share * sign)
            nodal[ia] -= reaction
            nodal[ib] += reaction

    def _stop_coordinate(self, edge: dict, position: np.ndarray,
                         velocity: np.ndarray) -> tuple[float, float]:
        """Signed common travel of every face in an equalized stop."""
        paths = self._paths(edge)
        refs = tuple(float(x) for x in
                     (edge.get("coupled_reference_separation_m") or ()))
        signs = tuple(float(x) for x in
                      (edge.get("coupled_motion_sign") or ()))
        coordinate = []
        coordinate_rate = []
        for i, (ia, ib, _share, declared_axis) in enumerate(paths):
            axis = self._path_axis(position, ia, ib, declared_axis)
            separation = float(axis @ (position[ib] - position[ia]))
            rate = float(axis @ (velocity[ib] - velocity[ia]))
            reference = refs[i] if len(refs) == len(paths) else 0.0
            sign = signs[i] if len(signs) == len(paths) else 1.0
            coordinate.append(sign * (separation - reference))
            coordinate_rate.append(sign * rate)
        return float(np.mean(coordinate)), float(np.mean(coordinate_rate))

    def _linear_spring_force(self, edge: dict, position: np.ndarray,
                             velocity: np.ndarray) -> float:
        """Complete force from a graph-declared linear spring/damper."""
        paths = self._paths(edge)
        if not paths:
            return 0.0
        compression = []
        for ia, ib, _share, declared_axis in paths:
            axis = self._path_axis(position, ia, ib, declared_axis)
            separation = float(axis @ (position[ib] - position[ia]))
            rest = float(edge.get("commanded_rest_length_m",
                                  edge["rest_length"]))
            compression.append(rest - separation)
        compression_m = float(np.mean(compression))
        compression_velocity = self._compression_velocity(
            edge, position, velocity)
        rate = float(edge.get(
            "stiffness_n_per_m",
            edge.get("spring_rate_n_per_m",
                     edge.get("stack_rate_n_per_m", 0.0))))
        common_damping = float(edge.get("linear_damping_n_s_per_m", 0.0))
        if compression_velocity >= 0.0:
            damping = float(edge.get("compression_damping_n_s_per_m",
                                     common_damping))
        else:
            damping = float(edge.get("rebound_damping_n_s_per_m",
                                     common_damping))
        preload = float(edge.get(
            "commanded_preload_n",
            edge.get("preload_force_n",
                     edge.get("spring_preload_n",
                              edge.get("preload_n", 0.0)))))
        force = (preload + rate * compression_m
                 + damping * compression_velocity)
        if edge.get("tension_only"):
            return min(0.0, force)
        return force

    def _linear_damping_force(self, edge: dict, position: np.ndarray,
                              velocity: np.ndarray) -> float:
        compression_velocity = self._compression_velocity(
            edge, position, velocity)
        common = float(edge.get("linear_damping_n_s_per_m", 0.0))
        damping = float(edge.get(
            "compression_damping_n_s_per_m" if compression_velocity >= 0.0
            else "rebound_damping_n_s_per_m", common))
        return damping * compression_velocity

    @staticmethod
    def _mr_force(edge: dict, compression_velocity: float) -> float:
        if abs(compression_velocity) <= 1.0e-12:
            return 0.0
        area = float(edge["piston_area_m2"])
        gap = max(float(edge["gap_m"]), 1.0e-9)
        active = float(edge.get("active_length_m", 0.060))
        current = min(1.0, max(0.0, float(edge.get("current_frac", 0.0))))
        tau_lo = float(edge.get("yield_min_pa", 1.0e3))
        tau_hi = float(edge.get("yield_max_pa", 6.0e4))
        tau = tau_lo + (tau_hi - tau_lo) * current
        yield_force = 2.0 * tau * active / gap * area
        viscosity = float(edge.get("plastic_viscosity_pa_s", 0.28))
        flow = area * abs(compression_velocity)
        dp = (12.0 * viscosity * active * flow
              / (math.pi * 0.12 * gap ** 3))
        magnitude = yield_force + dp * area
        return math.copysign(magnitude, compression_velocity)

    def _bump_stop_force(self, edge: dict, position: np.ndarray,
                         velocity: np.ndarray) -> float:
        """Unilateral progressive force after the physical gap closes."""
        coordinate, coordinate_rate = self._stop_coordinate(
            edge, position, velocity)
        penetration = self._stop_penetration(edge, coordinate)
        # A declared-open contact assembled from decimal node coordinates
        # can differ from its declared clearance by a few ulps.  That is not
        # physical penetration and must not manufacture a nanonewton contact.
        gap = float(edge["clearance_m"])
        if penetration <= max(1.0e-12, abs(gap) * 1.0e-12):
            return 0.0
        closing_speed = max(0.0, -coordinate_rate)
        direction = 1.0
        if edge.get("contact_side") == "maximum-separation":
            closing_speed = max(0.0, coordinate_rate)
            direction = -1.0
        linear = float(edge.get("linear_stiffness_n_per_m", 0.0))
        cubic = float(edge.get("cubic_stiffness_n_per_m3", 0.0))
        damping = float(edge.get("compression_damping_n_s_per_m", 0.0))
        return direction * (linear * penetration + cubic * penetration ** 3
                            + damping * closing_speed)

    def evaluate(self, dt: float, positions, velocities, *, coupled=False) -> np.ndarray:
        """Return balanced nodal forces for the current graph state."""
        position = np.asarray(positions, float).reshape(len(self.index), 3)
        velocity = np.asarray(velocities, float).reshape(len(self.index), 3)
        nodal = np.zeros_like(position)

        self._load_passive_velocities(position, velocity)
        advanced = (self.bank.step_coupled(float(dt)) if coupled
                    else self.bank.step(float(dt)))
        passive = advanced[:, -1] - self.rest_force
        if not np.isfinite(passive).all():
            bad = int(np.flatnonzero(~np.isfinite(passive))[0])
            raise FloatingPointError(
                f"{self.passive_edges[bad]['identity']}: nonfinite joint force")
        for edge, force in zip(self.passive_edges, passive):
            value = float(force)
            self.last_element_force[edge["identity"]] = value
            self._scatter(nodal, edge, value, position)

        for edge in self.linear_edges:
            force = self._linear_spring_force(edge, position, velocity)
            self.last_element_force[edge["identity"]] = force
            # LiveStructure's frame matrix contains this linear spring's
            # stiffness and its static solve contains the installed preload.
            # Only directional damping remains an explicit velocity law.
            if self.linear_embedded:
                installed = float(edge.get(
                    "preload_force_n",
                    edge.get("spring_preload_n",
                             edge.get("preload_n", 0.0))))
                commanded = float(edge.get("commanded_preload_n", installed))
                applied = (self._linear_damping_force(edge, position, velocity)
                           + commanded - installed)
            else:
                applied = force
            self._scatter(nodal, edge, applied, position)

        for edge in self.mr_edges:
            speed = self._compression_velocity(edge, position, velocity)
            force = self._mr_force(edge, speed)
            self.last_element_force[edge["identity"]] = force
            self._scatter(nodal, edge, force, position)
        for edge in self.bump_stop_edges:
            force = self._bump_stop_force(edge, position, velocity)
            self.last_element_force[edge["identity"]] = force
            self._scatter(nodal, edge, force, position)
        return nodal


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
